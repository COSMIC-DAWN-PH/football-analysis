# PLAN：足球检测优化（Ball Detection & Tracking）

> 状态：**待开始**
> 分支：`feature/pitch-goal-detection`（本文件所在分支）
> 前置计划：`TEAM_COLOR_CLASSIFICATION_DONE.md`、`REFEREE_DETECTION_DONE.md`（已验收关闭）；`YOLO_FINETUNE.md`（对象四类微调，与本研究并行、数据可共享）

数据源：`C:\Personal Profile\Profile\Video\` 共 6 部——`raw1.mp4` / `raw2.mp4`（同场上下半场，各 30min，前两轮已跑全量 tracks）、`demo1~4.mp4`（新场次，demo2 前两轮已用，demo1/3/4 本轮首次进入）。
硬件：Core Ultra 5 125H（Arc iGPU / NPU / CPU，无 CUDA）。

## 1. 背景与问题

现有球链路：`BallDetector`（`ball-detection.pt`，整帧 1920 + 2×2 重叠 tile 1280 双通道推理，`conf=0.02` 高召回 + NMS iou=0.5 + 宽高比 0.25~4 过滤）→ `BallTracker`（双卡尔曼 image/metric + track segment 分段 + 3 帧确认 + 0.15s 重启门 + 0.5s 预测桥接 + 尺寸上限 48px/5%）→ `BallToPlayerAssigner`（2.0m 归属 + 0.5s grace）。

前两轮（队色、裁判）只优化了颜色层，**球链路从未被系统评估过**。已知/预期问题：

| # | 现象 | 可疑环节 |
|---|------|----------|
| A | 假阳性（白鞋/白袜/场地线/灯光/头/手套被画成球三角） | conf=0.02 高召回 + 无硬负样本 |
| B | 漏检（远场小球 <10px、模糊、被身体遮挡） | 检测层召回不足 + tile 通道尺寸 |
| C | 轨迹断段（同一趟球被切成多段）与虚接（跳变候选接上错误新段） | restart_gap=0.15s / confirmations=3 / 运动门参数 |
| D | 球框闪烁、粘在球员脚上（归属错） | 尺寸门/运动门/归属距离 2.0m |
| E | 无任何量化基线，参数全靠经验值 | 缺评估设施 |

**结论：球链路需要一轮完整的"全量人工标注 → 基线量化 → 参数级修复 → 数据级微调"闭环，方法论沿用前两轮验证门，但人工审核范围扩大为 100%。**

## 2. 本轮相对前两轮的提升点（回答"如何提升"）

1. **全量人工审核，无抽样偏差**：上轮 referee 的 raw 部分只人工核了 116 张精选 + 9 张分歧，其余靠双模型共识；本轮**所有提取出来的球 crop 图片全部人工审核**（口径见 §5），`manual_label` 即权威 ground truth，精确率/召回率可真正统计。
2. **补上 FN 盲区**：上轮只评了"检出来的对不对"（precision），几乎没评"该检的漏没漏"（recall）。本轮增加**漏检扫描**：双模型在无检测帧上找"像球但被漏"的候选，同样进人工审核 → 召回率有真实度量。
3. **track 级指标**：球与裁判不同，观感问题主要在轨迹连续性。新增断段率、虚接率、确认延迟指标，直接对应"球框闪断/粘鞋"等用户可见问题。
4. **误差模式分类驱动修复**：每个 FN/FP 标注归因（远/小/模糊/遮挡/鞋袜/线/灯/头），修复按归因命中率排序，不做盲调。
5. **双模型预筛 + 仲裁降本**：qwen3.7plus 与 gpt5.6luna 先各预标一遍（人只做确认/纠错），分歧样本优先审，人工审核 100% 覆盖但单张耗时大幅下降。
6. **6 视频多场次**：raw1/2（同场）供训练/回归，demo1~4（多场次、含 demo1/3/4 全新场）做未见过场次评估与微调 test，split 纪律沿用 training/README。
7. **产出 ball 类微调数据集**（含硬负样本），对齐 training/README 门槛（≥600 球正样本、≥75% recall / ≥90% precision 才 promote 权重）。

## 3. 目标

- 全量人工标注集定稿：6 视频、所有候选 crop 100% 人工审核；
- crop 级：precision/recall 较基线显著提升（目标 P≥90%、R≥75%，沿用 training/README 口径）；
- track 级：断段率、虚接率下降，确认延迟下降，归属错配抽样下降；
- 前两轮成果不回退：referee 标注集、34-crop balanced acc=1.0、pytest 全绿；
- 端到端速度下降 ≤ 20%。

## 4. 工作方式（沿用前两轮验证门，模型名按本轮实际）

1. 一个 Phase 一次推进，单独 commit；Phase 开始前记基线，完成后同口径对比；
2. 实现模型：`opencode-go/deepseek-v4-pro`；验证模型：`opencode-go/qwen3.7plus` 与 `opencode-go/gpt5.6luna`（用户指定，代替上轮 kimi/luna 组合；**执行时发现 qwen3.7plus 在 openchamber 中不可用，经用户确认改用 `opencode-go/kimi-k2.7-code` 代替**）；
3. 验证模型**不得只读实现方结论**：必须自己看图核对判定、自己跑 eval 复算指标、复核 diff，给出"通过/不通过 + 理由"；
4. 标注环节：双模型只做**预标与仲裁辅助**，`manual_label`（人工）为唯一权威；分歧样本（双模型不一致）必须人工裁决；
5. 全部 Phase 完成后 qwen3.7plus 与 gpt5.6luna 各一次独立交叉复核，均通过才算整体完成；
6. 全程记录到本文档「推进记录」。

## 5. 人工审核口径（本轮核心变化：100% 审核）

**审核集合 = 以下三类 crop 的并集**（`tools/extract_ball_candidates.py` 产出）：

| 类别 | 内容 | 每段张数 |
|------|------|----------|
| confirmed | 已确认 track segment 的最清晰帧（小图放大、帧距 ≥15） | ≤3 张/segment |
| unconfirmed | 未确认的孤立候选/被运动门拒绝的候选（FP 高发区，**必审**） | 1 张/候选 |
| fn_sweep | 双模型在无检测帧上扫出的"像球但漏检"候选（FN 高发区，**必审**） | 1 张/候选 |

- 标注值：`ball` / `not_ball`（白鞋、袜、线、灯、头、手套等硬负样本写具体类别）/ `null`（太小、模糊到无法判定，不进评估但保留）；
- 双模型预填 `qwen_label` / `luna_label` 后，人工**逐张确认或纠错**，分歧样本优先审；
- 定稿产物：`eval/ball_crops/{src}/final_ball_labels.jsonl`（权威），附审核进度清单（缺审即验收不通过）。

## 6. Phase 划分

### Phase 0 — 全量检测+候选提取+双模型预标
- `tools/detect_ball_tracks.py`（新增）：6 视频全量球检测+跟踪（不含球员/关键点，GPU/NPU 并行），写 `raw/ball_tracks.jsonl`（含 segment/confidence/observed 全字段，供后续一切评估）；
- `tools/extract_ball_candidates.py`（新增）：按 §5 生成三来源候选 crop + auto 预填清单（conf、segment、尺寸、距预测点偏差等特征列）；
- `tools/prefill_ball_labels.py`（新增）：qwen3.7plus、gpt5.6luna 各写一份预标 jsonl，输出共识/分歧两份清单，分歧优先审；
- 分支 commit；产出即 Phase 1 审核材料。

### Phase 1 — 全量人工审核（用户执行，本轮不做抽样）
- 用户审完全部 crop（§5 集合），`final_ball_labels.jsonl` 定稿；
- 双模型对分歧样本出仲裁建议，用户最终裁决；
- 交付：标注定稿 commit（crop 图片按上轮惯例不入 git，清单+标签入）。

### Phase 2 — 基线评测与误差归因
- `tools/eval_ball.py`（新增）：在 final_ball_labels 上复算 crop 级 precision/recall/F1 + track 级（段召回、断段率、虚接率、确认延迟）+ 归属错配抽样报告；误差模式归因表（FN：远/小/模糊/遮挡；FP：鞋/袜/线/灯/头/手套/其他）；
- 记录基线数字到推进记录，作为所有后续 Phase 对比口径。

### Phase 3 — 参数级修复（按归因排序，逐项独立 commit + 验证门）
- 候选修复项（以实测归因为准，先按优先级）：conf 自适应/NMS 收紧（FP）；tile 重叠或分辨率调整、最小尺寸下限（FN）；尺寸门 48px/5% 与宽高比复核（FP/归属）；restart_gap / confirmations / max_prediction / 运动门距离曲线（断段/虚接）；归属距离与 grace（粘脚）；
- 每项修复后：`eval_ball.py` 同口径对比 + qwen3.7plus 或 gpt5.6luna 验证门裁决（不通过即回退重做）。

### Phase 4 — ball 微调数据集 + 云端训练（数据侧本轮必产出，训练可与 YOLO_FINETUNE 并行）
- `tools/build_ball_finetune_dataset.py`（新增/复用 build_finetune_dataset）：以 final_ball_labels 回填正确框；`ball` 正样本 + `not_ball` 硬负样本（鞋/袜/线/灯/头/手套）打包成 YOLO 二分类数据集；split 纪律：整场视频不跨 split、demo1~4 至少一场全新进 test、相邻帧不跨 split；
- 量级对齐 training/README：≥600 球正样本 + 硬负样本 ≥300；
- CUDA 云机 `python -m training.train_models --task ball`；未达 75%/90% 门槛不 promote（仅回传指标）。

### Phase 5 — 终验
- 6 视频全量重放：demo1~4 未见过场次看效果 + raw1/2 回归；track 级指标与归属抽看；老验收（referee 标注集、34-crop balanced acc、pytest）不回退；端到端速度 ≤20% 回退；
- qwen3.7plus + gpt5.6luna 双模型交叉复核（看图、复算、diff）均通过后关闭。

## 7. 涉及文件

| 文件 | 改动 |
|------|------|
| `tools/detect_ball_tracks.py` / `extract_ball_candidates.py` / `prefill_ball_labels.py` / `make_ball_sheets.py` / `run_ball_batch.ps1` / `eval_ball.py` / `build_ball_finetune_dataset.py` | 新增（Phase 0/2/4；网格图批量审核工具 make_ball_sheets 为本轮新增） |
| `tracking/ball_tracker.py` | Phase 3 参数级修复（NMS/尺寸门/运动门/确认与重启参数） |
| `ball_to_player_assignment/ball_to_player_assigner.py` | 归属距离/grace 复核（如归因命中） |
| `eval/ball_crops/{demo1..4,raw1,raw2}/` | 候选 crop + final_ball_labels.jsonl（图片不入 git） |
| `plan/BALL_DETECTION.md` | 本文档（推进记录） |
| 其余文件 | 不动；对外接口（BallDetector/BallTracker/assigner 构造签名）保持兼容 |

## 8. 验收标准

1. `final_ball_labels.jsonl` 覆盖 §5 三类候选且**审核进度 100%**（无未审项）；
2. crop 级 precision ≥90%、recall ≥75%（未达则须较基线显著提升并说明差距归因）；
3. track 级：断段率/虚接率较基线下降，确认延迟不劣化；
4. 前两轮回归：referee 标注集指标、34-crop balanced acc=1.0、全量 pytest 绿；
5. 端到端处理速度下降 ≤20%；
6. 双模型（qwen3.7plus / gpt5.6luna）交叉复核均通过。

## 9. 风险与对策

- **全量候选量大**（conf=0.02 高召回 + 6 视频）→ 段级聚合去重（每 segment ≤3 张）+ unconfirmed 只取每帧最强候选；仍超人工预算时按 §5 优先级顺序审（unconfirmed/fn_sweep 优先），confirmed 按清晰度降序——但清单完整性（每个 segment/候选至少 1 张）不妥协；
- **远场小球人无法判定** → `null` 类不进评估但保留，标注规则先定稿；recall 指标只算"可见可判定"样本；
- **双模型对球判定本身不可靠** → 模型标签仅作预填与仲裁建议，人工为唯一权威；分歧率过高时降低模型预标权重（纯排序用）；
- **raw 与 demo 光照/场地差异** → 参数修复用 6 视频全量验证，不以单场定参；
- **FN 扫描成本** → 双模型仅扫"无检测帧"的降采样子集 + 检测间隙窗口，不扫全片；
- **微调门槛未达** → Phase 4 只交付数据集与云机指标，promote 与上线另走 YOLO_FINETUNE 验收（不阻塞本计划关闭）。

## 10. 推进记录

| Phase | 实现模型 | 验证模型 | 基线指标 | 完成后指标 | 裁决 | 提交 |
|-------|----------|----------|----------|------------|------|------|
| 0 | opencode-go/deepseek-v4-pro | — | — | 工具链落地：`detect_ball_tracks.py`/`extract_ball_candidates.py`/`prefill_ball_labels.py`/`run_ball_batch.ps1`；demo4 600 帧冒烟：0.51 item/帧、77 段/20s（断段严重）；6 视频全量检测后台运行中（NPU 队列 + GPU 队列） | — | 见提交 |
