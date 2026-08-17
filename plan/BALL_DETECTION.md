# PLAN：足球检测优化（Ball Detection & Tracking）

> 状态：**Phase 0 已完成；Phase 1 人工审核进行中；Phase 3 修复专项（单球分层门控）已立项待执行，见 §11**
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
| F | 球员先验形同虚设：`_select_candidate` 仅 +0.10 权重、固定 150px 半径不分远近；评估回放传空球员字典，先验从未生效 | tracker 选型逻辑 + 评估设施缺口 |

**pilot 实证（2026-08-16~17，`eval/ball_crops/demo4/pilot/`）**：qwen3.6-flash/qwen3.7-plus 四轮试点（静态帧→全分辨率多帧→动态跟踪框视频→加场地判据）确认——①检测器最自信的 confirmed/bridge 假段集中在"球员脚边/身边的移动白点"与"场边垃圾"两簇；②"靠近球员"单独当判据会反噬（球员活动区附近的垃圾被两模型齐判为球）；③VLM 最好成绩仍 15% 假球率，只能当审核排序器。→ 修复必须叠加物理判据（透视/尺寸比/背景运动），详见 §11。

**结论：球链路需要一轮完整的"全量人工标注 → 基线量化 → 参数级修复 → 数据级微调"闭环，方法论沿用前两轮验证门，但人工审核范围扩大为 100%。**

## 2. 本轮相对前两轮的提升点（回答"如何提升"）

1. **全量人工审核，无抽样偏差**：上轮 referee 的 raw 部分只人工核了 116 张精选 + 9 张分歧，其余靠双模型共识；本轮**所有提取出来的球 crop 图片全部人工审核**（口径见 §5），`manual_label` 即权威 ground truth，精确率/召回率可真正统计。
2. **补上 FN 盲区**：上轮只评了"检出来的对不对"（precision），几乎没评"该检的漏没漏"（recall）。本轮增加**漏检扫描**：双模型在无检测帧上找"像球但被漏"的候选，同样进人工审核 → 召回率有真实度量。
3. **track 级指标**：球与裁判不同，观感问题主要在轨迹连续性。新增断段率、虚接率、确认延迟指标，直接对应"球框闪断/粘鞋"等用户可见问题。
4. **误差模式分类驱动修复**：每个 FN/FP 标注归因（远/小/模糊/遮挡/鞋袜/线/灯/头），修复按归因命中率排序，不做盲调。
5. **双模型预筛 + 仲裁降本**：kimi-k2.7-code 与 gpt-5.6-luna 先各预标一遍（人只做确认/纠错；qwen3.7plus 在 openchamber 不可用，经用户确认由 kimi-k2.7-code 顶替），分歧样本优先审，人工审核 100% 覆盖但单张耗时大幅下降。
6. **6 视频多场次**：raw1/2（同场）供训练/回归，demo1~4（多场次、含 demo1/3/4 全新场）做未见过场次评估与微调 test，split 纪律沿用 training/README。
7. **产出 ball 类微调数据集**（含硬负样本），对齐 training/README 门槛（≥600 球正样本、≥75% recall / ≥90% precision 才 promote 权重）。

## 3. 目标

- 全量人工标注集定稿：6 视频、所有候选 crop 100% 人工审核；
- crop 级：precision/recall 较基线显著提升（目标 P≥90%、R≥75%，沿用 training/README 口径）；
- track 级：断段率、虚接率下降，确认延迟下降，归属错配抽样下降；
- **单球门控专项（§11）**：pilot 20 项假段存活数显著下降（FP 段被 L1~L3 压制）；球员附近小球漏检由 L4 补救提升观测率；每层修复在 hold-out 场次不回退；
- 前两轮成果不回退：referee 标注集、34-crop balanced acc=1.0、pytest 全绿；
- 端到端速度下降 ≤ 20%。

## 4. 工作方式（沿用前两轮验证门，模型名按本轮实际）

1. 一个 Phase 一次推进，单独 commit；Phase 开始前记基线，完成后同口径对比；
2. 实现模型：`opencode-go/deepseek-v4-pro`；验证模型：`opencode-go/qwen3.7plus` 与 `opencode-go/gpt5.6luna`（用户指定，代替上轮 kimi/luna 组合；**执行时发现 qwen3.7plus 在 openchamber 中不可用，经用户确认改用 `opencode-go/kimi-k2.7-code` 代替**）；
3. 验证模型**不得只读实现方结论**：必须自己看图核对判定、自己跑 eval 复算指标、复核 diff，给出"通过/不通过 + 理由"；
4. 标注环节：双模型只做**预标与仲裁辅助**，`manual_label`（人工）为唯一权威；分歧样本（双模型不一致）必须人工裁决；
5. 全部 Phase 完成后 kimi-k2.7-code 与 gpt-5.6-luna 各一次独立交叉复核，均通过才算整体完成；
6. 全程记录到本文档「推进记录」。

## 5. 人工审核口径（本轮核心变化：100% 审核）

> **口径变更（2026-08-16，用户拍板）**：双模型预标/交叉标注整体取消（模型失误率过高，实际分歧率 ~50%，预标价值不成立）。全部候选由用户纯人工审核，`manual_label` 为唯一权威；已产出的模型标签文件留在磁盘但不再使用，`candidates.jsonl` 已剥离全部模型字段与 `priority`，`review_order.txt` 改为按视频时间顺序（帧号升序）。审核载体为网页工具 `tools/ball_review_server.py`（左=原图视频半秒窗口 + 红框定位，右=放大 crop；`/video/<src>` Range 流式服务原视频，无缓存）。

**审核集合 = 以下三类 crop 的并集**（`tools/extract_ball_candidates.py` 产出）：

| 类别 | 内容 | 每段张数 |
|------|------|----------|
| confirmed | 已确认 track segment 的最清晰帧（小图放大、帧距 ≥15） | ≤3 张/segment |
| unconfirmed | 未确认的孤立候选/被运动门拒绝的候选（FP 高发区，**必审**；过滤政策见下） | 1 张/簇 |
| fn_sweep | bridge（段内预测帧）+ gap（段间断档中点）+ global（空帧抽样）三类漏检候选（FN 高发区，**必审**） | 1 张/候选 |

- 过滤政策（用户确认）：`unconfirmed` 簇中 **conf<0.05 且单帧孤立**（cluster_frames≤1）不进审核集（`--drop-low-single` 默认开启），其余 100% 审核；demo4 实测该政策去掉 337/1132 簇，全量预估 ~91k → ~75k；
- 审核载体：`tools/make_ball_sheets.py` 生成的网格图（20 张/张，带 id 编号），双模型与人工都按网格图批量审，不确定的格子回 `crops/<id>.png` 放大看；

- 标注值：`ball` / `not_ball`（白鞋、袜、线、灯、头、手套等硬负样本写具体类别）/ `null`（太小、模糊到无法判定，不进评估但保留）；
- 双模型预填 `kimi_label` / `luna_label` 后，人工**逐张确认或纠错**，分歧样本优先审；
- 定稿产物：`eval/ball_crops/{src}/final_ball_labels.jsonl`（权威），附审核进度清单（缺审即验收不通过）。

## 5.2 视频球标注范式（2026-08-17 起，Phase 1 主载体）

> **口径变更 2**：Phase 1 改为**直接在视频里人工框球**（更符合人看视频的直觉，且产出帧级球位置真值，补齐 recall 评估盲区）。载体：`tools/video_annotate_server.py`（端口 8101，支持 demo1~4/raw1/raw2/raw4 七源）。crop 审核集保留为**可选**的非球原因归因工具（不再强制全审）。

**工具交互**：原画视频播放 + 画布叠加；点击=固定大小框（`[`/`]` 调大小）、拖拽=任意矩形、两种可拖动/改边；`h` 开关检测器候选提示（读 `ball_tracks.jsonl`）；关键帧实线绿框、关键帧间 ≤2s 自动插值虚线绿框实时显示；`Del` 删当前框、`Backspace` 撤销；实时保存。

**产出**：`eval/ball_gt/<src>/annotations.jsonl`（关键帧：frame/t/bbox/seg，仅存人工标注）→ `tools/interp_ball_gt.py` 展开为每帧 GT `frames_gt.jsonl`（同段相邻关键帧 ≤60 帧线性插值）。

**GT 使用纪律（Phase 2/4 通用）**：
1. **插值帧精度**：线性插值在球匀速时准，急停/变向/碰撞会偏——评估时关键帧与插值帧**分组统计**（关键帧组为准，插值帧组作参考）；
2. **recall 口径**：遮挡/出视野帧无 GT，recall 只算"GT 覆盖帧"（球可见帧）；漏检帧定位从关键帧组给出；
3. **帧号是权威**：工具以 29.97 折算 t，raw 实际 29.96——评估统一按 frame 索引对齐 ball_tracks.jsonl，不用 t；
4. **单球假设**：GT 为单目标轨迹，多检（水印/垃圾等）全部计 FP。

**工具待办优化（执行时逐项做）**：①关键帧间隔 >2s 时间轴标黄警示；②上一个/下一个关键帧跳转快捷键；③每场"已标覆盖时长/总时长"统计（防漏标导致 recall 偏差）；④时间轴按段着色+段时长；⑤GT 备份策略（`eval/ball_gt/` 定期导出或入 git，防数据丢失）。

**数据价值定位**：标注 GT = Phase 2 帧级基线评估（唯一 recall 真值）+ Phase 3 修复标尺 + Phase 4 微调正样本来源（按 GT 框裁原图，自然含远场/模糊/遮挡难度分布；负样本用检测 FP + 候选集 crops）。

## 6. Phase 划分

### Phase 0 — 全量检测+候选提取+双模型预标
- `tools/detect_ball_tracks.py`（新增）：6 视频全量球检测+跟踪（不含球员/关键点，GPU/NPU 并行），写 `raw/ball_tracks.jsonl`（含 segment/confidence/observed 全字段，供后续一切评估）；
- `tools/extract_ball_candidates.py`（新增）：按 §5 生成三来源候选 crop + auto 预填清单（conf、segment、尺寸、距预测点偏差等特征列）；
- `tools/prefill_ball_labels.py`（新增）：kimi-k2.7-code、gpt-5.6-luna 各写一份预标 jsonl，输出共识/分歧两份清单，分歧优先审；
- 检测模型口径（重要）：本计划全部 tracks 用 `ball-detection_openvino_model_1280_fp16`（1280 固定尺寸导出，比 main.py 默认的 640 fp16 导出召回更高、小球更清晰）；Phase 2 基线与此同口径，Phase 3 可把"生产默认模型换成 1280 导出"作为一项修复候选；

### Phase 1 — 全量人工审核（用户执行，本轮不做抽样）
- **主载体（2026-08-17 起）：视频球标注**——用户在 `tools/video_annotate_server.py`（8101）里直接框球，产出 `eval/ball_gt/<src>/annotations.jsonl`（关键帧）+ `frames_gt.jsonl`（插值展开，§5.2）；
- 顺序：demo4 先全标（出 Phase 2 基线）→ demo1/2/3 → raw1/2/4（可分多次续标，实时保存）；
- crop 审核集（`final_ball_labels.jsonl`）保留为可选的非球原因归因工具，不再强制全审；
- 交付：GT 定稿 commit（数据入 git 或备份，§5.2 纪律 ⑤）。

### Phase 2 — 基线评测与误差归因
- `tools/eval_ball.py`（新增）：**帧级对比为主**——GT 框（`frames_gt.jsonl`）vs 检测框（`ball_tracks.jsonl`）IoU≥0.3 匹配 → 帧级 precision/recall（关键帧/插值帧分组统计）、漏检帧定位、误检帧定位；crop 级指标与 track 级（段召回、断段率、虚接率、确认延迟）并存；
- 误差模式归因表（FN：远/小/模糊/遮挡；FP：鞋/袜/线/灯/头/手套/水印/垃圾）；
- 指标按源分报（§8.1 纪律 2）；记录基线数字到推进记录。

### Phase 3 — 参数级修复（按归因排序，逐项独立 commit + 验证门）

**3a. 单球分层门控专项（已立项，先行执行，详见 §11）**：不依赖全量标注定稿（用 pilot 20 项冒烟回归集替代快速反馈），与 Phase 1 人工审核并行推进——
- L1 球员邻域硬剪枝+保底（透视缩放门限半径）；L2 尺寸-深度一致性（球≈最近球员身高/8）；L3 静态背景抑制（球员脚点位移中值当全局运动代理）；L4 球员 ROI 补救检测（召回侧）；
- 测量先行：`tools/smoke_ball_fp.py` 冒烟回归 + `detect_ball_tracks.py` 球员轨迹回放（`--object-tracks`/`--rescue`）；迭代期只跑 demo4+raw1 子集，终验前才 6 视频全量。

**3b. 其余参数级修复项（以实测归因为准，按优先级）**：conf 自适应/NMS 收紧（FP）；tile 重叠或分辨率调整、最小尺寸下限（FN）；尺寸门 48px/5% 与宽高比复核（FP/归属）；restart_gap / confirmations / max_prediction / 运动门距离曲线（断段/虚接）；归属距离与 grace（粘脚）；
- 每项修复后：`eval_ball.py` 同口径对比 + kimi-k2.7-code 或 gpt-5.6-luna 验证门裁决（不通过即回退重做）。

### Phase 4 — ball 微调数据集 + 云端训练（数据侧本轮必产出，训练可与 YOLO_FINETUNE 并行）
- `tools/build_ball_finetune_dataset.py`（新增/复用 build_finetune_dataset）：**正样本以视频标注 GT 为准**——按 `frames_gt.jsonl` 框从原视频 1920×1080 裁球（自然含远场/模糊/遮挡难度分布），关键帧优先、插值帧补充去重；**负样本**用检测器 FP + 候选集 crops（鞋/袜/线/灯/头/手套/水印/垃圾）打包成 YOLO 二分类数据集；split 纪律：整场视频不跨 split、demo1~4 至少一场全新进 test、相邻帧不跨 split；
- 量级对齐 training/README：≥600 球正样本 + 硬负样本 ≥300；
- CUDA 云机 `python -m training.train_models --task ball`；未达 75%/90% 门槛不 promote（仅回传指标）。

### Phase 5 — 终验
- 6 视频全量重放：demo1~4 未见过场次看效果 + raw1/2 回归；track 级指标与归属抽看；老验收（referee 标注集、34-crop balanced acc、pytest）不回退；端到端速度 ≤20% 回退；
- kimi-k2.7-code + gpt-5.6-luna 双模型交叉复核（看图、复算、diff）均通过后关闭。

## 7. 涉及文件

| 文件 | 改动 |
|------|------|
| `tools/detect_ball_tracks.py` / `extract_ball_candidates.py` / `prefill_ball_labels.py` / `make_ball_sheets.py` / `run_ball_batch.ps1` / `eval_ball.py` / `build_ball_finetune_dataset.py` | 新增（Phase 0/2/4；网格图批量审核工具 make_ball_sheets 为本轮新增） |
| `tools/detect_ball_tracks.py` | §11 专项追加 `--object-tracks`（球员轨迹回放）/ `--rescue`（ROI 补救）开关 |
| `tools/video_annotate_server.py` / `tools/interp_ball_gt.py` | 新增（§5.2：视频球标注工具 + 关键帧→每帧 GT 展开） |
| `eval/ball_gt/{demo1..4,raw1,raw2,raw4}/` | 视频标注 GT（annotations.jsonl 关键帧 + frames_gt.jsonl 每帧，入 git 或定期备份） |
| `tools/smoke_ball_fp.py` | 新增（§11：pilot 20 项假段存活冒烟回归） |
| `tools/detect_tracks_only.py` | 复用（§11：为 demo1~4 生成 object_tracks.jsonl；raw1/raw2 已有） |
| `tracking/ball_tracker.py` | Phase 3 参数级修复（NMS/尺寸门/运动门/确认与重启参数）+ §11 L1~L4 门控栈 |
| `annotation/football_video_processor.py` | §11 L4：接入 rescue_fn（生产管线与回放工具共用） |
| `tests/test_ball_tracking.py` | §11 四层门控单测 |
| `ball_to_player_assignment/ball_to_player_assigner.py` | 归属距离/grace 复核（如归因命中） |
| `eval/ball_crops/{demo1..4,raw1,raw2}/` | 候选 crop + final_ball_labels.jsonl（图片不入 git）；`demo4/pilot/` 为已定稿冒烟 ground truth |
| `output_videos/{demo1..4}/raw/object_tracks.jsonl`、`*/ball_v2/` | §11 回放输入与重放产物（不入 git） |
| `plan/BALL_DETECTION.md` | 本文档（推进记录） |
| 其余文件 | 不动；对外接口（BallDetector/BallTracker/assigner 构造签名）保持兼容，§11 新增参数一律可选 |

## 8. 验收标准

1. `final_ball_labels.jsonl` 覆盖 §5 三类候选且**审核进度 100%**（无未审项）；
2. crop 级 precision ≥90%、recall ≥75%（未达则须较基线显著提升并说明差距归因）；
3. track 级：断段率/虚接率较基线下降，确认延迟不劣化；
4. 前两轮回归：referee 标注集指标、34-crop balanced acc=1.0、全量 pytest 绿；
5. 端到端处理速度下降 ≤20%；
6. 双模型（kimi-k2.7-code / gpt-5.6-luna）交叉复核均通过。

## 8.1 过拟合纪律（2026-08-16 追加，用户要求写入）

数据构成风险：raw1/raw2 为同一比赛上下半场（同一场地/机位/光照，共 ~60min），demo1~4 为不同场次但每场仅 ~2.5min——约 85% 数据来自单一场景，Phase 3 参数调优与 Phase 4 微调存在对该场景过拟合的真实风险。约束如下：

1. **Hold-out 场次**：至少 1 个 demo 全程不进 Phase 3 调参、不进 Phase 4 微调数据集，仅作最终验收——其指标即泛化估计；该场次从 Phase 2 基线起就固定，中途不得更换。
2. **指标按源分报**：Phase 2/3 的 crop 级与 track 级指标必须 per-source 列表（demo1/2/3/4/raw1/raw2 各一行），禁止只报汇总平均数拍板；"raw 上提升、hold-out demo 上下降"即过拟合信号，该项修复回退。
3. **参数修复需物理解释**：conf/尺寸门/宽高比等域相关参数的修复必须在 4 个不同场地全部同向改善才保留；restart_gap/确认帧数等物理类参数不要求多场地验证但需说明依据。
4. **微调不 promote 不回归**：云机训练结果必须在 hold-out 场次上 ≥ 生产基线才换权重；不达标只回传指标，`ball-detection.pt` 保持生产基线——最坏情况等于无提升，不允许变差。
5. **域覆盖补强**：后续补充 2~3 个不同球场的新视频进 Phase 4/5 训练与评估，再考虑放宽上述约束。

## 9. 风险与对策

- **全量候选量大**（conf=0.02 高召回 + 6 视频）→ 段级聚合去重（每 segment ≤3 张）+ unconfirmed 只取每帧最强候选；仍超人工预算时按 §5 优先级顺序审（unconfirmed/fn_sweep 优先），confirmed 按清晰度降序——但清单完整性（每个 segment/候选至少 1 张）不妥协；
- **远场小球人无法判定** → `null` 类不进评估但保留，标注规则先定稿；recall 指标只算"可见可判定"样本；
- **双模型对球判定本身不可靠** → 模型标签仅作预填与仲裁建议，人工为唯一权威；分歧率过高时降低模型预标权重（纯排序用）；
- **raw 与 demo 光照/场地差异** → 参数修复用 6 视频全量验证，不以单场定参；
- **FN 扫描成本** → 双模型仅扫"无检测帧"的降采样子集 + 检测间隙窗口，不扫全片；
- **微调门槛未达** → Phase 4 只交付数据集与云机指标，promote 与上线另走 YOLO_FINETUNE 验收（不阻塞本计划关闭）；
- **（§5.2）GT 数据丢失** → `eval/ball_gt/` 是关键资产（评估+微调都靠它），每次标注会话后自动落盘（工具实时保存），阶段完成即 commit 或导出备份；
- **（§11）硬剪枝误伤长传/空中球** → 保底规则：无候选进球员门时全保留；L1 只改候选集合不改运动门逻辑；
- **（§11）L3 全局运动代理失效**（球员极少/特写/回放镜头，脚点位移中值不代表相机运动）→ 球员数 <5 的帧禁用 L3；黑名单 ≤2s 自动过期，误杀后最多丢 2s 即可重建段；
- **（§11）回放口径与生产不一致** → 基线 ball_tracks 为空标定跑，回放对比沿用空标定保证可比；pitch 包含门等生产独有门的效果在报告中单独注明，不混入对比表；
- **（§11）剪枝改变段集合导致审核集失配** → 终验报告单独列出新增/消失段，必要时补审少量 crop；pilot 20 项冒烟集只测 FP 侧，召回侧结论以全量标注为准。

## 10. 推进记录

| Phase | 实现模型 | 验证模型 | 基线指标 | 完成后指标 | 裁决 | 提交 |
|-------|----------|----------|----------|------------|------|------|
| 0 | opencode-go/deepseek-v4-pro | — | — | 工具链落地（检测/提取/合并/网格图/批量脚本）+ 冒烟；6 视频全量检测后台运行（NPU/GPU 双队列）；demo4 完成：4523 帧、753 段（断段极严重）、2956 审核项/148 网格图（过滤政策去掉 337 低置信孤立簇）；kimi+luna 预标会话已起 | — | cb2c151→673aac6→a2cf126 |

### Phase 0 推进补充（2026-08-15 当日）

- 审核集产出：demo4 2956 项 / demo1 1763 项（89 网格图）/ demo3 3001 项（151 网格图）/ demo2 3004 项（151 网格图）/ raw1 30631 项（1532 网格图）/ raw2 30572 项（1529 网格图），**6 视频共 71,927 审核项**。
- 双模型预标进度：luna 完成 demo4（2956）+ demo1（1763）；demo3 由新会话续标（已 800，原会话上下文耗尽）；kimi 完成 demo4 2700/2956 后触发 opencode-go 5h 限额（约 18:40 重置后自动重试续标 demo4 剩余 256 条→demo1→demo3→demo2）。
- raw 预标队列已建：luna-raw1/raw2、kimi-raw1/raw2 四会话 + demo2 挂接 kimi/luna 主队列；全部采用"每 20 张 sheet 落盘、断点续标"协议。
- demo4 标签合并：2700 条共同标注中 **1186→1284 条分歧（~45%）**，主要模式 luna=not_ball/kimi=ball、luna=null/kimi=ball——双模型对"球"的宽容度差异大，人工全审价值被验证；分歧清单 `eval/ball_crops/demo4/disagreements.jsonl` + `review_order.txt` 已产出，**demo4 可开始人工审核**。
- 期间修复：merge 工具分歧清单作用域 bug；`--merge-labels PATH FIELD` 通用化（kimi 顶替 qwen）；低置信孤立簇过滤政策（用户确认）；标签文件去重；会话 scratch 文件 gitignore。
- Phase 2 工具先行：`tools/eval_ball.py` 已落地（crop 级 precision/采样召回代理 + track 级段统计/确认延迟），demo4 空标签自检通过。

### Phase 3a 立项记录（2026-08-17）

- **VLM 预标试点关闭**：`eval/ball_crops/demo4/pilot/` 四轮试点（report.md→report_v6.md）结论——qwen3.6-flash/qwen3.7-plus 在 20 个全假球困难样本上最好成绩仍 3/20 误判（v6 加判据后），动态框视频反而劣于静态帧；确认 VLM 不具备裁判资格，仅可作审核排序器，维持 §5 纯人工审核口径。
- **用户诉求立项**："场上只有一个球、基本在球员附近"→ 单球剪枝 + 识别率提升。pilot 证明单纯球员邻域判据会反噬（球员附近的场边垃圾），故方案升级为四层物理门控栈（L1 球员邻域硬剪枝+保底 / L2 尺寸-深度一致性 / L3 静态背景抑制 / L4 球员 ROI 补救），全文见 §11。
- **用户拍板六项**：①硬剪枝+保底（非纯软评分）；②剪枝与 ROI 补救同轮做；③评估配套做球员轨迹回放量化对比；④pilot 20 项作冒烟回归集；⑤L3 本轮做；⑥迭代期只跑 demo4+raw1 子集、终验前才 6 视频全量。
- 状态：**待执行**（执行顺序见 §11.4，每步独立 commit + 冒烟回归）。

## 11. Phase 3 修复专项：单球分层门控 + 球员 ROI 补救（已立项，待执行）

> 立项日期：2026-08-17（用户拍板）。状态：**待执行**。
> 动机：用户诉求"场上只有一个球，基本在球员附近——用球员邻域做剪枝、提高识别率"。
> 依据材料：`eval/ball_crops/demo4/pilot/`（qwen3.6-flash / qwen3.7-plus 四轮试点 report.md→report_v6.md）+ 联网检索佐证（MDPI 2021 micro-YOLOv3：静态背景持续假检靠运动信息抑制；Roboflow 球追踪实践：单球+时序质心一致性）。

### 11.1 pilot 试点结论（本专项的设计输入）

1. VLM 预标不可靠：20 个全假球困难样本，最好成绩（v6 加"球员聚集=球/场地边缘白点=垃圾"判据）仍 3/20=15% 误判为球；动态框视频（v5）反而比静态帧（v2）差。VLM 只能当审核排序器，不能当裁判——维持 §5 "纯人工审核"决定不变。
2. **"球员邻域"判据单独使用会反噬**：v6 残留盲区 b01722/b01723 是球员活动区附近的场边垃圾，两模型均误判为球。检测器最自信的 confirmed/bridge 假段恰恰集中在"球员脚边/身边的移动白点"簇——单纯"靠近球员就保留"救不了这簇 FP。
3. pilot 的 20 项是目前唯一已定稿的人工 ground truth（全 not_ball），可作快速冒烟回归集。
4. 口径注意：基线 ball_tracks 用空标定跑的，审核集混入了生产环境本会被 pitch 包含门（`_metric_candidates` + `geometry.contains`）过滤的场外候选；对比报告须注明此差异。

### 11.2 方案：分层门控栈（在现有 L0 之上叠加）

现有 L0（不动）：宽高比 0.25~4、尺寸上限 48px/5%、NMS iou=0.5、运动门、pitch 包含门（生产标定有效时）、3 帧确认/0.15s 重启/0.5s 桥接。

| 层 | 机制 | 目标 FP/FN 簇 | 依赖标定 |
|----|------|----------------|----------|
| L1 | **球员邻域硬剪枝+保底**：`_select_candidate` 固定 150px 半径 → `max(min_radius, k×最近球员bbox高)`（k≈2.5 可配）透视缩放；有候选进门→硬剪门外候选；无候选进门（长传空中球/无球员帧）→全保留不丢召回；球员先验评分权重 0.10→~0.25 并按门限半径归一化 | 远离比赛区的白点/灯光/观众（c00006/c00024 型） | 否 |
| L2 | **尺寸-深度一致性**：期望球尺寸≈最近球员 bbox 高/8（球 0.22m vs 人 ~1.8m 同深度像素比）；容差带内软评分、偏离 >3× 硬拒；无邻近球员不启用 | **球员脚边的鞋/袜/垃圾/头——pilot 盲区簇** | 否 |
| L3 | **静态背景抑制**：每帧用全体球员脚点位移中值估计全局相机运动；球轨迹连续 ~0.4s 残差位移≈背景且门内无球员 → 杀段 + 该位置 ~2s 黑名单；球员数 <5 的帧（特写/回放）禁用本层 | 场边持续存在的垃圾段（b01722/b01723 型） | 否 |
| L4 | **球员 ROI 补救（召回侧）**：`BallTracker.update(..., rescue_fn=None)`，未观测时对球员脚部区域+上次预测位置裁剪放大（≤6 个 ROI 按距预测点排序，imgsz≈640）二次推理，同帧重试选择（单次 Kalman predict 内完成，不重复推进状态）；补救候选同样过 L1/L2；`FootballVideoProcessor` 与回放工具共用同一 rescue_fn | 球员附近小球漏检（FN 侧） | 否 |

设计纪律：四层全部有物理解释（透视缩放/球人尺寸比/背景运动/单球先验），符合 §8.1 第 3 条；L1~L3 零额外推理开销，L4 仅丢球帧触发且 ROI 数封顶，速度预算 ≤20%（§3）。

### 11.3 测量与验收口径

1. **冒烟回归集（先行）**：`tools/smoke_ball_fp.py`（新增）——从 pilot manifest_v5/v6 + report_v6 人工结论提取 20 项帧号/段号，重放 demo4 报告假段存活数（confirmed/unconfirmed 分列）；基线先跑一次存档。每层改完即跑，几分钟出 FP 侧反馈，不等全量标注。
2. **回放改造**：`tools/detect_ball_tracks.py` 加 `--object-tracks PATH`（行号=帧号回放球员轨迹进 `tracker.update`）+ `--rescue` 开关；demo1~4 先用现成 `tools/detect_tracks_only.py` 生成 `object_tracks.jsonl`（raw1/raw2 已有）。
3. **迭代策略（用户拍板）**：迭代期只跑 demo4（有 pilot ground truth）+ raw1 子集；全部层完成后才 6 视频全量重放 → `ball_v2/ball_tracks.jsonl`（保留基线目录），`eval_ball.py` per-source 对比。
4. **回退红线**：§8.1 纪律照旧——raw 上提升但 hold-out demo 下降即回退该层；人工标注定稿后同步纳入对比；新增/消失段不在原审核集的部分单独列出，必要时补审少量 crop。

### 11.4 执行顺序（每步独立 commit）

1. 测量先行：smoke_ball_fp.py + detect_ball_tracks.py 回放改造 + demo1~4 object_tracks 生成 + 基线冒烟存档；
2. L1 球员邻域硬剪枝（+单测：硬剪枝选中低 conf 近球员候选、保底回退、无球员行为不变）；
3. L2 尺寸-深度一致性（+单测）；
4. L3 静态背景抑制（+单测：静态杀段、球员数不足禁用、黑名单过期恢复）；
5. L4 球员 ROI 补救（+单测：rescue 恢复观测）；
6. 终验：6 视频全量重放 + eval_ball per-source 对比 + pytest 全绿 + 速度核对；结果记入 §10 推进记录。

### 11.5 涉及文件

| 文件 | 改动 |
|------|------|
| `tracking/ball_tracker.py` | L1~L4（`_select_candidate` 门控栈 + `update` rescue_fn 参数） |
| `annotation/football_video_processor.py` | 接入 rescue_fn（构造/注入 rescue 检测器） |
| `tools/detect_ball_tracks.py` | `--object-tracks` / `--rescue` |
| `tools/smoke_ball_fp.py` | 新增（pilot 20 项冒烟回归） |
| `tests/test_ball_tracking.py` | 新增四层门控单测 |
| `plan/BALL_DETECTION.md` | 本节 + 推进记录 |
| 构造签名兼容性 | `BallDetector`/`BallTracker`/assigner 对外接口保持兼容（新增参数一律可选） |
