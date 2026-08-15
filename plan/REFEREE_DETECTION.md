# PLAN：裁判识别优化（Referee Detection & Club Assignment）

分支：`codex/team-color-classification`
数据源：`demo2.mp4`（2.5min）、`C:\Personal Profile\Profile\Video\raw1.mp4` / `raw2.mp4`（各 30min，同场比赛上下半场）
硬件：Core Ultra 5 125H（Arc iGPU 145ms/帧、AI Boost NPU 156ms/帧 @ object-detection 1280 fp16；无 CUDA）
配置参考色（本场，实测/用户确认）：Maroon 球员 (120,37,66)、Navy 球员 (31,72,127)、裁判黄 (168,156,74)、Maroon 门将黑 (30,30,30)、Navy 门将深紫 (48,37,68)

## 1. 背景与问题（demo2-30s-test 实测确认）

裁判球衣为**黄色**（hue≈26）。`object-detection.pt` 的 referee 类双向不可靠：

| 现象 | 数据证据（demo2 object_tracks.jsonl） |
|------|--------------------------------------|
| 真裁判被判成 `player` → `ClubAssignerModel.predict` 强制二选一（`np.argmin`，无"非两队色"拒绝出口），黄 hue26 距 Maroon(170)=36° 比 Navy(107)=79° 近 → 判给 Maroon → **画成红色队员** | player 类中位 hue 25-28 的 track：4(148帧)、104(117)、217(81)、83(42)、288(28)、203(10)、124(被判 Navy) 等 |
| 真球员被 YOLO 误判成 `referee` → 深灰虚线框、丢掉队色 | referee 类 track 197(153帧, hue107 藏青球员)、58(hue107)、243(hue172 栗色)、2(hue169) |
| referee 类 ID 碎片化（真裁判一段一 ID） | 48 个 referee ID，多数仅活 1-28 帧 |
| 门将需求（用户补充）：navy 门将深紫 (48,37,68)、maroon 门将黑 (30,30,30)，YOLO 把门将检成 player（p210 从未被检出 goalkeeper 类） | p210（navy 门将）91 帧全 player 类，hue 125-165 |

根因：`club_assignment/club_assigner.py` 分类无拒绝出口、无门将色支持；referee track 不参与颜色逻辑。

## 2. 目标

- 裁判稳定显示为裁判样式（深灰虚线），不再变红/蓝队员；红蓝队员不再被画成裁判；
- **两队门将（含非队色门将服）正确归属本队**；
- 上轮球队颜色成果不回退（34-crop 验证集 balanced acc 保持 1.0）；
- 以人工标注验证集为准，kimi-k2.7-code 与 gpt-5.6-luna 双模型多模态交叉验证通过；
- 为 Level B（YOLO 微调）产出三源标注数据集（本轮只备数据、不训练）。

## 3. 工作方式（沿用上轮团队颜色计划的验证门）

1. 一个 Phase 一次推进，单独 commit；
2. Phase 完成后由 openchamber 会话起 `opencode-go/kimi-k2.7-code` 与 `opencode-go/gpt-5.6-luna` 独立验证：看图核对判定 vs 人工标注、自己跑 eval 复算指标、复核 diff、给"通过/不通过"裁决；
3. 不通过 → 修复 → 重新验证；两票都过才进下一 Phase；
4. 全程记录到本文档「推进记录」。

## 4. Phase 0 — 候选截图 + 人工标注 + 备微调数据 ✅

- `tools/detect_tracks_only.py`：raw1/raw2 全量 tracks（GPU/NPU 并行，53872 帧/部）；
- `tools/extract_referee_candidates.py`：三源候选裁剪（demo2 113 张 + verdict 48 张；raw1 366 张、raw2 355 张），按 track 聚合、小图放大、auto 标签预填（`tools/prefill_referee_labels.py`）；
- demo2 两个清单已由用户人工核正；raw 候选已封装 116 张精选子集（`eval/referee_crops/raw_labeling/`），**人工标注暂缓，用户稍后标**。

## 5. Phase 1 — 颜色后处理 ✅（实现完成，Luna 复验通过）

`club_assignment/club_assigner.py`（详情见推进记录）：

1. `predict_referee()` 多参考色判定：裁判参考色（`--referee-color`）最近 → referee；门将参考色（`--club1/2-goalkeeper` 真实色）最近且 ≤60 → 该队门将；距两队都 >85 → referee；否则最近队；
2. 深色球衣路径 `extract_dark_jersey_stats()` + `_assign_dark_sample()`（黑门将等低亮低饱和样本，最近参考色取胜，队色需 ≤45）；
3. referee 类反向纠正（颜色匹配某队 → 恢复队色）；player 类被投 referee → 旗标 + 去 club；
4. 60 帧滑窗多数表决（referee/club 状态通用）；
5. 绘制层：`object_annotator.py` / `projection_annotator.py` 按旗标/队色切换样式。

## 6. Phase 2 — 双模型多模态交叉验证 🔄

- gpt-5.6-luna：首轮**不通过**（阈值 75 低于 maroon 实测最大 83）→ 修复（阈值 85 + get_player_club 统一 + 边界测试）→ **复验通过**；
- kimi-k2.7-code：会话运行中，待裁决；
- 双模型裁决齐后 Phase 2 关闭。

## 7. Phase 3 — 终验 ⏳（不需人工标注的部分先行）

- ✅ pytest 79 全绿（含门将/边界/深色用例）；
- ✅ demo2 重放 + 评估（见推进记录指标）；
- 🔄 raw1/raw2 重放（后台运行中）→ 完成后跑统计自洽报告（`tools/raw_selfcheck.py`，待写）：60 分钟全片 referee 旗标/队色恢复/门将归属/覆盖率/ID 稳定性；
- ⏳ demo2 端到端重跑（含门将支持后需重跑一次肉眼确认）；
- ⏳ 老 34-crop 验证集 balanced acc 保持 1.0（每次重放后复跑）。

## 8. Level B — YOLO 微调路线（数据侧本轮推进）

**为什么需要**：颜色后处理覆盖"颜色可分离"场景，但对 ID 污染 track（r74/p28）、YOLO 门将漏检（p210）、与队服同色裁判等无效，最终要提升 `object-detection.pt` 的 referee/goalkeeper 类召回与精度。

**步骤**（复用 `training/` 工作流）：

1. **数据**：`tools/build_finetune_dataset.py`（本轮新增）：三源视频抽帧 + object-detection.pt 预标四类 + 用本轮人工标注（demo2 已定稿；raw 待用户标注后回填）纠正 referee 类误标 → Roboflow 兼容 YOLO 目录 + manifest.csv（source_video/split 列）；
2. **格式**：四类 id 与现有模型一致（0=ball,1=goalkeeper,2=player,3=referee），整场视频不跨 split；
3. **本地校验**：`training/prepare_dataset.py` 增加 `object` task；
4. **训练**：`training/train_models.py` 增加 `object` task，CUDA 云机跑（本机无 CUDA）；
5. **验收**：未见过比赛上 referee 召回 ≥75%、精度 ≥90%，player/goalkeeper 无回退；导出 OpenVINO FP16；
6. **上线**：替换权重 + 端到端回归检查。

## 9. 涉及文件

| 文件 | 改动 |
|------|------|
| `tools/detect_tracks_only.py` / `extract_referee_candidates.py` / `prefill_referee_labels.py` / `eval_referee.py` / `make_raw_labeling_subset.py` / `package_raw_labeling_subset.py` | 新增（Phase 0/评估） |
| `tools/build_finetune_dataset.py` / `tools/raw_selfcheck.py` | 新增（本轮） |
| `club_assignment/club_assigner.py` | referee 拒绝出口/门将色/深色路径/反向纠正/投票 |
| `annotation/object_annotator.py` / `projection_annotator.py` | referee 旗标与队色绘制 |
| `main.py` / `tools/replay_assign.py` | `--referee-color` + 门将真实默认色 |
| `tests/test_club_assignment.py` | 79 项含 referee/门将/边界用例 |
| `eval/referee_crops/`、`eval/finetune_dataset/` | 标注集 + 微调数据集 |

## 10. 验收标准

1. demo2 人工标注集（134 crops）上 referee 精确率/召回率显著提升且双模型裁决通过；
2. 老 34-crop 验证集 balanced acc = 1.0（不回退）；
3. 有效分类覆盖率下降 ≤ 5pp；
4. 全量 pytest 绿；端到端处理速度下降 ≤ 20%；
5. raw 三源验证（统计自洽 + 双模型抽检 + 用户抽样标注）无明显崩坏。

## 11. 风险与对策

- **门将色与深色球员**：黑门将参考色对阴影深色样本有拉拢风险 → 深色路径队色阈值收紧到 45 + 滑窗多数表决吸收；
- **黄色与栗色 hue 只差 36°** → 阈值 85 落在实测安全区间（maroon 最大 83 / 黄衣最小 88）+ 裁判参考色路径兜底；
- **ID 污染 track**（r74/p28）→ Level B 微调提高 referee 类召回，让裁判拿到独立 referee 类 ID；
- **raw 标注暂缓** → 先用统计自洽 + 双模型多模态预标代替，用户稍后人工纠正。

## 12. 推进记录

| Phase | 实现模型 | 验证模型 | 基线指标 | 完成后指标 | 裁决 | 提交 |
|-------|----------|----------|----------|------------|------|------|
| 0 | opencode-go/deepseek-v4-pro | 用户人工标记 | — | 三源 tracks + demo2 标注定稿（113+48 张）+ raw 候选 721 张（精选 116 张待标） | 通过（demo2 标注完成） | 494596d |
| 1 | opencode-go/deepseek-v4-pro | gpt-5.6-luna（首轮不过→复验通过）；kimi 待裁决 | 裁判黄色 player tracks 全被判 Maroon/Navy | 用户标注口径：crop recall 0.907 / precision 1.0；track recall 0.941 / precision 1.0；club preservation 0.9875/0.975；老验证集 balanced_acc=1.0；pytest 79 绿；门将支持：p210 91/91 navy、黑门将合成测试 maroon | luna 通过；kimi 待裁决 | fc5a4ff→820ec31→77af117→6a3ee83→40e628f |

### Phase 1 实现要点与历次裁决

- **实现**：`predict_referee()` 判定顺序 = 裁判参考色（`referee_match_dist` 75，`--referee-color` 实测裁判黄 RGB(168,156,74)）→ 门将参考色（`gk_match_dist` 60，`--club1/2-goalkeeper` 默认黑 (30,30,30)/深紫 (48,37,68)）→ 两队距离阈值 85（luna 裁决重标定：maroon 实测最大 83 / 黄衣最小 88 的安全区）→ 最近队；`extract_dark_jersey_stats()` 深色路径（val 25-105 & sat<90）支持黑门将；60 帧滑窗多数表决；referee 类反向纠正 + player 类 referee 旗标；annotator/projection 绘制联动。
- **luna 首轮（不通过）**：阈值 75 < maroon 最大 83；get_player_club 无拒绝出口 → 修复提交 820ec31（阈值 85 + 边界测试 + predict_referee 统一）。
- **luna 复验（通过）**：独立复跑 pytest 75 绿、precision 1.0/recall 0.907/0.941、balanced acc 1.0；补充建议三参考色真实配置测试（已补 6a3ee83）。
- **门将支持**（40e628f，用户提供门将色）：navy 门将深紫/黑门将合成测试通过；p210 91/91 帧判 navy、0 帧 referee；gpgk 类 track 由"恒 maroon"变为按色分（17 navy / 12 maroon）。
- **已知残余**（Level B 范围）：r74/p28 ID 污染 track 滑窗滞后（4 张 FN）、p124 无数据帧（1 张）。

### p210 案例（用户两次确认）

- p210 = **navy 队门将**（非藏青球员），深紫门将服 (48,37,68)；YOLO 全程 player 类（goalkeeper 类漏检）。
- 标注词汇扩展 `navy_gk`/`maroon_gk`（按球队归属判定），`eval_referee.py` 支持；p210 3 张 verdict 图标 navy_gk，3/3 正确。

### 人工标注状态（暂缓中）

- demo2：✅ 已定稿（用户核正）。
- raw1/raw2：⏸ 暂缓。已产出 721 张候选（auto 预填）+ 116 张精选子集（`eval/referee_crops/raw_labeling/`）；双模型多模态预标已排期（luna 先标、kimi 交叉），用户稍后只做纠正。
