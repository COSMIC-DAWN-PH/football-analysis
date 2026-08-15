# PLAN：零训练泛化加固（自动标定 / 阴影不变 / 时序解码 / 碎片合并 / 弱标注器）

> **状态：待开始。**
> 定位：本机零 CUDA 即可全部完成的升级，与 `YOLO_FINETUNE.md`（需云机训练）互补、可并行。

分支：`feature/pitch-goal-detection`
基线：`TEAM_COLOR_CLASSIFICATION_DONE.md` + `REFEREE_DETECTION_DONE.md` 验收关闭后的 HEAD（a2cf126）
数据源：raw1/raw2（同场上下半场，已有全量 tracks 与标注集）、demo2（已定稿标注）、**demo1/3/4（全新场次，泛化验证主力，尚未做颜色层评估）**
硬件：Core Ultra 5 125H（Arc iGPU / NPU / CPU，无 CUDA）

## 1. 背景与问题

前两轮把颜色层对本场（demo2/raw）做到物理极限后，残余问题分四类，当前归属与缺口：

| # | 残余问题 | 证据 | 现有归属 | 缺口 |
|---|----------|------|----------|------|
| A | 换场次需手工配 5 个颜色参数，默认值是本场残留 | `--referee-color` 默认 None；club 默认色无普适意义 | 无 | 泛化性从未在新场次验证过；demo1/3/4 就在本地但没跑过 |
| B | 阈值 85/60/40/15° 全按本场光照手工标定 | luna 裁决记录：85 落在本场 83/88 安全区 | 无 | 换光照安全区漂移，无自动手段 |
| C | boundary 阴影漂移带（hue 87-94）队色判错 | raw 终评 club preservation 0.56，残差集中于此 | YOLO_FINETUNE（Level B） | 微调前无缓解手段；取色的光照不变性从未尝试 |
| D | ID 污染 track 滑窗滞后 ~1s；裁判 track 碎片化（一段一 ID，48 个 referee ID） | r74/p28 案例；REFEREE_DETECTION_DONE §1 | YOLO_FINETUNE（Level B） | 跟踪层本身的碎片合并从未尝试 |
| E | 工程债累积：votes 永不清理、平票不稳定、浅拷贝原地改、get_player_club 与 assign_clubs 口径不一致、绿掩码硬编码 | TEAM_COLOR_DONE §8「后续工作」清单 | **只列了清单，从未计划化** | 无归属、无验收 |
| F | YOLO 微调数据缺口（referee ≥600 张、gk 每队 ≥200 张）挖标成本高 | YOLO_FINETUNE §3 | YOLO_FINETUNE（数据目标以它为准） | 缺自动化标注工具，现状靠候选抽取 + 人工 |

**结论：E 是欠账，A/B 是泛化性主缺口，C/D 是微调前的低成本缓解尝试（成了赚、不成归档结论给 Level B），F 是给微调提速的杠杆。全部不需要 GPU。**

## 2. 与现有计划的关系（防重复声明）

| 现有计划 | 边界 |
|----------|------|
| `YOLO_FINETUNE.md` | 数据量目标、split 纪律、模型级/管线级验收**以它为准**；本计划 Phase 6 只提供伪标签工具，不重复其目标。若微调先行完成并解决 C/D，Phase 4/5 降级为可选 |
| `BALL_DETECTION.md` | 球链路完全不在本计划范围 |
| `FIELD_DETECTION_MOVING_CAMERA.md` | 场地注册完全不在本计划范围 |
| `TEAM_COLOR_CLASSIFICATION_DONE.md` §8 | 本计划 Phase 1 是该清单的正式计划化（该清单此前无归属） |
| `REFEREE_DETECTION_DONE.md` §12 遗留 | Phase 4/5 以零训练手段缓解其中 ID 污染/碎片化两项；同色裁判、红 bib、gk 类误检仍归 Level B，本计划不碰 |

## 3. 目标

- demo1/3/4（全新场次）：**零手工颜色参数**自动标定后跑通，标注集 balanced acc ≥ 0.9（目标值 Phase 0 基线出来后可修订）；
- 本场（demo2/raw）全指标零回退：referee precision 1.0 / recall 0.907/0.941、34-crop balanced acc 1.0、raw selfcheck（coverage 0.946 / flagged 321,331 / restore 1949,1943）；
- boundary 子集 club preservation 较 0.56 显著提升（Phase 4 验收口径见下）；
- ID 污染 track 切换延迟显著下降、裁判碎片 ID 数显著下降，错并率为 0（抽检）；
- 产出伪标签工具与首批数据，量级计入 YOLO_FINETUNE §3 进度；
- 全量 pytest 绿；端到端速度下降 ≤ 20%。

## 4. 工作方式（沿用验证门，模型名按 BALL_DETECTION.md 本轮口径）

1. 一个 Phase 一次推进，单独 commit；Phase 开始前记基线，完成后同口径对比；
2. 实现模型：`opencode-go/deepseek-v4-pro`；验证模型：`opencode-go/kimi-k2.7-code` 与 `opencode-go/gpt-5.6-luna`（实现与验证必须不同模型）；
3. 验证模型不得只读实现方结论：必须自己跑 eval 复算指标、复核 diff，给"通过/不通过 + 理由"；
4. 不通过 → 修复重验，两票通过才进下一 Phase；
5. 全部完成后 kimi 与 luna 各一次独立交叉复核；
6. 全程记录到本文档「推进记录」。

## 5. Phase 划分

### Phase 0 — 泛化基线与标注集（先做，给全场定口径）

- demo1/3/4 各跑一段 tracks-only（复用 `tools/detect_tracks_only.py`）+ 候选抽取；
- 每场抽 30~50 张清晰 crop 人工标两队/裁判/门将（口径沿用 demo2 标注规则），小标注集定稿入 `eval/`；
- 跑两组基线：①零调参（默认色直接跑）②手工调参（人工取色填参），量化"换场到底掉多少"——这是 Phase 2 的对比基线，也是本计划问题 A 的实证；
- 产出距离/hue 分布统计脚本（Phase 2 阈值拟合的数据源）。

### Phase 1 — 工程债清理（TEAM_COLOR_DONE §8 正式化）

1. `votes_by_track`/`club_by_track` 生命周期：track 消失 N 帧后清理，杜绝 ID 复用污染（上轮已知债）；
2. 滑窗平票 tie-break 确定性（固定优先级或保持旧值，禁 `most_common` 不稳定序）；
3. `tracks.copy()` 浅拷贝行为：修复为深拷贝或文档化 + 测试钉死；
4. `get_player_club()` 与 `assign_clubs()` 口径统一（统一走 `predict_referee`，外部调用者语义一致）；
5. 绿掩码 hue 范围（36-86）提为可配置参数（CLI 暴露，默认不变）。

验收：全部指标零回退；每项债新增回归测试；单独 commit。

### Phase 2 — 参考色 + 阈值自动标定（bootstrap，解决问题 A/B）

- `ClubAssigner` 增加 bootstrap 模式（CLI `--auto-calibrate [N]`，N=标定帧窗）：
  1. 前 N 有效帧的双亮色样本做圆形 2-means（复用 `_circular_2means`）→ 两队参考色；
  2. 距两队都超阈值的拒绝带样本中位 → 裁判参考色；
  3. 深色路径隔离出的样本 → 两队门将参考色（深色分档逻辑复用）；
  4. 输出标定报告（建议 RGB + 置信度 + 样本量），支持人工覆盖；
- 阈值自动拟合：对帧级距离分布做 2-类分割（直方图 Otsu 或 1D 2-means）自动定 `referee_assign_dist`（gk 阈值同法），保留手动覆盖优先；
- 验收：demo2/raw 上自动标定值与人工实测值 hue 偏差 ≤ 10°；demo2/raw 标注集指标零回退；demo1/3/4 零配置跑通并与 Phase 0 基线②（手工调参）对比不显著更差。

### Phase 3 — 阴影不变性取色（问题 C 的颜色层尝试，允许失败）

- A/B 同口径对比实验（只改 `extract_jersey_stats` 内部）：
  a. crop 级 CLAHE（HSV 的 V 通道均衡）后取色；
  b. 色度归一化 r/(r+g+b)（光照不变色度）替代原始 HSV 取色；
- 验收口径：raw final_raw_labels 的 boundary 标签子集 club preservation 较基线提升 ≥ 10pp 且其余子集不回退 → 上线最优方案；两方案都无提升 → **记录结论归档**（证明该残差只能靠 Level B），Phase 关闭不算失败。

### Phase 4 — 时序解码：HMM/Viterbi 替代滑窗多数表决（问题 D-1）

- 每个 track 维护状态序列（club1/club2/referee/abstain），发射=帧判定，转移惩罚抑制抖动，Viterbi 解码最优分段（手写，~几十行，无新依赖）；
- 滑窗保留为退化路径（HMM 参数不可信时）；
- 验收：合成 ID 切换测试切换延迟较 60 帧滑窗下降 ≥ 50%；demo2 r74 型污染 track 实测滞后下降；temporal_flips 不升；标注集指标零回退。

### Phase 5 — 轻量 ReID 碎片合并（问题 D-2）

- `tools/merge_tracks.py`（先离线、作用于 tracks JSONL）：track 级 HSV 直方图（躯干条带统计已有）+ 时空门（间隙短、位移合理）+ 匈牙利匹配，合并碎片 track；
- 可选运行时模式（`--merge-fragments`，默认关）；
- 验收：demo2 referee ID 数（现 48）显著下降；raw 自洽统计 flagged/restore 不回退；**错并抽检为 0**（人工 + 双模型各看 ≥ 30 对合并）；对 p210 型长 track 无误伤。

### Phase 6 — 弱标注器（问题 F，给 YOLO_FINETUNE 提速）

- `tools/auto_pseudo_label.py`：颜色层高置信判定（连续 ≥ K 帧一致 + Phase 4 解码平滑 + Phase 5 合并后的 track）自动产出 referee/gk 伪标签框，对接 `tools/build_finetune_dataset.py`；
- 纪律：伪标签**只进 train split**，test 永远人工/双模型核验（YOLO_FINETUNE split 纪律不变）；
- 验收：抽 ≥ 100 张伪标签人工核验 precision ≥ 95%；首批产出量计入 YOLO_FINETUNE §3 进度表（量级目标以该文档为准）。

### Phase 7 — 终验

- 6 视频全量重放（raw1/2 回归 + demo1~4 泛化）；零配置模式 vs 手工模式全指标对照表；
- 老验收全查：referee 标注集、34-crop balanced acc=1.0、raw selfcheck、全量 pytest、端到端速度；
- kimi + luna 双模型独立交叉复核均通过后关闭。

## 6. 涉及文件

| 文件 | 改动 |
|------|------|
| `club_assignment/club_assigner.py` | Phase 1 债清理；Phase 2 bootstrap + 阈值拟合；Phase 3 取色实验；Phase 4 HMM 解码 |
| `main.py` / `tools/replay_assign.py` | `--auto-calibrate` / `--green-hue-range` / `--merge-fragments` 等参数暴露 |
| `tools/eval_generalization.py`（新增） | Phase 0 泛化基线 + 零配置 vs 手工对照 |
| `tools/merge_tracks.py`（新增） | Phase 5 碎片合并 |
| `tools/auto_pseudo_label.py`（新增） | Phase 6 伪标签 |
| `tests/test_club_assignment.py` 等 | 各 Phase 回归测试（债清理 tie-break/生命周期、bootstrap、HMM 切换延迟、合并正确性） |
| `eval/new_match_crops/`（新增） | demo1/3/4 小标注集（清单入 git，图片不入） |

## 7. 验收标准（整体）

1. demo1/3/4 零配置 balanced acc ≥ 0.9（或较 Phase 0 手工基线差距 ≤ 3pp）；
2. 本场全指标零回退（§3 所列基线数字）；
3. boundary 子集 club preservation ≥ 0.66（0.56 + 10pp）或归档"颜色层不可解"结论；
4. ID 污染切换延迟下降 ≥ 50%、裁判碎片 ID 显著减少且错并为 0；
5. 伪标签 precision ≥ 95% 且只进 train；
6. 全量 pytest 绿；速度回退 ≤ 20%；
7. 双模型交叉复核通过。

## 8. 风险与对策

- **bootstrap 被观众/工作人员色污染** → 标定窗内做离群剔除（圆形中位距离门）+ 置信度不足时降级为"要求人工给两队色"并明确报告，不静默错标；
- **阈值拟合双峰不存在**（两队色距太小） → 退回默认 85 + 报告，不强拟合；
- **CLAHE/色度归一化拖慢取色** → 只在 crop 条带内做（几十像素量级），实测耗时入验收；
- **HMM 过度平滑真切换**（换人/裁判交替） → 转移惩罚调参以合成切换测试为准，保留滑窗退化路径；
- **ReID 色度相近队伍误并** → 深色/相近色对只走时空强门限（间隙 ≤ 0.5s 且位移 ≤ 阈值），宁缺勿并；
- **伪标签确认偏差**（模型错误自我强化） → 高置信门槛 + 只进 train + 抽检核验，test 集隔离不变；
- **与 BALL_DETECTION 并行抢本机算力** → 全量重放错峰（其球检测 GPU/NPU 队列与本计划 CPU 侧为主，冲突有限）。

## 9. 推进记录

| Phase | 实现模型 | 验证模型 | 基线指标 | 完成后指标 | 裁决 | 提交 |
|-------|----------|----------|----------|------------|------|------|
| 0 | — | — | — | — | — | — |
