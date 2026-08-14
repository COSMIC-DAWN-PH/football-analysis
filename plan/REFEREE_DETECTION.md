# PLAN：裁判识别优化（Referee Detection & Club Assignment）

分支：本工作从当前工作区直接推进（未开新分支，如需要可随时 `git branch`）
数据源：`demo2.mp4`（2.5min，已有 tracks）、`C:\Personal Profile\Profile\Video\raw1.mp4` / `raw2.mp4`（各 30min，同场比赛上下半场）
硬件：Core Ultra 5 125H（Arc iGPU 145ms/帧、AI Boost NPU 156ms/帧 @ object-detection 1280 fp16；无 CUDA）

## 1. 背景与问题（demo2-30s-test 实测确认）

裁判球衣为**黄色**（hue≈26）。`object-detection.pt` 的 referee 类双向不可靠：

| 现象 | 数据证据（demo2 object_tracks.jsonl） |
|------|--------------------------------------|
| 真裁判被判成 `player` → `ClubAssignerModel.predict` 强制二选一（`np.argmin`，无"非两队色"拒绝出口），黄 hue26 距 Maroon(170)=36° 比 Navy(107)=79° 近 → 判给 Maroon → **画成红色队员** | player 类中位 hue 25-28 的 track：4(148帧)、104(117)、217(81)、83(42)、288(28)、203(10)、124(被判 Navy) 等 |
| 真球员被 YOLO 误判成 `referee` → 深灰虚线框、丢掉队色 | referee 类 track 197(153帧, hue107 藏青球员)、58(hue107)、243(hue172 栗色)、2(hue169) |
| referee 类 ID 碎片化（真裁判一段一 ID） | 48 个 referee ID，多数仅活 1-28 帧 |

根因在 `club_assignment/club_assigner.py`：分类无拒绝出口；referee track 完全不参与颜色逻辑。

## 2. 目标

- 裁判稳定显示为裁判样式（深灰虚线），不再变红/蓝队员；红蓝队员不再被画成裁判；
- 上轮球队颜色成果不回退（34-crop 验证集 balanced acc 保持 1.0）；
- 以人工标注验证集为准，kimi-k2.7-code 与 gpt-5.6-luna 双模型多模态交叉验证通过；
- 为 Level B（YOLO 微调）产出三源标注数据集（本轮只备数据、不训练）。

## 3. 工作方式（沿用上轮团队颜色计划的验证门）

1. 一个 Phase 一次推进，单独 commit；
2. Phase 完成后由 openchamber 会话起 `opencode-go/kimi-k2.7-code` 与 `opencode-go/gpt-5.6-luna` 独立验证：**看图**核对 referee 判定 vs 人工标注、自己跑 eval 复算指标、复核 diff、给"通过/不通过"裁决；
3. 不通过 → 修复 → 重新验证；两票都过才进下一 Phase；
4. 全程记录到本文档「推进记录」。

## 4. Phase 0 — 候选截图 + 人工标注 + 备微调数据

### 4.1 数据获取
- `tools/extract_referee_candidates.py`：从 tracks JSONL + 视频裁剪候选到 `eval/referee_crops/{source}/`，按 track 聚合（每 track 选最清晰 k 帧，控制标注量）：
  - 黄色 hue（15-45）player tracks；
  - 全部 referee 类 tracks（含红蓝假阳性，供人工纠正）；
  - hue 边界样本（45-95 模糊带）；
  - 分层抽样正常红蓝球员作对照。
- demo2：复用 `output_videos/demo2-30s-test/raw/object_tracks.jsonl`，最快产出；
- raw1/raw2：`tools/detect_tracks_only.py`（只跑 object 模型检测+ByteTrack，不跑 keypoints/ball、不渲染视频）**全量**跑出 tracks，raw1→GPU、raw2→NPU 并行（各约 4.5h 墙钟）；跑完同脚本补出候选。

### 4.2 人工标注
- 用户逐张标 `referee / maroon / navy`（沿用 `eval/README.md` 规则：只收清晰无歧义样本，无法判定不进评估）；
- 定稿 `eval/referee_labels.json`（含 source/frame/track/yolo_class/hue 元数据）。

### 4.3 微调数据集（Level B 备用）
- 按标注结果整理 Roboflow 兼容 YOLO 格式数据集到 `eval/finetune_dataset/`：四类 `ball/goalkeeper/player/referee`，referee 类用本轮标注直接标，player/goalkeeper 由 YOLO 预标+人工抽查；
- 本轮不训练。

## 5. Phase 1 — 颜色后处理（核心，不动 YOLO 权重）

改 `club_assignment/club_assigner.py`：

1. **帧级"第三簇"**：现有 `_circular_2means` 扩展——若存在与两队参考色 hue 距离都 ≥ 阈值的簇（黄 26 vs Maroon 170/Navy 107）且人数 ≥2，该簇 tracks 判 referee（不分配 club），避免硬阈值误伤阴影栗色球员；
2. **单点兜底**：`predict()` 距两队 HSV 加权距离都超阈值 → 返回 `None`（referee，宁缺毋滥）；
3. **反向纠正**：referee 类 track 的颜色若与某队参考色相近 → 恢复为该队球员（带 club/club_color）；
4. **时序投票**：referee/player 状态做滑动窗口投票（复用 `votes_by_track` 机制），消除类间闪烁；
5. `annotation/object_annotator.py`：referee track 带 club_color 时按球员样式绘制（队色实线椭圆），否则维持深灰虚线；
6. `annotation/football_video_processor.py`：把 referee tracks 接入 assign_clubs 的采样与投票。

## 6. Phase 2 — 双模型多模态交叉验证

- openchamber 起两个会话：`opencode-go/kimi-k2.7-code`、`opencode-go/gpt-5.6-luna`；
- 各自独立：① 读 `eval/referee_crops/` 图片核对判定 vs 人工标注；② 跑 `tools/eval_referee.py` 复算；③ 复核代码 diff；④ 裁决通过/不通过。

## 7. Phase 3 — 终验

- `tests/test_club_assignment.py` 补 referee 合成用例（黄衣→referee、藏青→球员、ID 切换、第三簇）；全量 pytest 绿；
- demo2 全量端到端重跑，肉眼确认；raw1/raw2 用 tracks 重放验证稳健性；
- 老 34-crop 验证集 balanced acc = 1.0 不回退。

## 8. Level B — YOLO 微调路线（后续轮次，写清楚怎么做）

**为什么需要**：颜色后处理能覆盖"颜色可分离"的裁判（黄/黑），但对与某队服同色的裁判（如黑衣裁判 vs 深色队、红衣裁判 vs 红队）无效，最终要提升 `object-detection.pt` 的 referee 类召回与精度。

**步骤**（复用现有 `training/` 工作流，参考 `training/README.md`）：

1. **数据**：用 Phase 0 的三源标注集 + 继续从 raw1/raw2 全量 tracks 挖掘更多 referee 正样本（远/近景、光照、机位），补充 hard negative（红蓝球员）——目标首轮 ≥ 600 张 referee 正样本；
2. **格式**：Roboflow 兼容 YOLO 目录（`images/`+`labels/`，四类 id 与现有模型一致：0=ball,1=goalkeeper,2=player,3=referee），整场视频不跨 split（70/15/15，至少一场完全未见过的比赛进 test，相邻帧不跨 split），`manifest.csv` 带 `split`+`source_video` 列；
3. **本地校验**：`training/prepare_dataset.py` 增加 `object` task（validate/extract/unpack），检查类 id、坐标、跨 split 泄漏；或走 Roboflow 私有项目导出 zip 后 `unpack`；
4. **训练**：`training/train_models.py` 增加 `object` task（detect、4 类、沿用 `models/object_detection_train.ipynb` 的超参：YOLO 预训练权重、imgsz 1280、epochs/bs 按 CUDA 云机调整），在 **CUDA 云机**（本机无 CUDA）跑：
   `python -m training.train_models --task object --data training-data/object-yolo/data.yaml --device 0`
5. **验收**：未见过比赛上 referee 召回 ≥ 75%、精度 ≥ 90%，且 player/goalkeeper 无回退；导出 OpenVINO FP16（1280）后与本项目同样的 `intel:GPU/NPU/CPU` 路径验证；
6. **上线**：新权重替换 `models/weights/object-detection.pt` + OpenVINO 导出目录；端到端重跑 demo2/raw 片段做校准与速度回归检查。

**本轮交付的前置工作**：Phase 0.4.3 的标注数据集 + referee 候选挖掘脚本（可复用参数再挖一轮）。

## 9. 涉及文件

| 文件 | 改动 |
|------|------|
| `tools/extract_referee_candidates.py` | 新增：候选裁剪（三源） |
| `tools/detect_tracks_only.py` | 新增：raw 全量 tracks-only 运行 |
| `tools/eval_referee.py` | 新增：referee 评估脚本 |
| `club_assignment/club_assigner.py` | referee 拒绝出口/第三簇/反向纠正/投票 |
| `annotation/object_annotator.py` | referee 带队色时按球员样式绘制 |
| `annotation/football_video_processor.py` | referee tracks 接入颜色逻辑 |
| `tests/test_club_assignment.py` | 扩展 referee 合成用例 |
| `eval/referee_crops/`、`eval/referee_labels.json` | 新增人工标注集 |
| `eval/finetune_dataset/` | 新增微调数据集（Level B 备用） |

## 10. 验收标准

1. 三源人工标注集上 referee 精确率/召回率显著提升（以 Phase 0 基线为准），且双向误判都减少；
2. 老 34-crop 验证集 balanced acc = 1.0（不回退）；
3. 有效分类覆盖率下降 ≤ 5pp；
4. 全量 pytest 绿；demo2 端到端处理速度下降 ≤ 20%；
5. kimi-k2.7-code 与 gpt-5.6-luna 双模型裁决均通过。

## 11. 风险与对策

- **黄色与栗色 hue 只差 36°**：硬阈值易误伤阴影栗色球员 → 优先帧级第三簇（无监督）+ 标注集上标定阈值；
- **NPU/GPU 并发跑 raw 全量**：OpenVINO 双进程可能抢资源 → 分别绑定设备，若变慢改串行；
- **raw 时长 60min**：tracks-only 不渲染视频、分批写 JSONL，防止内存/磁盘膨胀；
- **标注负担**：按 track 聚合 + 每源限配额（referee 候选 ≤80 张、对照 ≤20 张）。

### p210 案例结论（用户人工确认 2026-08-15）
- p210 = **navy 方门将**，球衣为红紫色（hue 120-160，sat 141/val 68 中位），既不贴 navy(107) 也不贴 maroon(170)。
- YOLO 全程只给 player 类（从未检出 goalkeeper 类），与任何 gk/referee track 无 IoU 重叠（>0.1 均无命中）。
- 结论：**颜色后处理边界案例**——门将（YOLO 漏检 + 特殊球衣色）无法靠颜色区分，属 Level B（YOLO 微调：提高 goalkeeper 类召回）。可选缓解：用户传入真实门将参考色（--club1-goalkeeper 用实际色而非占位灰），把"距门将参考色近"排除出 referee 判定。
- 指标影响：p210 3 张标 navy 后，referee precision 1.0 -> 0.961（crop）/ 0.941（track）；recall 不变。

### Luna 首轮裁决与修复（2026-08-15）
- luna（gpt-5.6-luna，openchamber 会话）首轮裁决：**不通过**。阻断项：referee_assign_dist=75 低于 maroon 实测最大距离 83（存在 75-83 区间球员被误判为裁判的风险，评估集未覆盖该边界）；次要项：get_player_club() 仍走无拒绝出口的 predict()。
- 修复（提交 820ec31）：① 阈值 75→85（落在 maroon 最大 83 与黄衣裁判最小 88 的安全区间）；② get_player_club 改用 predict_referee；③ 新增边界合成测试 3 项（pytest 75 全绿）。
- 修复后指标：p210（navy 门将，用户标注）0 帧 referee 旗标（修复前 72/91），referee precision 恢复 1.0（crop/track），recall 0.907/0.941 不变，club preservation 0.9875/0.975，老验证集 balanced acc 1.0。残余误差仅剩 r74 混合 ID track 滞后 4 张与 p124 无数据帧 1 张（Level B 范围）。

### p210 标签语义修正（2026-08-15，用户二次确认）
- p210 不是藏青球员：他是 **navy 队（方）的门将**，穿红紫门将服（hue 120-160）。标注词汇因此扩展：`navy_gk` / `maroon_gk` = 某队门将（按球队归属判定，不按球衣色判定）。
- `tools/eval_referee.py` 支持 _gk 标签：预期 club = 所属队名（navy_gk → navy），计入 club preservation；referee 指标里按"非裁判"处理。
- p210 三张 verdict 图已改标 `navy_gk`；评估：navy_gk->navy 3/3 正确，指标不变（precision 1.0 / recall 0.907-0.941 / preservation 0.9875-0.975）。

## 12. 推进记录




| Phase | 实现模型 | 验证模型 | 基线指标 | 完成后指标 | 裁决 | 提交 |
|-------|----------|----------|----------|------------|------|------|
| 0 | opencode-go/deepseek-v4-pro | 用户人工标记 | — | 三源 tracks 就绪（demo2 900 帧；raw1 53872 帧 GPU 5.7fps/157min；raw2 53872 帧 NPU 6.2fps/145min）；demo2 候选 113 张 + verdict 48 张（含补抽 p210），auto 标签预填；用户已人工核正 demo2 两个清单 | 通过（用户标注完成） | 未提交 |
| 1 | opencode-go/deepseek-v4-pro | 待 kimi-k2.7-code + gpt-5.6-luna | demo2 重放：裁判黄色 player tracks 全被判 Maroon/Navy | **用户标注口径**：crop referee recall 0.907 / precision 1.0（FN 5 张：r74 混合 track 窗口滞后 4 + p124 无数据 1）；track referee recall 0.941 / precision 1.0；club preservation crop 0.987 / track 0.974；club coverage 1.0；老 34-crop balanced_acc=1.0 无回退；pytest 72 绿；新增 `--referee-color` 三参考色判定（用户提供裁判色场景通用化，demo2 实测裁判参考色 RGB(168,156,74)，加入后指标不变=无回归）；端到端冒烟跑通 | 待双模型验证 | 未提交 |

### 边界案例结论（人工标注确认）
- r74/p28 为 ByteTrack ID 污染 track（黄裁判↔藏青/栗色球员混用同一 ID），滑窗 60 帧在换人后滞后 ~1s——颜色逻辑无法修复，属 Level B（YOLO 微调提高 referee 类召回，让裁判拿到独立 referee 类 ID）范围；
- p210（hue 125-165 漂移、18x40px 远距）已补抽 3 张大图待判；p124 f264 该帧无有效球衣像素。

### Phase 1 实现要点
- `club_assigner.py`：`ClubAssignerModel.predict_referee()` 双模式——无裁判参考色时按加权 HSV 距离阈值 `referee_assign_dist=75`（demo2 实测：黄衣 min-dist≥88、maroon p90=18/max=83、navy p90=16/max=71）判"非两队色"；有 `--referee-color` 时改为三参考色最近邻（距裁判色最近且 ≤`referee_match_dist`→裁判）；`_assign_frame()` 先分离裁判样本直接投票（60 帧滑窗多数表决消噪），剩余样本走原 2-means 聚类/最近质心兜底；`assign_clubs()` 采样与写回扩展到 referee 类 track（颜色匹配某队→恢复队色；不匹配→保持裁判）；player 类多数票 referee → `track['referee']=True` 并移除 club。
- `object_annotator.py`：referee 类带 club → 队色实线球员样式；player 带 referee 旗标 → 深灰虚线裁判样式。
- `projection_annotator.py`：投影图 referee 旗标球员画虚线圆；恢复队色的 referee 画队色实心圆；Voronoi 排除 referee 旗标球员。
- `main.py` / `tools/replay_assign.py`：新增 `--referee-color R,G,B` 参数。
