# PLAN：移动机位下的鲁棒场地检测（Robust Pitch Registration for Moving Camera）

分支：`codex/robust-pitch-registration`
基线：`main` @ 1707b2e

## 1. 背景与问题

当前场地注册链路（`main.py` → `FootballVideoProcessor` → `ObjectPositionMapper` → `DynamicCameraCalibrator`）：

1. `KeypointsTracker`（`tracking/keypoints_tracker.py`）用 YOLO pose 模型（`models/weights/keypoints-detection.pt`，32 点场地关键点）每隔 5 帧检测一次；
2. `DynamicCameraCalibrator._estimate_direct()`（`position_mappers/camera_calibrator.py`）用 RANSAC 求 homography，并用跨度比（长度 ≥30%、宽度 ≥25%）、内点率、中位误差等阈值把关；
3. 检测间隔内用 `_estimate_flow()`（LK 光流 + 场内 GFTT 特征点）传播 homography，最多传播 1.0s；
4. 超出传播窗口或镜头切换（直方图相关性 <0.35 / 运动比 >0.25）就重置，靠 `detect_now()` 重新检测。

针对**视角持续变化**（XbotGo 类跟拍：变焦、俯仰、平移、切镜、局部视野）的视频，现有方案的主要缺陷：

| # | 缺陷 | 位置 |
|---|------|------|
| A | `_map_detection()` 只取 `keypoints.xy[0]`（第一个 pose 实例），不取置信度最高/关键点最多的实例，角度变化时容易拿到残缺或错误实例 | `tracking/keypoints_tracker.py:79` |
| B | 固定 5 帧检测间隔：机位快速变化时光流跟不上、漂移大；稳定时又白白浪费 1280px 推理算力 | `tracking/keypoints_tracker.py:21` |
| C | 跨度阈值直接淘汰近景/局部视野（禁区特写、半场画面），这些画面完全没有标定输出，位置映射全丢 | `camera_calibrator.py:222-229` |
| D | 只依赖稀疏关键点，不用场地线（边线/禁区线在任意视角下都是最强的几何证据）；光流传播纯视觉、无相机运动模型，1s 后强制失效 | `camera_calibrator.py:148-165` |
| E | 已有 `HomographySmoother`（`homography.py`）从未被使用，没有时域平滑/滤波 | `position_mappers/homography.py:5` |
| F | 无法评估：没有标定质量/复投影误差的量化与调试输出 | — |

## 2. 目标

- 对视角变化的视频，标定有效帧率（有 `image_to_pitch` 的帧占比）显著提升，且局部视野（禁区、半场）也能给出标定；
- 中位复投影误差不高于当前基线（先测基线，再定指标）；
- 静态转播机位视频不回归；现有测试（`tests/test_dynamic_calibration.py` 等）全绿；
- 保持 CPU/OpenVINO 可运行，推理开销可控。

## 3. 方案：分层融合 + 退化链

核心思想：**YOLO 关键点不是唯一来源**。构建一条「关键点 → 线特征 → 运动传播」的融合与退化链，每一层失效自动降级，恢复后自动回升。

```
┌─────────────────────────────────────────────────────────────┐
│ YOLO 关键点（多实例合并 + 置信度加权 + 自适应间隔）          │
│   ↓ 不可用/低质量                                          │
│ 场地线检测（LSD/HoughP → 消失点聚类 → 线与模板匹配求 H）    │  ← 新增
│   ↓ 不可用                                                  │
│ 相机运动传播（全局帧间单应 ECC/Farneback + 位姿滤波，长窗口）│  ← 增强
└─────────────────────────────────────────────────────────────┘
    有 YOLO 关键点且线检测成功时 → 联合优化精修 H（关键点 + 线 + 时域平滑）
```

### Phase 0 — 评估与调试基础设施（先做，给后续定指标）
- `--debug-calibration` 开关：输出调试视频（叠加关键点/检测线/重投影线）、逐帧 JSONL（H、status、质量分、耗时）。
- 评估脚本 `tools/eval_calibration.py`：对用户移动机位视频采样帧跑当前管线，统计：有效帧率、各状态（detected/fused/propagated/invalid）占比、可见地标人工标注重投影误差基线。
- 从用户视频收集 100~200 张多角度帧存档，供 Phase 1/2 调参与回归测试。

### Phase 1 — 关键点利用的快速修复（低风险、高收益）
1. `_map_detection()` 改为遍历全部 pose 实例，按 index 取最高置信度合并，并输出逐点置信度（供加权用）；
2. `_estimate_direct()` 使用 `findHomography` 的权重版本（置信度加权 + 迭代重加权 IRLS），替代现在的硬阈值一刀切；
3. 检测间隔自适应：质量分/传播年龄越差，间隔越短（最小 1 帧），稳定后逐步放宽；
4. 放宽跨度阈值的同时引入「部分视野先验」：只用可见侧的关键点 + 已知场地尺寸约束（利用禁区/球门区的已知几何补全不可见侧，允许 H 存在方向歧义时以最近一次有效 H 定夺）。

### Phase 1.5 — 放宽跨度阈值支持局部视野（低风险，可先行落地）

针对缺陷 C（`camera_calibrator.py:222-229` 硬拒绝长度跨度 ≥30% / 宽度跨度 ≥25%），禁区特写、半场、局部视野画面会被整帧判为无效，位置映射全丢。

**关键认识**：单应是全局投影变换，≥4 个非共线关键点即可解出全场 H。RANSAC + 中位误差 ≤5px + 内点率 ≥60% 已能保证可见区域内的定位精度；局部视野真正不可靠的只是"远离可见区域的外推"。YOLO 仅负责找点（32 个已知世界坐标的地标），不需要重新训练，也不能靠模型"脑补"出球场——那是单应矩阵的职责。

**改动**：

1. `camera_calibrator.py`：删除 30%/25% 两个跨度硬拒绝条件，替换为两个退化保护：
   - 反共线保护：内点世界坐标在 x、y 两个方向的跨度各 ≥ 6m（共线点求出的 H 无效）。注意：固定 6m 会误杀"仅球门区"特写（球门区纵深仅 5.5m，x 跨度不足 6m），实施时改用内点散布矩阵的奇异值比（如次大/最大 ≥ 0.25）或相对阈值；
   - 最小视野保护：至少一个方向跨度 ≥ 10% 球场（防止单点噪声被放大成"标定"）。
   - `quality` 公式中的 `min(1.0, span/阈值)` 因子已天然让局部视野质量下降（干净检测下禁区特写约 0.5 分，仍可过球速门槛 0.35，见 `ball_speed_estimator.py:139`），无需重复拒绝。
2. `object_position_mapper.py`：可选"可信区域"保护（默认开，`--allow-extrapolation` 可关闭）：
   - `CalibrationResult` 新增 `observed_world_bbox`（内点世界坐标包围盒），随 `serializable()` 自动进入 JSONL；
   - `map()` 中算出的位置若落在可见包围盒外扩余量（max(10m, 包围盒尺寸/2)）之外则跳过 `position_m`（默认，最安全）或写入但带 `extrapolated` 标记（便于事后分析外推帧）；
   - 全景视角下包围盒≈全场，行为不变、无回归。
3. `summarize_match.py:100-101`：旧格式 JSONL 回放路径的同门槛同步放宽，保持新老管线行为一致。
4. 测试（`tests/test_dynamic_calibration.py` 扩展）：
   - 仅可见禁区/球门区索引 → 标定有效且 quality > 0；
   - 共线/极小视野 → 拒绝；
   - 可信区域外的位置 → 不写 `position_m`（或带 `extrapolated`）；
   - 全景视角 → 既有行为无回归。

**不做**：检测间隔、光流传播、切镜检测、`min_keypoints=6` 均保持不变。

**验收**：局部视野帧能持续输出标定与位置数据；远处外推精度低于近处属预期，需在 JSONL/文档中体现质量差异。

### Phase 2 — 基于场地线的注册（新模块，不依赖 YOLO）
- 新增 `position_mappers/pitch_line_detector.py`：绿屏掩膜 → Canny/LSD 提取线段 → 按消失点聚类为「横线/竖线/圆（中圈弧）」。
- 新增 `position_mappers/pitch_line_fitter.py`：检测线 ↔ 模板线（`PitchGeometry`）的匹配 + RANSAC 求 H（线单应拟合）；对 2~3 条线的弱观测用「有限候选假设 + 上一帧先验」选出最稳解。
- 接入退化链：YOLO 关键点无效时先尝试线注册，成功同样写入 `CalibrationResult`（新状态 `line_detected`）。

### Phase 3 — 相机运动传播增强
- 新增 `position_mappers/camera_motion.py`：帧间全局单应（ECC 或 Farneback 光流 + RANSAC），与上一帧有效 H 复合；替代/补充现有场内 GFTT+LK 方案。
- 传播窗口 1.0s → 3~5s，质量分随传播年龄指数衰减（下游 `speed_usable` 已有 0.5s 窗口，速度不受影响）；
- 用 `HomographySmoother` 或 EKF（状态=单应参数，观测=直接检测）做时域平滑，消除关键点抖动；
- 保留现有切镜检测（直方图+运动比），补充渐变转场（长淡入淡出）处理。

### Phase 4 — 联合精修
- 当关键点与线检测同时可用：构造联合能量（关键点复投影残差 + 线残差 + 与上一帧 H 的时域平滑项），LM 优化 1~2 步精修 H。
- 入口：`camera_calibrator.py` 的 `update()` 在 fused 路径后追加 refine 步骤。

### Phase 5 — 数据回流（可选，视 Phase 1 结论）
- 若评测显示模型本身是瓶颈（关键点漏检/错检占主导），用现有 `training/prepare_dataset.py` + `train_models.py --task pitch` 在用户多角度数据上微调 `keypoints-detection.pt`。仅在前四阶段证明必要后启动。

## 4. 涉及文件

| 文件 | 改动 |
|------|------|
| `tracking/keypoints_tracker.py` | 多实例合并、自适应间隔 |
| `position_mappers/camera_calibrator.py` | 加权 RANSAC、退化链、部分视野先验、精修钩子；Phase 1.5：放宽跨度阈值 |
| `position_mappers/object_position_mapper.py` | Phase 1.5：可信区域（`observed_world_bbox`） |
| `position_mappers/pitch_line_detector.py` | **新增** 线检测 |
| `position_mappers/pitch_line_fitter.py` | **新增** 线→H 拟合 |
| `position_mappers/camera_motion.py` | **新增** 全局帧间运动估计 |
| `position_mappers/homography.py` | 启用平滑器 / 时域滤波 |
| `annotation/football_video_processor.py` | 退化链接线、调试输出 |
| `main.py` | `--debug-calibration`、线注册开关、Phase 1.5：`--allow-extrapolation` |
| `summarize_match.py` | Phase 1.5：放宽旧格式回放门槛 |
| `tests/` | `test_pitch_lines.py`、`test_camera_motion.py`、扩展 `test_dynamic_calibration.py` |

## 5. 验收标准

1. 用户移动机位样本集上：有效标定帧率 ≥ 90%（基线待 Phase 0 测定）；近景（禁区特写）也能输出标定；
2. 中位复投影误差 ≤ 基线水平（Phase 0 测定）；
3. 静态视频与现有测试无回归；
4. CPU 端到端处理速度下降 ≤ 20%；
5. 输出 tracks JSONL 中 calibration 状态字段向后兼容（新增状态值不破坏现有消费者）。

## 6. 风险与对策

- **线检测在草地纹理差/阴影重/观众席干扰下失败** → 用 H 约束的候选假设 + 时域先验兜底，仍失败则回到光流传播；
- **变焦主导时全局单应估计退化** → 用关键点/线直接检测纠正，传播仅作插值；
- **球员遮挡线** → 线拟合 RANSAC 本身抗遮挡；必要时加入前 N 帧线图累加；
- **性能** → 线检测可降频（每 2~3 帧）、降分辨率执行；Phase 0 起监控耗时。
