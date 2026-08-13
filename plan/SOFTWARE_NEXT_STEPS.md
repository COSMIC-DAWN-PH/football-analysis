# Software Next Steps：软件端状态与后续计划

> 更新日期：2026-08-14  
> 本文件是软件端工作的唯一权威计划。需要用户采集、确认和标注的内容见 [MANUAL_ACTIONS.md](MANUAL_ACTIONS.md)。

## 1. 当前结论

原始结果长期显示 `pending`，是三类问题叠加：

1. 相机与球场关系不稳定：关键点覆盖不足、RANSAC 内点率低，速度可用姿态很少。
2. 球检测质量低：低阈值候选会把白线、点球点及其他白色小目标当成球。
3. 轨迹与标定原来耦合：姿态丢失会打断球轨迹，无法积累速度样本。

单应性只能可靠描述地面平面。空中球速度必须依赖 Camera Profile、逐帧 Camera Pose、多帧射线和带重力约束的物理拟合；结果只能称为“单目物理模型估计”。

## 2. 已完成的软件能力

| 阶段 | 已完成 Implementation | 当前外部阻塞 |
|---|---|---|
| P0 诊断 | `warming_up / reliable / unavailable`、稳定原因码、逐帧 `diagnostics.jsonl`、`quality_summary.json`、正式画面隐藏不可靠速度、调试模式显示原因。 | 需要 M3/M4 的真实目标视频建立新基线。 |
| P1 球追踪 | 白线/白点证据、米制标线惩罚、可选 ball/non-ball verifier、最多 3 条候选路径、0.2 秒固定滞后、二维轨迹与姿态失效解耦、模型 promotion 检查。 | 需要 M5/M6 训练和未见球场验收。 |
| P2 相机几何 | `CameraProfile`、`PitchAnchorSet`、`CameraPoseResult`、棋盘格/锚点 CLI、关键点/标线/光流联合质量、人员区域屏蔽、分辨率和变焦 fail-closed。 | 需要 M1–M4 生成真实设备 Profile 与锚点；关键点模型仍需重训。 |
| P3 三维速度 | 地面射线求交、空中重力弹道、球尺寸弱约束、运动事件分段、速度向量、模式选择、不确定度和快速失败。 | 需要 M7 的雷达或第二机位参考集验收。 |

当前自动测试为 72 个全部通过，真实 1080p 两帧集成烟雾测试通过。现有正式权重仍是静态 640 导出，球模型为 YOLO11n，所以 3D 正式模式会主动拒绝这些权重。这是准确率优先的预期行为。

## 3. 当前公开 Interface

- `CameraProfile`：相机矩阵、畸变、分辨率、焦段、版本和 RMS。
- `PitchAnchorSet`：参考帧、4–8 个图像点及对应米制球场坐标。
- `CameraPoseResult`：旋转、平移、单应矩阵、重投影误差、不确定度、状态和失败原因。
- 球轨迹结果：`track_state`、候选评分、二维位置、轨迹段、候选路径数量和拒绝原因。
- `BallSpeedResult`：`warming_up | reliable | unavailable`、三维速度、速度向量、运动模式、不确定度和原因。

统一原因码：

```text
pose_invalid
pose_stale
track_tentative
low_ball_confidence
track_ambiguous
insufficient_samples
trajectory_unobservable
uncertainty_high
zoom_changed
no_camera_profile
no_ball
```

数据流固定为：

```text
视频帧
→ 人员/球场关键点检测
→ Camera Pose
→ 球候选
→ 白线与外观评分
→ 固定滞后多候选路径
→ 地面/空中三维轨迹
→ 速度与不确定度
→ 标注、JSONL 和质量摘要
```

## 4. 后续软件任务队列

任务编号 `S1–S8` 与人工资料编号 `M1–M7` 对应。缺少所列输入时不得用猜测值绕过。

### S1：人工输入审计

触发条件：M1–M4 完成并存在 `training-data/manual-input/READY.md`。

执行：

- 检查棋盘格照片与比赛视频的宽、高、方向、FPS 和元数据。
- 检查同一 `camera-profile-id` 是否混入不同分辨率、焦段或数字变焦。
- 检查视频是否被裁切、缩放、重编码或包含剪辑切换。
- 检查球场长宽、标线类型和参考点说明是否完整。
- 生成输入审计报告。

输出：

```text
output_videos/calibration/<camera-profile-id>/input_audit.json
output_videos/calibration/<camera-profile-id>/INPUT_AUDIT.md
```

完成判据：所有标定照片和目标视频可归属于唯一录制配置；发现冲突时停止 S2，不自动缩放内参。

### S2：生成相机内参与球场锚点

依赖：S1 通过，M1–M4 完整。

执行：

- 剔除无法识别完整棋盘格或明显模糊的照片。
- 运行棋盘格标定，检查 Camera Profile 的 RMS 和分辨率。
- 选择连续镜头参考帧，启动交互式 4–8 点球场锚点工具。
- 计算姿态和重投影误差；覆盖不足时请求重新选点。

输出：

```text
output_videos/calibration/<camera-profile-id>/camera_profile.json
output_videos/calibration/<pitch-id>/<shot-id>/pitch_anchors.json
output_videos/calibration/<pitch-id>/<shot-id>/calibration_report.json
```

主要命令：

```powershell
.\.venv\Scripts\python.exe -m calibration camera `
  --images "training-data\manual-input\cameras\<camera-profile-id>\checkerboard\*.jpg" `
  --columns 9 --rows 6 --square-size-m <实测米数> `
  --profile-id <camera-profile-id> `
  --output output_videos\calibration\<camera-profile-id>\camera_profile.json

.\.venv\Scripts\python.exe -m calibration pitch `
  --video input_videos\manual-validation\<pitch-id>\<shot-id>-original.mp4 `
  --frame 0 `
  --pitch-points "<按顺序排列的米制点>" `
  --output output_videos\calibration\<pitch-id>\<shot-id>\pitch_anchors.json
```

完成判据：分辨率完全匹配，锚点同时覆盖球场长宽，中位关键点重投影误差目标 ≤3 px。

### S3：运行真实视频诊断基线

依赖：S2 输出和 M3 原始视频。

执行：

- 先以 `ground` 模式运行，避免在模型未 promotion 前伪装成 3D。
- 统计 Camera Pose 覆盖率、候选数、轨迹段数、确认轨迹、原因分布和可靠速度覆盖率。
- 抽查白线、点球点、静止球、遮挡和镜头运动帧。
- 将新结果与此前 30 秒基线比较。

输出：

```text
output_videos/baselines/<pitch-id>-<shot-id>/
├── <shot-id>-analysis.mp4
├── BASELINE_REPORT.md
└── raw/
    ├── object_tracks.jsonl
    ├── keypoint_tracks.jsonl
    ├── calibration_tracks.jsonl
    ├── diagnostics.jsonl
    └── quality_summary.json
```

完成判据：失败原因可以量化，并能判断主要瓶颈属于球模型、Camera Pose 还是轨迹可观测性。

### S4：生成困难数据复核队列

依赖：S3 完成和 M5 多球场素材到位。

执行：

- 抽取球检测和 32 点球场关键点候选帧。
- 挖掘贴合标线、低置信度、候选跳变和已确认疑似假轨迹。
- 建立白线、点球点、白鞋袜、反光、垃圾、空场景和真球在线上的专项队列。
- 按完整视频和球场预分组，避免 train/val/test 泄漏。

输出给用户复核：

```text
training-data/review-queue/ball/
training-data/review-queue/ball-verifier/
training-data/review-queue/pitch-keypoints/
```

完成判据：至少覆盖 4 块球场，并保留一块训练阶段完全不可见的测试球场。

### S5：训练和 promotion 球模型与二级分类器

依赖：M6 的 `approved/ball-yolo` 和 `approved/ball-verifier`。

执行：

- 训练 YOLO11s 球检测模型，输入尺寸 1280，导出动态 OpenVINO FP16。
- 训练 ball/non-ball 分类器。
- 在验证集上选择“召回率 ≥75% 时精确率最高”的分类阈值，并写 sidecar。
- 在未见球场进行检测、假轨迹和白线专项验收。
- 训练脚本不自动覆盖正式权重。

训练输出：

```text
runs/xbotgo/xbotgo-ball/
runs/xbotgo/xbotgo-ball-verifier/
```

通过 promotion 后才人工复制到：

```text
models/weights/ball-detection-promoted-openvino/
models/weights/ball-verifier-promoted-openvino/
```

完成判据：未见球场精确率 ≥95%、可见球召回率 ≥75%、确认假轨迹 ≤0.5 次/分钟，白线/白点专项中没有持续超过 0.2 秒的确认假轨迹。

### S6：训练和 promotion 球场关键点模型

依赖：M6 的 `approved/pitch-yolo`。

执行：

- 重训 32 点球场模型，覆盖 XbotGo 远景、近景、俯视、逆光、平移和固定焦段镜头。
- 导出动态 1280 OpenVINO FP16。
- 验证镜头切换、变焦检测、短时缺点传播和人员区域屏蔽。
- 比较人工锚点和自动关键点两个 Adapter 的统一 `CameraPoseResult`。

通过 promotion 后才复制到：

```text
models/weights/keypoints-detection-promoted-openvino/
```

完成判据：支持视频 Camera Pose 可靠覆盖率 ≥80%，中位关键点重投影误差 ≤3 px；镜头切换或变焦后不得继续使用旧姿态。

### S7：启用并验收三维球速

依赖：S5/S6 通过，M7 参考数据到位。

执行：

- 使用原始视频、Camera Profile、Pitch Anchor Set 和 promotion 模型运行 `--speed-mode 3d`。
- 分别评估地面球、空中球、静止球、射门、遮挡、反弹和相机平移。
- 检查接触事件是否正确建立新运动段。
- 计算绝对误差、相对误差、不确定度覆盖和错误 reliable 数量。

主要命令：

```powershell
.\.venv\Scripts\python.exe main.py `
  --input <原始视频> --run-dir <输出目录> `
  --pitch-length-m <实测长度> --pitch-width-m <实测宽度> `
  --speed-mode 3d `
  --camera-profile <camera_profile.json> `
  --pitch-anchors <pitch_anchors.json> `
  --ball-model models\weights\ball-detection-promoted-openvino `
  --keypoints-model models\weights\keypoints-detection-promoted-openvino `
  --ball-verifier-model models\weights\ball-verifier-promoted-openvino `
  --debug-diagnostics --no-preview

.\.venv\Scripts\python.exe -m evaluation `
  --tracks-dir <输出目录>\raw `
  --ground-truth <参考 JSONL> `
  --fps <原始 FPS> --output <输出目录>\evaluation.json
```

完成判据：地面球中位误差 ≤3 km/h 或 ≤10%；空中球 ≤5 km/h 或 ≤15%；不满足观测条件的帧不得输出 `reliable`。

### S8：正式配置与回归发布

依赖：S5–S7 全部通过。

执行：

- 固定正式模型路径、阈值 sidecar、Camera Profile 和使用限制。
- 保存黄金视频指标，防止后续改动回退。
- 完整运行自动测试、模型元数据检查和目标视频回归。
- 在输出和文档中保留“单目物理模型估计”表述。

完成判据：正式模型不由训练脚本自动覆盖；72 个现有测试及新增真实黄金指标测试全部通过；质量门槛失败时保持 fail-closed。

## 5. 统一运行产物

每次正式运行的 `raw` 目录固定包含：

```text
object_tracks.jsonl
keypoint_tracks.jsonl
calibration_tracks.jsonl
diagnostics.jsonl
quality_summary.json
```

球速状态只允许：

```text
warming_up
reliable
unavailable
```

正常画面只显示 `reliable` 数值；调试模式才显示失败原因。

## 6. 验收总门槛

### 数据和模型

- 至少 4 块球场，按完整视频和球场划分。
- 测试集至少包含 1 块训练阶段完全未见的球场。
- 第一轮至少 800 帧球数据、400 帧球场关键点数据。
- 球精确率 ≥95%，可见球召回率 ≥75%。
- 确认假球轨迹 ≤0.5 次/分钟。

### 相机姿态

- Camera Pose 可靠覆盖率 ≥80%。
- 中位关键点重投影误差 ≤3 px。
- 镜头切换和变焦后不使用旧姿态。
- 短时姿态缺失不重置二维球轨迹。

### 三维速度

- 地面球至少 8 个观测且跨度 ≥0.25 秒。
- 空中球至少 12 个观测且跨度 ≥0.35 秒。
- 95% 不确定度半宽 >5 km/h 时不输出可靠速度。
- 地面误差 ≤3 km/h 或 ≤10%。
- 空中误差 ≤5 km/h 或 ≤15%。
- 单目结果不宣传为雷达级测量。

## 7. 不允许的软件捷径

- 不把现有静态 640 模型冒充动态 1280 promotion 模型。
- 不在缺少 Camera Profile 或 Pitch Anchor Set 时静默退回伪三维速度。
- 不用裁切、缩放或重编码视频做正式三维测量。
- 不跨触球、碰撞、反弹或方向突变平滑。
- 不用预测帧新增可靠速度样本。
- 不为了提高覆盖率降低可靠性门槛。
- 不自动覆盖 `models/weights` 中的正式权重。

## 8. 下一次继续工作的入口

当前首先等待 [MANUAL_ACTIONS.md](MANUAL_ACTIONS.md) 中的 M1–M4。

当 `training-data/manual-input/READY.md` 存在后，从 S1 开始；如果 S1 发现分辨率、焦段、变焦或视频来源不一致，则返回对应的 M 任务补充资料，不继续 S2。

