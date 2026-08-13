# Manual Actions：你需要准备的资料与放置位置

> 更新日期：2026-08-14  
> 本文件只记录需要你本人采集、确认或标注的内容。软件端后续工作见 [SOFTWARE_NEXT_STEPS.md](SOFTWARE_NEXT_STEPS.md)。

## 1. 使用方式

每项人工任务都有固定编号 `M1–M7`。完成后把资料放进本文指定的仓库目录，并告诉我“已完成 M1、M2……”即可。我会先检查文件，不会直接假设资料有效。

所有原始视频、照片、标注和测速参考数据都放在已经被 Git 忽略的 `input_videos/` 或 `training-data/` 下，避免误提交大文件和私人素材。

不要把资料放进：

- `models/weights/`：这里只放通过验收的正式模型。
- `output_videos/`：这里只放程序生成结果。
- `plan/`：这里只保留计划文档。

## 2. 第一批最小资料包：先完成 M1–M4

完成 M1–M4 后，就可以开始相机标定、人工锚点和真实视频基线，不需要先准备雷达或一次性标完 800 帧。

### M1：相机与录制设置说明

先为每一种固定录制配置取一个 `camera-profile-id`，例如：

```text
xbotgo-1080p30-1x
xbotgo-4k30-1x
```

分辨率、镜头、焦段/倍率、数字变焦或防抖裁切模式任意一项改变，都应该使用新的 profile，不可混用。

请创建并填写：

```text
training-data/manual-input/cameras/<camera-profile-id>/camera_info.md
```

模板：

```text
设备品牌和型号：
使用的镜头/摄像头：
原始录制分辨率：
原始录制 FPS：
横屏还是竖屏：
焦段或 App 中的倍率（例如 1×）：
数字变焦是否关闭：
自动跟拍/自动构图是否开启：
电子防抖/光学防抖状态：
对焦是否锁定：
曝光是否锁定：
相机安装高度（约数也可）：
相机距边线距离（约数也可）：
是否由 XbotGo/App 二次处理后保存：
其他说明：
```

如果你已经有正规标定结果，也放在同一目录，建议命名：

```text
training-data/manual-input/cameras/<camera-profile-id>/existing_intrinsics.json
```

至少应包含相机内参矩阵 `K`、畸变参数、对应分辨率和焦段标识。手机页面上的“等效焦距”只能作为说明，不能代替内参标定。

### M2：棋盘格标定照片

推荐使用 `9×6` 个内角点的棋盘格。实际格子边长可以是 24 mm、25 mm 或其他值，但必须用尺准确测量，不能采用打印模板的理论尺寸。

照片放在：

```text
training-data/manual-input/cameras/<camera-profile-id>/checkerboard/
```

同时创建：

```text
training-data/manual-input/cameras/<camera-profile-id>/checkerboard_spec.md
```

内容模板：

```text
内角点列数：9
内角点行数：6
实测单格边长 mm：
照片数量：
拍摄日期：
备注：
```

拍摄要求：

- 推荐 20–30 张；程序最低接受 8 张，但不建议只拍 8 张。
- 棋盘格平整贴在硬板上，不能弯曲。
- 使用与比赛视频完全相同的分辨率、方向、镜头、焦段、变焦和防抖设置。
- 覆盖画面中央、四角、四边，以及近、中、远和不同倾斜角度。
- 每张完整看到所有内角点，避免反光、运动模糊和过曝。
- 拍摄过程中不能改变倍率或分辨率。

如果只能提供棋盘格原始视频，放到：

```text
training-data/manual-input/cameras/<camera-profile-id>/checkerboard-video-original.mp4
```

我会在软件端抽取清晰帧，但照片通常更可靠。

### M3：第一段原始连续足球视频

视频放在：

```text
input_videos/manual-validation/<pitch-id>/<shot-id>-original.mp4
```

示例：

```text
input_videos/manual-validation/pitch-a/shot-001-ground-and-air-original.mp4
```

第一轮先提供 30–60 秒无剪辑连续片段，建议包含：

- 清楚可见的地面传球。
- 球经过或停在白线、点球点附近。
- 白鞋、白袜、反光等容易误检的目标。
- 一次较快射门或长传；有空中球更好。
- 正常平移、俯仰或旋转跟拍，但不要变焦。

这里不要求“固定文件大小”，真正必须固定的是：

- 全程像素分辨率和 FPS 不变。
- 全程使用同一个镜头、焦段/倍率和数字变焦状态。
- 不使用动态裁切、自动构图或会改变有效视场的自动取景。
- 尽量锁定对焦和曝光；无法锁定时写进 `camera_info.md`。
- 提供相机产生的原始文件，不经过 Clipchamp、微信、剪映等裁切、缩放、慢动作或重编码。

棋盘格与比赛视频必须属于同一个 `camera-profile-id`。如果原始视频有剪辑切换，请拆成连续片段，或在球场说明中记录每个切换时间。

### M4：球场尺寸和镜头说明

请创建：

```text
training-data/manual-input/pitches/<pitch-id>/pitch_info.md
```

模板：

```text
pitch-id：
对应视频相对路径：
对应 camera-profile-id：
实际边线长度 m：
实际球门线宽度 m：
禁区/球门区/中圈/点球点是否为标准尺寸：
是否存在额外小场线或其他颜色标线：
是否有缺失或模糊标线：
视频中的镜头切换时间（没有则写无）：
可确认的球场参照点：
其他说明：
```

不要默认写 `105×68 m`，除非这块场地确实是该尺寸。至少列出参考帧中可以确认的 4–8 个点，例如角点、中线与边线交点、禁区角或点球点。

你不需要手写像素坐标。我运行锚点工具后，需要你在交互窗口中按约定顺序点击并确认这些点。程序生成的结果会放到：

```text
output_videos/calibration/<pitch-id>/<shot-id>/pitch_anchors.json
```

## 3. 第一批完成后的目录应类似这样

```text
training-data/manual-input/
├── READY.md
├── cameras/
│   └── xbotgo-1080p30-1x/
│       ├── camera_info.md
│       ├── checkerboard_spec.md
│       └── checkerboard/
│           ├── 001.jpg
│           ├── 002.jpg
│           └── ...
└── pitches/
    └── pitch-a/
        └── pitch_info.md

input_videos/manual-validation/
└── pitch-a/
    └── shot-001-ground-and-air-original.mp4
```

最后创建：

```text
training-data/manual-input/READY.md
```

模板：

```text
已完成任务：M1、M2、M3、M4
camera-profile-id：
pitch-id：
shot-id：
待确认问题：无 / 请填写
```

`READY.md` 只是交接标记，不代表资料一定合格。我会继续执行软件计划中的 S1 输入审计。

## 4. 第二批资料：M5–M6，用于解决白线误检和姿态覆盖

### M5：多球场原始素材

正式替换模型前至少需要 4 块不同球场的视频。继续按下面结构放置：

```text
input_videos/training-source/<pitch-id>/<video-id>-original.mp4
training-data/manual-input/pitches/<pitch-id>/pitch_info.md
```

要求：

- 同一原视频及相邻帧只能属于一个数据集划分。
- 至少一块球场完整保留给测试集，训练阶段不可见。
- 保留远景、近景、俯视、逆光、平移、白线/白点、遮挡和空场景。
- 不要只保留模型容易识别的片段。

第一轮目标为至少 800 帧球数据和 400 帧球场关键点数据。

### M6：人工复核标注

我会先将预标注和困难负样本放到：

```text
training-data/review-queue/ball/
training-data/review-queue/ball-verifier/
training-data/review-queue/pitch-keypoints/
```

你复核完成后分别放到：

```text
training-data/approved/ball-yolo/
training-data/approved/ball-verifier/
training-data/approved/pitch-yolo/
```

人工要求：

- 球检测：真实足球使用紧致 bbox；无球负样本保持空标签。
- 二级分类：每张裁剪图确认属于 `ball` 或 `non_ball`。
- 球场姿态：确认 32 个关键点的身份，不只是确认点落在白线上。
- 真球压在线上、静止球、多人遮挡必须专项复核。
- 不得把画面外或完全不可见的球强行标注出来。

数据集最终拆分和目录转换由软件端完成，你不要自行把同一视频拆到 train/val/test。

## 5. 第三批资料：M7，用于速度精度验收

没有参考设备也可以运行单目物理估计，但不能证明达到误差目标。正式验收优先使用雷达；第二机位作为备选。

### 雷达方案

原始视频放到：

```text
input_videos/reference-speed/<session-id>/<video-id>-original.mp4
```

参考记录放到：

```text
training-data/reference-speed/<session-id>/measurements.csv
```

CSV 建议列：

```text
event_id,video_file,event_frame,event_time_seconds,motion_mode,reference_speed_kmh,device,occluded,notes
```

采集要求：

- 分别覆盖贴地传球、射门、空中球和反弹。
- 每类建议至少 10 个有效事件，包含慢、中、快三档。
- 雷达方向尽量接近球运动方向，并记录摆放位置，减少余弦误差。
- 记录每次触球/起球对应的帧号或时间。

### 同步第二机位方案

两台相机的原始视频分别放到：

```text
input_videos/reference-speed/<session-id>/camera-a-original.mp4
input_videos/reference-speed/<session-id>/camera-b-original.mp4
```

两台设备各自的 M1/M2 标定资料仍放在：

```text
training-data/manual-input/cameras/<camera-profile-id>/
```

两机位需要共同可见的同步事件，例如闪光、拍板或清晰触球帧，并记录两机位与球场的摆放关系。

## 6. 人工任务与软件任务对应关系

| 你的任务 | 完成后触发的软件任务 | 目的 |
|---|---|---|
| M1 + M2 | S1、S2 | 检查录制配置并生成 Camera Profile |
| M3 + M4 | S1、S2、S3 | 生成 Pitch Anchor Set 并建立真实基线 |
| M5 | S4 | 抽帧、预标注和困难负样本挖掘 |
| M6 | S5、S6 | 训练并验收球模型、分类器和关键点模型 |
| M7 | S7、S8 | 地面/空中速度误差验收和正式启用 |

软件任务的具体执行、输出路径和验收门槛见 [SOFTWARE_NEXT_STEPS.md](SOFTWARE_NEXT_STEPS.md)。

## 7. 现在最先做什么

按顺序完成：

1. M1：相机设置说明。
2. M2：20–30 张棋盘格照片和格子实测尺寸。
3. M3：一段 30–60 秒无剪辑、无变焦的原始视频。
4. M4：该球场实际长宽和可确认参照点。
5. 创建 `training-data/manual-input/READY.md`，然后告诉我资料已经放好。

第一批完成前不需要买雷达，也不需要一次性标完训练集。

