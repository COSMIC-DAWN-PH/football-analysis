# 足球视频分析

[English](README.md) | [简体中文](README.zh-CN.md)

这是一个足球视频分析流水线，可检测并跟踪球员、守门员、裁判和足球，依据球衣颜色分队，将目标位置映射到俯视球场，并输出标注视频与逐帧 JSONL 数据。

> 本仓库是在 [Mihailo Radović 的 `mradovic38/football-analysis`](https://github.com/mradovic38/football-analysis) 基础上维护的衍生版本。项目、教程、数据集与论文的具体来源见[来源与致谢](#来源与致谢)。

## 功能

- 使用 YOLO 检测足球、球员、守门员和裁判。
- 使用 ByteTrack 生成短期目标跟踪 ID。
- 检测 32 个球场关键点，并通过单应性变换进行位置映射。
- 结合草地遮罩与 K-Means，根据球衣颜色分配球队。
- 生成球员俯视投影与动态 Voronoi 控制区域。
- 可选球员速度估算与模型推断的累计控球率叠加。
- 输出标注 MP4，以及逐帧目标和关键点 JSONL。
- 可选战术摘要、热图、时间线和质量指标。
- 支持无窗口运行与 OpenVINO 模型目录。

## 当前维护版本新增内容

与上游原始演示代码相比，当前分支把写死配置的示例改造成了可配置、容错性更好的视频分析流水线：

| 增强项 | 具体变化 | 带来的作用 |
|---|---|---|
| 可配置 CLI | 无须修改 Python，即可设置视频路径、模型路径、轨迹目录、批量大小、起始偏移、球队名称、四组球衣颜色、预览、速度和控球标注。 | 同一份代码可以分析不同比赛，也便于脚本化运行。 |
| 模型自动回退 | 两个检测器都会依次尝试 OpenVINO FP16、普通 OpenVINO 和 PyTorch `.pt`；显式传入 `--object-model` 或 `--keypoints-model` 可覆盖自动选择。 | 有优化模型时自动利用，没有时仍兼容 `.pt`。 |
| 流式 JSONL 轨迹 | 每个已处理帧作为一行紧凑 JSON 追加到目标和关键点文件。 | 长视频不再每帧读取并重写不断增大的 JSON 数组。 |
| 更稳健的球场映射 | 关键点保持原视频 16:9 几何；允许关键点缺失和单应矩阵计算失败，弱检测帧可沿用上一帧平滑后的单应矩阵。 | 某一帧漏掉球场标志时，不容易导致整段任务退出。 |
| 稳定的球队分配 | 按“目标类型 + track ID”缓存球队判断。 | 不必每帧对同一短期轨迹重复执行 K-Means，也能减少球队标签闪烁。 |
| 独立的分析图层 | 速度估算与控球率标注可分别开关，并且默认关闭。 | 低采样率战术分析不会默认展示不可靠的瞬时指标。 |
| 无窗口处理 | `--no-preview` 会跳过 OpenCV 窗口。 | 可在服务器或无人值守环境运行。 |
| 环境检查 | `check_setup.py` 显示 Python、PyTorch、设备和 OpenCV 信息，按哈希检查必要素材，并可验证模型元数据。 | 开始长任务前发现文件缺失或模型不兼容。 |
| 战术报告 | `summarize_match.py` 将 JSONL 转换为经过质量筛选的摘要、逐分钟指标、热力图和阵型时间线。 | 不另写分析程序也能查看轨迹产物。 |
| 容错测试与仓库卫生 | 覆盖空投影、关键点缺失等场景，并忽略视频、权重、输出、环境和本地密钥，保留必要的空目录占位。 | 本地数据与密钥不会进入 Git，常见的部分检测结果也可持续测试。 |

相对上游需要注意的默认行为变化：

- 目标检测置信度默认为 `0.25`，足球为 `0.05`，球场检测为 `0.20`，关键点为 `0.50`。
- 默认 batch size 为 `1`；预览默认开启，速度估算和控球率标注默认关闭。
- 投影缺失时，投影绘制、控球分配和标注阶段会跳过对应目标，而不是把它当成致命错误。
- track ID 仍然只是 ByteTrack 的短期编号；缓存球队标签并不会让它变成永久球员身份。

## 仓库文件策略

视频、分析产物、模型权重、OpenVINO 导出目录、本地虚拟环境和 API 密钥均不会提交到 Git。仓库只保留源码、训练 notebook、测试、文档，以及运行所需的小型球场底图。

## 环境要求

- 推荐 Python 3.11。
- 支持 Windows、Linux 和 macOS。
- 目标检测模型必须依次包含 `ball`、`goalkeeper`、`player`、`referee` 四类。
- 球场姿态模型必须输出 32 个关键点。

Python 依赖版本见 [`requirements.txt`](requirements.txt)。CPU 可以运行，但完整视频处理较慢；兼容硬件上的 CUDA 或 OpenVINO 通常能显著提高速度。

## 安装

```bash
git clone https://github.com/COSMIC-DAWN-PH/football-analysis.git
cd football-analysis
python -m venv .venv
```

激活虚拟环境并安装依赖。

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

```bash
# Linux 或 macOS
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 模型文件

本仓库不重新分发模型权重。请使用仓库内的 notebook 训练，或自行提供兼容权重：

```text
models/weights/object-detection.pt
models/weights/keypoints-detection.pt
```

每个检测器都会按以下顺序选择第一个存在的路径：

```text
models/weights/object-detection_openvino_model_fp16/
models/weights/object-detection_openvino_model/
models/weights/object-detection.pt

models/weights/keypoints-detection_openvino_model_fp16/
models/weights/keypoints-detection_openvino_model/
models/weights/keypoints-detection.pt
```

处理视频前检查本地环境：

```powershell
.\.venv\Scripts\python.exe check_setup.py --load-models
```

`check_setup.py` 会按哈希校验已知的本地权重。主动替换权重后出现哈希不一致是预期现象，但仍须确认目标类别顺序和 32 点结构兼容。

## 分析视频

把视频放入 `input_videos/`，或传入任意可读取的本地路径。两队颜色应填写具有代表性的球员和守门员球衣 RGB 值。

```powershell
.\.venv\Scripts\python.exe main.py `
  --input input_videos\match.mp4 `
  --output output_videos\match-analysis.mp4 `
  --tracks-dir output_videos\match-tracks `
  --batch-size 1 `
  --club1-name Red --club1-player 220,30,30 --club1-goalkeeper 20,20,20 `
  --club2-name Blue --club2-player 30,80,220 --club2-goalkeeper 240,220,30 `
  --estimate-speed --no-annotate-possession --no-preview
```

完整 CLI 参数：

| 参数 | 默认值 | 用途 |
|---|---|---|
| `--input PATH` | 必填 | 输入比赛视频。 |
| `--output PATH` | `output_videos/analysis.mp4` | 标注 MP4 路径；父目录会自动创建。 |
| `--object-model PATH` | 自动回退 | 目标检测 `.pt` 文件或 OpenVINO 模型目录。 |
| `--keypoints-model PATH` | 自动回退 | 球场姿态 `.pt` 文件或 OpenVINO 模型目录。 |
| `--field-image PATH` | `input_videos/field_2d_v2.png` | 俯视投影面板使用的球场底图。 |
| `--tracks-dir PATH` | `output_videos` | 两个轨迹 JSONL 的保存目录。 |
| `--batch-size N` | `1` | 推理批量，最小为 1。 |
| `--skip-seconds N` | `0` | 跳过源视频开头的秒数，不能为负数。 |
| `--estimate-speed` / `--no-estimate-speed` | 关闭 | 计算并绘制平滑后的球员速度。 |
| `--annotate-possession` / `--no-annotate-possession` | 关闭 | 绘制模型估算的累计控球率。 |
| `--preview` / `--no-preview` | 开启 | 显示或关闭 OpenCV 实时窗口。 |
| `--club1-name NAME` | `Club1` | 第一支球队名称；若要运行当前汇总脚本，应使用 `Red`。 |
| `--club1-player R,G,B` | `232,247,248` | 第一队普通球员代表色。 |
| `--club1-goalkeeper R,G,B` | `6,25,21` | 第一队守门员代表色。 |
| `--club2-name NAME` | `Club2` | 第二支球队名称；若要运行当前汇总脚本，应使用 `Blue`。 |
| `--club2-player R,G,B` | `172,251,145` | 第二队普通球员代表色。 |
| `--club2-goalkeeper R,G,B` | `239,156,132` | 第二队守门员代表色。 |

每组 RGB 必须正好包含三个 `0–255` 的整数。

执行 `python main.py --help` 可查看完整命令行接口。

## 输出

主流水线会生成：

- `--output` 指定的标注视频。
- `--tracks-dir` 下的 `object_tracks.jsonl`。
- `--tracks-dir` 下的 `keypoint_tracks.jsonl`。

JSONL 每一行对应一个已处理帧。根据目标类型和启用的分析功能，目标记录可能包含 `bbox`、`club`、`club_color`、`projection`、`has_ball` 和 `speed`。

`object_tracks.jsonl` 每行包含四类对象。JSON 序列化后，数字 track ID 会变成字符串：

```json
{"ball":{"3":{"bbox":[915.2,511.0,928.4,525.7],"projection":[271.6,173.1]}},"goalkeeper":{},"player":{"12":{"bbox":[804.1,392.5,858.8,556.2],"club":"Red","club_color":[220,30,30],"projection":[244.9,181.3],"has_ball":true,"speed":18.4}},"referee":{}}
```

对已经检出的目标，只有 `bbox` 是必有字段。`projection`、`club`、`club_color`、`has_ball` 和 `speed` 都是条件性字段，写入器不会额外生成 `confidence`。`keypoint_tracks.jsonl` 每行是一个对象：字符串键为球场关键点编号，值为图像坐标，例如 `{"0":[312.4,104.8],"13":[960.1,221.7]}`。

上游旧版 `object_tracks.json` 数组属于历史格式。新版采用 JSON Lines，应逐行解析，不能把整个文件当成一个 JSON 数组。使用同一轨迹目录启动新任务时，程序会先删除该目录中上次生成的两个 JSONL 文件。

输出视频会组合原画面、目标 ID、球队颜色标记、球场关键点和俯视投影。速度文字与控球率图层可分别启用或关闭。

生成视频固定为 `1920×1080`，处理完成后编码为 MP4；不会复制源视频音轨。

## 可选战术摘要

生成轨迹后，可使用 `summarize_match.py` 输出经过质量筛选的球队指标和可视化。

该可选脚本还需要 Matplotlib：`python -m pip install matplotlib`。

```powershell
.\.venv\Scripts\python.exe summarize_match.py `
  --tracks-dir output_videos\match-tracks `
  --output-dir output_videos\match-summary `
  --fps 1 `
  --source input_videos\match.mp4
```

`--fps` 表示 JSONL 轨迹每秒包含多少个样本，不应盲目填写原视频帧率。例如，若轨迹来自“源视频每秒抽取 1 帧”，应填写 `--fps 1`；逐帧处理时通常才填写源视频 FPS。

当前汇总实现只识别球队名 `Red` 和 `Blue`。如果后续需要汇总轨迹，运行主程序时应把 `--club1-name` 和 `--club2-name` 设置成这两个准确名称。脚本同时要求 `object_tracks.jsonl` 和 `keypoint_tracks.jsonl`，两者长度不同时只分析较短文件覆盖的帧。

一个画面只有同时满足以下门槛才会进入战术统计：至少 8 个关键点、至少 6 个 RANSAC 内点、内点率不低于 40%、内点中位重投影误差不高于 4 像素、目标球场 X/Y 跨度分别不低于 150/100 像素。之后，球员或守门员脚点还必须落在固定的 `100 × 50 m` 球场范围内。

汇总目录包含：

- `summary.json`：来源信息、采样率、有效单应性比例、球队汇总，以及阵型最拉开/最压缩的分钟。
- `minute_metrics.csv`：逐分钟可见人数、重心、宽度、长度、紧凑度和两队重心间距。
- `team_heatmaps.png`：`Red` 和 `Blue` 通过质量筛选后的位置密度。
- `team_centres_timeline.png`：固定球场坐标下的逐分钟纵向重心。
- `team_shape_timeline.png`：逐分钟球队宽度和长度。
- `REPORT.md`：根据当前样本计算结果生成的中文解读。

这些摘要只是单个移动机位下的观测估计，不能等同于经过标定的多机位数据或 GPS 追踪数据。低帧率样本适合观察球队整体队形趋势，不适合计算跑动距离、冲刺次数或瞬时速度。

## 模型训练

仓库包含：

- [`models/object_detection_train.ipynb`](models/object_detection_train.ipynb)
- [`models/keypoints_detection_train.ipynb`](models/keypoints_detection_train.ipynb)

训练 notebook 通过 Roboflow 下载数据，因此需要你自己的 API 密钥。请把 `config.example.py` 复制为 `config.py`，只在本地填写密钥，不要提交。

两个 notebook 直接引用了 **Mihailo** 在 Roboflow Universe 发布的数据集：

- [football-players-detection 第 2 版](https://universe.roboflow.com/mihailo/football-players-detection-3zvbc-7ocfe/dataset/2)：目标检测，CC BY 4.0。
- [football-field-detection 第 1 版](https://universe.roboflow.com/mihailo/football-field-detection-f07vi-apxzb/dataset/1)：32 点关键点检测，CC BY 4.0。

下载、训练或重新分发衍生权重前，请检查数据集页面上的最新条款。

## 已知限制

- 视频中的 ID 是短期跟踪 ID，不是球员身份或球衣号码。
- 单个移动机位会带来投影误差和不完整的球场覆盖。
- 足球在远景中像素很小，会影响检出率和控球判断。
- 速度来自相邻帧投影位置，跟踪或单应矩阵跳变时可能触及速度上限。
- 分队依赖代表色，可能受到光照、背心、观众和相近球衣颜色干扰。
- 用户须自行确认模型权重及其许可证是否适用于具体用途。

## 来源与致谢

### 直接上游

本代码库直接基于 [Mihailo Radović 的 `mradovic38/football-analysis`](https://github.com/mradovic38/football-analysis)。目标跟踪、球场关键点、分队、单应性映射、俯视投影、Voronoi、控球和速度估算等初始架构与实现均来自该上游项目。

仓库内的 `input_videos/field_2d_v2.png` 球场底图同样来自该上游仓库。其 MIT 版权声明保留在 [`LICENSE`](LICENSE) 中。

当前 [`COSMIC-DAWN-PH/football-analysis`](https://github.com/COSMIC-DAWN-PH/football-analysis) 分支增加了可配置 CLI、长视频处理改进、JSONL 轨迹写入、OpenVINO 模型选择、无窗口执行、环境检查、健壮性修复、测试和可选战术摘要。

### 教程与技术参考

上游项目将以下资料列为灵感或技术参考。本仓库保留这些署名以便追溯；列在这里不表示当前分支直接复制了每个来源的代码。

- [Code In a Jiffy：使用 YOLO、OpenCV 和 Python 构建足球分析系统](https://www.youtube.com/watch?v=neBZ6huolkg)
- [Roboflow：Football AI Tutorial](https://www.youtube.com/watch?v=aBVGKoNZQUw)
- [Roboflow Football AI notebook](https://github.com/roboflow/notebooks/blob/main/notebooks/football-ai.ipynb)
- [DonsetPG/Narya](https://github.com/DonsetPG/narya)：足球跟踪与单应性变换工作。
- [PiotrGrabysz/PitchGeometry](https://github.com/PiotrGrabysz/PitchGeometry)：足球场关键点检测。
- R. Alhejaily 等人，[《Automatic Team Assignment and Jersey Number Recognition in Football Videos》](https://doi.org/10.32604/iasc.2023.033062)，2023。

核心运行库包括 [Ultralytics](https://github.com/ultralytics/ultralytics)、[Supervision](https://github.com/roboflow/supervision)、OpenCV、PyTorch、SciPy、scikit-learn 和 pandas。各项目的作者与许可证分别适用。

## 许可证

本仓库源码按照 [MIT License](LICENSE) 分发，并保留上游版权声明。

Ultralytics 软件和 YOLO 模型另有许可证条款，包括 AGPL-3.0 和商业许可。Roboflow 数据集以及用户自行取得的模型权重也保留各自许可证，请根据实际用途检查全部适用条款。
