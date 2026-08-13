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

以下 OpenVINO 目录存在时，程序会优先使用：

```text
models/weights/object-detection_openvino_model_fp16/
models/weights/keypoints-detection_openvino_model_fp16/
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

常用参数：

| 参数 | 用途 |
|---|---|
| `--estimate-speed` | 在视频和目标 JSONL 中加入平滑速度估计。 |
| `--annotate-possession` | 在视频中显示模型估算的累计控球率。 |
| `--no-preview` | 关闭 OpenCV 预览窗口，适合无界面运行。 |
| `--skip-seconds N` | 跳过视频开头 `N` 秒。 |
| `--batch-size N` | 设置推理批量；资源有限的 CPU 建议使用 `1`。 |
| `--object-model PATH` | 指定目标模型或 OpenVINO 目录。 |
| `--keypoints-model PATH` | 指定球场关键点模型或 OpenVINO 目录。 |

执行 `python main.py --help` 可查看完整命令行接口。

## 输出

主流水线会生成：

- `--output` 指定的标注视频。
- `--tracks-dir` 下的 `object_tracks.jsonl`。
- `--tracks-dir` 下的 `keypoint_tracks.jsonl`。

JSONL 每一行对应一个已处理帧。根据目标类型和启用的分析功能，目标记录可能包含 `bbox`、`club`、`club_color`、`projection`、`has_ball` 和 `speed`。

输出视频会组合原画面、目标 ID、球队颜色标记、球场关键点和俯视投影。速度文字与控球率图层可分别启用或关闭。

## 可选战术摘要

生成轨迹后，可使用 `summarize_match.py` 输出经过质量筛选的球队指标和可视化。

该可选脚本还需要 Matplotlib：`python -m pip install matplotlib`。

```powershell
.\.venv\Scripts\python.exe summarize_match.py `
  --tracks-dir output_videos\match-tracks `
  --output-dir output_videos\match-summary `
  --fps 30 `
  --source input_videos\match.mp4
```

这些摘要只是单个移动机位下的观测估计，不能等同于经过标定的多机位数据或 GPS 追踪数据。

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
