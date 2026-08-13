# Windows 本地配置

这份目录已使用 Python 3.11 创建独立环境 `.venv`，不会影响系统 Python。

## 已配置

- 项目：`Mando-03/Football-Analysis`
- 依赖：`requirements.txt` 全部安装完成
- 运行设备：CPU（本机 Intel Arc 不能被当前 PyTorch CUDA 版本直接使用）
- 加速后端：OpenVINO FP16（可由 Intel CPU / Arc GPU / NPU 自动选择）
- 目标检测权重：公开兼容 YOLO11m，4 类顺序为 `ball / goalkeeper / player / referee`
- 球场关键点权重：公开兼容 YOLOv8x-pose，32 个关键点
- 专用足球权重：单类 `ball` OpenVINO FP16 / OpenVINO / PT 自动回退
- 球场底图：来自上游 `mradovic38/football-analysis`

## 先检查

```powershell
.\.venv\Scripts\python.exe check_setup.py --load-models
```

## 分析视频

1. 把视频复制到 `input_videos`，例如 `input_videos\match.mp4`。
2. 根据球衣实际颜色填写两个队的 RGB；示例：

```powershell
.\.venv\Scripts\python.exe main.py `
  --input input_videos\match.mp4 `
  --run-dir output_videos\match `
  --pitch-length-m 105 --pitch-width-m 68 `
  --batch-size 1 `
  --club1-name Red --club1-player 220,30,30 --club1-goalkeeper 20,20,20 `
  --club2-name Blue --club2-player 30,80,220 --club2-goalkeeper 240,220,30
```

默认产物统一保存在 `output_videos\match`：分析视频位于根目录，原始 JSONL 位于 `raw`，战术摘要位于 `summary`。

服务器或不需要实时窗口时加 `--no-preview`。NVIDIA GPU 服务器可把 `--batch-size` 提高到 8 或 10；本机 CPU 建议保持 1，并先用 10-30 秒短片验证。

## Roboflow（仅重新训练时需要）

```powershell
Copy-Item config.example.py config.py
```

然后把 `config.py` 中的占位符替换成自己的 API key。`config.py` 已被 Git 忽略，不要提交密钥。

## 权重说明

原项目没有提交 `.pt` 文件，作者给出的 Roboflow 下载入口需要账号/API key。本地默认使用同类公开模型；如果以后取得作者原始权重，直接覆盖：

- `models/weights/object-detection.pt`
- `models/weights/keypoints-detection.pt`
- `models/weights/ball-detection.pt`

替换后 `check_setup.py` 会报告哈希不同，这是预期现象；再检查类别顺序和 32 点结构即可。
