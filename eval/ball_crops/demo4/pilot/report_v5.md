# 试点对比报告 v5：动态跟踪框视频 + 前后静态帧 + 放大图（双模型）

- 方式：2.5s 视频（阿里 API 最短可接受，已实测）每帧**红框+红十字跟踪目标**（LK 光流，候选帧绿色加粗）+ 候选帧前后 7 张静态原图（-1.0/-0.5/-0.25/0/+0.25/+0.5/+1.0s，同样画跟踪框）+ 放大 crop
- **人工 ground truth（你提供）：20 项全部不是足球（0 个 ball）**
- 模型：qwen3.6-flash / qwen3.7-plus（原生视频 read_media 通道）

## 逐项对照（v5）

| # | id | 类别 | qwen3.6-flash | qwen3.7-plus | 人工(全非球) |
|---|---|---|---|---|---|
| 1 | demo4_c00006 | confirmed | ball ✗ | ball ✗ | |
| 2 | demo4_b01721 | bridge | ball ✗ | ball ✗ | |
| 3 | demo4_b01722 | bridge | not_ball(player) ✓ | ball ✗ | |
| 4 | demo4_b01723 | bridge | ball ✗ | null ~ | |
| 5 | demo4_c00009 | confirmed | ball ✗ | ball ✗ | |
| 6 | demo4_b01724 | bridge | not_ball(other) ✓ | null ~ | |
| 7 | demo4_c00012 | confirmed | not_ball(debris) ✓ | null ~ | |
| 8 | demo4_b01725 | bridge | ball ✗ | null ~ | |
| 9 | demo4_c00015 | confirmed | not_ball(other) ✓ | not_ball(light) ✓ | |
| 10 | demo4_b01727 | bridge | not_ball(light) ✓ | null ~ | |
| 11 | demo4_c00017 | confirmed | not_ball(debris) ✓ | not_ball(shoe) ✓ | |
| 12 | demo4_c00020 | confirmed | not_ball(watermark) ✓ | not_ball(watermark) ✓ | |
| 13 | demo4_b01731 | bridge | not_ball(watermark) ✓ | not_ball(watermark) ✓ | |
| 14 | demo4_b01732 | bridge | not_ball(watermark) ✓ | not_ball(watermark) ✓ | |
| 15 | demo4_c00024 | confirmed | ball ✗ | null ~ | |
| 16 | demo4_b01733 | bridge | not_ball(debris) ✓ | null ~ | |
| 17 | demo4_c00027 | confirmed | not_ball(other) ✓ | not_ball(other) ✓ | |
| 18 | demo4_c00030 | confirmed | ball ✗ | ball ✗ | |
| 19 | demo4_c00033 | confirmed | not_ball(debris) ✓ | not_ball(debris) ✓ | |
| 20 | demo4_b01737 | bridge | not_ball(other) ✓ | null ~ | |

## 统计（ground truth：20/20 全非球）

| 指标 | qwen3.6-flash v4(静态框) | **qwen3.6-flash v5** | **qwen3.7-plus v5** |
|---|---|---|---|
| 误判 ball | 5 | **7** | **5** |
| 正确 not_ball | 14 | **13** | 7 |
| null | 1 | 0 | **8** |
| 水印识别 | ✓×3 | ✓×3 | ✓×3 |

## 分析

1. **动态框没有帮助 qwen3.6 减少假球，反而略增**（5→7）：c00024（观众区）、b01725（观众）被误判。可能原因：LK 跟踪的框随目标微小移动，模型看到"框在动+周围有人"反而更确信是球；或者照片+视频多源信息让模型过度自信（null 从 1 降到 0）。
2. **qwen3.7-plus 更保守**：8 个 null（"太小太糊"），只硬判 12 个、其中 7 个正确。它和 qwen3.6 的错位完全不同：3.7 错在"球员小白点"簇（c00006/b01721-23/c00009/c00030），3.6 除了同一簇还多了 c00024/b01725。
3. **水印识别两模型都稳**（c00020/b01731/b01732 全对）——动态框+多帧照片对"固定叠加层 vs 场上物体"的区分有效。
4. **共同盲区仍是"球员脚边/身边的移动白点"**：检测器最自信的 confirmed/bridge 段候选（绿框=跟踪确认的球），两模型都有 5-7 个误判成球——这些正是需要人工看放大图+运动轨迹才能判的难点。
5. **结论：动态框方案对静止类误检（水印/杂物/观众）有效，但对"运动中疑似球"反而增加误判**；两模型依然不能替代人工。若用于预标排序：**两模型都判 ball 的项**（本组 c00006/b01721/c00009/c00030 共 4 个）可排人工最优先——本组它们都是假球，正好是需要人工重点核查的对象。

## 产物

- 视频：`eval/ball_crops/demo4/pilot/clips_v5/`（20 个 2.5s 动态框 H.264）+ Library `video/pilotv5/`
- 静态帧：`eval/ball_crops/demo4/pilot/frames_v5/`（140 张 1920×1080 带框）
- 标签：`qwen36_v5_labels.jsonl` / `qwen37_v5_labels.jsonl`
- 工具：`tools/extract_ball_pilot_v5.py`（LK 动态跟踪框，--window/--min-duration 可调）
