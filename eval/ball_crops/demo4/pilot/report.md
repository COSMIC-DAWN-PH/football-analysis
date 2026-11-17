# 试点对比报告：demo4 前 20 项 × qwen3.7-plus vs qwen3.6-flash

- 素材：`clips/<id>.mp4`（±0.25s H.264 视频，红框标注）+ `context/<id>.png`（5 帧并排帧条）+ `crops/<id>.png` 放大图
- 实际观看方式：两个模型在 openchamber CLI 环境都无法播放 mp4，**全部退化为帧条 + 放大图判定**（视频运动信息未用上）
- 判定口径与人工一致：ball / not_ball(12类原因) / null
- **人工对照列留空，请你审完后自行填写**（对照素材：审核工具 http://127.0.0.1:8100 按时间顺序前 20 个格子，左视频右放大图）

## 逐项对照

| 顺序 | id | 类别 | qwen3.7-plus | qwen3.6-flash | 一致? | 人工判定 |
|---|---|---|---|---|---|---|
| 1 | demo4_c00006 | confirmed | ball | ball | ✓ | |
| 2 | demo4_b01721 | bridge | ball | ball | ✓ | |
| 3 | demo4_b01722 | bridge | ball | ball | ✓ | |
| 4 | demo4_b01723 | bridge | ball | ball | ✓ | |
| 5 | demo4_c00009 | confirmed | ball | ball | ✓ | |
| 6 | demo4_b01724 | bridge | debris | line | ✗ | |
| 7 | demo4_c00012 | confirmed | ball | ball | ✓ | |
| 8 | demo4_b01725 | bridge | debris | debris | ✓ | |
| 9 | demo4_c00015 | confirmed | ball | null | ✗ | |
| 10 | demo4_b01727 | bridge | debris | null | ✗ | |
| 11 | demo4_c00017 | confirmed | ball | ball | ✓ | |
| 12 | demo4_c00020 | confirmed | ball | ball | ✓ | |
| 13 | demo4_b01731 | bridge | debris | ball | ✗ | |
| 14 | demo4_b01732 | bridge | debris | null | ✗ | |
| 15 | demo4_c00024 | confirmed | ball | ball | ✓ | |
| 16 | demo4_b01733 | bridge | debris | debris | ✓ | |
| 17 | demo4_c00027 | confirmed | other | player | ✗ | |
| 18 | demo4_c00030 | confirmed | ball | ball | ✓ | |
| 19 | demo4_c00033 | confirmed | ball | ball | ✓ | |
| 20 | demo4_b01737 | bridge | debris | debris | ✓ | |

## 统计

- 完全一致：14 / 20（70%）
- 分歧：6 / 20（30%）——其中 4 项是"球 vs 难判/杂物"性质分歧（b01731 直接相反），2 项是原因细节分歧（debris vs line、other vs player）
- qwen3.7-plus：ball 12 / not_ball 8 / null 0
- qwen3.6-flash：ball 13 / not_ball 5 / null 3

## 初步观察

1. **qwen3.6-flash 更保守**：3 个 null（太小太糊），qwen3.7 一个 null 都没给（全部硬判，模糊的 b01727/b01732 也给了 debris）
2. 分歧集中在 bridge_sweep 和模糊帧（b01731/b01732/c00015/c00027 等），这正是"没有视频运动信息"最难判的样本——**如果模型能真看视频，分歧率大概率会显著下降**
3. 两个模型都没用上视频（CLI 限制），本次试点实际测的是"帧条+放大图"方式，不是"±0.5s 视频"方式

## 下一步选项

- A. 保持帧条方式（模型 CLI 看不了 mp4，帧条已含 5 帧运动信息，比上轮单张 sheet 强）
- B. 尝试让模型看视频：找支持视频输入的会话方式（如把片段合成 GIF 动图 / 或模型多图输入逐帧看 15 帧）
- C. 你看完人工判定列后，再决定哪个模型/哪种方式值得扩全量
