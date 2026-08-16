# 球检测审核集（ball_crops/）

## 标注规则（先读这个）

- 来源：`tools/detect_ball_tracks.py` 全量球检测+跟踪（`output_videos/<src>/ball/ball_tracks.jsonl`，生产参数 conf=0.02、整帧+2×2 tile），由 `tools/extract_ball_candidates.py` 提取。
- 目录结构：`eval/ball_crops/<src>/crops/<id>.png` 为图片；`candidates.jsonl` 为清单（`manual_label` / `manual_reason` 字段）。
- 审核对象 = 清单里的**每一张**图片（本轮 100% 人工审核，无抽样）。
- 过滤政策（用户 2026-08-15 确认）：`unconfirmed` 中 **conf < 0.05 且只出现在单帧的孤立簇**不进入审核集（`extract_ball_candidates.py` 默认 `--drop-low-single`），其余全部人工审核。
- **口径（2026-08-16 起）**：纯人工审核，不使用任何模型预标/交叉标注（模型预标失误率过高，已整体取消）。`manual_label` 是唯一权威。

### 五类候选

| category | 含义 | 重点 |
|----------|------|------|
| `confirmed` | 跟踪器确认的 segment 清晰帧（红框=当时跟踪框） | 判定框内是否真球；确认虚接/错段 |
| `unconfirmed` | 跟踪器**没收**的孤立候选（聚类后每簇 1 张） | FP 高发：鞋/袜/线/灯/头/手套 |
| `bridge_sweep` | segment 内预测帧（检测漏了、跟踪器硬桥接） | 窗口里有没有真球？位置在哪 |
| `gap_sweep` | 两 segment 之间断档中点 | 球是不是在这里丢了 |
| `global_sweep` | 无候选空帧抽样（全帧缩小） | 盲查漏检 |

### 标签取值

- `ball`：确实有一颗球在图中（写在 `manual_label`）
- `not_ball`：不是球（可在 `manual_reason` 写：`shoe`/`sock`/`line`/`light`/`head`/`hand`/`other`/`penalty_spot` 点球点/`debris` 球场杂物）
- `null`：太小/太糊无法判定（不进评估但保留）

### 网页审核工具（推荐）

- 启动：`python tools/ball_review_server.py --port 8100`，浏览器打开 http://127.0.0.1:8100
- 按 `review_order.txt` **视频时间顺序**导航（帧号升序）
- 点击格子弹出双视图：**左 = 原图视频**（自动播候选帧前 0.25s→后 0.25s 半秒片段，红框定位；`v` 重播、`l` 循环、可拖动进度条看任意帧），**右 = 放大 crop**；`/video/<src>` 以 HTTP Range 流式服务本地视频，无磁盘缓存
- 弹窗内快捷键：`1`=ball `2`=not_ball `3`=null `q/w/e/r/t/y/u`=非球原因（鞋/袜/线/灯/头/手套/其他） `p`=点球点 `d`=杂物 `[`/`]`=已标间切换 `g`=输入 id 或序号跳转 `Backspace`=回退上一标 `0`=清除 `Esc`=关闭；判定后自动跳到下一个未审格子
- 标注即时保存进 `candidates.jsonl` 的 `manual_label` / `manual_reason` 字段

## 流程

1. 人工在网页工具上按时间顺序全量审核（左原图视频 + 右放大 crop 双视图）；
2. `python tools/prefill_ball_labels.py --report --manifest candidates.jsonl` 检查审核进度；
3. 全部审完后 `tools/eval_ball.py` 复算指标。

## 评估口径

- 主口径（crop 级）：precision / recall / F1，`tools/eval_ball.py` 计算；
- track 级：段召回、断段率、虚接率、确认延迟；
- 归属：抽样人工核对；
- 过拟合纪律：至少留 1 个 demo 全程不进调参/微调做 hold-out；指标按源分报，不用汇总平均数拍板；微调 hold-out 不达标不 promote。
