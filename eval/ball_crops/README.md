# 球检测审核集（ball_crops/）

## 标注规则（先读这个）

- 来源：`tools/detect_ball_tracks.py` 全量球检测+跟踪（`output_videos/<src>/ball/ball_tracks.jsonl`，生产参数 conf=0.02、整帧+2×2 tile），由 `tools/extract_ball_candidates.py` 提取。
- 目录结构：`eval/ball_crops/<src>/crops/<id>.png` 为图片；`candidates.jsonl` 为清单（含 `manual_label` / `qwen_label` / `luna_label` 字段）。
- 审核对象 = 清单里的**每一张**图片（本轮 100% 人工审核，无抽样）。
- 过滤政策（用户 2026-08-15 确认）：`unconfirmed` 中 **conf < 0.05 且只出现在单帧的孤立簇**不进入审核集（`extract_ball_candidates.py` 默认 `--drop-low-single`），其余全部人工审核。

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
- `not_ball`：不是球（可在 `manual_reason` 写：`shoe`/`sock`/`line`/`light`/`head`/`hand`/`other`）
- `null`：太小/太糊无法判定（不进评估但保留）
- 注意：模型预填字段按源不同——demo1~4 用 `kimi_label`（kimi 系）+ `luna_label`（gpt-5.6-luna）；raw1/raw2 用 `kimi_label`（kimi-k2.6）+ `qwen_label`（qwen3.7-plus，阿里 token plan，落盘文件沿用了 `luna_labels.jsonl` 旧名）。人工以 `manual_label` 为准，**逐张确认或纠错**，分歧样本（`disagreements.jsonl`）优先审。

### 网页审核工具（推荐）

- 启动：`python tools/ball_review_server.py --port 8100`，浏览器打开 http://127.0.0.1:8100
- 按 `review_order.txt` 分歧优先排序导航；格子上直接着色显示双模型一致/分歧状态
- 点击格子放大单张 crop，弹窗内快捷键：`1`=ball `2`=not_ball `3`=null `q/w/e/r/t/y/u`=非球原因（鞋/袜/线/灯/头/手套/其他） `Esc`=关闭；判定后自动跳到下一个未审格子
- 标注即时保存进 `candidates.jsonl` 的 `manual_label` / `manual_reason` 字段（人工即权威）

## 流程

1. 双模型预填：openchamber 会话（kimi-k2.7-code / gpt-5.6-luna）按 `sheets/` 网格图逐张看图（每张 20 个 crop，格子左上角有 id 编号），分别写 `kimi_labels.jsonl` / `luna_labels.jsonl`（每行 `{"id": ..., "label": "ball"|"not_ball"|"null", "reason": "..."}`）；网格图由 `tools/make_ball_sheets.py` 生成，单张核对可用 `crops/<id>.png`；
2. `python tools/prefill_ball_labels.py --manifest candidates.jsonl --merge-labels kimi_labels.jsonl kimi_label --merge-labels luna_labels.jsonl luna_label` 合并并产出 `disagreements.jsonl` 与 `review_order.txt`；
3. 人工按 review_order 全量审核（建议同样按 sheets 网格图浏览，不确定的格子回 `crops/<id>.png` 放大看），直接改 `candidates.jsonl` 里的 `manual_label`（不存在的字段加上即可）；
4. `python tools/prefill_ball_labels.py --report --manifest candidates.jsonl` 检查审核进度。

## 评估口径

- 主口径（crop 级）：precision / recall / F1，`tools/eval_ball.py` 计算；
- track 级：段召回、断段率、虚接率、确认延迟；
- 归属：抽样人工核对。
