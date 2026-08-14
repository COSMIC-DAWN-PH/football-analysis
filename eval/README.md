# 球队颜色验证集（demo2-30s-test）

## 标注规则

- 来源：`output_videos/demo2-30s-test/demo2-30s-test-input.mp4`，按 `raw/object_tracks.jsonl` 的 track id 与帧号裁剪。
- 收录标准（只收清晰、无歧义样本）：
  1. bbox 宽 ≥ 40px、高 ≥ 90px（近景大球员）；
  2. 躯干中心条带（横向 12%~88%、纵向 30%~60%）过滤绿草(H35-95)/暗(V<60)/低饱和(S<60)后，有效像素 ≥ 100；
  3. 中位 hue 明确落在两队峰值区（≥150 或 95~130），不在中间模糊带。
- 每 track 至多 6 张 crop，全部人工核对过队服颜色。
- `manual_label` 取值：`Maroon`（栗色/红队）或 `Navy`（藏青/蓝队）；无歧义样本不收录，不允许"无法判定"进入评估。
- 评估脚本：`tools/eval_team_colors.py`。验收主口径 = 本验证集上的 balanced accuracy 与双向混淆矩阵。

## 基线（Phase 0，自动预标注经人工确认）

- 10 个 track、34 张 crop。
- 现有 pipeline 在该集上：balanced accuracy 0.667（Maroon 召回 0.5、Navy 召回 0.833）；Maroon→Navy 2 例、Navy→Maroon 1 例。

# 裁判识别验证集（referee_crops/）

## 标注规则

- 来源：demo2-30s-test（900 帧）与 raw1/raw2 全量 tracks（`output_videos/<src>/raw/object_tracks.jsonl`），由 `tools/extract_referee_candidates.py` 按 track 聚合裁剪（每 track 最清晰的 ≤3 帧、帧间距 ≥15）。
- 目录结构：`eval/referee_crops/{demo2,raw1,raw2}/candidates.jsonl` 为候选清单；`.../demo2/verdict/candidates.jsonl` 为新 assigner 改判案例（flag_player = 我判成裁判的球员、restore_referee = 我恢复队色的 referee 类），**人工重点核对这两类**。
- `manual_label` 取值：`referee`（裁判）/ `maroon`（栗色红队）/ `navy`（藏青蓝队）；看不清或有歧义保持 `null`（不进评估）。
- 清单已用规则预填 auto 结论，**你只需纠错**：把预填错误的字段改掉即可。
- 评估脚本：`tools/eval_referee.py`（主口径 = referee 精确率/召回率 + 球队色保持率）；回归门槛 = 上节 34-crop 验证集 balanced acc 保持 1.0。
