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
