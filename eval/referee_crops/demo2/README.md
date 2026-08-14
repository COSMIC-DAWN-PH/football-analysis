# 裁判候选标注说明

请逐张查看本目录下的 PNG 图片，在 `candidates.jsonl` 中为每行填写 `manual_label`：

- `referee`：裁判（黄衣或黑衣）
- `maroon`：栗色/红队球员
- `navy`：藏青/蓝队球员
- 看不清/有歧义：保持 `null`（该样本不进评估）

文件名格式：`{source}_{p|r}{track_id}_f{帧号}_h{躯干中位hue}_{category}.png`；p=player 类，r=referee 类。
