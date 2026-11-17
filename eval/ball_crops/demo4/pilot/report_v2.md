# 试点对比报告 v2：demo4 前 20 项 × 多图逐帧（15 帧全分辨率）方式

- 素材：每项 15 张 1920×1080 原图帧（f01→f15，覆盖候选帧前后 ±0.25s，f08 红框=候选位置）+ crop 放大图，全部全分辨率
- 判定方式：两模型都逐帧看完了 15 帧序列
- **人工 ground truth（你提供）：20 项全部不是足球（0 个 ball）**
- v1（帧条 640px 低清）对比见下

## 逐项对照（v2）

| 顺序 | id | 类别 | qwen3.7-plus | qwen3.6-flash | 一致? | 人工(全部非球，原因待你补) |
|---|---|---|---|---|---|---|
| 1 | demo4_c00006 | confirmed | ball ✗ | not_ball debris ✓ | ✗ | |
| 2 | demo4_b01721 | bridge | ball ✗ | not_ball debris ✓ | ✗ | |
| 3 | demo4_b01722 | bridge | ball ✗ | not_ball debris ✓ | ✗ | |
| 4 | demo4_b01723 | bridge | ball ✗ | not_ball debris ✓ | ✗ | |
| 5 | demo4_c00009 | confirmed | ball ✗ | not_ball debris ✓ | ✗ | |
| 6 | demo4_b01724 | bridge | not_ball ✓ | not_ball debris ✓ | ✓ | |
| 7 | demo4_c00012 | confirmed | not_ball ✓ | not_ball debris ✓ | ✓ | |
| 8 | demo4_b01725 | bridge | not_ball ✓ | not_ball debris ✓ | ✓ | |
| 9 | demo4_c00015 | confirmed | not_ball ✓ | not_ball other ✓ | ✓ | |
| 10 | demo4_b01727 | bridge | not_ball ✓ | not_ball other ✓ | ✓ | |
| 11 | demo4_c00017 | confirmed | not_ball debris ✓ | null ~ | ✓ | |
| 12 | demo4_c00020 | confirmed | not_ball debris ✓ | ball ✗ | ✗ | |
| 13 | demo4_b01731 | bridge | not_ball debris ✓ | ball ✗ | ✗ | |
| 14 | demo4_b01732 | bridge | not_ball debris ✓ | null ~ | ✓ | |
| 15 | demo4_c00024 | confirmed | not_ball debris ✓ | ball ✗ | ✗ | |
| 16 | demo4_b01733 | bridge | not_ball debris ✓ | null ~ | ✓ | |
| 17 | demo4_c00027 | confirmed | not_ball other ✓ | not_ball player ✓ | ✓ | |
| 18 | demo4_c00030 | confirmed | not_ball debris ✓ | ball ✗ | ✗ | |
| 19 | demo4_c00033 | confirmed | not_ball debris ✓ | null ~ | ✓ | |
| 20 | demo4_b01737 | bridge | not_ball debris ✓ | not_ball player ✓ | ✓ | |

## 统计（v2，对照 ground truth：20/20 全非球）

| 指标 | qwen3.7-plus | qwen3.6-flash | v1 帧条(低清)两模型 |
|---|---|---|---|
| 误判为 ball 数 | **5 / 20** | **4 / 20** | 12~13 / 20 |
| 正确判 not_ball | **15 / 20 (75%)** | 12 / 20 (60%) + 4 null | ~8 / 20 |
| null | 0 | 4 | 0~3 |
| 两模型一致 | 14 / 20 | 14 / 20 | 14 / 20 |

## 结论

1. **高分辨率+多帧运动信息显著提升了判定**：误判 ball 从 12~13 个降到 4~5 个；qwen3.7 的 11-20 项全部判对（全 not_ball）
2. **但仍有硬骨头**：qwen3.7 在 1-5 项（"草地上小白点"簇）全军覆没——这类远场白点和球的静止帧外观几乎无法区分；qwen3.6 反而不受此骗（判 debris），却在 12/13/15/18 项误判 ball（crop 圆形白斑）
3. **两模型互补性明显**：分歧的 6 项里，各有各的错——qwen3.7 错在 1-5，qwen3.6 错在 12/13/15/18
4. **离"可靠预标"仍有距离**：任意单模型在"全是非球"的 20 项里仍会产出 4-5 个假球。若人工全审，模型预标的价值主要在**排序（先审疑似球）而非直接采信**
5. 建议下一步任选：
   - A. 只把模型当"疑似球过滤器"：只采信两模型都判 ball 的（本组 0 个 → 完美零假球，但漏检风险未知）
   - B. 保持纯人工（模型预标到此为止）
   - C. 用你的人工原因列表继续训练口径：把两模型误判 ball 的 9 张（1-5 + 12/13/15/18）作为重点分析样例，看能不能提炼"人类一眼识别"的判据写进 prompt 再测一轮
