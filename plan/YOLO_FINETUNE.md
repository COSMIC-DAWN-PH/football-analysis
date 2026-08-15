# PLAN：YOLO 微调（对象检测四类：ball / goalkeeper / player / referee）

状态：**待开始**（数据侧前置已完成，见 §4）
依赖的前置计划：`TEAM_COLOR_CLASSIFICATION_DONE.md`、`REFEREE_DETECTION_DONE.md`（均已验收关闭）
前置数据资产：`eval/finetune_dataset/`（2032 图、213 个修正框、四类预标、split=raw1/raw2/demo2；不入 git）

## 1. 背景：颜色后处理解决不了的四类问题

两个已关闭计划把颜色层做到极限后，剩余误差全部指向 YOLO 检测层：

| # | 现象 | 证据（前两个计划的遗留清单） | 微调如何解决 |
|---|------|------------------------------|--------------|
| A | referee 类误检：红蓝球员被判 referee，真裁判被判 player | demo2：r197(153帧藏青)、r243/r2(栗色) 假阳性；raw 全片 1949/1943 条 referee 类被颜色层反向纠正 | 提升 referee 类召回与精度，减少双向混淆 |
| B | ID 污染 track：黄裁判与球员混用同一 ByteTrack ID → 滑窗滞后 ~1s | demo2 r74/p28；raw 终评 referee 召回 0.79 的漏判主因 | referee 类变可靠 → 裁判拿到独立 referee 类 ID，不再混进 player 类 |
| C | 门将漏检：navy 门将 p210 全程 player 类；gk 类反而误检球员/裁判（raw 全片 gk 类 742/742 帧判 maroon 的根源） | p210 案例；raw gk 类分布 | goalkeeper 类召回提升 + 误检下降，门将路径真正可用 |
| D | boundary 阴影漂移：hue 87-94 的栗色球员颜色层判错（raw club preservation 0.56 主因） | raw 终评 boundary 类 | YOLO 学球衣外观而非单点颜色，阴影下仍可分 |
| E | 与队服同色裁判、非两队人员（红 bib 替补等） | raw1 红球衣误判案例（已缓解但归队仍错） | referee 类按外观识别；非两队人员可用低置信度过滤 |

**结论：颜色后处理已到物理极限，下一步只能微调 `object-detection.pt`。**

## 2. 目标

- referee 类：未见过比赛上召回 ≥ 75%、精度 ≥ 90%（沿用 training/README 的 ball 类推广标准）；
- goalkeeper 类：召回显著提升（p210 型漏检消除），误检下降；
- player 类：mAP 无回退（≥ 现有基线，训练前后对比）；
- 与颜色后处理叠加后：demo2 标注集 + raw 终评指标不回退，ID 污染/B 类问题缓解；
- 导出 OpenVINO FP16（1280，dynamic），在本机 `intel:GPU/NPU/CPU` 路径验证后替换运行时权重。

## 3. 数据计划

现状：`eval/finetune_dataset/` 2032 图（demo2 75 + raw1 938 + raw2 935，2 秒抽帧 + 标注帧强制包含），四类预标 + 213 个修正框（demo2 人工 + raw 双模型共识+用户裁决）。首轮训练**不够**，需扩充：

| 项 | 目标 | 做法 |
|----|------|------|
| referee 正样本 | ≥ 600 张（training/README 的 ball 类同款口径） | 从 raw1/raw2 全量 tracks 按 hue/类别挖更多黄裁判帧（`tools/extract_referee_candidates.py` 可复用）；必要时用户补标 |
| goalkeeper 正样本 | 每队门将 ≥ 200 张 | 从 raw 全片挖 gk 类 + p210 型深紫/黑门将帧，人工/双模型核标 |
| hard negatives | 每类 ≥ 100 张易混样本 | boundary 阴影样本、红 bib、同色系干扰（已有一批 boundary/dark 标签） |
| 新场次 | 至少 1 场**完全不同**的比赛（新球场/新队色/新裁判色）进 test | 用户提供新视频；train 可用现有三源，test 必须未见过（训练工作流强制校验） |
| 扩充方法 | 抽帧间隔加密 + 每帧全框标注 | `tools/build_finetune_dataset.py` 已有；Roboflow 平台可做人工精修 |

split 纪律（沿用 training/README）：整场视频不跨 split；70/15/15；test 含至少一场未见过比赛；相邻帧不跨 split（抽帧间隔 ≥2s 已满足）。

## 4. 代码改造（本机完成，无需 GPU）

`training/prepare_dataset.py` 增加 `object` task：
- `validate --task object`：四类 id 0-3 校验（复用 `_validate_label`，expected=5）；
- `extract --task object`：与 ball 相同的抽帧+预标，预标写四类 box（class_id 保留，不写 0）；
- `unpack` 不变。

`training/train_models.py` 增加 `object` task：
- 权重起点：`models/weights/object-detection.pt`（沿用现有四类模型的预训练权重，避免从 COCO 重新学）；
- `task="detect"`，`mosaic=1.0`、`scale=0.5`、`fliplr=0.5`（与 ball 类同款增强）；
- 训练后同样导出 OpenVINO FP16 1280 dynamic + `training_result.json`（不自动替换运行时权重）。

## 5. 训练（CUDA 云机，本机无 CUDA）

```bash
# 仓库根，CUDA 机器上
python -m training.prepare_dataset validate --task object --data training-data/object-yolo/data.yaml
python -m training.train_models --task object --data training-data/object-yolo/data.yaml --device 0 --batch 8 --epochs 200
```

数据路径：把 `eval/finetune_dataset/`（或 Roboflow 精修版导出 zip `unpack` 后）放到云机 `training-data/object-yolo/`。云机依赖 `requirements.txt`（torch+ultralytics CUDA 版）。

## 6. 验收与上线

1. **模型级验收**（云机 held-out）：referee 召回 ≥75% / 精度 ≥90%；goalkeeper 类无漏检回归；player mAP 不降；
2. **管线级验收**（本机）：OpenVINO FP16 导出 → `intel:GPU/NPU/CPU` 三设备推理冒烟 → 替换 `models/weights/object-detection*`（.pt + openvino 目录，保留旧权重备份）；
3. **回归验收**：demo2 全量端到端重跑（`--referee-color 168,156,74 --club1-player 120,37,66 --club1-goalkeeper 30,30,30 --club2-player 31,72,127 --club2-goalkeeper 48,37,68`）+ 新场次视频跑一遍，对照：
   - demo2 标注集 referee precision/recall 不回退；
   - 老 34-crop balanced acc = 1.0；
   - raw 终评 referee precision 1.0 不回退、club preservation 应因 boundary 改善而上升；
   - ID 污染 track（r74 型）滑窗滞后消失或大幅减少；
   - 端到端速度下降 ≤ 20%；
4. 双模型交叉验证门（沿用惯例）：kimi-k2.7-code 与 gpt-5.6-luna 独立复核以上验收项后裁决通过才上线。

## 7. 涉及文件

| 文件 | 改动 |
|------|------|
| `training/prepare_dataset.py` | 新增 `object` task（validate/extract） |
| `training/train_models.py` | 新增 `object` task |
| `eval/finetune_dataset/` | 扩充数据（referee/gk 正样本挖掘 + 新场次） |
| `models/weights/object-detection*` | 验收通过后替换（含 OpenVINO 导出） |
| `training/README.md` | 补充 object task 说明 |

## 8. 风险与对策

- **数据量不足** → 首轮可用三源 2032 图跑通流程验证代码，正式轮等数据扩充达标；Roboflow 精修可快速补标；
- **微调后颜色层与 YOLO 叠加冲突**（如 referee 类误检增多）→ 管线回归验收强制覆盖，颜色层参数可调（`referee_assign_dist`/`gk_match_dist` 等已在 `ClubAssigner` 暴露）；
- **新场次缺失** → test 暂时用 demo2（不同拍摄段），但正式上线前必须补一场完全未见过比赛（计划硬性要求）；
- **云机环境差异** → 训练脚本本地可先 dry-run validate；导出 dynamic shape 保证本机 1280/1920 全分辨率可用。

## 9. 推进记录

| Phase | 实现模型 | 验证模型 | 基线指标 | 完成后指标 | 裁决 | 提交 |
|-------|----------|----------|----------|------------|------|------|
| 0（数据扩充） | — | — | 2032 图 / 213 修正框 | — | — | — |
