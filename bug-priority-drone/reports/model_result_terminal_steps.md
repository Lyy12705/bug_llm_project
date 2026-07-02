# 查看模型結果的終端機步驟

本文件記錄如何在終端機查看目前最佳模型的優先級判定結果。目前最佳模型為 `recall_balanced_priority_model`，它在 DRONE/REP- 與 P2 keyword features 的基礎上，加入 cost-sensitive class sample weights 與 P1/P2 boundary classifier，並用 macro recall、minimum recall、macro F1 等目標選出較平衡的模型。

## 1. 進入專案資料夾

```bash
cd /Users/linyaying/Documents/bug-priority-drone-v2
```

## 2. 查看目前最佳模型結果

只顯示主要指標：

```bash
python3 scripts/show_model_metrics.py
```

查看原始 CSV 與完整各類別分類報告：

```bash
cat reports/recall_balanced_best_eval.csv
cat reports/recall_balanced_best_class_report.csv
```

## 3. 用表格方式查看重點指標

```bash
python3 - <<'PY'
import pandas as pd

df = pd.read_csv("reports/recall_balanced_best_eval.csv")
cols = [
    "accuracy",
    "macro_f1",
    "off_by_one_accuracy",
    "mae",
    "p1_recall",
    "p2_recall",
    "p3_recall",
    "p4_recall",
    "p5_recall",
]
print(df[cols].to_string(index=False))
PY
```

目前 `natural_holdout` 結果：

| 指標 | 數值 |
|---|---:|
| Accuracy | 0.7266 |
| Macro F1 | 0.7265 |
| Off-by-one accuracy | 0.9045 |
| MAE | 0.4040 |
| P1 recall | 0.6583 |
| P2 recall | 0.6869 |
| P3 recall | 0.8350 |
| P4 recall | 0.7136 |
| P5 recall | 0.7387 |

## 4. 查看改良實驗比較結果

```bash
cat reports/recall_balanced_improvement_summary.md
cat reports/recall_balanced_comparison.csv
cat reports/recall_balanced_priority_model.md
```

## 5. 查看 P2 錯誤分析

查看改良後真實 P2 被錯判成 P1 或 P3 的案例摘要：

```bash
cat reports/p2_error_analysis_improved_priority_p2_keywords_summary.md
```

查看人工分類後的錯誤類型：

```bash
cat reports/p2_error_manual_categories_improved_priority_p2_keywords.md
```

## 6. 重新產生目前最佳模型

如果特徵資料已存在，只要重新訓練與評估目前最佳設定：

```bash
python3 scripts/train_recall_balanced_priority_model.py
```

此指令會輸出：

- `models/recall_balanced_priority_model.joblib`
- `reports/recall_balanced_best_eval.csv`
- `reports/recall_balanced_best_class_report.csv`
- `reports/recall_balanced_priority_model.md`

## 7. 指標判讀

`accuracy` 是整體完全判對的比例，越高越好。

`macro F1` 是 P1-P5 五個類別各自 F1 的平均，越高代表模型不是只偏向某一類。

`off-by-one accuracy` 是預測差一級也算可接受的比例，例如 P2 判成 P1 或 P3。Priority 本身是序位等級，所以這個指標很適合補充說明。

`MAE` 是預測 priority 與真實 priority 的平均距離，越低越好。

`P1-P5 recall` 代表每個真實 priority 被模型成功抓出的比例。目前最佳模型的 P2 recall 是 `0.6869`，P4 recall 是 `0.7136`，比前一版 `0.6834` 明顯改善。

## 8. 報告建議

報告時可以把目前模型說成：

本研究先依據 DRONE/GRAY 建立六類因素，包含 textual、temporal、author、related-report、severity、product/component。接著利用 Bugzilla 的 `dupe_of` 欄位訓練 REP- related-report 權重，並進一步調整 BM25/BM25F 中 summary 與 description 的欄位權重。最後根據錯誤分析加入 keyword features、cost-sensitive class sample weights 與 P1/P2 boundary classifier，並以 macro recall、minimum recall、macro F1 等目標選出較平衡的模型。

目前最佳模型是 `recall_balanced_priority_model`，在 `natural_holdout` 上 accuracy 為 `0.7266`，macro F1 為 `0.7265`，P2 recall 為 `0.6869`。
