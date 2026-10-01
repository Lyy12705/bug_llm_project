# Ticket classification and historical retrieval

## Duplicate detection／retrieval

| 實作 | 方法及輸出 | 整合狀態 |
|---|---|---|
| [MAIN DuplicateDetector](../bug_tracking_llm_system/bug_tracking_llm_system/src/modules/duplicate_detector.py) | title／description／error similarity + component bonus；`is_duplicate`、`duplicate_of`、`needs_review`、`top_k_candidates` | Implemented，MAIN default |
| [DUP package](../bug-duplicate-detection/bug-duplicate-detection/src/duplicate_ticket_detection/) | dataset／duplicate groups、TF-IDF、title/content SBERT triplets、hard negatives、metadata／stable rerank、decision classifier | Implemented but not integrated（相對 MAIN default）；模型推論／重訓 Not verified |

MAIN 從設定的 historical JSONL 讀資料；不存在時回空集合，不能解讀成已搜尋完整歷史庫。預設 threshold 0.82、review margin 0.05、Top-k 5，來自 `PipelineConfig`。positive duplicate 直接結束；review margin 或 detector exception 暫停於人工複核。候選也傳入 priority 的 related-report factor。

DUP 的歷史 retrieval 是 duplicate candidate ranking，不是 code retrieval，也不是 assignee history。其 CLI `evaluate`／`rank`／`build-triplets` 和研究 runner 在獨立 cwd 執行。SBERT 存在不表示 MAIN 使用 SBERT；DUP 的 ranking 分數不能套到 MAIN 0.82 gate。

半自動候選推薦與二元判定是兩種驗收：Top-k／MAP／MRR 不能替代 duplicate precision／recall。按需讀 [DUP README](../bug-duplicate-detection/bug-duplicate-detection/README.md)、[literature_method](../bug-duplicate-detection/bug-duplicate-detection/docs/literature_method.md) 與 [experiments](../bug-duplicate-detection/bug-duplicate-detection/docs/experiments.md)。目前已有使用者未提交的 metrics／stable-rerank 修改，不應覆蓋。

## Priority

[MAIN PriorityClassifier](../bug_tracking_llm_system/bug_tracking_llm_system/src/modules/priority_classifier.py) 是 `deterministic_drone_gray_baseline`。先保留有效 existing priority，否則用 textual、temporal、author、related-report、severity、component 六種 factor 加權成 P1–P5。exception 由 orchestrator 回 P3、confidence 0、fallback_used。已有 priority 的保留結果不能混算成從無標籤文字預測的模型準確率。

[PRI scripts](../bug-priority-drone/bug-priority-drone/scripts/) 有 cleaning、split、REP-／BM25 features、classifier／boundary／recall-balanced training、natural-holdout prediction export；本機存在 `models/recall_balanced_priority_model.joblib`。它們是 Implemented but not integrated：MAIN 沒有載入此 joblib 或 feature pipeline。

PRI 的 feature matrix、vectorizer、metadata encoder／scaler 與分類器是一起使用的契約，不能只將模型檔放進 MAIN 就算完成 integration。重現需要 numpy／pandas／scipy／scikit-learn／joblib 等依賴；以 [requirements](../bug-priority-drone/bug-priority-drone/requirements.txt) 和 [README](../bug-priority-drone/bug-priority-drone/README.md) 為準。

## 驗收與邊界

Duplicate 任務要驗證 Top-k、review、positive、missing history／invalid JSON、exception short-circuit；priority 任務要驗證 P1–P5、existing priority 與無標籤輸入、fallback。替換模型屬另一次明確授權的 implementation 任務，不在文件整理中進行。Split、label leakage 和不同指標分母見 [datasets](datasets.md)、[evaluation](evaluation.md)。Assignee 後段另見 [assignee_routing](assignee_routing.md)。
