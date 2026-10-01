# Fault localization

## 先選正確入口

| 位置 | 關鍵程式 | 用途 |
|---|---|---|
| [MAIN BugLocalizer](../bug_tracking_llm_system/bug_tracking_llm_system/src/modules/bug_localizer.py) | `utils/fault_localization_stage3.py`、`symbol_localization.py` | 完整 ticket pipeline 的目前預設定位 |
| [FL](../fault_localization_feature_files/src/) | `utils/fault_localization.py`、`modules/bug_localizer.py`；`scripts/` | 獨立研究、凍結實驗與目前 merged-gold evaluator |
| [REPAIR pipeline](../patch/integrated-fixed-patch/src/fim_patch/pipeline.py) | `build_code_index` → `localize_ticket` → `localization_for_patch` | 與 FIM 綁定同一 clean base snapshot 的定位 |

MAIN 同時保留 `utils/fault_localization.py` 與 `fault_localization_stage3.py`。不可因檔名相似，就認定 MAIN CLI、demo、獨立 FL script 與 REPAIR 使用同一版本。任務先追 import/call graph；不要修改 `tmp/` 副本來代替真實入口。

## Pipeline 與 handoff

Repository source → index（file path、symbol、line range、source fingerprint）→ ticket evidence／TF-IDF 或可選 SBERT retrieval → Stage-1 file pool → 可選 file rerank／Stage-2 file candidates → Stage-3 structured AST/symbol pool → ranked symbols + confidence。

MAIN `BugLocalizer.localize()` 明確使用 `symbol_localization=True`、`symbol_llm_rerank=False`、`symbol_candidate_k=30`、`symbol_top_k=5`、`symbol_retrieval_mode=b1-structured`、`symbol_selection_mode=coverage-aware-v1`；file backend 預設 tfidf。這與獨立研究「最佳 frozen experiment」不是同一配置敘述。

Patch consumer 讀 `stage3_ranked_symbols`，不要把 legacy `localized_candidates` 的 file ranking 冒充 symbol ranking。每個 symbol 至少有 `file_path`、`symbol_qualified_name`、`start_line`、`end_line`。MAIN 額外帶 `repo`、`base_commit`、`source_file_sha256`；REPAIR 的 fresh adapter 加上 `schema_version=LocalizationForPatchV1`。

保留原始 `confidence_level`、`should_manual_review`、`recommend_patch_generation`。空 symbol、低信心、hash／commit／行號不合必須如實報告，不能為了 patch 成功而捏造 high confidence 或以 gold 補齊 location。Context／FIM 細节见 [patch_generation](patch_generation.md)。

## 已實作與研究限制

Retrieval、index、file/symbol ranking、MAIN Stage-3 integration 是 Implemented；本次 handoff contracts 與 REPAIR 的真實 FL＋可控 backend smoke 通過。大型 repository／SBERT／Ollama／正式 cohort 的本次結果皆 Not verified。

Symbol LLM reranking 是 Experimental，並非 MAIN selected default。[FL README](../fault_localization_feature_files/README.md) 記錄 selected coverage-aware-v1、G2 尚未達標與 exploratory WP4 的限制；這些是凍結研究證據，不是本次重算。非 Python symbol extraction 使用啟發式；不能由 Python AST smoke 外推所有語言。

## 按需深入

| 任務 | 進一步文件 |
|---|---|
| 模型、訓練／資料契約 | [FAULT_LOCALIZATION_MODEL_SPEC](../fault_localization_feature_files/FAULT_LOCALIZATION_MODEL_SPEC.md) |
| Stage-2 → AST | [STAGE2_TO_AST_IMPLEMENTATION_PLAN](../fault_localization_feature_files/STAGE2_TO_AST_IMPLEMENTATION_PLAN.md) |
| Symbol reranker／WP4 | [Stage-3 plan](../fault_localization_feature_files/STAGE3_SYMBOL_RERANKER_IMPLEMENTATION_PLAN_ZH.md) |
| MAIN 使用者介面 | [fault_localization_user_guide](../bug_tracking_llm_system/bug_tracking_llm_system/docs/fault_localization_user_guide.md)（需留意歷史設定） |
| Gold／metrics | [datasets](datasets.md)、[evaluation](evaluation.md)，再依入口選 evaluator |

獨立 FL 已有 merged-repair-ground-truth-v1 policy；MAIN／REPAIR 的歷史 evaluator 不可默認具有相同政策。既有 oracle analysis 是定位候選池診斷，不是允許建立 Oracle → FIM 的理由。
