# Datasets, schemas and provenance

這份文件描述目前磁碟上的資料與保護邊界，不重新定義 dataset／labels／splits。`data`、`dataset`、`reports`、`runs`、模型 cache 與臨時 upstream repositories 不可視為同一种 artifact。

## 位置與用途

| Domain | 位置 | 實際內容／schema |
|---|---|---|
| Extraction | [ToJson/dataset](../ToJson/dataset/) | raw issues、processed bug_report、labeled gold、predicted_json；`id` 為對齊鍵，gold 在 `json_ground_truth` |
| Duplicate | [DUP data](../bug-duplicate-detection/bug-duplicate-detection/data/) | Mozilla/Eclipse CSV，含 ticket_id/title/description/duplicate_of/duplicate_group 與 metadata；examples 是小型 fixture |
| Priority | [PRI data](../bug-priority-drone/bug-priority-drone/data/) | raw、clean_with_dupe、experiment_splits_with_dupe、train/validation/natural_test features；matrix／labels／feature metadata 配套 |
| Main／assignee | [MAIN data](../bug_tracking_llm_system/bug_tracking_llm_system/data/)、[assignee research](../bug_tracking_llm_system/bug_tracking_llm_system/assignee_triage_accuracy/) | historical tickets、raw example、processed checkpoints；BMO paper-grade history/validation/test、rosters、temporal manifests、shadow evidence |
| FL／repair | [FL data](../fault_localization_feature_files/data/fault_localization/)、[REPAIR data](../patch/integrated-fixed-patch/data/fault_localization/) | SWE-bench Lite/full/live tickets、gold、splits／sealed artifacts；兩處 evaluator policy 不同 |

2026-09-16 抽查：`ToJson/dataset/labeled/test.jsonl` 56 筆、六個 gold fields（bug_type/component/os/version/priority/error_message）；`public_test.jsonl` 100 筆，現存 gold 只有 component/os/version/priority。擴充 schema 不代表既有 gold 已擴充。MAIN 小型 historical JSONL 的 keys 包含 ticket_id/title/description/component/priority/assignee，不能當大型完整 ticket database。

FL full `test_tickets.jsonl` 的第一筆除了公開 ticket fields 也有 `fail_to_pass`、`pass_to_pass`、`hints_text`。因此「tickets 檔」並不天然等於可直接送 model 的 public-only record；跨 repair boundary 必須遵守現有 sanitizer／allowlist。REPAIR batch 的 `sanitize_ticket` 與 generator `evidence()` 分別處理這個邊界。

## Gold 與 evaluation policy

[FL MERGED_GROUND_TRUTH](../fault_localization_feature_files/docs/MERGED_GROUND_TRUTH.md) 描述目前獨立 FL 的 `merged-repair-ground-truth-v1`：PR/merge/base provenance、完整 base→merge diff、same-snapshot test receipts／hashes；只有 merged 不足以驗證 repair。準備器／evaluator 已有程式，但真實完整 cohort 的 verified receipts 本次 Not verified。

該 policy 的 all-changed-paths 可含 tests，與 default source-only retrieval 的候選範圍有差異；不能修改分母或悄悄刪 gold targets 消除差異。歷史 developer-patch 指標在該 evaluator 必須明示 legacy mode，不能將歷史分數改稱新 policy 成績。此要求不應直接套用到 MAIN 或 REPAIR 的另一份 evaluator，先檢查程式。

Patch reference／hidden tests／gold symbols 不可進 FIM localization input。REPAIR 單案 reference comparison 在生成後讀取，描述性 patch/file overlap 不能代替 functional success。已有 oracle diagnostics 保留為 evaluation analysis，不能擴充成第二條 generation pipeline。

## Split 與保存規則

Extraction manual gold 與 source consistency 分開；DUP duplicate family／query relevance、PRI validation selection 與 natural holdout、assignee label-availability time、FL development／sealed holdout 各有不同實驗單位。沒有一個跨 domain 通用「accuracy」。

不讀 sealed labels 作調參，不重切 frozen cohort，不重寫 gold、不改 metrics definition。若未來任務需要資料建置，輸出到新路徑，保留來源、schema、ID、split、SHA／seed／環境及未知狀態。檔案存在、LFS pointer、cache 目錄或 README 的舊數字都不能證明資料可用／完整／未曝光。

本次僅盤點目錄、讀取 schema／非 sealed 樣本及現有 manifests；未下載 dataset、執行 gold builder、載入 sealed holdout labels、重訓或改寫資料。

## 文獻與其他 artifacts

[文獻目錄](../文獻/) 有 extraction、duplicate embedding、priority、triage、Code Llama、FIM、test oracle、regression testing 等 11 份 PDF；根目錄另有研究申請 PDF 與架構圖。已盤點其位置／用途，沒有把論文方法當作已實作證據。`ToJson/dataset/literature/` 是 HumanEval／MBPP code-generation benchmark；根目錄 `test_predictions_local.jsonl` 是獨立大型輸出，未有證據可將它等同目前主系統 live result。

評分及命令使用 [evaluation](evaluation.md)，只有任務真的涉及某資料集時再讀該子專案詳細 README。
