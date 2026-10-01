# AI Bug Tracker / Automated Program Repair

本工作區涵蓋 bug report extraction／normalization、
ticket classification／retrieval／routing，以及
fault localization／patch generation／validation。

各研究子專案與主系統預設實作不完全相同；
不要把研究成果等同預設整合功能。

## Working principles

- 先確認任務的實際專案根目錄與入口，以目前程式、設定與測試為準。
  根目錄是多專案工作區，並非單一 Git checkout；保留既有未提交修改。

- 只讀取任務需要的文件。跨模組或初次辨識入口時讀
  [總覽](docs/project_overview.md)，不要每次掃讀全部文件。

- 小範圍變更、沿用既有介面；區分 production default、
  opt-in adapter、experiment、歷史 artifact。
  未完成功能如實記錄。

- 目前主要 Patch Generation pipeline 使用現有 Fault Localization
  的預測結果作為 localization input。
  Ground truth／reference patch 僅用於 dataset 與 evaluation；
  除非任務明確要求，不新增 Ground-truth／Oracle localization
  → FIM pipeline。

## Documentation routing

| 任務 | 先讀 |
|---|---|
| 入口、全系統流程、模組狀態 | [project_overview](docs/project_overview.md) |
| Bug report → JSON | [extraction_normalization](docs/extraction_normalization.md) |
| Priority、duplicate、歷史 ticket retrieval | [ticket_classification](docs/ticket_classification.md) |
| Assignee、Top-k confirmation、feedback、deployment gate | [assignee_routing](docs/assignee_routing.md) |
| Code indexing、file／symbol localization | [fault_localization](docs/fault_localization.md) |
| FL handoff、FIM、diff、repair CLI | [patch_generation](docs/patch_generation.md) |
| Dataset、schema、split、gold provenance | [datasets](docs/datasets.md) |
| Tests、metrics、validation、test／commit generation | [evaluation](docs/evaluation.md) |
| 撰寫任務、驗收 | [task_prompt_templates](docs/task_prompt_templates.md)、[done_criteria](docs/done_criteria.md) |

更細的舊文件由各模組文件按需連結；
[本次盤點與差異](docs/restructure_review.md)
只在追查已知限制時讀。

## Validation principles

執行與變更相關的測試及 Input → Output 驗證，
記錄 cwd、指令、環境、結果、fallback 與限制。

Mock／synthetic smoke 不能證明真實模型或 benchmark 成功；
generated、apply passed、plausible、official resolved 不可互換。

無法驗證者標示 `Not verified`，
不得宣稱該功能 Done。

## Decision boundaries

可自行搜尋、進行 read-only analysis、執行相關驗證，
以及修改當前任務明確授權範圍內的程式碼與文件。

未獲任務授權，不修改 dataset／ground truth、
evaluation definition、核心模型、研究方法或重大架構，
也不大量刪除資料。

文件整理任務不順便修改 production code 或實作缺失功能；
sealed holdout 不拿來調參。

新實驗輸出使用新路徑，保留 frozen artifacts。