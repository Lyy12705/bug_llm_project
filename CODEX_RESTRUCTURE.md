# Repository Documentation Restructuring

## Goal

請重新整理目前 repository 的 coding-agent documentation architecture，
使本專案更適合 Codex / coding agent 長期開發與維護。

本次工作的目標是建立：

- 精簡的 `AGENTS.md`
- 模組化 `docs/`
- Task Prompt Templates
- 明確的 Done Criteria

本次任務主要是「分析現有 repository 並整理文件架構」，
不是重新實作或修改研究功能。


## Project Scope

本 repository 是一個完整的 AI-assisted Bug Tracking and
Automated Program Repair system。

不要只把它理解為 Fault Localization / Patch Generation 專案。

目前整體研究架構大致包含三個 domain：

### 1. Bug Report Extraction and Normalization

User Bug Report
→ LLM Information Extraction
→ Structured JSON Bug Information

### 2. Automated Ticket Classification and Routing

Structured Ticket
→ Priority Classification
→ Duplicate Detection
→ Historical Ticket Retrieval / Similarity
→ Ticket Routing / Assignment

### 3. Fault Localization, Patch Generation and Validation

Bug / Ticket
→ Fault Localization
→ Predicted Buggy Location / Span
→ Context Construction
→ FIM Patch Generation
→ Patch Application
→ Validation / Testing
→ Evaluation

以上為研究架構，不代表 repository 中所有功能都已完成。

必須根據實際程式碼判斷 implementation status。


## Source of Truth

請先完整檢查 repository 的實際：

- directory structure
- source code
- scripts
- modules
- README
- existing documentation
- configuration
- datasets
- evaluation scripts
- tests
- literature / references（若存在）

不要因為 README、架構圖、論文、註解或計畫書中出現某功能，
就直接判定該功能已經 Implemented。

如果文件與實際程式碼不一致：

以目前 repository 的實際 implementation 為主要依據，
並記錄差異。


## Implementation Status

分析各模組時，請使用：

- Implemented
- Partially implemented
- Not implemented
- Experimental
- Implemented but not integrated
- Not verified

不要把「存在一個 script / function」直接視為完整功能已完成。


## Documentation Architecture

請根據 repository 的實際模組建立適合的 documentation architecture。

至少應考慮：

- `AGENTS.md`
- `docs/project_overview.md`
- extraction / normalization documentation
- ticket classification / duplicate detection documentation
- fault localization documentation
- patch generation / FIM documentation
- dataset documentation
- evaluation documentation
- task prompt templates

但不要為不存在、尚未實作或內容過少的功能，
強制建立大量空白文件。

請根據 repository 實際複雜度決定合理拆分方式。


## AGENTS.md Principles

`AGENTS.md` 必須保持精簡。

它只應保存：

- Project Overview
- 長期 Repository Working Principles
- Documentation Routing
- Validation Principles
- Coding Agent Decision Boundaries

不要把各模組大量技術細節放進 `AGENTS.md`。

技術細節應放入對應的 `docs/`。


## Progressive Disclosure

Codex 不應該每次任務都讀取所有 documentation。

請建立 documentation routing，使 coding agent：

只在任務涉及某個模組時，
才讀取該模組相關文件。

例如：

Fault Localization task
→ 讀取 Fault Localization documentation

Patch Generation task
→ 讀取 Patch Generation documentation

Duplicate Detection task
→ 讀取 Duplicate Detection documentation

Evaluation task
→ 讀取 Evaluation documentation


## Patch Generation Research Constraint

目前 Patch Generation 使用：

Current Fault Localization
→ Predicted Buggy Location / Span
→ FIM Patch Generation
→ Patch Application
→ Validation
→ Evaluation

Patch Generation 的 localization input
必須來自目前 Fault Localization 模組的預測結果。

目前不要建立：

Ground-truth Localization → FIM

或：

Oracle Localization → FIM

作為第二條 Patch Generation pipeline。

Ground truth 可以存在於 dataset / evaluation，
但不是目前 Patch Generation 的 localization input。


## Task Prompt Templates

請建立適合未來 Codex 任務使用的 Prompt Templates。

Template 應以以下結構為主：

- Goal
- Scope
- Relevant Documentation
- Constraints
- Expected Output
- Done Criteria

避免建立過度詳細、死板的 step-by-step SOP。


## Done Criteria Principles

Coding agent 不應只因程式碼已寫完就宣稱 Done。

任務完成時應確認：

- 實際 implementation 與文件一致
- 相關程式可以執行
- Input → Output 可以實際走通
- 已執行與修改相關的測試
- 沒有此次修改造成的 regression
- 無法驗證的部分標示 `Not verified`

如果無法驗證：

不要宣稱已完成。


## Decision Boundaries

Codex 可以自行：

- 搜尋 repository
- 閱讀程式碼
- 閱讀相關 documentation
- 建立或修改本次要求的 documentation
- 執行必要的 read-only analysis

以下情況不要自行進行：

- 修改 dataset ground truth
- 改變 evaluation definition
- 更換核心模型
- 改變研究方法
- 大規模修改 repository architecture
- 刪除大量資料
- 修改與本次 documentation restructuring 無關的 production code


## Scope of This Task

本次可以：

- 分析 repository
- 建立 / 修改 `AGENTS.md`
- 建立 / 修改 `docs/`
- 整理 module relationships
- 整理 pipeline
- 建立 Task Prompt Templates
- 建立 Done Criteria

本次不要：

- 重寫研究功能
- 修改 Fault Localization algorithm
- 修改 Patch Generation implementation
- 優化模型 accuracy
- 修改 dataset
- 修改 ground truth
- 改變 evaluation definition
- 更換模型
- 大規模 refactor production code

如果發現 implementation 缺失：

記錄在 documentation。

不要在這次任務順便實作。


## Completion

完成後請自行 review documentation，確認：

- `AGENTS.md` 沒有過度膨脹
- 技術細節已放到適當 docs
- 沒有不必要的文件重複
- 文件與程式碼沒有明顯矛盾
- Planned feature 沒有被寫成 Implemented
- 未驗證功能沒有被寫成 Verified
- 沒有要求 agent 每次讀取所有 docs
- 沒有建立 Ground-truth / Oracle → FIM 第二條 pipeline


## Final Report

完成後請回報：

1. 建立或修改哪些文件
2. 最終 documentation architecture
3. `AGENTS.md` 的內容與用途
4. 各 docs 的用途
5. 目前 repository 實際辨識出的主要模組
6. 各主要模組的 implementation status
7. 目前完整 system pipeline
8. Fault Localization → Patch Generation 的串接狀態
9. 尚未完成或 Not verified 的功能
10. 發現的文件 / 程式碼不一致
11. 下一個建議處理的 implementation task