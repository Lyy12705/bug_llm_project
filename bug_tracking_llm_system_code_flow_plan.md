# 基於 LLM 的軟體錯誤追蹤與修復系統：程式碼流程與系統整合計畫書

## 一、整體系統架構

本專題的目標是建立一套半自動化的軟體錯誤追蹤與修復系統。使用者提交 bug report、issue 或 ticket 後，系統會先將非結構化文字轉換為標準 JSON，再判斷是否為歷史重複問題。如果不是 Duplicate，系統會繼續進行優先級預測、負責人分派、錯誤定位、補丁生成、測試案例生成、回歸測試，最後產生 Commit Message 草稿。

整體流程如下：

```text
使用者提交 Raw Ticket
    ↓
Module 1：Ticket 資料擷取與 JSON 正規化
    ↓
Module 2：Duplicate Ticket 偵測
    ├── 若為 Duplicate：回傳 duplicate_of，流程結束
    ↓ 若不是 Duplicate
Module 3：Ticket 優先級分類
    ↓
Module 4：Ticket 自動分派負責人
    ↓
Module 5：錯誤定位 Bug Localization
    ↓
Module 6：補丁生成 Patch Generation
    ↓
Module 7：測試案例生成 Test Case Generation
    ↓
Module 8：回歸測試 Regression Testing
    ↓
Module 9：Commit Message 生成
    ↓
Module 10：Pipeline Orchestrator 輸出 final_pipeline_result.json
```

### Module 1：Ticket 資料擷取與 JSON 正規化

功能目的：

將使用者提交的自然語言 bug report、log、環境資訊與截圖文字整理成標準 JSON 格式，方便後續模組處理。

輸入資料：

- `raw_ticket.json`
- 使用者輸入的標題、描述、log、環境資訊、截圖 OCR 文字

輸出資料：

- `structured_ticket.json`

使用的模型或技術：

- Code Llama-Instruct 或其他 instruction-tuned LLM
- Prompt Engineering
- JSON Schema Validation
- JSON repair fallback

整合方式：

此模組是整個 pipeline 的第一步，輸出的 structured ticket 會提供給 Duplicate Detection、Priority Classification、Assignee Triage、Bug Localization 等後續模組。

對應程式檔案：

- `src/modules/ticket_extractor.py`
- `src/utils/json_schema.py`
- `src/prompts/extract_ticket_prompt.txt`

未來可評估指標：

- exact match
- field-level F1
- schema valid rate

### Module 2：Duplicate Ticket 偵測

功能目的：

判斷新提交的 Ticket 是否與歷史 Ticket 重複，避免重複修復同一個問題。

輸入資料：

- `structured_ticket.json`
- `historical_tickets.jsonl`

輸出資料：

- `duplicate_detection_result.json`

使用的模型或技術：

- SBERT
- Sentence Embedding
- Cosine Similarity
- Triplet Network / Triplet Loss 微調
- top-k retrieval

整合方式：

若最高相似度超過 threshold，pipeline 直接回傳 Duplicate 結果；若不是 Duplicate，結果中的 top-k candidates 可提供給 Priority Classification 作為 related report factor。

對應程式檔案：

- `src/modules/duplicate_detector.py`
- `src/utils/embedding_utils.py`

未來可評估指標：

- accuracy
- precision
- recall
- F1
- top-k hit rate
- duplicate_of accuracy

### Module 3：Ticket 優先級分類

功能目的：

預測 Ticket 的優先級，例如 P1、P2、P3、P4、P5，協助團隊決定處理順序。

輸入資料：

- `structured_ticket.json`
- Duplicate Detection 的 top-k similar tickets
- 歷史優先級資料

輸出資料：

- `priority_prediction_result.json`

使用的模型或技術：

- DRONE / GRAY framework
- Multi-Factor Analysis
- Linear Regression
- Thresholding
- Text feature extraction

整合方式：

優先級結果會傳給 Assignee Triage，作為分派負責人時的參考，也會進入最終 pipeline 結果。

對應程式檔案：

- `src/modules/priority_classifier.py`
- `models/priority_gray/`

未來可評估指標：

- accuracy
- macro-F1
- weighted-F1
- per-class recall，尤其是 P1 / P2
- confusion matrix

### Module 4：Ticket 自動分派負責人

功能目的：

根據 Ticket 內容、component、歷史修復紀錄與優先級，自動建議最適合的負責人。

輸入資料：

- `structured_ticket.json`
- `priority_prediction_result.json`
- 歷史 assignee 資料

輸出資料：

- `assignee_prediction_result.json`

使用的模型或技術：

- Code Llama-Instruct
- DeepSeek-R1-Distill-Llama-8B + LoRA
- JSONL Instruction Tuning
- Supervised Fine-Tuning
- component-owner mapping fallback

整合方式：

此模組輸出負責人與理由，供團隊 triage 使用。即使後續自動修復失敗，也能回傳人工處理建議。

對應程式檔案：

- `src/modules/assignee_triager.py`
- `src/prompts/triage_prompt.txt`
- `models/triage_llm/`

未來可評估指標：

- top-1 accuracy
- top-3 accuracy
- MRR

### Module 5：錯誤定位 Bug Localization

功能目的：

根據 Ticket 描述、error message、stack trace 與 repository 程式碼，找出最可能出錯的檔案、函式與行號範圍。

輸入資料：

- `structured_ticket.json`
- repository path

輸出資料：

- `bug_location_result.json`

使用的模型或技術：

- Code Llama-Instruct
- Repository-Level Reasoning
- Long Context
- Code Retrieval
- Stack trace search
- ripgrep
- AST parser 或 tree-sitter

整合方式：

錯誤定位結果會傳給 Patch Generation，作為生成修補程式的主要上下文。

對應程式檔案：

- `src/modules/bug_localizer.py`
- `src/utils/code_retriever.py`
- `src/prompts/localization_prompt.txt`

未來可評估指標：

- file-level accuracy
- function-level accuracy
- top-k localization accuracy

### Module 6：補丁生成 Patch Generation

功能目的：

根據錯誤定位結果與 Ticket 內容，產生建議修補程式。

輸入資料：

- `structured_ticket.json`
- `bug_location_result.json`
- repository path

輸出資料：

- `patch_result.json`

使用的模型或技術：

- Code Llama-Instruct
- FIM Fill-In-the-Middle
- PSM Format
- unified diff generation
- patch apply validation

整合方式：

產生的 patch 會傳給 Test Case Generation 與 Regression Testing。若 patch 無法套用或編譯失敗，系統需回饋錯誤訊息給模型重新生成。

對應程式檔案：

- `src/modules/patch_generator.py`
- `src/prompts/patch_prompt.txt`
- `src/utils/git_utils.py`

未來可評估指標：

- compile success rate
- test pass rate
- patch correctness rate
- human review acceptance rate

### Module 7：測試案例生成 Test Case Generation

功能目的：

根據 bug report 與 patch 產生能重現錯誤的測試案例，並檢查測試是否符合 FIB 條件。

FIB 指的是：

- Fail In Buggy：在修復前版本應該失敗
- Pass In Fixed：在修復後版本應該通過

輸入資料：

- `structured_ticket.json`
- `patch_result.json`
- repository path

輸出資料：

- `generated_tests_result.json`

使用的模型或技術：

- LIBRO
- Few-shot Prompting
- Bug Reproduction Test Generation
- pytest / unittest
- FIB validation

整合方式：

產生的測試會交給 Regression Testing 使用，也會成為 patch 是否可信的重要依據。

對應程式檔案：

- `src/modules/test_generator.py`
- `src/prompts/test_generation_prompt.txt`

未來可評估指標：

- FIB success rate
- generated test compile rate
- bug reproduction rate

### Module 8：回歸測試 Regression Testing

功能目的：

確認 patch 修復錯誤後，沒有破壞既有功能，並分析測試是否覆蓋修改過的程式碼。

輸入資料：

- `patch_result.json`
- generated tests
- repository path

輸出資料：

- `regression_test_result.json`

使用的模型或技術：

- CLEVEREST
- Feedback-Directed Test Generation
- RIPR model
- pytest / unittest
- coverage.py

RIPR 包含：

- Reaching：測試是否執行到修改過的程式碼
- Infecting：是否造成程式狀態差異
- Propagating：差異是否傳播到輸出
- Revealing：測試是否揭露錯誤或確認修復

整合方式：

若 regression failed，pipeline 不應產生可直接提交的結果，而應標記為 `patch_unverified` 或 `patch_unsafe`。

對應程式檔案：

- `src/modules/regression_tester.py`
- `src/utils/git_utils.py`

未來可評估指標：

- changed-code coverage
- RIPR coverage
- regression failure detection rate

### Module 9：Commit Message 生成

功能目的：

根據 Ticket、bug localization、patch diff 與測試結果，自動產生 Commit Message 草稿。

輸入資料：

- `structured_ticket.json`
- `bug_location_result.json`
- `patch_result.json`
- `generated_tests_result.json`
- `regression_test_result.json`

輸出資料：

- `commit_message_result.json`

使用的模型或技術：

- Code Llama-Instruct
- Zero-shot Instruction Following
- Code Summary
- Prompt Engineering
- Conventional Commit template

整合方式：

Commit Message 是 pipeline 的最後輸出之一。若 LLM 生成失敗，可使用固定模板 fallback。

對應程式檔案：

- `src/modules/commit_message_generator.py`
- `src/prompts/commit_message_prompt.txt`

未來可評估指標：

- BLEU
- ROUGE
- BERTScore
- human readability score
- completeness score

### Module 10：Pipeline Orchestrator 系統整合流程

功能目的：

串接所有模組，負責流程控制、錯誤處理、log 紀錄、fallback 策略與最終結果輸出。

輸入資料：

- raw ticket
- repository path
- config

輸出資料：

- `final_pipeline_result.json`

使用的模型或技術：

- Python pipeline orchestration
- logging
- exception handling
- step-level checkpoint

整合方式：

所有模組都由 `PipelineOrchestrator` 呼叫。每一步都輸出中間 JSON，方便 debug、展示與評估。

對應程式檔案：

- `src/pipeline/orchestrator.py`
- `src/main.py`
- `src/config.py`

未來可評估指標：

- pipeline success rate
- average runtime
- fallback usage rate
- manual review rate

## 二、每個功能使用的模型與技術

### 1. Ticket 資料擷取與 JSON 正規化

使用模型：

- Code Llama-Instruct
- 或其他 instruction-tuned LLM

使用技術：

- Prompt Engineering
- JSON Schema Validation
- JSON repair
- rule-based fallback

目標：

將自然語言 bug report 轉成結構化 JSON。

建議 JSON 欄位：

```json
{
  "ticket_id": "BUG-001",
  "title": "Login fails when token is null",
  "description": "User cannot login when auth token is missing.",
  "bug_type": "runtime_error",
  "component": "authentication",
  "os": "macOS",
  "version": "1.2.0",
  "priority": null,
  "error_message": "TypeError: token is None",
  "steps_to_reproduce": [
    "Open login page",
    "Submit login without token"
  ],
  "expected_behavior": "System should show validation error.",
  "actual_behavior": "System crashes.",
  "logs": "TypeError at auth.py:42",
  "screenshots_text": "Login page shows blank error area."
}
```

### 2. Duplicate Ticket 偵測

使用模型：

- SBERT

使用技術：

- Sentence Embedding
- Cosine Similarity
- Triplet Network / Triplet Loss 微調
- top-k retrieval

方法：

1. 將新 Ticket 的 `title + description + error_message` 轉成 embedding。
2. 將歷史 Ticket 資料庫中的 Ticket 也轉成 embedding。
3. 使用 cosine similarity 找出 top-k 相似 Ticket。
4. 若最高分超過 threshold，標記為 Duplicate。

輸出格式：

```json
{
  "is_duplicate": true,
  "duplicate_of": "BUG-038",
  "similarity_score": 0.87,
  "top_k_candidates": [
    {
      "ticket_id": "BUG-038",
      "title": "Login crash when token missing",
      "similarity": 0.87
    }
  ]
}
```

### 3. Ticket 優先級分類

使用方法：

- DRONE / GRAY framework

使用技術：

- Multi-Factor Analysis
- Linear Regression
- Thresholding

特徵包含：

- Textual Factor：標題與描述文字特徵
- Temporal Factor：近期相似問題數量與分布
- Author Factor：回報者過去提交紀錄
- Related Report Factor：相似歷史 Ticket 的優先級
- Severity Factor：使用者描述的嚴重性
- Product / Component Factor：該元件過去錯誤頻率

輸出格式：

```json
{
  "predicted_priority": "P2",
  "confidence": 0.78,
  "factor_scores": {
    "textual_factor": 0.75,
    "temporal_factor": 0.6,
    "author_factor": 0.4,
    "related_report_factor": 0.82,
    "severity_factor": 0.9,
    "component_factor": 0.7
  }
}
```

### 4. Ticket 自動分派負責人

使用模型：

- Code Llama-Instruct
- DeepSeek-R1-Distill-Llama-8B + LoRA

使用技術：

- JSONL Instruction Tuning
- Supervised Fine-Tuning
- component-owner mapping fallback

訓練格式：

```json
{
  "system": "You are an expert bug triager.",
  "user": "Title: Login fails\nDescription: token is null\nComponent: authentication",
  "assistant": "developer_email_or_name"
}
```

輸出格式：

```json
{
  "assignee": "dev@example.com",
  "confidence": 0.81,
  "reason": "此問題與 authentication module 有關，該開發者過去處理過相關模組。"
}
```

### 5. 錯誤定位 Bug Localization

核心原則：

- 採用 retrieval-first，而不是直接要求 LLM 讀完整 repository。
- LLM / Code Llama-Instruct 只作為 optional reranker，用於少量候選片段。
- 預設流程必須能在沒有 LLM 的情況下執行，避免模型幻覺或本機推論成本過高。

使用技術：

- Code Retrieval
- TF-IDF / SBERT candidate reranking
- Stack Trace Search
- AST parser
- file-level aggregation
- identifier / path / domain-aware ranking signals
- optional Code Llama-Instruct rerank

方法：

1. 讀取 structured ticket JSON。
2. 掃描 repository 並建立 code index，切成 file/function/class/chunk。
3. 根據 bug report、error message、stack trace、path hint 與 identifier 召回候選程式碼片段。
4. 使用 TF-IDF 或 TF-IDF + SBERT rerank 排序候選片段。
5. 將同一檔案的多個 chunk 聚合成 file-level Top-k candidates。
6. 可選擇只把 Top-10 或 Top-20 candidates 交給 Code Llama-Instruct rerank。
7. LLM rerank 需使用 timeout、JSONL response cache 與 batch checkpoint，避免重跑成本過高。
8. 若 LLM 逾時、輸出格式錯誤或服務不可用，回退到 retrieval ranking 結果。

這個設計符合真實專案限制：大型 repository 不適合整包送入 LLM，且錯誤定位應先提供可追蹤、可評估、可重跑的候選檔案與判斷依據。

輸出格式：

```json
{
  "bug_location": {
    "file": "src/auth/validator.py",
    "function": "validate_token",
    "line_start": 38,
    "line_end": 52,
    "reason": "stack trace 指向 validate_token，且此函式未處理 token 為 None 的情況。"
  },
  "candidates": [
    {
      "file": "src/auth/validator.py",
      "score": 0.91,
      "reason": "stack trace and identifier signals match this file"
    }
  ]
}
```

評估方式：

- 使用 file-level Top-1、Top-3、Top-5 accuracy 與 MRR。
- 若 patch hunk 可對應到 function/class，後續再加入 symbol-level 評估。
- 目前 SWE-bench Lite 準備資料提供合法的 file-level ground truth。
- 目前正式 300 筆 SWE-bench Lite test split 結果：Top-1 0.5500、Top-3 0.7567、Top-5 0.8133、MRR 0.6559，且 300 筆皆可完成定位輸出。

### 6. 補丁生成 Patch Generation

使用模型：

- Code Llama-Instruct

使用技術：

- FIM Fill-In-the-Middle
- PSM Format
- unified diff

PSM 格式範例：

```text
<PRE>
def validate_token(token):
<SUF>
    return decoded_user
<MID>
    if token is None:
        raise ValueError("Token is required")
<EOT>
```

輸出格式：

```json
{
  "patch": "diff --git a/src/auth/validator.py b/src/auth/validator.py\n...",
  "modified_files": [
    "src/auth/validator.py"
  ],
  "explanation": "新增 token 為 None 時的檢查，避免後續 decode 造成 TypeError。"
}
```

### 7. 測試案例生成 Test Case Generation

參考方法：

- LIBRO

使用技術：

- Few-shot Prompting
- Bug Reproduction Test Generation
- FIB 判準

方法：

1. 根據 bug report 生成多個候選測試案例。
2. 在修復前版本執行測試，應該失敗。
3. 在修復後版本執行測試，應該通過。
4. 篩選符合 FIB 的測試案例。

輸出格式：

```json
{
  "generated_tests": [
    {
      "file": "tests/test_auth_validator.py",
      "test_name": "test_validate_token_none",
      "content": "def test_validate_token_none(): ..."
    }
  ],
  "fib_passed_tests": [
    "tests/test_auth_validator.py::test_validate_token_none"
  ],
  "test_result": "passed"
}
```

### 8. 回歸測試 Regression Testing

參考方法：

- CLEVEREST

使用技術：

- Feedback-Directed Test Generation
- RIPR model
- pytest / unittest
- coverage.py

方法：

1. 根據 patch 和 commit message 生成回歸測試。
2. 比對修復前後版本行為。
3. 若測試不足，將執行結果回饋給 LLM 重新生成。

輸出格式：

```json
{
  "regression_tests": [
    "tests/test_auth_validator.py",
    "tests/test_login_flow.py"
  ],
  "ripr_analysis": {
    "reaching": true,
    "infecting": true,
    "propagating": true,
    "revealing": true
  },
  "regression_result": "passed",
  "coverage": {
    "changed_code_coverage": 0.86
  }
}
```

### 9. Commit Message 生成

使用模型：

- Code Llama-Instruct

使用技術：

- Zero-shot Instruction Following
- Code Summary
- Prompt Engineering
- Conventional Commit

輸入：

- Ticket JSON
- Bug localization 結果
- Patch diff
- Test result

輸出格式：

```json
{
  "commit_message": "fix(auth): handle missing token during login validation",
  "commit_body": "This patch adds validation for missing authentication tokens before decoding. It prevents a TypeError and adds a reproduction test for the missing-token login case."
}
```

## 三、專案資料夾結構

建議 Python 專案架構如下：

```text
bug_tracking_llm_system/
│
├── data/
│   ├── raw_tickets/
│   ├── processed_tickets/
│   ├── historical_tickets.jsonl
│   ├── duplicate_ground_truth.jsonl
│   ├── priority_dataset.jsonl
│   ├── assignee_dataset.jsonl
│   └── evaluation_results/
│
├── models/
│   ├── sbert_duplicate/
│   ├── priority_gray/
│   ├── triage_llm/
│   └── cache/
│
├── src/
│   ├── main.py
│   ├── config.py
│   ├── pipeline/
│   │   └── orchestrator.py
│   ├── modules/
│   │   ├── ticket_extractor.py
│   │   ├── duplicate_detector.py
│   │   ├── priority_classifier.py
│   │   ├── assignee_triager.py
│   │   ├── bug_localizer.py
│   │   ├── patch_generator.py
│   │   ├── test_generator.py
│   │   ├── regression_tester.py
│   │   └── commit_message_generator.py
│   ├── utils/
│   │   ├── json_schema.py
│   │   ├── embedding_utils.py
│   │   ├── git_utils.py
│   │   ├── code_retriever.py
│   │   ├── llm_client.py
│   │   ├── logger.py
│   │   └── evaluation.py
│   └── prompts/
│       ├── extract_ticket_prompt.txt
│       ├── triage_prompt.txt
│       ├── localization_prompt.txt
│       ├── patch_prompt.txt
│       ├── test_generation_prompt.txt
│       └── commit_message_prompt.txt
│
├── experiments/
│   ├── evaluate_json_extraction.py
│   ├── evaluate_duplicate_detection.py
│   ├── evaluate_priority_classification.py
│   ├── evaluate_assignee_triage.py
│   ├── evaluate_bug_localization.py
│   ├── evaluate_patch_success.py
│   ├── evaluate_test_generation.py
│   ├── evaluate_regression_testing.py
│   └── evaluate_commit_message.py
│
├── tests/
│   ├── test_ticket_extractor.py
│   ├── test_duplicate_detector.py
│   ├── test_priority_classifier.py
│   ├── test_assignee_triager.py
│   ├── test_pipeline.py
│   └── fixtures/
│
├── requirements.txt
├── pyproject.toml
└── README.md
```

資料夾與檔案用途：

| 路徑 | 用途 |
|---|---|
| `data/raw_tickets/` | 存放使用者原始 bug report |
| `data/processed_tickets/` | 存放結構化後的 Ticket JSON |
| `data/historical_tickets.jsonl` | 歷史 Ticket 資料庫 |
| `data/duplicate_ground_truth.jsonl` | Duplicate Detection 評估標註資料 |
| `data/priority_dataset.jsonl` | 優先級分類訓練與測試資料 |
| `data/assignee_dataset.jsonl` | 負責人分派訓練資料 |
| `models/sbert_duplicate/` | Duplicate Detection 的 SBERT 模型 |
| `models/priority_gray/` | 優先級分類模型與 factor 權重 |
| `models/triage_llm/` | 負責人分派 LLM 或 LoRA adapter |
| `src/main.py` | CLI 或主程式入口 |
| `src/config.py` | 儲存模型路徑、threshold、資料路徑等設定 |
| `src/pipeline/orchestrator.py` | 串接完整流程 |
| `src/modules/` | 各功能模組主程式 |
| `src/utils/` | 共用工具，例如 JSON schema、embedding、git、retrieval、evaluation |
| `src/prompts/` | LLM prompt 模板 |
| `experiments/` | 各模組實驗與評估程式 |
| `tests/` | 單元測試與整合測試 |

## 四、完整 Pipeline 流程

以下是一個較完整的 Python pseudocode，包含錯誤處理、log 紀錄、fallback 與每個模組的串接方式。

```python
import logging
from typing import Any, Dict


class PipelineOrchestrator:
    """
    負責串接 Ticket 正規化、Duplicate 偵測、優先級分類、
    負責人分派、錯誤定位、補丁生成、測試生成、回歸測試與 Commit Message 生成。
    """

    def __init__(
        self,
        ticket_extractor,
        duplicate_detector,
        priority_classifier,
        assignee_triager,
        bug_localizer,
        patch_generator,
        test_generator,
        regression_tester,
        commit_message_generator,
        logger: logging.Logger,
    ):
        self.ticket_extractor = ticket_extractor
        self.duplicate_detector = duplicate_detector
        self.priority_classifier = priority_classifier
        self.assignee_triager = assignee_triager
        self.bug_localizer = bug_localizer
        self.patch_generator = patch_generator
        self.test_generator = test_generator
        self.regression_tester = regression_tester
        self.commit_message_generator = commit_message_generator
        self.logger = logger

    def run_pipeline(self, raw_ticket: Dict[str, Any], repository_path: str) -> Dict[str, Any]:
        result = {
            "status": "running",
            "errors": [],
            "warnings": []
        }

        try:
            self.logger.info("Step 1: Extract ticket JSON")
            structured_ticket = self.ticket_extractor.extract(raw_ticket)
            result["structured_ticket"] = structured_ticket
        except Exception as exc:
            result["status"] = "failed"
            result["failed_step"] = "ticket_extraction"
            result["errors"].append(str(exc))
            return result

        try:
            self.logger.info("Step 2: Detect duplicate")
            duplicate_result = self.duplicate_detector.detect(structured_ticket)
            result["duplicate"] = duplicate_result

            if duplicate_result.get("is_duplicate"):
                result["status"] = "duplicate"
                result["duplicate_of"] = duplicate_result.get("duplicate_of")
                return result
        except Exception as exc:
            result["warnings"].append(f"Duplicate detection failed: {exc}")
            duplicate_result = {
                "is_duplicate": False,
                "top_k_candidates": [],
                "fallback_used": True
            }
            result["duplicate"] = duplicate_result

        try:
            self.logger.info("Step 3: Predict priority")
            priority_result = self.priority_classifier.predict(
                structured_ticket,
                duplicate_result.get("top_k_candidates", [])
            )
            result["priority"] = priority_result
        except Exception as exc:
            result["warnings"].append(f"Priority prediction failed: {exc}")
            priority_result = {
                "predicted_priority": "P3",
                "confidence": 0.0,
                "fallback_used": True
            }
            result["priority"] = priority_result

        try:
            self.logger.info("Step 4: Assign developer")
            assignee_result = self.assignee_triager.assign(
                structured_ticket,
                priority_result
            )
            result["assignee"] = assignee_result
        except Exception as exc:
            result["warnings"].append(f"Assignee triage failed: {exc}")
            assignee_result = {
                "assignee": "manual_triage",
                "reason": "模型分派失敗，需人工指派。",
                "fallback_used": True
            }
            result["assignee"] = assignee_result

        try:
            self.logger.info("Step 5: Localize bug")
            bug_location = self.bug_localizer.localize(
                structured_ticket,
                repository_path
            )
            result["bug_location"] = bug_location
        except Exception as exc:
            result["status"] = "needs_manual_review"
            result["failed_step"] = "bug_localization"
            result["errors"].append(str(exc))
            return result

        try:
            self.logger.info("Step 6: Generate patch")
            patch = self.patch_generator.generate(
                structured_ticket,
                bug_location,
                repository_path
            )
            result["patch"] = patch
        except Exception as exc:
            result["status"] = "needs_manual_patch"
            result["failed_step"] = "patch_generation"
            result["errors"].append(str(exc))
            return result

        try:
            self.logger.info("Step 7: Generate reproduction tests")
            reproduction_tests = self.test_generator.generate_tests(
                structured_ticket,
                patch
            )
            result["tests"] = reproduction_tests
        except Exception as exc:
            result["warnings"].append(f"Test generation failed: {exc}")
            reproduction_tests = {
                "generated_tests": [],
                "fib_passed_tests": [],
                "test_result": "not_generated",
                "fallback_used": True
            }
            result["tests"] = reproduction_tests

        try:
            self.logger.info("Step 8: Run regression tests")
            regression_result = self.regression_tester.run(
                patch,
                repository_path
            )
            result["regression"] = regression_result
        except Exception as exc:
            result["status"] = "patch_unverified"
            result["failed_step"] = "regression_testing"
            result["errors"].append(str(exc))
            return result

        try:
            self.logger.info("Step 9: Generate commit message")
            commit_message = self.commit_message_generator.generate(
                structured_ticket,
                patch,
                {
                    "reproduction_tests": reproduction_tests,
                    "regression_result": regression_result
                }
            )
            result["commit_message"] = commit_message
        except Exception as exc:
            result["warnings"].append(f"Commit message generation failed: {exc}")
            result["commit_message"] = {
                "commit_message": "fix: resolve reported bug",
                "commit_body": "Generated fallback commit message.",
                "fallback_used": True
            }

        result["status"] = "completed"
        return result
```

### Pipeline 失敗處理策略

| 步驟 | 失敗情況 | 系統處理方式 |
|---|---|---|
| Ticket JSON 正規化 | LLM 輸出不是合法 JSON | JSON repair、重新 prompt、rule-based fallback |
| Duplicate Detection | embedding 或 index 失敗 | 視為非 duplicate，但加上 warning |
| Priority Classification | 模型無法預測 | 預設 P3，標記 fallback |
| Assignee Triage | 無法分派負責人 | 指派 `manual_triage` |
| Bug Localization | 找不到錯誤位置 | 結束自動修復，標記 `needs_manual_review` |
| Patch Generation | patch 無法生成 | 結束流程，標記 `needs_manual_patch` |
| Test Generation | 測試生成失敗 | 繼續回歸測試，但標記 warning |
| Regression Testing | 測試失敗或 patch 不安全 | 標記 `patch_unverified` |
| Commit Message | LLM 生成失敗 | 使用 template fallback |

## 五、各模組 Class Interface

### TicketExtractor

```python
class TicketExtractor:
    """
    負責將原始 bug report、issue 或 ticket 轉換成標準 structured ticket JSON。

    可能用到的套件：
    - jsonschema
    - pydantic
    - transformers
    - ollama 或 openai-compatible client

    後續實作建議：
    1. 先建立 ticket JSON schema。
    2. 使用 prompt 要求 LLM 只輸出 JSON。
    3. 驗證 schema。
    4. 若格式錯誤，執行 JSON repair 或重新 prompt。
    """

    def extract(self, raw_ticket: dict) -> dict:
        """
        Args:
            raw_ticket:
                原始 Ticket，例如 title、description、logs、screenshots_text。

        Returns:
            structured_ticket:
                符合 JSON schema 的標準化 Ticket。
        """
        raise NotImplementedError
```

### DuplicateDetector

```python
class DuplicateDetector:
    """
    負責將 Ticket 轉為 embedding，並與歷史 Ticket 進行相似度比對。

    可能用到的套件：
    - sentence-transformers
    - faiss-cpu
    - numpy
    - scikit-learn

    後續實作建議：
    1. 先使用 pretrained SBERT 建立 baseline。
    2. 將歷史 Ticket embedding cache 起來。
    3. 使用 cosine similarity 找 top-k。
    4. 後續再加入 Triplet Loss 微調。
    """

    def detect(self, ticket_json: dict, historical_db: list | None = None) -> dict:
        """
        Args:
            ticket_json:
                結構化 Ticket。
            historical_db:
                歷史 Ticket 清單；若為 None，從設定檔載入。

        Returns:
            {
                "is_duplicate": bool,
                "duplicate_of": str | None,
                "similarity_score": float,
                "top_k_candidates": list
            }
        """
        raise NotImplementedError
```

### PriorityClassifier

```python
class PriorityClassifier:
    """
    使用 DRONE / GRAY framework 的 multi-factor 分析預測 Ticket 優先級。

    可能用到的套件：
    - scikit-learn
    - pandas
    - numpy

    後續實作建議：
    1. 將每個 factor 實作成獨立計算函式。
    2. 使用 regression 或 classification model 預測 priority score。
    3. 透過 threshold 將分數轉為 P1-P5。
    """

    def predict(self, ticket_json: dict, duplicate_candidates: list) -> dict:
        """
        Args:
            ticket_json:
                結構化 Ticket。
            duplicate_candidates:
                Duplicate detector 找到的相似歷史 Ticket。

        Returns:
            {
                "predicted_priority": "P1/P2/P3/P4/P5",
                "confidence": float,
                "factor_scores": dict
            }
        """
        raise NotImplementedError
```

### AssigneeTriager

```python
class AssigneeTriager:
    """
    負責根據 Ticket component、描述、歷史修復紀錄與優先級分派負責人。

    可能用到的套件：
    - transformers
    - peft
    - datasets
    - pandas

    後續實作建議：
    1. 初期使用 component-owner mapping。
    2. 中期加入歷史 assignee 統計。
    3. 後期使用 LoRA fine-tuned LLM。
    """

    def assign(self, ticket_json: dict, priority_result: dict) -> dict:
        """
        Args:
            ticket_json:
                結構化 Ticket。
            priority_result:
                優先級預測結果。

        Returns:
            {
                "assignee": str,
                "reason": str,
                "confidence": float
            }
        """
        raise NotImplementedError
```

### BugLocalizer

```python
class BugLocalizer:
    """
    負責根據 Ticket 內容與 repository 程式碼定位可能錯誤位置。

    可能用到的工具或套件：
    - ripgrep
    - tree-sitter
    - sentence-transformers
    - LLM client

    後續實作建議：
    1. 先用 stack trace 與 keyword search 找候選檔案。
    2. 再用 embedding retrieval 擴充候選。
    3. 最後交給 LLM 排序並輸出 file/function/line range。
    """

    def localize(self, ticket_json: dict, repo_path: str) -> dict:
        """
        Args:
            ticket_json:
                結構化 Ticket。
            repo_path:
                專案程式碼路徑。

        Returns:
            {
                "bug_location": {
                    "file": str,
                    "function": str,
                    "line_start": int,
                    "line_end": int,
                    "reason": str
                },
                "candidates": list
            }
        """
        raise NotImplementedError
```

### PatchGenerator

```python
class PatchGenerator:
    """
    負責根據錯誤位置與上下文程式碼生成 unified diff patch。

    可能用到的套件：
    - transformers
    - unidiff
    - subprocess
    - gitpython

    後續實作建議：
    1. 擷取 bug location 前後程式碼。
    2. 使用 FIM / PSM prompt 生成修補內容。
    3. 要求模型輸出 unified diff。
    4. 嘗試 apply patch 並檢查語法。
    """

    def generate(self, ticket_json: dict, bug_location: dict, repo_path: str) -> dict:
        """
        Args:
            ticket_json:
                結構化 Ticket。
            bug_location:
                錯誤定位結果。
            repo_path:
                專案程式碼路徑。

        Returns:
            {
                "patch": str,
                "modified_files": list[str],
                "explanation": str
            }
        """
        raise NotImplementedError
```

### TestGenerator

```python
class TestGenerator:
    """
    負責生成 bug reproduction tests，並篩選符合 FIB 的測試案例。

    可能用到的套件：
    - pytest
    - unittest
    - subprocess
    - LLM client

    後續實作建議：
    1. 從 bug report 的 steps_to_reproduce 產生測試。
    2. 在 buggy version 執行測試，確認失敗。
    3. 在 fixed version 執行測試，確認通過。
    4. 只保留符合 FIB 的測試。
    """

    def generate_tests(self, ticket_json: dict, patch: dict) -> dict:
        """
        Args:
            ticket_json:
                結構化 Ticket。
            patch:
                補丁結果。

        Returns:
            {
                "generated_tests": list,
                "fib_passed_tests": list,
                "test_result": "passed/failed/not_generated"
            }
        """
        raise NotImplementedError
```

### RegressionTester

```python
class RegressionTester:
    """
    負責套用 patch 後執行既有測試與新增測試，並進行 RIPR 分析。

    可能用到的套件：
    - pytest
    - coverage.py
    - subprocess
    - gitpython

    後續實作建議：
    1. 建立臨時 worktree 或 sandbox。
    2. 套用 patch。
    3. 執行既有測試與 generated tests。
    4. 收集 coverage 與 changed-code coverage。
    5. 輸出 RIPR 分析。
    """

    def run(self, patch: dict, repo_path: str) -> dict:
        """
        Args:
            patch:
                補丁結果。
            repo_path:
                專案程式碼路徑。

        Returns:
            {
                "regression_tests": list,
                "ripr_analysis": dict,
                "regression_result": "passed/failed"
            }
        """
        raise NotImplementedError
```

### CommitMessageGenerator

```python
class CommitMessageGenerator:
    """
    負責根據 Ticket、patch diff 與測試結果產生 commit message。

    可能用到的套件：
    - transformers
    - LLM client

    後續實作建議：
    1. 使用 Conventional Commit 格式。
    2. commit title 控制在一行。
    3. commit body 說明修復原因、修改內容與測試結果。
    4. 若 LLM 失敗，使用 template fallback。
    """

    def generate(self, ticket_json: dict, patch: dict, test_result: dict) -> dict:
        """
        Args:
            ticket_json:
                結構化 Ticket。
            patch:
                補丁結果。
            test_result:
                reproduction 與 regression 測試結果。

        Returns:
            {
                "commit_message": str,
                "commit_body": str
            }
        """
        raise NotImplementedError
```

## 六、資料格式設計

### 1. `raw_ticket.json`

```json
{
  "ticket_id": "RAW-001",
  "title": "Login page crashes",
  "description": "When I try to login without token, the app crashes.",
  "logs": "TypeError: token is None at auth/validator.py:42",
  "environment": {
    "os": "macOS",
    "version": "1.2.0"
  },
  "screenshots_text": "Blank error dialog"
}
```

### 2. `structured_ticket.json`

```json
{
  "ticket_id": "BUG-001",
  "title": "Login page crashes when token is missing",
  "description": "User cannot login when authentication token is missing.",
  "bug_type": "runtime_error",
  "component": "authentication",
  "os": "macOS",
  "version": "1.2.0",
  "priority": null,
  "error_message": "TypeError: token is None",
  "steps_to_reproduce": [
    "Open login page",
    "Submit login without token"
  ],
  "expected_behavior": "Show validation error.",
  "actual_behavior": "Application crashes.",
  "logs": "TypeError: token is None at auth/validator.py:42",
  "screenshots_text": "Blank error dialog"
}
```

### 3. `historical_ticket.jsonl`

```json
{"ticket_id":"BUG-038","title":"Login crash when token missing","description":"Missing token causes TypeError","component":"authentication","priority":"P2","assignee":"alice@example.com"}
{"ticket_id":"BUG-039","title":"Password reset email not sent","description":"SMTP timeout","component":"email","priority":"P3","assignee":"bob@example.com"}
```

### 4. `duplicate_detection_result.json`

```json
{
  "is_duplicate": true,
  "duplicate_of": "BUG-038",
  "similarity_score": 0.87,
  "threshold": 0.82,
  "top_k_candidates": [
    {
      "ticket_id": "BUG-038",
      "similarity": 0.87,
      "title": "Login crash when token missing"
    }
  ]
}
```

### 5. `priority_prediction_result.json`

```json
{
  "predicted_priority": "P2",
  "confidence": 0.78,
  "factor_scores": {
    "textual_factor": 0.75,
    "temporal_factor": 0.6,
    "author_factor": 0.4,
    "related_report_factor": 0.82,
    "severity_factor": 0.9,
    "component_factor": 0.7
  }
}
```

### 6. `assignee_prediction_result.json`

```json
{
  "assignee": "alice@example.com",
  "confidence": 0.81,
  "reason": "此 Ticket 屬於 authentication component，且 alice@example.com 過去多次修復 token validation 相關問題。"
}
```

### 7. `bug_location_result.json`

```json
{
  "bug_location": {
    "file": "src/auth/validator.py",
    "function": "validate_token",
    "line_start": 38,
    "line_end": 52,
    "reason": "stack trace 與錯誤訊息皆指向 token validation。"
  },
  "candidates": [
    {
      "file": "src/auth/validator.py",
      "score": 0.91
    }
  ]
}
```

### 8. `patch_result.json`

```json
{
  "patch": "diff --git a/src/auth/validator.py b/src/auth/validator.py\n...",
  "modified_files": [
    "src/auth/validator.py"
  ],
  "explanation": "新增 token 為 None 時的防呆判斷。"
}
```

### 9. `generated_tests_result.json`

```json
{
  "generated_tests": [
    {
      "file": "tests/test_auth_validator.py",
      "test_name": "test_validate_token_none",
      "content": "def test_validate_token_none(): ..."
    }
  ],
  "fib_passed_tests": [
    "tests/test_auth_validator.py::test_validate_token_none"
  ],
  "test_result": "passed"
}
```

### 10. `regression_test_result.json`

```json
{
  "regression_tests": [
    "tests/test_auth_validator.py",
    "tests/test_login_flow.py"
  ],
  "ripr_analysis": {
    "reaching": true,
    "infecting": true,
    "propagating": true,
    "revealing": true
  },
  "regression_result": "passed",
  "coverage": {
    "changed_code_coverage": 0.86
  }
}
```

### 11. `commit_message_result.json`

```json
{
  "commit_message": "fix(auth): handle missing token during login validation",
  "commit_body": "This patch adds validation for missing authentication tokens before decoding. It prevents a TypeError and adds a reproduction test for the missing-token login case."
}
```

### 12. `final_pipeline_result.json`

```json
{
  "status": "completed",
  "structured_ticket": {
    "ticket_id": "BUG-001"
  },
  "duplicate": {
    "is_duplicate": false
  },
  "priority": {
    "predicted_priority": "P2"
  },
  "assignee": {
    "assignee": "alice@example.com"
  },
  "bug_location": {
    "bug_location": {
      "file": "src/auth/validator.py"
    }
  },
  "patch": {
    "modified_files": [
      "src/auth/validator.py"
    ]
  },
  "tests": {
    "test_result": "passed"
  },
  "regression": {
    "regression_result": "passed"
  },
  "commit_message": {
    "commit_message": "fix(auth): handle missing token during login validation"
  },
  "warnings": [],
  "errors": []
}
```

## 七、評估指標

### 1. JSON 結構化

- exact match：完整 JSON 是否與標準答案一致。
- field-level F1：逐欄位計算 precision、recall、F1。
- schema valid rate：輸出 JSON 通過 schema validation 的比例。

### 2. Duplicate Detection

- accuracy：Duplicate / Non-Duplicate 判斷正確率。
- precision：被判斷為 Duplicate 的 Ticket 中真正 Duplicate 的比例。
- recall：真正 Duplicate 的 Ticket 中被成功找出的比例。
- F1：precision 與 recall 的綜合指標。
- top-k hit rate：正確 duplicate ticket 是否出現在 top-k candidates。
- duplicate_of accuracy：`duplicate_of` 指到正確 Ticket 的比例。

### 3. Priority Classification

- accuracy：優先級整體分類正確率。
- macro-F1：每個類別平均後的 F1。
- weighted-F1：依類別數量加權後的 F1。
- per-class recall：各優先級召回率，特別注意 P1 / P2。
- confusion matrix：觀察模型是否常將高優先級誤判為低優先級。

### 4. Assignee Triage

- top-1 accuracy：第一名建議是否正確。
- top-3 accuracy：前三名候選是否包含正確負責人。
- MRR：Mean Reciprocal Rank，評估正確負責人在排序中的位置。

### 5. Bug Localization

- file-level accuracy：是否找對檔案。
- function-level accuracy：是否找對函式。
- top-k localization accuracy：正確位置是否在 top-k 候選中。

### 6. Patch Generation

- compile success rate：patch 套用後是否能通過編譯或語法檢查。
- test pass rate：patch 套用後測試是否通過。
- patch correctness rate：patch 是否真正修復問題。
- human review acceptance rate：人工審查接受比例。

### 7. Test Generation

- FIB success rate：測試是否能在 buggy version 失敗、fixed version 通過。
- generated test compile rate：生成測試是否能執行。
- bug reproduction rate：測試是否能重現 bug。

### 8. Regression Testing

- changed-code coverage：測試是否覆蓋修改過的程式碼。
- RIPR coverage：是否滿足 Reaching、Infecting、Propagating、Revealing。
- regression failure detection rate：是否能偵測補丁造成的新錯誤。

### 9. Commit Message

- BLEU / ROUGE / BERTScore：與人工標準 commit message 比對。
- human readability score：人工評估可讀性。
- completeness score：是否包含修復原因、修改內容、測試結果。

## 八、開發階段 Roadmap

### Phase 1：資料前處理與 JSON 正規化

開發目標：

完成原始 Ticket 到 structured JSON 的轉換。

要完成的程式檔案：

- `src/modules/ticket_extractor.py`
- `src/utils/json_schema.py`
- `src/prompts/extract_ticket_prompt.txt`
- `experiments/evaluate_json_extraction.py`

輸入資料：

- raw ticket
- log
- screenshots OCR text

輸出成果：

- `structured_ticket.json`

測試方式：

- schema validation
- field-level F1
- 人工抽樣檢查

可能風險：

- LLM 輸出格式錯誤。
- 欄位缺漏。
- 自然語言描述過於模糊。

fallback 方法：

- JSON repair
- retry prompt
- rule-based extraction

### Phase 2：Duplicate Detection

開發目標：

完成 Ticket embedding 與歷史 Ticket 相似度比對。

要完成的程式檔案：

- `src/modules/duplicate_detector.py`
- `src/utils/embedding_utils.py`
- `experiments/evaluate_duplicate_detection.py`

輸入資料：

- structured ticket
- historical tickets
- duplicate ground truth

輸出成果：

- `duplicate_detection_result.json`

測試方式：

- accuracy
- precision
- recall
- F1
- top-k hit rate

可能風險：

- 相似描述不一定代表同一 bug。
- threshold 不容易設定。

fallback 方法：

- 回傳 top-k candidates 給人工確認。
- 對 hard negative 進行 Triplet Loss 微調。

### Phase 3：Priority Classification

開發目標：

完成 P1-P5 優先級預測。

要完成的程式檔案：

- `src/modules/priority_classifier.py`
- `experiments/evaluate_priority_classification.py`

輸入資料：

- structured ticket
- related historical tickets
- priority dataset

輸出成果：

- `priority_prediction_result.json`

測試方式：

- macro-F1
- weighted-F1
- P1 / P2 recall
- confusion matrix

可能風險：

- P1 / P2 資料量較少。
- 優先級可能受團隊政策影響。

fallback 方法：

- class weight
- rule-based severity mapping
- 預設 P3 並標記低信心

### Phase 4：Assignee Triage

開發目標：

完成自動分派負責人。

要完成的程式檔案：

- `src/modules/assignee_triager.py`
- `src/prompts/triage_prompt.txt`
- `experiments/evaluate_assignee_triage.py`

輸入資料：

- structured ticket
- priority result
- historical assignee dataset

輸出成果：

- `assignee_prediction_result.json`

測試方式：

- top-1 accuracy
- top-3 accuracy
- MRR

可能風險：

- 歷史 assignee 資料不足。
- 開發者職責會變動。

fallback 方法：

- component-owner mapping
- 最近 commit 貢獻者統計
- manual triage

### Phase 5：Bug Localization

開發目標：

找出可能錯誤檔案、函式與行號。

要完成的程式檔案：

- `src/modules/bug_localizer.py`
- `src/utils/code_retriever.py`
- `src/prompts/localization_prompt.txt`
- `experiments/evaluate_bug_localization.py`

輸入資料：

- structured ticket
- repository path

輸出成果：

- `bug_location_result.json`

測試方式：

- file-level accuracy
- function-level accuracy
- top-k localization accuracy

可能風險：

- repository 太大，LLM context 不足。
- stack trace 不完整。

fallback 方法：

- keyword search
- stack trace search
- embedding retrieval top-k
- 人工 review candidates

### Phase 6：Patch Generation

開發目標：

根據錯誤位置生成可套用的 patch。

要完成的程式檔案：

- `src/modules/patch_generator.py`
- `src/prompts/patch_prompt.txt`
- `experiments/evaluate_patch_success.py`

輸入資料：

- structured ticket
- bug location
- repository path

輸出成果：

- `patch_result.json`

測試方式：

- patch apply
- compile / syntax check
- existing unit tests

可能風險：

- patch 格式錯誤。
- patch 修錯方向錯誤。
- 只修表面症狀。

fallback 方法：

- 要求 LLM 重新輸出 unified diff。
- 將編譯錯誤回饋給 LLM。
- 標記 `needs_manual_patch`。

### Phase 7：Test Generation

開發目標：

產生可重現 bug 的測試案例，並檢查 FIB。

要完成的程式檔案：

- `src/modules/test_generator.py`
- `src/prompts/test_generation_prompt.txt`
- `experiments/evaluate_test_generation.py`

輸入資料：

- structured ticket
- patch
- repository path

輸出成果：

- `generated_tests_result.json`

測試方式：

- 在 buggy version 測試應失敗。
- 在 fixed version 測試應通過。
- 計算 FIB success rate。

可能風險：

- 測試無法重現 bug。
- 測試與專案測試框架不相容。

fallback 方法：

- few-shot prompt
- 提供現有測試範例給 LLM
- 將執行錯誤回饋給 LLM 重新生成

### Phase 8：Regression Testing

開發目標：

確認 patch 不破壞既有功能。

要完成的程式檔案：

- `src/modules/regression_tester.py`
- `src/utils/git_utils.py`
- `experiments/evaluate_regression_testing.py`

輸入資料：

- patch
- generated tests
- repository path

輸出成果：

- `regression_test_result.json`

測試方式：

- 執行完整測試或相關測試。
- 收集 coverage。
- 分析 RIPR。

可能風險：

- 測試時間太長。
- 測試環境不穩定。

fallback 方法：

- 只執行受影響模組測試。
- 先跑 smoke tests。
- 標記 `patch_unverified`。

### Phase 9：Commit Message Generation

開發目標：

產生清楚、可讀、符合格式的 commit message。

要完成的程式檔案：

- `src/modules/commit_message_generator.py`
- `src/prompts/commit_message_prompt.txt`
- `experiments/evaluate_commit_message.py`

輸入資料：

- structured ticket
- bug location
- patch
- test result

輸出成果：

- `commit_message_result.json`

測試方式：

- 人工可讀性評分。
- completeness score。
- 與歷史 commit message 比對。

可能風險：

- message 太籠統。
- 未說明測試結果。

fallback 方法：

- 使用 Conventional Commit template。

### Phase 10：Full Pipeline Integration

開發目標：

完成從 raw ticket 到 final result 的完整自動化流程。

要完成的程式檔案：

- `src/pipeline/orchestrator.py`
- `src/main.py`
- `src/config.py`
- `tests/test_pipeline.py`

輸入資料：

- raw ticket
- repository path

輸出成果：

- `final_pipeline_result.json`

測試方式：

- end-to-end integration test
- mock LLM response test
- fault injection test

可能風險：

- 任一模組失敗會中斷流程。
- 中間資料格式不一致。

fallback 方法：

- step-level checkpoint
- 統一 JSON schema
- 每步驟獨立 try/except

### Phase 11：Experiment and Evaluation

開發目標：

完成各模組與整體系統評估。

要完成的程式檔案：

- `experiments/evaluate_json_extraction.py`
- `experiments/evaluate_duplicate_detection.py`
- `experiments/evaluate_priority_classification.py`
- `experiments/evaluate_assignee_triage.py`
- `experiments/evaluate_bug_localization.py`
- `experiments/evaluate_patch_success.py`
- `experiments/evaluate_test_generation.py`
- `experiments/evaluate_regression_testing.py`
- `experiments/evaluate_commit_message.py`

輸入資料：

- 各模組 dataset
- pipeline outputs

輸出成果：

- metrics report
- error analysis report
- demo results

測試方式：

- 自動計算評估指標。
- 人工審查部分案例。

可能風險：

- 標註資料不足。
- 評估資料與真實情境有落差。

fallback 方法：

- 小規模人工標註。
- 使用公開資料集補充。
- 先以 case study 方式展示。

## 九、模型結果錯誤時的處理策略

| 錯誤情境 | 處理方式 |
|---|---|
| LLM 輸出不是合法 JSON | JSON repair、重新 prompt、schema fallback |
| Duplicate threshold 判斷不確定 | 回傳 top-k candidates，標記 `needs_review` |
| Priority 信心過低 | 使用預設 P3，並保留 factor scores |
| Assignee 無法判斷 | 使用 component-owner mapping 或 manual triage |
| Bug localization 找不到檔案 | 使用 error message、stack trace、component keyword 重新檢索 |
| Patch 無法 apply | 要求 LLM 重新輸出 unified diff |
| Patch 編譯失敗 | 將 compiler error 回饋給 LLM 重新修正 |
| 測試案例無法重現 bug | 重新生成測試，加入更明確的 reproduction steps |
| Regression failed | 標記 patch unsafe，不產生正式 commit |
| Commit message 品質不佳 | 使用 template 產生 fallback message |

## 十、專題成果建議

本專題最後建議交付以下成果：

1. 一套可執行的 Python pipeline。
2. 每個模組的 class interface 與可測試實作。
3. 每個階段的 JSON 輸入與輸出格式。
4. Duplicate、Priority、Assignee、Localization、Patch、Testing 等模組的評估腳本。
5. 一份完整 demo case，展示從 raw ticket 到 patch、test、commit message 的完整流程。
6. 一份實驗報告，說明各模組使用模型、資料集、指標與錯誤分析。

## 十一、總結

本系統將 LLM、NLP、ML 與軟體工程自動化流程整合成完整的 bug tracking and repair pipeline。系統並非只做單一分類任務，而是從使用者提交 Ticket 開始，依序完成資料結構化、Duplicate 偵測、優先級預測、負責人分派、錯誤定位、補丁生成、測試生成、回歸測試與 Commit Message 產生。

程式設計上，建議採用模組化架構，讓每個功能都有獨立 class、輸入輸出 JSON、測試檔案與評估程式。這樣可以讓專題團隊分工開發，也方便逐步實驗不同模型，例如 SBERT、Code Llama、DeepSeek-R1-Distill-Llama-8B、LoRA、DRONE / GRAY framework、LIBRO 與 CLEVEREST。

透過此計畫書，專題團隊可以依照 Phase 逐步完成系統，先建立資料處理與分類 baseline，再逐步加入 repository-level reasoning、patch generation 與 regression testing。最終成果可作為展示 LLM 在軟體維護、自動化 triage 與程式修復領域應用的完整專題系統。
