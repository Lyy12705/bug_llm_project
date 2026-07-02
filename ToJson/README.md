# 第一階段：資料前處理與標準化架構建置

本專案目前聚焦第一階段工作：將非結構化的軟體錯誤 Ticket 轉換為可分析、可評估的 JSON 結構化資料。

## 目標

解決非結構化 Ticket 難以分析的問題，建立資料蒐集、清理、欄位抽取與評估流程。

### 步驟 1.1：資料集建置

資料來源以可統一成 Ticket/bug report 格式為原則，包含：

- 開源專案中的錯誤報告與相關提交紀錄。
- 提供真實軟體錯誤案例的資料集。
- 技術論壇資料，例如 Stack Overflow。

目前程式已實作 GitHub Issues 與 Jira Search API 的資料擷取骨架，並輸出成統一 raw schema。其他來源可依相同格式匯入。

### 步驟 1.2：非結構化資訊擷取與資料正規化

此步驟使用 Code Llama-Instruct 讀取使用者回報的自然語言描述、錯誤訊息與環境資訊，透過提示工程輸出固定 JSON schema。目標欄位包含：

- `ticket_id`：Ticket / issue 編號。
- `title`：問題標題。
- `description`：主要問題描述。
- `product`：產品、專案或 issue tracker product。
- `severity`：嚴重程度。
- `bug_type`：問題類型。
- `component`：受影響模組或元件。
- `os`：作業系統或平台環境。
- `version`：版本號。
- `priority`：優先級。
- `error_message`：錯誤訊息或 log 摘要。
- `steps_to_reproduce`：重現步驟。
- `expected_behavior`：預期行為。
- `actual_behavior`：實際行為。
- `logs`：較完整的 log 或 stack trace 摘要。
- `screenshots_text`：截圖或畫面文字描述。

其中 `product`、`component`、`title`、`description`、`severity`、`priority`、`bug_type` 會供後續 duplicate detection、priority classification 與 assignee triage 共用。

## 文獻依據與實作對應

本專案依據 Roziere et al. (2024)《Code Llama: Open Foundation Models for Code》的方法設計第一階段資訊抽取流程。論文指出 Code Llama-Instruct 具備 instruction-following 能力，可在程式相關自然語言任務中進行 zero-shot 指令遵循；同時 Code Llama 支援較長上下文，適合處理包含錯誤描述、log 與環境資訊的 bug report。

Code Llama 論文的主要測試資料集包含：

- `HumanEval`：Python description-to-code generation benchmark。
- `MBPP`：Mostly Basic Python Problems，用於 Python 程式生成評估。
- `APPS`：程式競賽與面試風格題目。
- `MultiPL-E`：HumanEval 的多語言版本，包含 C++、Java、PHP、C#、TypeScript、Bash 等。
- `GSM8K`：數學推理能力評估。
- `Long Code Completion (LCC)` 與長上下文檢索任務：用於評估長上下文能力。

本專案目前採用其中最小、公開、容易重現且與 Code Llama 核心結果直接相關的 `HumanEval` 與 `MBPP`，建立文獻 benchmark 標準化資料集。

注意：文獻 benchmark 模式不是在處理使用者 bug 回報訊息。它是讓 Code Llama 讀取程式題目並生成 Python 程式，再用官方 tests 判斷程式是否正確。使用者回報訊息轉 JSON 的任務仍在 `scripts/To_Json/`。

| 文獻方法 | 本專案實作 |
| --- | --- |
| 使用 Code Llama-Instruct 處理程式相關自然語言任務 | `extract_to_json.py` 預設使用 `codellama:7b-instruct` |
| 以 instruction prompt 驅動模型遵循任務 | 使用 `[INST] ... [/INST]` 格式建立欄位抽取提示 |
| zero-shot instruction following | 預設 `PROMPT_MODE=zero_shot`，直接要求模型依 schema 輸出 |
| 支援長上下文輸入 | 預設保留最多 `6000` 字元 bug report，並設定 `CODE_LLAMA_NUM_CTX=8192` |
| pass@1 使用 greedy decoding | 資訊抽取採 `temperature=0.0`、`top_p=1.0`，降低隨機性並提高可重現性 |
| 以程式/錯誤領域資料提升模型適配性 | 資料來源聚焦 issue、bug report、log、commit 與技術論壇內容 |

若需要與提示範例比較，可將 `PROMPT_MODE` 設為 `few_shot`，程式會加入少量欄位抽取範例；正式方法預設仍採文獻中的 zero-shot instruction setting。

## Ticket JSON 評估資料集

Code Llama 論文沒有提供「使用者 bug 回報訊息轉 JSON 欄位」的公開測試資料集。論文中的 `HumanEval`、`MBPP`、`APPS`、`MultiPL-E` 等資料集主要用來評估程式生成、程式補全或數學/長上下文能力，因此不能直接拿來計算本專案 Ticket JSON 正規化任務的準確率。

本專案的 Ticket JSON 準確率必須使用 `json_ground_truth` 人工標註資料計算。目前 `dataset/labeled/test.jsonl` 已完成 56 筆人工標註，既有正式評估欄位包含 `bug_type`、`component`、`os`、`version`、`priority`、`error_message`。專案內另保留 `dataset/labeled/test_backup.jsonl` 作為同一批標註資料的備份；`evaluate_results.py` 也會在主標註檔沒有有效標註時，自動改用這份備份標註檔，避免把空白欄位誤算成準確率。

為了和後續流程整合，目前 Code Llama 輸出 schema 已擴充為 `dataset/schema.json` 中的完整欄位。舊六欄評估結果仍可作為第一階段欄位抽取 baseline；若要評估新欄位，需要重新補人工標註或使用含官方欄位的公開資料集。

人工標註規則整理於 `docs/manual_annotation.md`。標註時以 `bug_report` 的標題與描述為主要依據，將問題類型、受影響模組、作業系統、版本、優先級與錯誤訊息等欄位填入 `json_ground_truth`。若分類欄位無法判斷，使用 `unknown`；若 `version`、`error_message` 或其他文字欄位沒有明確出現在原文或來源欄位，則保留空字串。

若要使用公開資料集測試第一階段，可採用 issue tracker 原本的結構化欄位作為 ground truth。最適合本專案的是 [Eclipse issue report dataset](https://zenodo.org/records/15348468)：它包含 `Summary`、`Description` 以及 `Component`、`Version`、`Op sys`、`Priority` 等官方 Bugzilla 欄位。這不是已經整理好的「JSON 抽取資料集」，但可以轉換成本專案的 `bug_report -> json_ground_truth` 格式。

其他可參考資料：

- [GitBugs](https://github.com/av9ash/gitbugs/)：整合 GitHub、Bugzilla、Jira 的 bug reports，部分專案含 `Summary`、`Description`、`Priority`、`Affects Version/s` 等欄位，但不一定都有 `Component` / `OS`。
- [Mozilla bmobugs](https://mozilla.github.io/data-docs/datasets/other/bmobugs/reference.html)：Mozilla Bugzilla mirror，包含 `component`、`priority`、`version` 等欄位，但主要透過 BigQuery/STMO 存取。
- [Mozilla and Eclipse Defect Tracking Dataset](https://github.com/ansymo/msr2013-bug_dataset)：較舊的 MSR 2013 XML 資料集，包含 bug report 快照與欄位更新歷史。

準備公開 Eclipse sample 評估集：

```bash
python3 scripts/To_Json/prepare_public_bug_report_dataset.py --limit 500
```

輸出：

```text
dataset/raw/eclipse_issue_sample.csv
dataset/processed/public_bug_reports.jsonl
dataset/labeled/public_test.jsonl
```

用公開資料集跑 Code Llama 抽取：

```bash
python3 scripts/To_Json/extract_to_json.py \
  --input dataset/processed/public_bug_reports.jsonl \
  --output dataset/predicted/public_predicted.jsonl
```

評估公開資料集結果：

```bash
python3 scripts/To_Json/evaluate_results.py \
  --gold dataset/labeled/public_test.jsonl \
  --pred dataset/predicted/public_predicted.jsonl \
  --fields product,severity,component,os,version,priority \
  --skip-missing-ground-truth \
  --no-source-consistency \
  --no-backup
```

說明：公開 Eclipse sample 有官方 ground truth 的欄位主要是 `product`、`severity`、`component`、`os`、`version`、`priority`。`bug_type` 與 `error_message` 不是該資料集的官方標籤，所以若重新產生公開資料集，可使用 `--fields product,severity,component,os,version,priority` 評估 tracker 官方欄位。若使用舊版已產生資料，`product` 與 `severity` 可能尚未寫入 `json_ground_truth`，可先只評估 `component,os,version,priority`。

公開資料集改善流程：

```bash
python3 scripts/To_Json/postprocess_predictions.py \
  --input-records dataset/processed/public_bug_reports.jsonl \
  --pred dataset/predicted/public_predicted.jsonl \
  --output dataset/predicted/public_predicted_postprocessed.jsonl \
  --mode public_eclipse

python3 scripts/To_Json/evaluate_results.py \
  --gold dataset/labeled/public_test.jsonl \
  --pred dataset/predicted/public_predicted_postprocessed.jsonl \
  --fields component,os,version,priority \
  --skip-missing-ground-truth \
  --no-source-consistency \
  --no-backup
```

改善後結果，以下為舊版四欄官方欄位評估：

```text
component : 79/100 = 0.7900
os        : 7/100  = 0.0700
version   : 1/100  = 0.0100
priority  : 55/100 = 0.5500

Overall field accuracy : 142/400 = 0.3550
Exact match accuracy   : 0/100 = 0.0000
```

改善主要來自 `component` 欄位：後處理會把 `managedBuilder`、`C/C++ Build`、`Scanner Discovery` 等模型抽出的自然語言元件名稱映射回 Eclipse 官方 component，例如 `cdt-build`。`os` 與 `version` 仍偏低，原因是 text-only 輸入中多數資料沒有明確寫出 OS 或版本。

若只想評估原文中真的看得到 ground truth 值的欄位，可使用：

```bash
python3 scripts/To_Json/evaluate_results.py \
  --gold dataset/labeled/public_test.jsonl \
  --pred dataset/predicted/public_predicted_postprocessed.jsonl \
  --fields component,os,version,priority \
  --skip-missing-ground-truth \
  --observable-only \
  --no-source-consistency \
  --no-backup
```

若要測 metadata normalization，可先產生含 issue tracker metadata 的輸入，再重新抽取：

```bash
python3 scripts/To_Json/prepare_public_bug_report_dataset.py \
  --limit 200 \
  --input-mode metadata \
  --processed-out dataset/processed/public_bug_reports_metadata.jsonl \
  --labeled-out dataset/labeled/public_test_metadata.jsonl

CODE_LLAMA_NUM_PREDICT=256 python3 scripts/To_Json/extract_to_json.py \
  --input dataset/processed/public_bug_reports_metadata.jsonl \
  --output dataset/predicted/public_predicted_metadata.jsonl \
  --postprocess public_eclipse
```

目前使用 `dataset/labeled/test.jsonl` 評估 `dataset/predicted/predicted.jsonl` 的結果：

```text
Overall field accuracy : 192/336 = 0.5714
Exact match accuracy   : 3/56 = 0.0536

bug_type      : 37/56 = 0.6607
component     : 11/56 = 0.1964
os            : 52/56 = 0.9286
version       : 30/56 = 0.5357
priority      : 33/56 = 0.5893
error_message : 29/56 = 0.5179
```

判讀：模型對 `os` 與 `bug_type` 較穩定，`component` 明顯偏低，代表目前提示詞或資料格式還不夠能讓模型精準抓出受影響模組。此處的 `exact match` 是針對既有六欄人工標註，嚴格要求六個欄位全部正確，因此只有 3/56。

## 文獻資料集模式

由於 Code Llama 論文的公開測試資料集是程式生成 benchmark，而不是 bug ticket 欄位抽取資料集，本專案將其整理為獨立的 literature dataset，以避免把程式題目硬轉成 bug 欄位。

準備文獻資料集：

```bash
python3 scripts/Literature_Datasets/prepare_code_llama_benchmarks.py
```

輸出位置：

```text
dataset/literature_schema.json
dataset/literature/raw/humaneval.jsonl
dataset/literature/raw/mbpp.jsonl
dataset/literature/processed/code_tasks.jsonl
dataset/literature/processed/summary.json
```

標準化後每筆資料包含：

- `benchmark`：HumanEval 或 MBPP。
- `task_type`：目前為 code_generation。
- `language`：目前為 python。
- `prompt`：自然語言/程式題目描述。
- `canonical_solution`：文獻資料集提供的標準解答，也就是 ground truth solution。
- `tests`：文獻資料集提供的官方測試案例，也就是評估 ground truth。
- `metadata`：文獻來源、原始資料集網址、split 與原始 ID。

### 文獻資料集準確度

Code generation benchmark 不使用欄位 accuracy 或字串完全比對，而是使用測試通過率。若模型只產生一個答案，指標為 `pass@1`：

```text
pass@1 = 通過官方 tests 的題目數 / 全部題目數
```

先確認文獻資料集的 ground truth 可用：

```bash
python3 scripts/Literature_Datasets/evaluate_code_generation.py --use-canonical
```

目前 canonical ground truth 檢查結果：

```text
HumanEval : 164/164 = 1.0000
MBPP      : 427/427 = 1.0000
Total     : 591/591 = 1.0000
```

產生 Code Llama 預測：

```bash
python3 scripts/Literature_Datasets/run_code_llama_result.py --generate --regenerate
```

如果只想先測一小部分：

```bash
python3 scripts/Literature_Datasets/run_code_llama_result.py \
  --benchmark HumanEval \
  --limit 10 \
  --generate \
  --regenerate
```

使用文獻 ground truth tests 評估模型輸出：

```bash
python3 scripts/Literature_Datasets/run_code_llama_result.py
```

評估報告會輸出到：

```text
dataset/literature/processed/pass_at_1_report.json
```

目前完整 591 題模型輸出結果：

```text
Total     : 58/591 = 0.0981
HumanEval : 44/164 = 0.2683
MBPP      : 14/427 = 0.0328
```

判讀：目前模型在 HumanEval 函式補全任務上通過率較高；MBPP 從自然語言題目生成完整 Python 程式的通過率較低，因此整體 `pass@1` 被拉低。這代表目前 prompt/model 設定對「完整題目轉程式」掌握不足，不代表資料集沒有 ground truth。

Code Llama 論文表 2 的參考結果如下：

```text
Code Llama-Instruct 7B  : HumanEval pass@1 34.8%, MBPP pass@1 44.4%
Code Llama-Instruct 13B : HumanEval pass@1 42.7%, MBPP pass@1 49.4%
Code Llama-Instruct 34B : HumanEval pass@1 41.5%, MBPP pass@1 57.0%
Code Llama-Instruct 70B : HumanEval pass@1 67.8%, MBPP pass@1 62.2%
```

本專案目前使用本機 `codellama:7b-instruct`，且可能是 Ollama 量化模型；文獻結果通常使用該模型的標準 evaluation harness。兩者不一定能完全等同。論文中 HumanEval 為 zero-shot，MBPP 為 3-shot；因此本專案的 `generate_code_llama_solutions.py` 預設使用 `paper_aligned` prompt style：HumanEval zero-shot，MBPP 3-shot。

改善分數可從以下方向調整：

- 使用 `paper_aligned` prompt，尤其讓 MBPP 使用 3-shot 題目範例。
- 改用較大的模型，例如 `codellama:13b-instruct`、`codellama:34b-instruct`，或 Python 專用模型。
- 將 HumanEval 和 MBPP 分開報告，不要只看總分。
- 增加 `num_predict` 或改善 prompt，避免模型輸出不完整程式。
- 若要對齊論文 pass@10/pass@100，需要改成多次 sampling，而不是目前的 greedy pass@1。

## 專案結構

```text
dataset/
  raw/issues.json              # 原始 issue / ticket 資料
  processed/unlabeled.jsonl    # 清理後的 bug_report
  labeled/test.jsonl           # 人工標註模板或測試資料
  predicted/predicted.jsonl    # Code Llama-Instruct 抽取結果
  schema.json                  # 目標欄位 schema
  literature_schema.json       # Code Llama 文獻 benchmark schema
  literature/                  # HumanEval / MBPP 文獻資料集

scripts/To_Json/
  fetch_issue_fields.py        # 擷取 GitHub/Jira issue 並轉為 raw schema
  prepare_public_bug_report_dataset.py # 轉換公開 bug report dataset 成 JSON 評估集
  transform_issues.py          # 清理 title/body，建立 bug_report
  sample_test_set.py           # 建立人工標註模板
  extract_to_json.py           # 呼叫 Code Llama-Instruct 抽取 JSON
  postprocess_predictions.py   # 將模型輸出映射回公開資料集官方欄位
  evaluate_results.py          # 評估 predicted_json 與標註/來源欄位差異

scripts/Literature_Datasets/
  prepare_code_llama_benchmarks.py  # 下載並標準化 HumanEval / MBPP
  generate_code_llama_solutions.py   # 用 Code Llama 生成文獻 benchmark 解答
  evaluate_code_generation.py        # 用官方 tests 計算 pass@1
  run_code_llama_result.py           # 主程式：生成/評估/判讀結果
```

## 執行流程

### A. 文獻 benchmark 資料集

```bash
python3 scripts/Literature_Datasets/run_code_llama_result.py --canonical
python3 scripts/Literature_Datasets/run_code_llama_result.py --generate --regenerate
python3 scripts/Literature_Datasets/run_code_llama_result.py
```

若要用 zero-shot prompt 與 3-shot prompt 比較：

```bash
python3 scripts/Literature_Datasets/run_code_llama_result.py \
  --generate \
  --regenerate \
  --prompt-style zero_shot

python3 scripts/Literature_Datasets/run_code_llama_result.py \
  --generate \
  --regenerate \
  --prompt-style paper_aligned
```

### B. Ticket JSON 正規化資料集

1. 擷取資料：

```bash
python3 scripts/To_Json/fetch_issue_fields.py
```

2. 清理與正規化輸入文字：

```bash
python3 scripts/To_Json/transform_issues.py
```

3. 建立人工標註模板：

```bash
python3 scripts/To_Json/sample_test_set.py
```

4. 啟動本機 Ollama，並使用 Code Llama-Instruct 抽取 JSON：

```bash
python3 scripts/To_Json/extract_to_json.py
```

可用環境變數調整文獻式推論設定：

```bash
CODE_LLAMA_MODEL=codellama:7b-instruct \
PROMPT_MODE=zero_shot \
CODE_LLAMA_NUM_CTX=8192 \
python3 scripts/To_Json/extract_to_json.py
```

5. 評估抽取結果：

```bash
python3 scripts/To_Json/evaluate_results.py
```

若只想確認評估結果完全使用主標註檔、沒有自動套用備份標註，可停用備份標註檔：

```bash
python3 scripts/To_Json/evaluate_results.py --no-backup
```

也可以明確指定人工標註檔與預測檔：

```bash
python3 scripts/To_Json/evaluate_results.py \
  --gold dataset/labeled/test.jsonl \
  --pred dataset/predicted/predicted.jsonl \
  --no-backup
```

## 注意

`extract_to_json.py` 預設呼叫 `http://localhost:11434/api/generate`，模型名稱為 `codellama:7b-instruct`。執行前需確認 Ollama 服務與模型已可用。輸出的 `predicted.jsonl` 會保留 `extraction_meta`，記錄文獻來源、模型、prompt style、prompt mode 與 decoding 設定，方便後續撰寫研究方法與重現實驗。

## 參考文獻

Roziere, B., Gehring, J., Gloeckle, F., et al. (2024). Code Llama: Open Foundation Models for Code. arXiv:2308.12950.
