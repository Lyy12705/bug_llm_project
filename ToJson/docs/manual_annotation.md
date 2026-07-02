# 人工標註說明

本專案為了評估「使用者 bug 回報訊息轉 JSON 欄位」的效果，建立一份人工標註測試集。此標註集只作為測試與評估使用，不作為模型訓練資料。標註檔位於：

```text
dataset/labeled/test.jsonl
```

每一筆資料皆包含原始 `bug_report`，以及人工填寫的 `json_ground_truth`。標註目標是讓非結構化的 issue / ticket 描述轉換成固定 JSON schema，作為 Code Llama-Instruct 抽取結果的評估標準。

## 測試用途

由於 Code Llama 論文沒有提供「bug 回報訊息轉 JSON 欄位」的完整公開測試集，本專案以人工標註建立測試用 ground truth。模型執行時只讀取 `bug_report`，產生 `predicted_json`；評估時再將 `predicted_json` 與人工標註的 `json_ground_truth` 逐欄比較。

此測試集用於檢驗第一階段資料正規化流程是否能完成以下任務：

1. 將非結構化 bug report 轉成合法 JSON。
2. 抽取或判斷固定欄位內容。
3. 對每個欄位計算準確率。
4. 檢查一筆資料是否指定評估欄位皆正確。

因此，本測試集的角色是「評估標準答案」，不是模型輸入的一部分，也不會提供給 Code Llama 作為提示內容。

## 報告用文字

可在報告中使用以下描述：

```text
為了評估第一階段「非結構化 Ticket 轉 JSON 正規化」的效果，本研究建立人工標註測試集作為 ground truth。測試資料共 56 筆，來源包含 GitHub issue / pull request 與 Jira issue。每筆資料保留原始使用者回報文字作為模型輸入，既有 baseline 由人工標註六個 JSON 欄位：bug_type、component、os、version、priority 與 error_message。

目前為了和後續 duplicate detection、priority classification、assignee triage 整合，Code Llama 輸出 schema 已擴充。新建立的人工標註檔會包含完整欄位；舊 56 筆標註結果仍可用於六欄 baseline 評估。

模型推論時只讀取 bug_report，不讀取人工標註答案；評估階段再將模型輸出的 predicted_json 與人工標註的 json_ground_truth 進行逐欄比對。評估指標包含 field accuracy 與 exact match accuracy，其中 field accuracy 用於衡量各欄位抽取是否正確，exact match accuracy 則要求單筆資料的所有欄位皆與人工標註一致。

此人工標註測試集用於補足目前公開資料集中缺少完整 bug report JSON ground truth 的問題，使本專題能夠針對第一階段資料正規化功能進行量化評估。
```

## 標註資料

目前人工標註資料共 56 筆，來源包含 GitHub issue / pull request 與 Jira issue。既有 baseline 標註以下六個欄位：

```json
{
  "bug_type": "",
  "component": "",
  "os": "",
  "version": "",
  "priority": "",
  "error_message": ""
}
```

其中 `bug_type`、`component`、`os`、`priority` 為主要分類欄位；`version` 與 `error_message` 若原文沒有明確資訊，允許留空字串。

新 schema 的完整標註模板如下：

```json
{
  "ticket_id": "",
  "title": "",
  "description": "",
  "product": "",
  "severity": "",
  "bug_type": "",
  "component": "",
  "os": "",
  "version": "",
  "priority": "",
  "error_message": "",
  "steps_to_reproduce": [],
  "expected_behavior": "",
  "actual_behavior": "",
  "logs": "",
  "screenshots_text": ""
}
```

其中 `title`、`description`、`product`、`component`、`severity`、`priority`、`bug_type` 會直接或間接提供給後續流程使用，特別是負責人分派階段會使用 `product`、`component` 與文字相似度欄位。

## 標註規則

### ticket_id

`ticket_id` 表示 issue / bug / ticket 的識別碼。若資料來源已有 `id`、`issue_id` 或 `bug_id`，直接填入；若只是一般使用者輸入且沒有編號，留空字串。

### title

`title` 表示問題標題。若來源資料有標題，直接複製；若只有一段使用者輸入，可用第一句或最能代表問題的短句作為標題。

### description

`description` 表示主要問題描述。標註時保留使用者回報的核心內容，不需要包含完整長 log；log 可放在 `logs`。

### product

`product` 表示 issue tracker 的 product、專案名稱或 repository 名稱。例如 Mozilla Bugzilla 的 `Firefox`、`Core`，或 GitHub repository 名稱。若無法判斷，標為 `unknown`。

### severity

`severity` 表示嚴重程度，優先使用 issue tracker 原生欄位，例如 `blocker`、`critical`、`major`、`normal`、`minor`、`trivial`、`enhancement`。若沒有明確資訊，標為 `unknown`。

### bug_type

`bug_type` 表示問題類型，只能標為以下類別：

| 類別 | 標註條件 |
| --- | --- |
| `crash` | 程式崩潰、例外、執行中斷，例如 `RuntimeError`、`TypeError`、`NullPointerException` |
| `logic_error` | 行為或結果錯誤，例如狀態追蹤錯誤、輸出錯誤、邏輯不一致 |
| `performance` | 效能下降、CI 過慢、memory leak、資源未釋放、overhead 過高 |
| `build_error` | build、compile、docker、CI 環境建置或 dependency resolution 失敗 |
| `compatibility` | 平台、版本、套件或相依性不相容 |
| `ui_bug` | 顯示、介面互動或 UI 行為異常 |
| `non_bug` | PR、重構、版本升級、文件修改、feature request、release task、純維護工作 |
| `unknown` | 原文不足以判斷問題類型 |

### component

`component` 表示受影響模組或元件。標註時優先使用標題或描述中明確出現的模組名稱，例如 `dynamo`、`inductor`、`torch.compile`、`hadoop-aws`。若 issue tracker 原生欄位可提供更明確的元件，也可作為輔助判斷。

若無法從文字或來源欄位判斷，標為 `unknown`。

### os

`os` 表示作業系統或執行平台。若原文或來源欄位明確指出 `windows`、`linux`、`macos` 等平台，標成對應值；若未提及，標為 `unknown`。

### version

`version` 表示受影響版本、套件版本或 issue tracker 中標示的版本。若文字中出現明確版本號，例如 `3.5.0`、`2.10.0`、`0.8.1`，則填入該值。若沒有明確版本資訊，留空字串。

### priority

`priority` 使用五級標準：

| 類別 | 標註條件 |
| --- | --- |
| `P1` | Blocker，系統無法運作或嚴重阻斷主要流程 |
| `P2` | Critical，嚴重影響核心功能，但仍可能有有限替代方式 |
| `P3` | Major，一般 bug 或功能缺陷，預設 bug 等級 |
| `P4` | Minor，影響較小、範圍有限或可接受 workaround |
| `P5` | Trivial / non-bug，文件、維護、重構、建議或非錯誤類型 |

若 `bug_type` 為 `non_bug`，通常標為 `P5`。

### error_message

`error_message` 表示原文中明確出現的錯誤訊息、exception 名稱或 log 摘要。例如：

```text
java.lang.NullPointerException
RuntimeError
Segmentation fault
```

若原文沒有明確錯誤訊息，留空字串。不要自行補出原文沒有的錯誤訊息。

## 標註分布

目前 56 筆人工標註資料的欄位分布如下：

### bug_type

```text
non_bug       : 31
logic_error   : 9
crash         : 6
performance   : 5
build_error   : 3
compatibility : 2
```

### os

```text
unknown : 51
linux   : 4
windows : 1
```

### priority

```text
P5 : 31
P3 : 17
P4 : 6
P2 : 1
P1 : 1
```

`version` 有 27 筆非空值，`error_message` 有 10 筆非空值。

## 評估方式

完成模型抽取後，使用以下指令評估：

```bash
python3 scripts/To_Json/evaluate_results.py \
  --gold dataset/labeled/test.jsonl \
  --pred dataset/predicted/predicted.jsonl \
  --no-backup
```

目前完整六欄評估結果：

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

## 限制

此人工標註集目前規模較小，適合作為第一階段 proof-of-concept 評估。由於部分 issue / pull request 並非真正 bug report，因此 `non_bug` 比例偏高。後續若要提升評估可靠度，建議擴充更多真實 bug report，並由兩位以上標註者交叉標註以計算一致性。
