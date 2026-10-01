# Fault Localization：第二階段到 AST 後續實作計畫書

> 文件日期：2026-08-23  
> 分析範圍：Stage-2 Code Llama File Reranker → Top-5 files → Python AST → Symbol records  
> 原則：沿用現有架構，以修正、補強與測試為主，不進行全面重寫。

## 先做什麼

1. 先完成 **P0-1：統一 AST extractor 與 ParseResult**。
2. 接著完成 **P0-2：snapshot commit 與路徑安全驗證**。
3. 再修正 **P0-3：nested scope、qualified name 與 symbol ID**。
4. 完成上述三項後，重新執行 10 筆測資並產生新版輸出。

預估 P0 核心修正約 2 個工作日；完整 Roadmap 約 9–11 個工作日，不包含 2,294 筆完整模型推論時間。

---

## 一、結論

目前可以描述為：

> 「第二階段到 AST 的功能性原型已完成，可以跑通基本流程；但 AST 正確性、錯誤處理、資料契約與測試尚未補齊，因此還不能視為完整可交付版本。」

**估計整體完成度：約 60%。**

目前已驗證：

- 現有 4 個測試全部通過。
- 實際 Code Llama 單筆執行成功完成 Top-20 → Top-5。
- 該筆資料從 5 個檔案抽出 160 個 AST symbols。
- 該筆 Stage-2 Hit@1 = 0、Hit@3 = 1、Hit@5 = 1。
- 單筆結果只能證明 pipeline 可執行，不能代表整體準確率。

目前已重現的核心錯誤：

- Syntax Error 會被誤判成單純 `empty`。
- `../escape.py` 能讀到 repository 外的 Python 檔案。
- method 內的 nested function 會被錯標為 class method。
- 專案內既有 10 筆 prediction 是舊格式，沒有新版 Stage-2 與 AST 欄位。

---

## 二、目前程式碼與架構盤點

### 2.1 相關檔案與職責

| 檔案 | Class／Function／Module | 目前職責 |
|---|---|---|
| `scripts/run_stage1.py` | `CodeLlamaClient` | 呼叫 Ollama Code Llama；要求固定 JSON Schema；設定 timeout、model 與 Top-K。 |
| `scripts/run_stage1.py` | `BatchRunConfig` | 保存 tickets、gold、snapshot、index、output、Ollama 等執行設定。 |
| `scripts/run_stage1.py` | `run_batch()` | 主批次流程：取得 index、執行 Stage-1/2、執行 AST、寫入 predictions/metrics/manifest。 |
| `scripts/run_stage1.py` | `load_or_build_index()` | 載入新版或舊版 index；必要時建立指定 commit 的 repository snapshot 與 code index。 |
| `scripts/run_stage1.py` | `resolve_ast_repository_root()` | 從目前環境的 `data/snapshots/` 動態取得 AST 使用的 repository path。 |
| `scripts/run_stage1.py` | `extract_ast_symbols()` | 讀取 Stage-2 Top-5 Python 檔案，使用 AST 抽出 class、function、method。 |
| `src/utils/fault_localization.py` | `CodeChunk` | 表示一段程式碼及其檔案、symbol、行號與原始文字。 |
| `src/utils/fault_localization.py` | `CodeIndex` | 保存 repository code chunks 與索引設定。 |
| `src/utils/fault_localization.py` | `LocalizationCandidate` | 表示候選檔案／chunk 的 retrieval score、reason 與 signals。 |
| `src/utils/fault_localization.py` | `FaultLocalizer` | 驗證 ticket、Stage-1 排名、file aggregation、Stage-2 rerank、fallback 與輸出。 |
| `src/utils/fault_localization.py` | `build_code_index()` | 掃描 repository source files 並建立 CodeIndex。 |
| `src/utils/fault_localization.py` | `rank_code_chunks()` | TF-IDF 與額外 signals 排名，最後聚合成候選檔案。 |
| `src/utils/fault_localization.py` | `rerank_candidates_with_llm_result()` | 呼叫 LLM、解析回覆、驗證 Top-K、重試、fallback。 |
| `src/utils/fault_localization.py` | `_python_symbol_ranges()` | `ast.parse()`、AST traversal、輸出 Python symbol 範圍。 |
| `src/utils/fault_localization.py` | `extract_symbols_from_files()` | 第二套 AST 擷取實作，目前與 runner 的版本重複且行為不同。 |
| `src/utils/fault_localization.py` | `rerank_symbols_with_llm()` | 尚未接入 pipeline 的 Stage-3 Symbol Reranker 雛形。 |
| `scripts/evaluate_stage1.py` | `evaluate_records()` | 計算 Stage-1、Stage-2 file metrics，以及 AST 是否有 symbols。 |
| `tests/test_stage1_smoke.py` | `Stage1SmokeTest` | 目前 4 個 Stage-1、Stage-2 retry/fallback、基本 AST 測試。 |

### 2.2 完整資料流程

```text
data/test_tickets.jsonl
        │
        ▼
選取 ticket、驗證欄位、組合 bug report
        │
        ▼
取得指定 repo + base_commit snapshot
        │
        ▼
掃描 source files
Python AST / 其他語言 regex
        │
        ▼
建立 CodeChunk + CodeIndex
        │
        ▼
Stage-1 TF-IDF + signals + file aggregation
        │
        ▼
Top-20 candidate files
        │
        ▼
Stage-2 Code Llama File Reranker
JSON Schema → validate → retry / fallback
        │
        ▼
Top-5 files
        │
        ▼
重新讀取 data/snapshots/ 內的 Python files
        │
        ▼
ast.parse() → NodeVisitor → Symbol ranges
        │
        ▼
ast_extracted_symbols + ast_diagnostics
        │
        ▼
predictions.jsonl + metrics.json + manifest.json
```

### 2.3 模組串接方式

1. `run_batch()` 呼叫 `load_or_build_index()` 取得 `CodeIndex`。
2. `run_batch()` 呼叫 `localize_ticket()` 執行 Stage-1 與 Stage-2。
3. `FaultLocalizer` 呼叫 `rerank_candidates_with_llm_result()` 取得 Top-5 或 fallback。
4. `run_batch()` 將 `localized_candidates` 傳給 `extract_ast_symbols()`。
5. `evaluate_records()` 讀取批次輸出，計算 file metrics 與 AST readiness。

目前 Python 原始碼會被 AST parse 兩次：

- 建立 Stage-1 index 時 parse 整個 repository。
- Stage-2 完成後，再 parse Top-5 files。

### 2.4 功能狀態

| 狀態 | 功能 |
|---|---|
| 已完成 | Stage-2 Code Llama 呼叫、固定 Top-5 Schema、最多 2 次嘗試、fallback、動態 snapshot 路徑、基本 Python AST、批次輸出。 |
| 部分完成 | AST traversal、symbol identity、source mapping、per-file diagnostics、多檔案 partial result、Stage-2 評估。 |
| 尚未完成 | AST 正確率評估、symbol ground truth、多語言 AST、穩定的下一階段介面、完整 edge-case tests。 |
| 已完成但需修改 | 重複 AST 實作、Stage-2 evidence、輸出大小、模型重現性、舊 predictions。 |

---

## 三、目前程式碼問題

### 3.1 P0：會影響 pipeline 正確性的問題

#### P0-A：現有 snapshot 未重新驗證 commit

- **問題位置**：`scripts/run_stage1.py::resolve_ast_repository_root()`。
- **問題原因**：只要預期的 snapshot path 存在就直接回傳，沒有確認 Git HEAD。
- **可能影響**：AST 可能解析到錯誤 commit；行號、symbols 與 benchmark gold 全部可能錯位。
- **建議修改**：即使 path 已存在，也執行 `git rev-parse HEAD`，確認等於解析後的 `base_commit`。

#### P0-B：Stage-2 file path 沒有 repository containment 檢查

- **問題位置**：`scripts/run_stage1.py::extract_ast_symbols()`。
- **問題原因**：直接使用 `repo_root / relative_path`，沒有拒絕 absolute path、`..` 或外部 symlink。
- **可能影響**：可以讀取 repository 外檔案；亦可能讓錯誤路徑污染 AST 結果。
- **建議修改**：對 root 與 file 執行 `resolve()`，再檢查 file 必須位於 root 之內且為一般檔案。

#### P0-C：Parser failure 被誤判為沒有 symbols

- **問題位置**：`src/utils/fault_localization.py::_python_symbol_ranges()`。
- **問題原因**：`SyntaxError`、`RecursionError` 都直接回傳空 list。
- **可能影響**：Syntax Error、parser failure、空檔案、合法但沒有 function/class 的檔案，全被標示為 `empty`。
- **建議修改**：回傳 `AstParseResult`，保存 `status`、`error_type`、`message`、`line`、`column` 與 symbols。

#### P0-D：Nested scope 分類錯誤

- **問題位置**：`src/utils/fault_localization.py::Visitor._visit_function()`。
- **問題原因**：只有 `class_stack` 與 function depth，沒有完整 class/function scope stack。
- **可能影響**：method 內的 nested function 可能被錯標為 method；nested class 也失去完整名稱。
- **建議修改**：以 scope stack 建立 `Outer.Inner.run`、`Class.method.<locals>.inner` 等 qualified name，並保留 parent symbol。

#### P0-E：沒有 symbol ground truth

- **問題位置**：`data/test_gold.jsonl`。
- **問題原因**：2,294 筆資料的 `fixed_symbols` 全部為空。
- **可能影響**：目前只能評估 AST 是否有輸出，不能評估正確 symbol 是否被抽出。
- **建議修改**：從 developer patch changed hunks 映射 base-commit AST，建立 symbol-level gold。

### 3.2 P1：功能與介面問題

#### P1-A：AST 有兩套重複實作

- **問題位置**：`scripts/run_stage1.py::extract_ast_symbols()` 與 `src/utils/fault_localization.py::extract_symbols_from_files()`。
- **問題原因**：兩份程式碼分開維護；第二份漏掉 top-level `async_function`。
- **可能影響**：呼叫不同函式時結果不同，修正容易只套用到其中一份。
- **建議修改**：建立唯一 public AST extractor，舊函式只保留相容 wrapper。

#### P1-B：Stage-2 prompt 只看到每個檔案的一個 primary chunk

- **問題位置**：`_aggregate_file_candidates()`、`_llm_rerank_prompt()`。
- **問題原因**：file aggregation 雖保存 supporting chunks，prompt 沒有提供它們的 code evidence。
- **可能影響**：正確檔案可能因 primary chunk 不相關而被 Code Llama 淘汰。
- **建議修改**：每個檔案提供 2–3 個最相關 snippets，並限制每檔與整體 token budget。

#### P1-C：Stage-2 混合分數的語意不完整

- **問題位置**：`_parse_llm_rerank_candidates()`。
- **問題原因**：模型先選 5 檔，再對這 5 檔混合 retrieval/LLM score；未入選的 15 檔不參與混合。
- **可能影響**：0.7 retrieval + 0.3 LLM 並不是對全部 Top-20 的真正 rerank。
- **建議修改**：明確選擇並凍結其中一種政策：
  1. LLM 選 5 檔，retrieval 只負責排列這 5 檔。
  2. LLM 評分全部 20 檔，再混合分數取 Top-5。

#### P1-D：Stage-2 metric 存在成功子集偏差

- **問題位置**：`scripts/evaluate_stage1.py::evaluate_records()`。
- **問題原因**：Stage-2 Hit@K 的分母只有 `stage2_used_rows`。
- **可能影響**：如果 LLM 在困難 ticket 上容易 fallback，成功子集指標會看起來過度樂觀。
- **建議修改**：同時回報 conditional、all-attempted、coverage、fallback pipeline、Recall@5。

#### P1-E：AST output 不足以穩定支援下一階段

- **問題位置**：`ast_extracted_symbols` output。
- **問題原因**：缺少 commit、language、parent、唯一 ID、column、Stage-2 file score 與來源狀態。
- **可能影響**：同名 symbols 衝突；Stage-3 無法安全融合 file score 與 symbol score。
- **建議修改**：建立版本化 `SymbolRecord` schema。

### 3.3 P2：效能、穩定性與程式品質

#### P2-A：Stage-3 雛形不能直接啟用

- **問題位置**：`src/utils/fault_localization.py::rerank_symbols_with_llm()`。
- **問題原因**：一次可能把數百個 symbols 放入 prompt；fallback import 的 `utils.json_schema` 不存在。
- **可能影響**：context 超限、ImportError、回覆不完整。
- **建議修改**：先標記 experimental 或移至獨立模組，本階段不要接入主 pipeline。

#### P2-B：Prediction 輸出過大

- **問題位置**：`LocalizationCandidate.to_dict()`。
- **問題原因**：預設輸出完整 code、signals、supporting chunks。
- **可能影響**：實際單筆 prediction 約 70 KB；完整資料集輸出與分析成本增加。
- **建議修改**：分離 production schema 與 debug schema；正式輸出只保留 file-level 必要欄位。

#### P2-C：模型執行缺少可重現資訊

- **問題位置**：`CodeLlamaClient`、run manifest。
- **問題原因**：沒有 model digest、seed、Ollama 版本、每階段耗時與錯誤類型。
- **可能影響**：不同機器或不同時間的結果難以比較。
- **建議修改**：將完整推論環境、seed 與 timings 寫入 manifest。

---

## 四、完成度評估

| 項目 | 完成度 | 判斷依據 |
|---|---:|---|
| 基本架構 | 85% | Stage-1、Stage-2、AST、輸出已接通。 |
| Stage-2 File Reranker | 75% | Schema、retry、fallback 可運作；evidence 與評估仍不足。 |
| AST 建立 | 70% | Python `ast.parse()` 可執行，但 parser errors 被吞掉。 |
| AST traversal | 55% | 能遍歷 class/function/async；scope 判斷不完整。 |
| Symbol extraction | 60% | 基本 symbols 可抽出；nested、同名、identity 不穩定。 |
| Source location mapping | 55% | 有 decorator-aware 行號；缺 column、commit、source validation。 |
| Error handling | 45% | Stage-2 較完整；AST 只處理 missing/non-Python/empty。 |
| 多語言／多檔案 | 45% | 多檔案已支援；AST 只支援 Python。 |
| 與下一階段介接 | 50% | 有 symbol list；缺穩定 schema、ID、provenance、score。 |
| Testing | 35% | 4 個測試通過；AST 只有一個正常案例。 |

**整體完成度：約 60%。**

---

## 五、P0–P3 後續實作 Tasks

### P0：必須先完成

#### P0-1：統一 AST extractor 與 ParseResult

- **目的**：消除重複實作，正確區分正常、空檔案與 parser error。
- **對應檔案**：`scripts/run_stage1.py`、`src/utils/fault_localization.py`，或新增小型 `src/utils/ast_extraction.py`。
- **目前問題**：兩套 AST extractor；`_python_symbol_ranges()` 只回傳 list。
- **修改內容**：建立 `AstParseResult`、`FileParseDiagnostic`、唯一 public extractor。
- **建議方法**：保留舊函式 wrapper，避免一次修改全部呼叫端。
- **預期輸入／輸出**：source text + metadata → symbols + parse diagnostics。
- **驗收標準**：每個檔案必有明確狀態；Syntax Error 不可再顯示為 `empty`。
- **影響模組**：runner、tests、未來 Stage-3。
- **難度**：中。

#### P0-2：snapshot commit 與路徑安全驗證

- **目的**：確保 AST 只讀取指定 commit 的 repository 內容。
- **對應檔案**：`scripts/run_stage1.py`。
- **目前問題**：現有 snapshot 未確認 HEAD；candidate path 可逃逸 root。
- **修改內容**：驗證 Git HEAD；拒絕 absolute、`..`、外部 symlink、非一般檔案。
- **建議方法**：建立 `resolve_repository_file(repo_root, relative_path)` 共用 helper。
- **預期輸入／輸出**：root + relative path → 已驗證的 file path 或結構化錯誤。
- **驗收標準**：錯 commit 和 `../escape.py` 測試必須失敗；正常檔案保持可讀。
- **影響模組**：runner、AST diagnostics。
- **難度**：中。

#### P0-3：修正 scope、qualified name 與 symbol identity

- **目的**：讓 nested、同名 symbols 可被正確定位。
- **對應檔案**：Python AST visitor、`CodeChunk` 或新的 `SymbolRecord`。
- **目前問題**：缺完整 scope stack；`symbol_qualified_name` 實際等於簡單名稱。
- **修改內容**：維護 class/function scope；新增 parent、qualified name、deterministic symbol ID。
- **建議方法**：ID 使用 `repo@commit:file:kind:qname:start:end` 的 hash。
- **預期輸入／輸出**：AST node + scopes → 唯一且穩定的 SymbolRecord。
- **驗收標準**：nested function 不被標成 method；同名 symbols 的 ID 不相同。
- **影響模組**：AST output、evaluator、Stage-3。
- **難度**：中。

#### P0-4：凍結輸出 schema 與評估規則

- **目的**：讓不同執行結果可比較，避免 fallback 被誤當 LLM 成果。
- **對應檔案**：`scripts/run_stage1.py`、`scripts/evaluate_stage1.py`。
- **目前問題**：舊 predictions 與新程式不一致；缺 `schema_version` 與 AST input provenance。
- **修改內容**：新增 schema version、`ast_input_source`、all-attempted metrics、coverage。
- **建議方法**：新舊 schema 不混用；新版輸出使用新資料夾或明確 migration。
- **預期輸入／輸出**：prediction JSONL → 可驗證的版本化 records 與完整 metrics。
- **驗收標準**：重新跑的 10 筆全部包含 Stage-2/AST diagnostics；evaluator 能拒絕錯誤 schema。
- **影響模組**：runner、evaluator、outputs。
- **難度**：中。

### P1：重要功能補強

#### P1-1：強化 Stage-2 file evidence

- **目的**：讓 Code Llama 看到每個檔案內多個相關位置。
- **對應檔案**：`_aggregate_file_candidates()`、`_llm_rerank_prompt()`。
- **目前問題**：prompt 只提供 primary chunk code。
- **修改內容**：加入 2–3 個 supporting snippets 與 file summary。
- **建議方法**：每檔固定字元上限；先以 50 筆比較 current 與 enhanced prompt。
- **預期輸入／輸出**：Top-20 file evidence → validated Top-5。
- **驗收標準**：used-rate ≥95%；Stage-2 Hit@5 不低於 Stage-1 Top-5。
- **影響模組**：Stage-2、執行時間、prompt size。
- **難度**：中。

#### P1-2：完成版本化 SymbolRecord

- **目的**：提供 Fault Localization 下一階段穩定輸入。
- **對應檔案**：AST extractor、runner output。
- **目前問題**：缺 repo、commit、language、parent、columns、file score。
- **修改內容**：定義 schema 並加入 validation。
- **建議方法**：使用 dataclass，再以 `to_dict()` 統一序列化。
- **預期輸入／輸出**：AST node + file candidate → SymbolRecord。
- **驗收標準**：所有 output records 通過 schema validation；同名 symbols 可區分。
- **影響模組**：Stage-3、metrics、debug tools。
- **難度**：中。

#### P1-3：建立 symbol-level gold

- **目的**：評估 AST 是否涵蓋真正修改位置。
- **對應檔案**：新增 `scripts/build_symbol_gold.py`、`data/test_gold.jsonl` 或獨立 symbol gold。
- **目前問題**：`fixed_symbols` 全空。
- **修改內容**：解析 developer patch hunks，映射到 base-commit AST symbol ranges。
- **建議方法**：無法映射到 function/class 的修改標成 `<module>`。
- **預期輸入／輸出**：ticket + patch + snapshot → fixed symbol records。
- **驗收標準**：人工抽查 50 筆，映射正確率 ≥95%。
- **影響模組**：evaluator、dataset。
- **難度**：高。

#### P1-4：支援 per-file partial AST

- **目的**：單一檔案失敗時保留其餘檔案結果。
- **對應檔案**：AST extractor、runner diagnostics。
- **目前問題**：總狀態只有 `ok` 或 `empty`。
- **修改內容**：每檔記錄 `ok/no_symbols/syntax_error/unsupported/missing/invalid_path`。
- **建議方法**：總狀態由 per-file 狀態聚合成 `ok/partial/failed`。
- **預期輸入／輸出**：Top-5 → symbols + 5 個 file diagnostics。
- **驗收標準**：其中一檔失敗時，其餘檔案仍輸出，總狀態為 `partial`。
- **影響模組**：runner、metrics。
- **難度**：中。

### P2：效能、穩定性與品質

#### P2-1：建立 AST cache

- **目的**：避免 Stage-1 與 AST 階段重複 parsing。
- **對應檔案**：index/cache、AST extractor。
- **修改內容**：以 repo、commit、file hash 快取 SymbolRecord。
- **驗收標準**：第二次執行 AST 時間降低至少 50%，結果完全一致。
- **影響模組**：index、storage；難度中。

#### P2-2：精簡 prediction output

- **目的**：降低輸出體積與後續分析成本。
- **對應檔案**：`LocalizationCandidate.to_dict()`、runner。
- **修改內容**：分離 production/debug schema，正式輸出不附完整 code。
- **驗收標準**：50 筆 prediction 體積降低至少 50%，metrics 不變。
- **影響模組**：outputs、debugging；難度低。

#### P2-3：加入可觀測性與重現性資訊

- **目的**：判斷 timeout、fallback、parser failure 與執行瓶頸。
- **對應檔案**：`CodeLlamaClient`、runner manifest。
- **修改內容**：model digest、seed、Ollama version、request/parse/AST timings、錯誤分類。
- **驗收標準**：manifest 足以重建設定；每個 fallback 有明確原因。
- **影響模組**：runner；難度中。

### P3：未來擴充

#### P3-1：多語言 AST adapter

- **目的**：支援 Stage-1 已索引的 JS/TS/Java/C/C++/Go/Rust。
- **建議方法**：採 Tree-sitter adapter，輸出同一套 SymbolRecord。
- **驗收標準**：每種語言至少 10 個 fixtures，名稱與位置全部正確。
- **難度**：高。

#### P3-2：程式關係圖與 Symbol Reranker

- **目的**：加入 import、call、inheritance evidence，支援第三階段。
- **前提**：P0、P1 與 symbol gold 完成後才開始。
- **驗收標準**：指定 symbol 可查詢相鄰 symbols，且不破壞現有 schema。
- **難度**：高。

---

## 六、Fault Localization 所需的 AST 資料契約

建議下一階段至少接收：

```json
{
  "schema_version": "symbol-record-v1",
  "symbol_id": "stable-hash",
  "repo": "owner/repo",
  "base_commit": "commit",
  "file_path": "package/module.py",
  "language": "python",
  "symbol_kind": "method",
  "qualified_name": "Outer.Inner.run",
  "parent_symbol": "Outer.Inner",
  "start_line": 10,
  "start_column": 4,
  "end_line": 20,
  "end_column": 16,
  "source_file_rank": 2,
  "source_file_score": 0.73,
  "ast_input_source": "stage2_llm",
  "parse_status": "ok"
}
```

關鍵欄位：

1. `base_commit`：避免映射到錯誤版本。
2. `symbol_id`：區分同名 symbols。
3. `qualified_name`：保留完整 scope。
4. `source_file_score`：讓下一階段融合 file 與 symbol 分數。
5. `ast_input_source`：區分 Stage-2 LLM 成功與 Stage-1 fallback。

---

## 七、Testing 計畫

### 7.1 Unit Tests：基本 AST

| 測試 | 測試目的 | Input | Expected Output | 驗收方式 |
|---|---|---|---|---|
| 正常程式碼 | 確認基本 extraction | class、method、function、async function | kind、name、range 正確 | 比對完整 SymbolRecord |
| Syntax Error | 區分 parser error | `def broken(:` | `syntax_error` 與錯誤行列 | assert diagnostics |
| 空檔案 | 區分合法空檔案 | 0 bytes 或只有註解 | `no_symbols` | status/count 精確 |
| Nested class/function | 驗證 scope traversal | class method 內 nested function | 完整 qname；nested function 不誤標 method | 比對 qname/kind |
| 多個同名 function | 驗證 symbol identity | 同檔兩個 `handler()` | 名稱可同，ID 不同 | 比對 ID 與行號 |

### 7.2 Unit Tests：位置與錯誤

| 測試 | 測試目的 | Input | Expected Output | 驗收方式 |
|---|---|---|---|---|
| AST location mapping | 驗證來源位置 | decorator、多行 signature、async method | start/end line/column 精確 | source slice 與 node 一致 |
| 大型檔案 | 驗證資源限制 | 超過限制或數千 symbols | parse 或 `too_large` | 時間與記憶體不超標 |
| Parser failure | 驗證非 SyntaxError 失敗 | mock `RecursionError` | `parser_error` | 其他檔案仍保留 |
| 路徑逃逸 | 驗證 repository containment | `../escape.py`、absolute、symlink | `invalid_path` | 不得讀取外部檔案 |
| 非 Python | 明確記錄不支援語言 | `.c`、`.js` | `unsupported_language` | diagnostics 精確 |

### 7.3 Integration Tests

| 測試 | 測試目的 | Input | Expected Output | 驗收方式 |
|---|---|---|---|---|
| 不同 repository structure | 驗證 snapshot/path | src-layout、namespace package、深層路徑 | repo、commit、relative path 正確 | 比對 metadata |
| 多檔案 partial | 驗證局部成功 | 3 正常、1 missing、1 syntax error | 正常 symbols 保留，總狀態 `partial` | file diagnostics 合計為 5 |
| Stage-2 malformed response | 驗證 response validation | duplicate、超界、NaN、少於 5 筆 | retry 或 fallback | 比對 attempts/reason |
| Stage-2 timeout/HTTP failure | 驗證 retry policy | timeout、500、400 | retryable 才重試 | mock request 次數 |
| 完整 pipeline | 驗證模組串接 | fake LLM + temporary git snapshot | Top-20 → Top-5 → AST → JSONL → metrics | schema 與 metrics 全通過 |

### 7.4 真實模型驗證

1. 先固定 10 個 ticket IDs，避免每次 sample 不同。
2. 記錄 model digest、Ollama version、seed 與 timeout。
3. 回報 Stage-2 used-rate、fallback-rate、Hit@1/3/5、Recall@5。
4. 回報 AST file coverage、parse error、unsupported、symbol count。
5. 10 筆成功後再擴至 50 筆，不直接執行完整 2,294 筆。

---

## 八、Roadmap

| Phase | 要完成的 Tasks | 修改檔案 | 完成條件 | 依賴／估時 |
|---|---|---|---|---|
| Phase 1：修正核心問題 | P0-1～P0-4 | `run_stage1.py`、AST utility、evaluator | parser 狀態、commit、路徑、schema 全部驗收 | 無；約 2 工作日 |
| Phase 2：補齊 AST / Symbol Extraction | P1-2、P1-4 | AST utility、runner output | nested、同名、async、partial 全部正確 | Phase 1；約 2 工作日 |
| Phase 3：與 Fault Localization 串接 | P1-1、P1-3 | Stage-2 prompt、symbol gold、metrics | 可產生 50 筆準確率報告 | Phase 2；約 2–3 工作日 |
| Phase 4：Testing 與 Edge Cases | 完成 Unit、Integration、Ollama smoke | `tests/`、fixtures | CI 通過；10 筆新版輸出完成 | Phase 1–3；約 2 工作日 |
| Phase 5：Performance / Refactoring | P2-1～P2-3 | cache、output、manifest、README | 效能、體積、重現性達標 | Phase 4；約 1–2 工作日 |

依賴順序：

```text
Phase 1
   ↓
Phase 2
   ↓
Phase 3
   ↓
Phase 4
   ↓
Phase 5
```

---

## 九、組員執行 Checklist

### 先做：P0

- [ ] P0-1 統一唯一 AST extractor 與 ParseResult。
- [ ] P0-2 驗證 snapshot HEAD。
- [ ] P0-3 禁止 `../`、absolute path 與外部 symlink。
- [ ] P0-4 修正 nested scope、qualified name、symbol ID。
- [ ] P0-5 加入 schema version、AST input provenance、完整 Stage-2 metrics。

### 接著做：P1

- [ ] P1-1 Stage-2 prompt 加入每檔多段 evidence。
- [ ] P1-2 完成版本化 SymbolRecord。
- [ ] P1-3 建立 developer patch → symbol ground truth。
- [ ] P1-4 支援 per-file diagnostics 與 partial success。
- [ ] P1-5 重跑固定 10 筆並保存新版 outputs/manifest。

### 之後做：P2 / P3

- [ ] P2-1 建立 AST cache，避免重複 parsing。
- [ ] P2-2 分離 production/debug output。
- [ ] P2-3 加入 model digest、seed、timings、錯誤分類。
- [ ] P3-1 使用 Tree-sitter 擴充多語言 AST。
- [ ] P3-2 再接 Symbol Reranker 與程式關係圖。

---

## 十、P0 完成後的驗證指令

先執行本機測試：

```bash
python3 -m unittest discover -s tests -v
```

再執行固定 10 筆：

```bash
python3 scripts/run_stage1.py --clone-missing --limit 10
```

重新計算 metrics：

```bash
python3 scripts/evaluate_stage1.py \
  --gold data/test_gold.jsonl \
  --pred outputs/stage1_test/test_predictions.jsonl \
  --output outputs/stage1_test/test_metrics_recalculated.json
```

最小驗收條件：

1. 10 筆 prediction 全部有 `schema_version`、`stage2_rerank`、`ast_diagnostics`。
2. 每個 AST requested file 都有一筆 file diagnostic。
3. Syntax Error 不得顯示為單純 `empty`。
4. 所有 symbol IDs 在單筆 ticket 內唯一。
5. metrics 同時包含 Stage-2 coverage、conditional Hit@K 與 all-attempted Hit@K。
