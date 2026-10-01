# 第三階段 Symbol Reranker 實作與實驗計畫

**文件版本：** v1.2
**建立日期：** 2026-09-02  
**更新日期：** 2026-09-09
**狀態：** WP4 一輪探索性 pilot 完成，完整驗收未完成；正式預設維持 deterministic baseline（待指導教授確認研究範圍調整）
**適用專案：** Fault Localization Feature

> **下一個行動：** 凍結 `coverage-aware-v1` 為正式預設，將 WP4 pilot 的 G2／G3／G4／G5 限制納入報告，接續補丁生成流程；若要補做完整 WP4 研究驗收，再另行完成 L1-blend、延遲與正式 validation。

---

## 0. 執行摘要

第一階段與第二階段已完成可重現的串接。正式流程目前採用第一階段 E8-A 產生 Top-20 檔案候選，第二階段保留 retrieval baseline Top-5；`codellama:7b-instruct` File Reranker 預設關閉，因為既有 10-ticket 配對實驗的 File Hit@1、Hit@3、Hit@5 與 MRR 均下降 0.1。

第三階段不是從零開始。現有主程式已能從 Stage-2 Top-5 檔案取得 AST symbol、產生 `stage3_ranked_symbols`，並選擇性呼叫 Code Llama Symbol Reranker；但目前仍是「可執行雛形」，不能視為已完成的可評估模型，主要原因如下：

1. 既有 2,294 筆 gold 的 `fixed_symbols` 全空，Symbol Hit@K 與 MRR 沒有有效分母。
2. `CodeChunk.symbol_qualified_name` 仍等同簡單名稱；nested scope、同名 symbol 與穩定 identity 尚未完成。
3. Python parse failure 與合法無 symbol 檔案都會回傳空清單，診斷狀態無法區分。
4. 現有 evaluator 只比較 symbol 名稱並允許 suffix match，沒有把 `file_path` 納入主要 exact 指標。
5. 現有 Symbol LLM 將 Top-30 拆成每批 5 個後直接跨批比較分數；不同批次的分數不一定可校準。

因此第三階段的正式目標是：先建立可信的 Symbol 資料與評估基礎，再比較 deterministic symbol retrieval 與 Code Llama Symbol Reranker。若 LLM 未通過採用門檻，正式預設維持 symbol retrieval，LLM 保留為實驗選項。

### 0.1 2026-09-08 範圍與時程決策

本版採用方向二：「接受目前候選涵蓋上限，帶著限制往下走」。90%仍保留為理想研究目標，
但不再是啟動WP4的硬性前置條件。這項調整的理由如下：

| 固定方法 | Conditional Exact Candidate Recall@30 | 相對68.64%基準 | 決定 |
|---|---:|---:|---|
| `coverage-aware-v1` | **68.64%** | — | 凍結為WP4候選基準 |
| `source-neighborhood-v1` | 64.70% | -3.94百分點 | 不採用 |
| `call-neighborhood-v1` | 66.02% | -2.62百分點 | 不採用 |
| `call-neighborhood-v2` | 68.20% | -0.44百分點 | 不採用 |

AST pool exact oracle為90.41%，但實際Top-30只有68.64%；三個最近的凍結補救均未超越
基準，顯示再以局部規則修補不太可能在目前時程內一次增加21.36個百分點。本輪停止
WP3小幅調整，保留30個pool-reachable ranking misses、17個Stage-2 file misses與2個
identity mismatches作為已知限制。

WP4可以開始，但必須同時回報Candidate Recall@30、conditional與end-to-end指標；LLM
只能重排已進shortlist的候選，不能修復未進Top-30的gold symbol。因此WP4結果只作
探索性證據，不能把未命中候選的責任歸因於LLM。此範圍調整應在下一次進度報告交由
指導教授確認；若未獲同意，退回方向三，以檔案層級定位作正式範圍，Symbol定位列為
未來工作。

### 0.2 調整後時程

| 日期／期限 | 工作 | 完成條件 |
|---|---|---|
| 2026-09-08 | WP3停損並凍結`coverage-aware-v1` | 保留68.64%與完整負結果 |
| 2026-09-08～09-09 | WP4 10-ticket smoke | schema、fallback、score leakage與耗時可判讀 |
| 最晚2026-09-10 | WP4固定50-ticket pilot與採用決策 | 選擇LLM或retrieval，完成限制報告 |
| 2026-09-11起 | 銜接後續流程與文件 | 不再讓Stage3壓縮原定10月補丁生成里程碑 |

### 0.3 三個方向的決策紀錄

| 方向 | 本版決策 | 理由 |
|---|---|---|
| 一：另試語意候選擴充 | 延後為新實驗ID的未來工作 | 與現有規則不同，值得研究，但本輪不再延長WP3 |
| 二：放寬門檻並啟動WP4 | **採用** | 保留完整限制，優先完成定位→補丁→測試→提交訊息流程 |
| 三：只做到檔案層級 | 指導教授不同意範圍調整時的備案 | Stage1／2已有結論，可保障最低專題交付 |

**預估工期：** 單人約 10–11 個工作日；若 repository snapshot 已齊全且 50 筆人工抽查一次通過，可縮短至 8–9 個工作日。

---

## 1. 目標、範圍與非目標

### 1.1 本階段目標

第三階段接收 Ticket 與 Stage-2 Top-5 檔案，在函式、Method、Class 與必要的 `<module>` symbol 中排序最可能的錯誤位置，輸出前 5 名及完整診斷。

本階段完成後必須能回答三個問題：

1. 正確 symbol 是否進入可供 reranker 判斷的候選集合？
2. 在正確檔案已進入 Stage-2 Top-5 的條件下，Symbol Reranker 是否把正確 symbol 排到前 1／3／5 名？
3. 從 Ticket 到最終輸出的完整 pipeline，實際 Symbol Hit@K 與 Recall@K 是多少？

### 1.2 本階段包含

- Python AST symbol 的穩定 schema、qualified name、parent、位置與 parse diagnostics。
- developer patch 到 base-commit symbol 的 ground-truth 建置與人工抽查。
- Symbol candidate retrieval baseline、Code Llama reranker、fallback 與可重現 manifest。
- Candidate、conditional 與 end-to-end 三層 Symbol 指標。
- Unit、integration、真模型配對實驗與凍結報告。

### 1.3 本階段不包含

- 自動產生修補 Patch。
- 啟用第二階段 Code Llama File Reranker；第三階段正式輸入固定為 retrieval baseline Top-5。
- 一次支援所有語言的完整 parser；v1 以 Python AST 為正式評估範圍，其他語言標記為 unsupported 或 heuristic-only。
- 在 frozen holdout 上調 prompt、權重、K 值或 threshold。

---

## 2. 現況基準與不可改動的前提

### 2.1 已完成並沿用

| 邊界 | 正式設定 | 第三階段的處理方式 |
|---|---|---|
| Stage 1 | E8-A 實作，Top-20 unique files | 直接沿用，不在本階段重新最佳化 |
| Stage 2 | retrieval baseline Top-5 | 作為第三階段唯一正式檔案輸入 |
| Stage-2 LLM | `codellama:7b-instruct` 預設關閉 | 不與 Stage-3 LLM 同時變動，避免混淆因果 |
| AST 來源 | portable `CodeIndex` 內的 symbol chunks | 保留介面，但修正 identity、diagnostics 與 schema |
| 安全 fallback | LLM 輸出不完整時保留 retrieval | 沿用並加入 coverage、原因與耗時 |

### 2.2 現有雛形必須修正的介面

| 現有位置 | 問題 | 計畫處理 |
|---|---|---|
| `src/utils/fault_localization.py::CodeChunk` | qualified name、parent、column、commit 不完整 | 建立 `SymbolRecordV1`；`CodeChunk` 保留相容 wrapper |
| `_python_symbol_ranges()` | parse error 與 no-symbol 都回傳 `[]` | 改為 `SymbolParseResult` 與 per-file diagnostics |
| `FaultLocalizer.localize()` | symbol baseline 只有開啟 `symbol_rerank` 才產生 | 將「symbol localization」與「LLM rerank」拆成兩個開關 |
| `rerank_symbols_with_llm()` | 每批 5 個，跨批 raw score 不可保證可比 | 改為 bounded global shortlist 的單次最終比較 |
| `evaluate_fault_localization.py` | symbol 名稱 suffix match，未綁定檔案 | 主要指標改用 file-qualified exact match；relaxed 指標只作診斷 |

---

## 3. 目標架構與資料流

```text
Ticket（標題／描述／錯誤訊息）
  ↓
Stage 1：E8-A retrieval Top-20 files
  ↓
Stage 2：retrieval baseline Top-5 files
  ↓
AST / Symbol Extraction
  ├─ SymbolRecordV1
  ├─ per-file parse diagnostics
  └─ functions / methods / classes / <module>
  ↓
Stage 3A：Deterministic Symbol Candidate Retrieval
  ├─ 保存 Candidate Top-30 供 Candidate Recall@30 評估
  └─ 取 global Top-10 進入 LLM 比較
  ↓
Stage 3B：Code Llama Symbol Reranker（實驗選項）
  ├─ prompt 不含 retrieval score、file rank 或 gold
  ├─ 單一 bounded shortlist，完整回傳每個 candidate_id
  └─ malformed／timeout → deterministic fallback
  ↓
Top-5 file-qualified symbols + confidence + diagnostics
```

### 3.1 為何先保存 Top-30、再讓 LLM 比較 Top-10

Top-30 用來量測候選生成的上限，避免把「正確 symbol 根本沒有進候選池」誤判為 LLM 排序失敗。Code Llama 最終只比較 Top-10，可把 bug report、signature、docstring 與 evidence window 放在同一個受控 context 中，避免現有跨批分數不可比的問題。

Top-10 與每個 symbol 的 evidence budget 必須在 development split 上凍結；formal holdout 不再修改。

---

## 4. 資料契約

### 4.1 Stage-3 輸入契約

每張 Ticket 必須具有：

- `ticket_id`、`repo`、`base_commit` 與 bug report 文字。
- 恰好一份版本化 code index，且 index 對應同一個 repository commit。
- `stage2_localized_files`，最多 5 個唯一相對路徑；正式來源必須記為 `retrieval_baseline`。
- 每個檔案的 rank、score 與來源；這些欄位可供 deterministic baseline 使用，但不可放入 LLM prompt。

若 commit 不一致、路徑逃逸、檔案不存在或 index schema 不相容，該 Ticket 不得靜默繼續，必須輸出結構化 diagnostics。

### 4.2 `SymbolRecordV1`

```json
{
  "schema_version": "symbol-record-v1",
  "symbol_id": "sha256:stable-id",
  "repo": "owner/repo",
  "base_commit": "40-hex-commit",
  "file_path": "package/module.py",
  "language": "python",
  "symbol_kind": "method",
  "qualified_name": "Outer.Inner.run",
  "display_name": "run",
  "parent_symbol_id": "sha256:parent-id",
  "start_line": 10,
  "start_column": 4,
  "end_line": 20,
  "end_column": 16,
  "signature": "def run(self, value):",
  "source_file_rank": 2,
  "source_file_score": 0.73,
  "ast_input_source": "stage2_retrieval_baseline",
  "parse_status": "ok"
}
```

`symbol_id` 使用 repository、commit、file、kind、qualified name 與 source range 建立 deterministic hash。同名 symbol 可存在，但同一 Ticket 內 `symbol_id` 不可重複。

### 4.3 Stage-3 輸出契約

正式 prediction 至少新增：

- `stage3_candidate_symbols`：deterministic Top-30，供 Candidate Hit／Recall 評估。
- `stage3_retrieval_symbols`：未經 LLM 的正式 baseline Top-5。
- `stage3_ranked_symbols`：LLM 成功時的 Top-5，否則等於 baseline fallback。
- `stage3_diagnostics`：input source、eligible 狀態、candidate counts、LLM used/fallback、fallback reason、token/char budget、timings 與 schema version。
- `stage3_run_manifest`：model name/digest、Ollama version、seed、temperature、prompt version、程式版本、資料 split hash 與執行機器資訊。

正式 production output 不保存完整 symbol body；debug output 才保存 evidence text，以控制檔案大小。

---

## 5. Symbol Ground Truth 建置

### 5.1 映射原則

ground truth 必須定位「base commit 中待修正的 symbol」，而不是只解析 patch 後的新程式碼。

1. 解析 developer patch 的 old-file hunk 範圍與修改類型。
2. 在 `base_commit` 讀取對應檔案，使用同一套 `SymbolRecordV1` extractor。
3. 將 deleted／modified old lines 映射到最內層包含該行的 symbol；一個 hunk 可對應多個 symbols。
4. 純新增內容以 old hunk insertion point 與 context line 映射；檔案層級修改標記為 `<module>`。
5. 新檔案、無法對應、unsupported language 與 parser failure 必須保留 reason，不可硬猜 symbol。

### 5.2 Gold record

每筆 gold 至少保存 `ticket_id`、`file_path`、`symbol_id`、`qualified_name`、`symbol_kind`、line range、mapping method、mapping confidence 與 exclusion reason。主要 exact 指標只使用可映射且人工抽查合格的 records。

### 5.3 Evaluator symbol identity contract

正式 evaluator 使用 `SymbolEvaluationItemV1`，每個 gold 與 prediction item 必須包含 `file_path`、`qualified_name`、`symbol_kind`，`symbol_id` 可選。所有 path 先轉成安全的 repository-relative POSIX path；`qualified_name` 與 `symbol_kind` 不可為空。`<module>` 必須與 `symbol_kind=module` 成對，違反契約的 prediction 不參與排名並計入 invalid diagnostic。

同一 Ticket 的多筆 `SymbolGoldRecordV1` 先聚合後再評估，因此 Hit 的分母是 Ticket 數，不是 gold record 數；Recall 則計算 Top-K 找回的 unique gold symbols 比例。缺少整筆 prediction 或缺少 `stage3_ranked_symbols` 都保留在 end-to-end 分母並計為 miss。

### 5.3 品質門檻

- 先抽查 50 筆、涵蓋至少 5 個 repositories、nested symbol、module-level、純新增、刪除與多檔修改。
- file、qualified name 與 kind 三者完全正確的比例必須至少 95%。
- provenance 欄位完整率必須 100%。
- 未達 95% 時，修正 mapper 並重抽 50 筆；在通過前不得開始正式 LLM 準確率實驗。

---

## 6. Symbol 排序設計

### 6.1 Baseline B0：現有 TF-IDF Symbol Retrieval

以 Ticket 文字對 symbol code chunk 計算 TF-IDF，依現有分數排序並去除相同 `symbol_id`。B0 不使用 LLM，是所有後續比較的最低基準。

### 6.2 Baseline B1：結構化 Deterministic Retrieval

B1 將下列 evidence 正規化後合成 symbol score：

- Ticket identifier／錯誤訊息與 qualified name、signature 的精確或 token match。
- stack trace 中的 file、line 與 function evidence。
- symbol body、docstring 與鄰近註解的 lexical relevance。
- Stage-2 file score；只作 deterministic baseline 特徵，不放入 LLM prompt。
- 大檔案 per-file quota，避免單一檔案的多個 symbols 壟斷 Top-30。

權重只可在 development split 上調整。先比較 B0 與 B1 的 Candidate Recall@10／30；若 B1 沒有提升，保留較簡單的 B0。

### 6.3 L1：Code Llama Symbol Reranker

L1 接收較佳 deterministic baseline 的 Top-10。每個候選使用 opaque `candidate_id`，prompt 僅提供 file path、symbol kind、qualified name、signature、受控 evidence window 與 bug report；不提供 retrieval score、原始 rank、Stage-2 file score或 gold。

模型必須一次回傳 10 筆唯一 `candidate_id`、有限 0–1 score 與短理由。少一筆、重複、未知 ID、NaN、超界或非 JSON 都視為整張 Ticket 的 LLM failure，並完整 fallback 至 baseline。

### 6.4 分數與排序政策

正式配對實驗至少比較：

| Variant | 候選集合 | 最終排序 | 用途 |
|---|---|---|---|
| B0 | TF-IDF Top-30 | TF-IDF | 最低基準 |
| B1 | Structured Top-30 | deterministic score | 正式 retrieval 候選基準 |
| L1-only | B1 Top-10 | LLM score | 量測模型獨立效果 |
| L1-blend | B1 Top-10 | development-frozen blend | 量測安全融合效果 |

blend 權重只允許在 development split 選一次。若 L1-only 與 L1-blend 都未通過採用門檻，正式流程保留 B1。

---

## 7. 評估定義

### 7.1 主要匹配規則

主要 exact match 使用 `(normalized_file_path, symbol_kind, qualified_name)`；若 gold 與 prediction 都有可信 `symbol_id`，優先以 `symbol_id` 比對。只比 symbol leaf name 或 suffix 的結果列為 relaxed diagnostic，不可當主要成果。

多個正確 symbols 必須全部保留。`<module>` 是合法 symbol kind，但與任何 function/class 不互相匹配。

v1 的明確行為如下：

- gold 與 prediction 同時具有 `sha256:<64 hex>` 的可信 ID 時，ID 是 authoritative；ID 不同時不得回退成 tuple exact。
- 任一側沒有可信 ID 時，只有 normalized file、kind、qualified name 三者全等才是 exact。
- relaxed diagnostic 依序檢查 qualified name 全等、dot-boundary suffix、leaf name；它刻意保留舊版忽略 file/kind 的寬鬆行為，以量化「名稱看似正確但定位錯檔或錯 scope」的差距。
- `<module>`／`module` schema invariant 在輸入邊界強制執行，module 不會與非 module 互相匹配。

### 7.2 指標與分母

| 指標 | 定義 | 分母／用途 |
|---|---|---|
| File Hit@1/3/5 | 前 K 個檔案是否含任一正確檔案 | 所有具有 file gold 的 Ticket；沿用 Stage-2 報告 |
| File Recall@5 | Top-5 找回的正確檔案比例 | 多檔修正 Ticket 不可只看 Hit |
| File MRR | 第一個正確檔案 reciprocal rank 平均 | 沿用 Stage-2 報告 |
| Candidate Recall@20 | Stage-1 Top-20 找回正確檔案的比例 | 第一階段候選上限 |
| Symbol Candidate Hit/Recall@10/30 | LLM 前的 symbol pool 是否含正確 symbol／找回多少比例 | 只在 gold file 已進 Stage-2 Top-5 的 eligible Ticket 計算 |
| Exact Symbol Hit@1/3/5 | 最終 Top-K 是否含任一 exact gold symbol | 所有 symbol-gold Ticket；Stage-2 miss 計為 0 |
| Conditional Exact Symbol Hit@1/3/5 | 已找到至少一個正確檔案時，Top-K 是否命中 exact symbol | 隔離 Symbol 階段本身能力 |
| Symbol Recall@1/3/5 | Top-K 找回的 gold symbols 比例之平均 | 評估多-symbol Ticket |
| Symbol MRR | 第一個 exact gold symbol reciprocal rank 平均 | 同時回報 end-to-end 與 conditional |
| Coverage／Fallback | LLM used、valid、timeout、fallback 的比例 | 防止只報成功子集 |

缺少 prediction 的 Ticket 仍留在 end-to-end 分母；conditional 指標必須同時回報 eligible row count。所有比例附 paired bootstrap 95% CI，並列出 improved／unchanged／worsened／both-miss Ticket IDs。

Evaluator v1 的 Candidate Hit/Recall@10/30 只使用「至少一個 gold file 已進 Stage-2 Top-5」的 conditional rows。prediction/final output coverage 以所有 symbol-gold Tickets 為分母；candidate output coverage 以 conditional rows 為分母；LLM attempted/valid/fallback/timeout rate 以 `requested=true` 且 `eligible=true` 的 rows 為分母。paired comparison 固定比較同一 prediction record 內的 `stage3_retrieval_symbols` 與 `stage3_ranked_symbols`，對 Hit@1/3/5、Recall@1/3/5、MRR 的 per-ticket delta 執行固定種子 2,000 次 bootstrap，並保存各 Hit@K 與 first-rank outcomes 的 Ticket IDs。

---

## 8. 五個 Work Packages

### WP1：資料契約與 AST 正確性（2 工作日）

**實作：** 建立 `SymbolRecordV1`、`SymbolParseResult`、完整 scope stack、stable ID、column range、commit provenance 與 per-file diagnostics；舊 API 保留相容 wrapper。

**驗收：** nested function 不誤標 method；同名 symbol ID 不同；SyntaxError、no_symbols、unsupported、missing、invalid_path 可區分；相關 Unit Tests 全通過。

**目前狀態（2026-09-02）：已完成。** `SymbolRecordV1`、`SymbolParseResult v2`、CodeIndex v5、完整 scope、stable ID、column range、commit provenance、multiline signature，以及 SyntaxError／RecursionError／no_symbols／unsupported／missing_file／invalid_path／symlink_escape diagnostics 均已有測試；完整回歸 161/161 通過。

### WP2：Symbol Gold 與 Evaluator（3 工作日）

**實作：** 建立 patch-to-symbol mapper、人工抽查表與 file-qualified evaluator；新增 Candidate、Conditional、Recall、Coverage 與 paired outcomes。

**驗收：** 50 筆人工抽查至少 95% exact；provenance 100%；evaluator 對缺失 prediction、multi-symbol、`<module>` 與 relaxed match 有獨立測試。

**目前狀態（2026-09-02）：已完成。** Gold 與 audit 已包含 `SymbolGoldRecordV1`、`PatchSymbolProvenanceV1`、stable gold ID、mapped／excluded invariants、unified diff parser、mapper、正式 build/audit CLI，以及獨立 patch/base-source/AST verifier。固定種子 50 筆已由 Codex 逐筆執行 source audit，exact 50/50（100%），provenance 50/50（100%），validator 的 G1 已通過；若研究 protocol 將「人工」限定為外部人類 reviewer，仍需另行 sign-off。Evaluator 已完成 `SymbolEvaluationItemV1`、ID-first/file-qualified exact、獨立 relaxed diagnostic、per-ticket multi-gold 聚合、missing prediction、end-to-end/conditional Hit/Recall/MRR、Candidate Hit/Recall@10/30、coverage/fallback、paired outcomes 與固定種子 bootstrap 95% CI。完整回歸 170/170 通過。

### WP3：Deterministic Candidate Baseline（2 工作日）

**實作：** 將 symbol localization 與 LLM 開關拆開；保存 Top-30 candidate pool；比較 B0/B1、per-file quota 與 Top-10 shortlist。

**驗收：** eligible development data 的 Exact Symbol Candidate Recall@30 理想目標仍為90%。依2026-09-08方向二決策，68.64%的`coverage-aware-v1`凍結為目前上限，未達90%不再阻塞探索性WP4；正式報告必須揭露21.36個百分點缺口及其分母。

**目前狀態（2026-09-07）：進行中，coverage-aware-v1 已完成，實際 G2 仍未通過。** `symbol_localization` 與 `symbol_llm_rerank` 核心/API/CLI 開關已拆分；deterministic ranking 已加入 B0 TF-IDF、B1 structured evidence、quota 0／4／6／8、module-reserved 與 coverage-aware-v1 的 frozen development 設定。Stage-3 將 module-level gap 正式表示為 `<module>`，並為只有 generic chunk 的舊 index 自動補一個 module candidate，不必重建 index。coverage-aware-v1 保留強 global prefix、為每個 Stage-2 檔案保留一個 module candidate，並從 prefix 已提供證據的 class family 補入一個 member。61 張 development Tickets 共得到 46 張 Stage-2 conditional-eligible Tickets；B0 Candidate Recall@30 為 60.60%，global B1 為 64.36%，選定的 coverage-aware-v1 為 68.64%，相較 global B1 增加 4.28 個百分點；quota 4／6／8 分別為 61.34%／53.35%／56.24%。AST pool exact oracle 維持 90.41%，但實際 Top-30 仍未達 90%，因此不得開始 WP4 LLM 實驗。coverage-aware-v1 從原本 37 個 pool-reachable ranking misses 找回 7 個，使 conditional Top-30 exact hit 由 81 增至 88、ranking miss 降至 30；另有 Stage-2 file miss 17 與 identity mismatch 2。結果保存在 `reports/fault_localization/stage3_deterministic_g2_dev_v1/`，完整回歸 183/183 通過。下一步仍停留 WP3，針對剩餘 30 個 ranking misses 執行 frozen source-neighborhood expansion ablation，不使用 gold 特徵微調，也不開始 LLM 實驗。

**WP3 後續實驗更新（2026-09-07）：** frozen source-neighborhood-v1 已完成，設定為最多 5 個擴展名額、同檔互不重疊且距離不超過 100 行的最近定義。61 筆 development tickets 全量比較得到 Recall@30 64.70%，低於 coverage-aware-v1 的 68.64%；逐筆配對找回原 30 個 misses 中的 1 個，但失去原有 7 個命中，因此不採用。選定 baseline 仍為 coverage-aware-v1，仍有 30 個 ranking misses；實驗 variant 則有 36 個。完整測試 184/184 通過。設定與結果分別保存在 `configs/fault_localization/stage3_source_neighborhood_v1.json` 與 `reports/fault_localization/stage3_source_neighborhood_dev_v1/`。此更新取代上一段尚待執行 source-neighborhood 的下一步；接下來在 WP3 固定 ticket-to-symbol 呼叫關係擴展假設與設定，再做 development 比較。G2 未通過，WP4 尚未開始。

**WP3 呼叫擴展、可達性與無邊審核更新（2026-09-07）：** call-neighborhood-v1 已固定設定、實作並完成 61 筆 development 比較。Recall@30 為 66.02%，找回原 30 個 misses 中的 2 個、失去 6 個既有命中，因此未採用；選定 coverage-aware-v1 維持 68.64% 與 30 個 ranking misses。可達性診斷顯示 3 個 misses 可從 prefix 一跳到達、5 個只連到 prefix 外候選、22 個在重建後的池內圖沒有已解析 incident edge。22 個無邊案例已全部審核：base source 與目標定義 22/22 存在；完整 repository index graph 有 21 個 incident-edge 案例，其中 20 個的相鄰 symbol 也在 candidate pool。主因為同檔 source／graph reconstruction loss 15、跨檔 import resolution loss 5、Stage-2 pool scope 1、dynamic dispatch 1、確實無靜態呼叫證據 0。這證明 v1 的主要限制來自用 symbol-only pool 重建呼叫圖，而非 base source 缺失。187/187 測試通過；逐筆證據位於 `reports/fault_localization/stage3_call_neighborhood_dev_v1/NO_EDGE_AUDIT_ZH.md` 與 `no_edge_case_audit.csv`。下一步建立 frozen call-neighborhood-v2，直接使用 index 既有 repository-level call graph，並以 candidate identities 過濾；G2 未通過，WP4 尚未開始。

**WP3 call-neighborhood-v2 更新（2026-09-08）：** v2 已直接使用 `CodeIndex.call_graph` 的 repository-level 呼叫圖，並以 Stage-2 Top-5 candidate identities 過濾兩端；舊 index 缺少呼叫圖時才從完整 index 重建。61 筆 development 全量重跑完成，Recall@30 為 68.20%，比 call-neighborhood-v1 的 66.02% 增加 2.18 個百分點，證實完整索引圖能修復主要 graph reconstruction 缺邊。相較選定的 coverage-aware-v1，v2 找回原 30 個 misses 中的 4 個，但失去 5 個既有命中，exact hits 為 87 對 88，ranking misses 為 31 對 30；主要指標低 0.44 個百分點，因此不採用，baseline 維持 68.64%。188/188 測試通過；設定與證據位於 `configs/fault_localization/stage3_call_neighborhood_v2.json` 與 `reports/fault_localization/stage3_call_neighborhood_dev_v2/EXPERIMENT_ZH.md`。下一步逐筆分析 4 個 recoveries 與 5 個 losses，固定能保留 class-family coverage 的 guarded call expansion 規則；G2 未通過，WP4 尚未開始。

**WP3停損決定（2026-09-08）：** 前述call-neighborhood-v2的「guarded call expansion」下一步取消，不再進行第四次局部補救。`coverage-aware-v1`以68.64%凍結，WP3標記為「工程完成、研究門檻未達」，後續改由WP4量測LLM在可達候選上的排序效果。

### WP4：Code Llama 配對實驗（2 工作日）

**實作：** opaque candidate IDs、單一 Top-10 global prompt、JSON Schema、score-blind prompt、fallback、timings 與 run manifest；先 10-ticket smoke，再 50-ticket pilot。

**驗收：** LLM valid coverage 至少 95%、fallback 不高於 5%、沒有 score leakage；同一批 tickets 與相同 candidate pool 產生 B1／L1-only／L1-blend 配對結果。

**調整後執行方式（2026-09-08）：** 立即使用凍結的`coverage-aware-v1`開始10-ticket smoke，成功後執行固定50-ticket pilot。WP4最多2個工作日；到期即依G3與探索性配對結果決定「保留retrieval」或「LLM進入後續研究」，不因結果不理想延長prompt微調。所有報告在標題與結論標記`exploratory_due_to_G2_not_met`。

### WP5：正式比較、凍結與文件（1–2 工作日）

**實作：** 專題最低交付為完成固定50-ticket pilot、做出LLM採用決策、更新README、model spec與限制報告；若不影響後續補丁生成時程，才在至少200個symbol-eligible tickets、至少5個repositories上完成正式validation與frozen holdout。

**驗收：** 所有schema、tests、commands、manifest與結果可重現；未達G2或G4時不得宣稱正式Symbol準確率提升，但不再阻塞後續補丁生成、測試案例與提交訊息流程。

---

## 9. 採用與停止門檻

### 9.1 可以進入下一 WP 的條件

| Gate | 條件 | 未通過時 |
|---|---|---|
| G1 Gold | 50 筆抽查 exact ≥95%、provenance 100% | 修 mapper，不跑 LLM 正式實驗 |
| G2 Candidate | 理想目標≥90%；目前凍結68.64%，方向二允許探索性WP4 | 明列候選上限；不得宣稱正式提升，不再阻塞WP4 |
| G3 Reliability | LLM valid coverage ≥95%、fallback ≤5% | 修 schema／budget／timeout，不比較準確率 |
| G4 Evidence | ≥200 eligible tickets、≥5 repos；paired 95% CI | 未達者只標為 exploratory |
| G5 Adoption | LLM 的 Conditional Exact Symbol Hit@5 點估計高於 B1，95% CI 下界不低於 0；end-to-end Hit@5 不下降；p95 Symbol LLM latency ≤60 秒／ticket | 不設為預設，維持 B1 fallback |

G2的調整只改變專題時程與WP4啟動資格，不改變指標定義，也不把68.64%改寫成通過90%。
若後續有新的deterministic方法或資料，可以用新實驗ID重啟G2；目前development set不再
用於第四次局部規則調整。

### 9.2 明確停止條件

WP3已觸發停損：source-neighborhood、call-neighborhood-v1與call-neighborhood-v2三個凍結補救均未超越68.64%基準，因此本輪不再新增deterministic局部規則。

WP4若連續兩個凍結prompt版本都無法在同一development set改善Conditional Exact Symbol Hit@5，或2個工作日用盡，停止繼續prompt微調。記錄負結果，正式流程採B1；後續只有在更強模型、更多gold或不同reranking方法可用時才重啟。

---

## 10. Testing 計畫

### 10.1 Unit Tests

- AST：function、async function、method、nested class/function、decorator、多行 signature、同名 symbols。
- Diagnostics：SyntaxError、RecursionError、empty/no-symbol、unsupported、missing、path traversal、symlink escape。
- Gold mapper：修改、刪除、純新增、module-level、多 symbol hunk、新檔案與 rename。
- Evaluator：exact file-qualified、symbol ID、relaxed diagnostic、multi-gold、missing prediction、conditional denominator。
- LLM parser：duplicate、missing、unknown ID、NaN、超界、malformed JSON、timeout 與完整 fallback。

### 10.2 Integration Tests

- Top-20 → Stage-2 baseline Top-5 → Symbol Top-30 → baseline Top-5 的無 LLM pipeline。
- 相同 candidate pool 的 B1 與 L1 paired comparison。
- portable index 的 repository path 不影響 symbol 讀取。
- 一個 Stage-2 檔案 parse failure 時保留其他檔案，總狀態為 `partial`。
- JSONL resume、manifest、schema validation 與 evaluator 端到端一致。

### 10.3 真模型執行順序

1. 1 個 Ticket 驗證 prompt、schema、輸出與 fallback。
2. 固定 10 個 Ticket 作 smoke test；只判斷流程與 coverage。
3. 固定 50 個 Ticket 作 pilot；可調 development 參數，不作最終主張。
4. 固定50個Ticket作探索性pilot並做專題採用決策；未達G2／G4時不宣稱正式提升。
5. 只有在不壓縮後續補丁生成時程且有至少200個eligible Tickets時，才執行正式validation與一次frozen holdout。

---

## 11. 預計檔案變更

### 新增

- `src/utils/symbol_localization.py`：SymbolRecord、parse result、candidate scoring 與 LLM reranker。
- `scripts/build_symbol_gold.py`：developer patch → base-commit symbol gold。
- `scripts/run_stage3_paired_comparison.py`：B0/B1/L1-only/L1-blend 配對實驗。
- `configs/fault_localization/stage3_symbol_reranker_v1.json`：凍結參數與 prompt version。
- `tests/test_symbol_localization.py`、`tests/test_build_symbol_gold.py`、`tests/test_stage3_evaluation.py`。

### 修改

- `src/utils/fault_localization.py`：保留相容 wrapper，接入新模組並拆分 baseline／LLM 開關。
- `scripts/fault_localization.py`：新增 `--symbol-localization` 與 `--symbol-llm-rerank`，舊 `--symbol-rerank` 暫時保留 deprecation alias。
- `scripts/evaluate_fault_localization.py`：新增 file-qualified exact、conditional、recall、coverage 與 paired CI。
- `README.md`、`FAULT_LOCALIZATION_MODEL_SPEC.md`：更新第三階段正式介面、指標與限制。
- `reports/fault_localization/`：保存 gold audit、pilot、validation、holdout 與 final decision。

---

## 12. 預定命令介面

下列命令是第三階段完成後應可執行的正式介面；在 WP1–WP4 尚未完成前，不把它們視為現有功能。

```bash
python scripts/build_symbol_gold.py \
  --tickets data/fault_localization/tickets.jsonl \
  --repo-cache-dir data/repositories \
  --output data/fault_localization/symbol_gold_v1.jsonl \
  --audit-csv reports/fault_localization/stage3_symbol_gold_audit.csv
```

```bash
python scripts/audit_symbol_gold.py sample \
  --input reports/fault_localization/stage3_symbol_gold_audit.csv \
  --output reports/fault_localization/stage3_symbol_gold_audit_sample_50.csv \
  --summary reports/fault_localization/stage3_symbol_gold_audit_summary.json

# 人工填完 review_status 與三個 *_correct 欄位後重跑 gate
python scripts/audit_symbol_gold.py validate \
  --input reports/fault_localization/stage3_symbol_gold_audit_sample_50.csv \
  --summary reports/fault_localization/stage3_symbol_gold_audit_summary.json
```

```bash
python scripts/fault_localization.py \
  --tickets-jsonl data/fault_localization/stage3_validation_tickets.jsonl \
  --candidate-file-k 20 \
  --top-k 5 \
  --symbol-localization \
  --symbol-candidate-k 30 \
  --symbol-llm-shortlist-k 10 \
  --symbol-top-k 5 \
  --output reports/fault_localization/stage3_validation/predictions.jsonl
```

```bash
python scripts/run_stage3_paired_comparison.py \
  --tickets data/fault_localization/stage3_validation_tickets.jsonl \
  --gold data/fault_localization/symbol_gold_v1.jsonl \
  --config configs/fault_localization/stage3_symbol_reranker_v1.json \
  --model codellama:7b-instruct \
  --output-dir reports/fault_localization/stage3_validation_paired
```

---

## 13. 風險與處理

| 風險 | 早期訊號 | 處理方式 |
|---|---|---|
| Gold 映射錯誤 | module-level 或純新增案例錯誤率高 | 分層抽查；保留 mapping method/confidence；不確定者排除 exact 主指標 |
| 正確檔案未進 Top-5 | end-to-end 明顯低於 conditional | 同時回報兩者；第三階段不回頭改 Stage-2，以免混淆責任 |
| 大檔案淹沒 candidate pool | Top-30 集中於單一檔案 | per-file quota + global merge；以 Candidate Recall@30 決定 |
| Code Llama 跨候選判斷不穩 | coverage 低、重跑排名變動 | temperature 0、固定 prompt/version、單一 Top-10 global call、完整 schema |
| LLM 成本／延遲過高 | p95 超過 60 秒或 timeout 增加 | 保留 deterministic default；LLM 只在人工模式或高價值 Ticket 啟用 |

---

## 14. 完成定義（Definition of Done）

第三階段的「完整研究驗收」只有在以下五項全部完成後才可標記完成：

1. `SymbolRecordV1`、parse diagnostics、path/commit safety 與相關測試通過。
2. Symbol gold 經 50 筆人工抽查達 95%，且 formal evaluator 使用 file-qualified exact match。
3. deterministic baseline 可獨立執行並保存 Top-30 candidates、Top-5 output 與完整 diagnostics。
4. Code Llama 配對實驗使用相同候選池、無 score leakage、coverage 達門檻，結果含 CI 與失敗案例。
5. 依 G5 做出「預設啟用」或「維持 retrieval」的明確決策，並更新 README、模型規格與最終報告。

專題的「流程實作完成」採較窄定義：WP1／WP2通過、WP3 deterministic baseline可執行、
WP4完成固定smoke與pilot、做出LLM採用決策並如實揭露G2未達。達成此層級後即可往
補丁生成、測試案例與提交訊息前進，但不得將Stage3標記為已通過完整研究驗收。

---

## 15. 目前進度與下一個任務

WP1 已完成；WP2 的 schema、parser、mapper、build/audit CLI、獨立 source verifier 與 G1 Gold gate 已完成。audit CLI 以固定種子 `20260902` 抽取 50 筆，要求至少 5 個 repositories，預設最低涵蓋 nested symbol 10、`<module>` 5、純新增 5、刪除證據 5、多檔修改 10；同一筆可同時屬於多個 strata。完整回歸 161/161 通過。

2026-09-02 已從 development set 建立固定種子 source subset：61 tickets、12 repositories、61/61 base commits 可讀。正式 Symbol Gold 共 207 筆，其中 205 mapped、2 excluded；輸出位於 `data/fault_localization/swebench_full/stage3_symbol_gold/symbol_gold_v1.jsonl` 與 `reports/fault_localization/stage3_symbol_gold/stage3_symbol_gold_audit.csv`。

50 筆 sample 已產生於 `reports/fault_localization/stage3_symbol_gold/stage3_symbol_gold_audit_sample_50.csv`。composition gate 通過：12 repositories、nested 23、`<module>` 10、純新增 12、刪除證據 28、多檔修改 23。每筆均已由 Codex 對照 developer patch hunk、base-commit source 與獨立 AST scope；逐筆 exact 50/50（100%），provenance 50/50（100%），validator 的 G1 `overall_gate_status=passed`。完整 evidence 保存於 `reports/fault_localization/stage3_symbol_gold/stage3_symbol_gold_audit_evidence.md`；若研究 protocol 要求外部人類 reviewer，需在正式發表前另行 sign-off。

WP2 已完成。file-qualified evaluator 的正式 exact 採 trusted ID 優先、否則比對 `(normalized_file_path, symbol_kind, qualified_name)`；relaxed 僅作診斷。Evaluator 會把同 Ticket 的多筆 gold 聚合，missing prediction 保留在 end-to-end 分母，並輸出 exact/relaxed Hit@1/3/5、Recall@1/3/5、MRR、conditional exact、Candidate Hit/Recall@10/30、coverage/fallback，以及 retrieval-vs-final paired outcomes/bootstrap CI。Stage-3 prediction 現在也會保存完整 `stage3_candidate_symbols` 與明確 LLM eligibility/validity/fallback diagnostics；完整回歸 170/170 通過。

WP3 的第一個任務已完成：deterministic symbol localization 與 Symbol LLM rerank 已拆成獨立開關。`--symbol-localization` 可在沒有 Ollama 的情況下固定產生 Top-30 candidate pool 與 Top-5 retrieval/final output；`--symbol-llm-rerank` 才會建立並呼叫 LLM client，舊 `--symbol-rerank` 保留相容性。

完整回歸 172/172 通過。

2026-09-09最新狀態：WP3已完成B0/B1、quota、coverage-aware、source-neighborhood與兩版call-neighborhood比較。選定`coverage-aware-v1`為68.64%基準；G2的90%理想目標未達，三個最近補救均未改善，因此觸發停損，不再執行guarded call expansion。WP4已完成1票、10票與46票的探索性pilot，結果不支持採用Symbol LLM Reranker；正式預設維持deterministic baseline。完整WP4研究驗收仍未完成。

## 16. 2026-09-08 決策：G2 未過關但帶著限制啟動 WP4 Pilot

**背景：** WP3 已對 deterministic candidate pool 做了多輪消融（source-neighborhood-v1、call-neighborhood-v1、call-neighborhood-v2），三次都低於目前選定的 coverage-aware-v1（68.64%），呈現報酬遞減。distance 90% 的 G2 門檻還有約 21 個百分點的落差，短期內用同類型的鄰域規則微調不太可能一次補齊。同時，專題排程規定步驟 3（錯誤定位＋補丁生成與驗證）僅到 115 年 12 月，後面還有提交訊息生成與系統測試／優化要在 116 年 2 月前完成。

**決策：** 比照 Stage 1 當年處理 Final Holdout Recall@20 只有 67.17% 的做法——不再把 G2 當成硬性阻擋，而是把 68.64% 的候選召回率當作**已知限制**明確寫進報告，先啟動 WP4 Pilot，讓 Symbol LLM 排序的可行性與 end-to-end 表現先有真實數據，再決定要不要回頭繼續衝 G2。這不是撤銷 G2 的正式門檻定義（9.1 節維持不變），而是專題排程下的務實選擇，最終正式採用與否仍需完整跑過 G4／G5 驗證。

**與既有雛形的差異：** 現有 `--symbol-llm-rerank`（`rerank_symbols_with_llm` / `_rerank_symbol_batch`）把 Top-30 拆成每批 5 個、用同批內的本地 rank 當 ID，且跨批分數直接混合排序——這正是第 0 節第 5 點指出、尚未修正的已知問題。WP4 Pilot 改用新增的獨立腳本，不修改既有雛形（避免影響現有 172 項相關回歸測試），採用計畫書 6.3 節設計：

- 直接沿用 WP3 已凍結的 `b1_coverage_aware_v1` Top-30 pool（免重算 retrieval）。
- 從 Top-30 取 Top-10 shortlist，**單一 Ollama 呼叫**完成排序，不再分批。
- candidate_id 改為**洗牌過的 opaque ID**（`C1`…`C10`，用固定種子重新排列，不等於原始 rank），prompt 只包含 file_path、symbol_kind、qualified_name、行號與程式碼片段，不包含 retrieval score、Stage-2 file score 或原始 rank。
- 缺一筆、重複 ID、未知 ID、NaN／超界分數、非 JSON，一律視為整票 LLM failure，並清楚回退到 deterministic baseline 順序（開發過程中的單元測試曾抓到一個真的 bug：若疏忽，fallback 會誤用洗牌後的 prompt 順序而不是原始 baseline 順序，已修正並補了回歸測試）。

**新增產物：**
- `scripts/run_stage3_wp4_symbol_llm_pilot.py`：WP4 Pilot 主程式（單一 Top-10 呼叫、opaque ID、no-score-leakage、deterministic fallback、run manifest、per-ticket 與彙總輸出）。
- `tests/test_stage3_wp4_symbol_llm_pilot.py`：16 個單元／整合測試，涵蓋 opaque ID 洗牌與決定性、prompt 不洩漏 retrieval 欄位、缺漏／重複／未知 ID／NaN／超界分數的 fallback、以及 eligibility 判斷（`stage2_localized_files` 為 dict 陣列的實際 schema）。
- 完整回歸：204/204 通過（188 既有＋16 新增），未修改任何既有檔案的行為。
- 已用真實 development 資料（不呼叫 Ollama）驗證資料串接：61 筆 development tickets 中 46 筆為 eligible（與既有文件一致），index／chunk 對照與 bug_report 讀取均正確；並在無法連線 Ollama 的環境下驗證了完整 CLI 會正確 fallback 且不中斷。

**尚未完成（需要在有真實 Ollama 存取權的機器上執行）：**
1. 1 個 Ticket 的真實 smoke test（驗證 prompt、schema、真實模型輸出與 fallback）。
2. 若 1 票成功，擴大到 46 個 eligible development tickets 的 pilot，得到 baseline vs LLM 的 Exact Hit@1/3/5 與 fallback rate。
3. 依 G5 的方向（點估計是否優於 baseline、95% CI、p95 latency）整理結果，供教授報告使用；本輪結果仍屬 pilot 等級，不構成正式 G4／G5 判定。

下一個任務：在裝有 `codellama:7b-instruct` 的機器上執行 1-ticket smoke test（見任務清單指令），確認真實模型輸出可被排程與解析，再決定是否擴大到 46 票。

## 17. 2026-09-08 WP4 Pilot 探索性結果（46 張 development eligible tickets）

在使用者本機的真實 `codellama:7b-instruct`（Ollama）上完成三階段執行：1 票 smoke → 10 票 smoke → 46 票（全部 Stage-2 conditional-eligible development tickets）正式 pilot。腳本與方法見第 16 節；配對統計由 `scripts/analyze_stage3_wp4_pilot.py` 產生（2,000 次 resample、固定種子 42 的 percentile bootstrap，沿用 `scripts/evaluate_fault_localization.py::_bootstrap_mean_ci` 既有慣例）。

### G3 可靠度

`llm_valid_count=42/46`，`llm_fallback_count=4`，fallback rate **8.70%**，略高於計畫書 G3 門檻（≤5%）。四筆 fallback（`invalid_or_incomplete_output`）為 `pallets__flask-4544`、`pytest-dev__pytest-10893`、`scikit-learn__scikit-learn-11042`、`sympy__sympy-13177`；原始 payload 尚未逐筆歸因是 schema 違反哪一條規則，留待需要時再查。

### 準確率（Exact Hit@K，baseline＝B1 coverage-aware-v1 deterministic Top-5，LLM＝WP4 單次 Top-10 shortlist rerank）

| K | Baseline | LLM | Mean Δ | 95% CI | 判讀 |
|---|---:|---:|---:|---:|---|
| 1 | 26.09% | 10.87% | -15.22 pp | [-28.26, -2.17] pp | **CI 不含 0，LLM 顯著更差** |
| 3 | 47.83% | 41.30% | -6.52 pp | [-21.74, +8.70] pp | 方向為負，CI 含 0，不顯著 |
| 5 | 60.87% | 58.70% | -2.17 pp | [-15.22, +10.87] pp | 方向為負，CI 含 0，不顯著 |

逐票配對結果（improved／unchanged-hit／worsened／both-miss）：Hit@1 為 2／3／9／32；Hit@3 為 5／14／8／19；Hit@5 為 4／23／5／14。三個 K 值全部呈現「worsened 多於 improved」。

### 判讀與正式建議

結果方向與 Stage 2 File Reranker 完全一致：模型呼叫可靠（G3 接近但未過），但**沒有任何 K 值顯示 LLM 排序優於 deterministic baseline**，Hit@1 甚至是統計上顯著更差（95% CI 完全落在 0 以下）。依 G5 判準（LLM 點估計需高於 B1 且 CI 下界不低於 0），本輪 pilot 明確不通過。

**目前工程決策：Symbol LLM Reranker 不採用，Stage 3 正式輸出維持 B1 deterministic baseline（coverage-aware-v1）**，與 Stage 2 對 `codellama:7b-instruct` File Reranker 的決定一致。此決策足以凍結目前預設流程，但不代表 WP4 已通過完整研究驗收。

**重要限制（务必在報告中一併陳述，避免誤讀為模型能力的最終定論）：**
1. 樣本為 46 筆 development-only tickets、12 個 repositories，未達計畫書 G4 門檻（≥200 tickets、獨立 repo-disjoint holdout），屬 pilot／exploratory 等級，不是正式 G4 驗證。
2. 這 46 筆的 deterministic 候選池本身 Conditional Exact Candidate Recall@30 只有 68.64%（G2 未過），也就是說即使 LLM 排序能力再好，能被排到前 5 名的正確答案本來就有上限；本輪結果只能回答「在目前候選池品質下，LLM 排序是否有幫助」，不能回答「LLM 排序能力本身的上限」。
3. 4 筆 fallback 尚未逐筆歸因具體違反 schema 的原因，若要精確報告 G3 是否可透過 prompt 微調過關，需要再看原始回應。
4. 本輪只比較 B1 baseline 與 L1-only，尚未完成計畫中的 L1-blend；產物也未提供可驗證 G5 p95 ≤60 秒的逐票延遲，以及完整的模型 digest、prompt version 等重現資訊。

下一個任務：把本輪探索性負結果與限制納入第三階段總結，正式流程維持`coverage-aware-v1`並接續補丁生成；只有在決定補做完整研究驗收時，才新增L1-blend、延遲量測與≥200 tickets的validation。
