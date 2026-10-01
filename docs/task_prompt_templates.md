# Task prompt templates

複製與任務最接近的一個模板，填入 `<...>`。只讀 Relevant Documentation 列出的文件；跨 boundary 才追加相鄰模組文件。這些是可裁剪的任務契約，不是逐步 SOP；模板沒有授權修改 gold、模型或研究方法。

## 1. Extraction／normalization

**Goal**：改善 `<具體輸入問題>`，使 `<raw ticket / predicted_json>` 能產生 `<預期欄位結果>`。

**Scope**：`<ToJson/scripts/To_Json 或 MAIN TicketExtractor/json_schema>`；明示修改獨立抽取或主系統 adapter。

**Relevant Documentation**：[extraction_normalization](extraction_normalization.md)；涉及欄位評分再讀 [evaluation](evaluation.md)。

**Constraints**：保持既有欄位／下游契約；不得更換核心模型、改 labeled data；区分 normalization、LLM extraction 與 source consistency。

**Expected Output**：最小修改、可執行輸入輸出範例、缺值／fallback 行為及同步文件。

**Done Criteria**：實際走通指定入口，驗證 raw／predicted_json／缺值與下游相容性；相關測試通過；未跑真實模型則該能力標記 Not verified；符合 [共通驗收](done_criteria.md)。

## 2. Ticket classification／duplicate retrieval

**Goal**：修正 `<priority／duplicate 的明確行為>`。

**Scope**：`<MAIN baseline / DUP research / PRI feature-model pipeline>`；列明 affected input/output 與不包含的訓練／整合工作。

**Relevant Documentation**：[ticket_classification](ticket_classification.md)；涉及 split／metrics 再讀 [datasets](datasets.md)、[evaluation](evaluation.md)。

**Constraints**：不把研究模型當成 default；維持 duplicate review／short-circuit；不改 gold、holdout 或 metric 定義；不以保留輸入 priority 的結果冒充預測 accuracy。

**Expected Output**：可重現案例、最小實作、gates／ranking／fallback 的影響，以及所用模型與入口的清楚標示。

**Done Criteria**：驗證正例／負例／review／缺失歷史或 P1–P5／fallback；下游 contract 與相關 regression 通過；真實模型或訓練未驗證時分開列 Not verified。

## 3. Assignee routing／feedback

**Goal**：完成 `<候選推薦／人工選擇／時間安全／部署 gate 的單一能力>`。

**Scope**：`<MAIN hybrid / LTR / rolling / phase9 protocol>`，明示使用的 artifacts 與實驗 ID。

**Relevant Documentation**：[assignee_routing](assignee_routing.md)；只追加該 phase 的 protocol 文件。

**Constraints**：保留 roster／calibration／open-set fail-closed；不捏造 live feedback／組織簽核；protected holdout 不調參；不因離線 accuracy 高而核准部署。

**Expected Output**：候選／routing／feedback JSON 或 protocol artifact 的變更、時間與來源證據、失敗 fallback。

**Done Criteria**：as-of history、inactive／missing artifact、人工確認、feedback provenance 測試符合任務；缺少 live shadow 或正式審核時部署仍未完成，不能宣稱 production ready。

## 4. Fault localization

**Goal**：修正 `<index／file ranking／symbol ranking／handoff 的明確問題>`。

**Scope**：`<MAIN / FL / REPAIR>` 的 `<真實入口>`，限定要改的 stage。

**Relevant Documentation**：[fault_localization](fault_localization.md)；影響 handoff 再讀 [patch_generation](patch_generation.md)；評分再讀 [evaluation](evaluation.md)。

**Constraints**：不改研究方法、selected default、gold 或 evaluation definition；維持 source/commit/hash provenance；不以 ground truth 補預測 symbol。

**Expected Output**：ticket + snapshot → predicted files/symbols 的可重現結果、confidence/fallback、最小修改與測試。

**Done Criteria**：驗證正確 snapshot、空結果、source drift、不同 file/symbol 欄位語意；若有 patch consumer，驗其 handoff；mock／單案不外推正式 Recall/MRR。

## 5. FIM／repair integration

**Goal**：完成 `<指定 generation／validation／接合缺口>`。

**Scope**：`<MAIN opt-in attach_fim / 外層 FIM / REPAIR single CLI / REPAIR batch>`，指定目標入口與 output 目錄。

**Relevant Documentation**：[patch_generation](patch_generation.md)、[evaluation](evaluation.md)；只有改 FL contract 才追加 [fault_localization](fault_localization.md)。

**Constraints**：localization 只能來自目前 FL 預測；禁止 Oracle／Ground-truth → FIM；保留 confidence、base/hash、AST／scope gates；不更換核心模型；reference 只在生成後隔離評估；不覆寫原始 source 或 frozen runs。

**Expected Output**：input → localization → candidate diff → validation 的可追蹤 artifacts、選擇／拒絕原因、所用 backend 與執行結果。

**Done Criteria**：相關 handoff／generation／CLI tests 通過；真實 I/O 達到此次要求的驗證層級；generated／plausible／official resolved 分開；runtime 未驗證不得宣稱整體 repair Done。

## 6. Dataset／evaluation audit 或 documentation

**Goal**：釐清 `<schema／provenance／分母／文件落差>`，交付 `<audit report／文件修改>`。

**Scope**：`<指定資料版本、evaluator、文件>`；預設 read-only，不建新 dataset／重跑 frozen holdout。

**Relevant Documentation**：[datasets](datasets.md)、[evaluation](evaluation.md)；修改說明時僅追加相關模組文件。

**Constraints**：不改 ground truth／split／evaluation definition；不讀 sealed labels 作方法選擇；歷史數字與本次重跑分開；文件不把 planned feature 改稱 Implemented。

**Expected Output**：可定位至程式／schema／artifact 的差異、局部檢查結果、修正文案、Not verified 清單。

**Done Criteria**：routing 與連結有效、來源與結論一致、文件範圍符合授權；確認沒有非文件修改；未知項保持未知。套用 [documentation-only criteria](done_criteria.md)。
