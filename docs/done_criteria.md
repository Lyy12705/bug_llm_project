# Done criteria

## 共通完成條件

任務開始時把 Goal、Scope 與所需驗證層級寫清楚。以下條件要有實際證據，不以「程式已寫完」代替。

1. **契約一致**：實際入口、預設設定、input/output、fallback 與文件一致；不同 checkout／研究版本已標明。
2. **可執行且 I/O 走通**：使用指定環境走通任務所需 Input → Output，包含適當失敗分支；報告 cwd、command、backend／artifact 身分。
3. **相關測試與 regression**：執行與變更相關的既有測試；必要時補行為測試，不以重寫實作的測試代替驗證。既有 failure 和此次引入的 failure 分開。
4. **研究／資料邊界保持**：未越權改 dataset/gold/holdout/metrics/core model；FL 預測是唯一 patch localization input；frozen artifact 不被覆寫。
5. **結論與證據相符**：unknown／未執行／blocked 標 `Not verified`；列明原因、影響與下一個驗證動作。未滿足目標驗證層級的功能不能宣稱 Done。

## 各 domain 必須補充的證據

| Domain | Done 的額外證據 |
|---|---|
| Extraction／normalization | raw/predicted_json、missing fields、aliases、invalid model JSON；核對 downstream 欄位；真實 extraction 任務需真實模型樣本，不能只驗 normalization |
| Duplicate／priority | ranking 與 classification 分開；duplicate review／exception short-circuit；priority label-preservation 與無 label 預測分開；模型和 feature/schema 配套 |
| Assignee | timestamp/provenance、candidate/roster、manual fallback、calibration/open-set；部署任務另外需要現有 protocol 指定的 live evidence 與授權，不以 offline test 取代 |
| FL／handoff | ticket+repo+base_commit 身分、source fingerprint、file/symbol 欄位、confidence、空/低信心/來源變動；只傳預測，不偷補 gold |
| FIM／patch | AST/scope/source/apply gates、候選拒收／選擇、原 repo 不變；功能修復需 before-fail/after-pass 及 regression；benchmark claim 需正式 evaluator verdict |
| Dataset／evaluation | 來源、schema、split、ID 對齊、完整分母、missing/failed records、protocol 版本；不能改定義使結果通過 |
| Test／regression／commit text | supplied/generated/plan 分開；測試確實有執行且非零測試假成功；commit text 不暗示已 commit／push；常數 coverage 不當實測 |

## Documentation-only task

本次 documentation architecture 的驗收不要求把所有研究能力實作或部署完成。它要求如實記錄現況；module runtime failure 可以是已完整記錄的發現，但不能被寫成該功能已驗證。

1. `AGENTS.md` 保持入口用途，只含 overview、長期原則、routing、validation、decision boundaries；細節在 docs。
2. 全 repository 的 domain 與真實入口都涵蓋，沒有把研究計畫當實作、把獨立模型當 default。
3. Progressive disclosure 有效：只讀相關模組；不要求每次讀全部 docs；子專案既有規格由連結引用，避免抄寫大量歷史數值。
4. Prompt templates 有 Goal／Scope／Relevant Documentation／Constraints／Expected Output／Done Criteria；不變成死板 SOP。
5. 自行檢查連結、source claims、狀態和修改範圍；未修改 production/data/gold/evaluation definition，未建立 Oracle → FIM。報告檢查過程與未驗證範圍。

## 狀態用語

| 狀態 | 用法 |
|---|---|
| Implemented | 所列具體能力有實作；另列測試、integration 與 runtime 證據 |
| Partially implemented | 功能或端到端 acceptance 尚有缺口 |
| Not implemented | 尚無該能力的實作（不是說整個模組不存在） |
| Experimental | 診斷／研究路徑，尚未採為正式 default／驗收方法 |
| Implemented but not integrated | 能力在別的入口存在，但未接目標整合路徑 |
| Not verified | 指定環境、模型、資料或 acceptance 尚未確認；可與 implementation status 同時出現 |

完成回報要指出交付內容、驗證證據、限制與一個最優先後續任務。不要把 Not verified 改寫成無條件成功。
