# Repository documentation restructuring review

日期：2026-09-16。依 [CODEX_RESTRUCTURE.md](../CODEX_RESTRUCTURE.md) 進行全工作區文件整理；本頁是 Final Report 的可追溯版本。

## 1. 交付與最終架構

只新增 root `AGENTS.md` 與以下 12 份 root `docs/` 文件；既有子專案 README／規格／報告保留原位。

```text
AGENTS.md
docs/
  README.md
  project_overview.md
  extraction_normalization.md
  ticket_classification.md
  assignee_routing.md
  fault_localization.md
  patch_generation.md
  datasets.md
  evaluation.md
  task_prompt_templates.md
  done_criteria.md
  restructure_review.md
```

[AGENTS](../AGENTS.md) 是精簡常駐入口：全系統 overview、working principles、按任務 routing、validation 與 decision boundaries。技術細節集中到 domain docs，不要求每次讀全目錄。[文件用途表](README.md) 說明每份 docs；templates 覆蓋 extraction、classification、assignee、FL、FIM 與 dataset/evaluation/documentation，皆含指定六欄。

## 2. 實際盤點範圍與證據

盤點六個顶層程式目錄及內嵌 REPAIR checkout，核對各 README、entry points／imports、orchestrator、config／dependencies、module implementations、scripts、tests、schema、非 sealed 資料樣本、selected saved artifacts 及既有研究文件。保留多專案重複目錄結構，沒有搬動 code 或 datasets。

| 檢查面向 | 具體來源／方法 |
|---|---|
| 全系統 | MAIN `src/main.py`、`config.py`、`pipeline/orchestrator.py`、九個主要 module／assignee helpers、tests／experiments／demo／docs inventory |
| Extraction／classification | ToJson scripts/schema/README/manual annotation；DUP package／pyproject/tests；PRI feature/training/export scripts、requirements、模型與報告位置 |
| FL／repair | 三份 FL 入口、Stage-3 adapter、patch context、兩份 FIM CLI/generator、REPAIR pipeline/evaluator/batch、merged-gold policy 與 tests |
| Dataset／evaluation | 非 sealed ticket/gold/schema samples、ToJson gold 計數、CSV headers、assignee shadow artifact、FL protocol、synthetic smoke evaluation artifacts；未讀 sealed labels 做分析 |
| 文獻／背景 | 根目錄 flow plan、架構圖／研究申請檔案 inventory、11 份文獻 PDF 題名與位置；沒有逐頁重審論文，也沒有引用論文結果作 runtime 證據 |

這是架構與契約盤點，不宣稱逐位元審核所有大型 dataset、cache、upstream repository 或每一份歷史 report。曾遇到全域 `rg --files` 被大量 `runs/` 上游檔案淹沒，後續改採明確的 owned source／scripts／tests 路徑。沒有把 `tmp/` staging copy 當現役模組。

## 3. 主要模組、狀態與完整 pipeline

完整狀態矩陣與 source routing 在 [project_overview](project_overview.md)。關鍵結論：

| 範圍 | 實際狀態 |
|---|---|
| Extraction、duplicate／historical retrieval、priority | MAIN baseline 已實作並接合；ToJson LLM、DUP SBERT、PRI trained model 為 Implemented but not integrated（相對 default MAIN） |
| Assignee | Hybrid／manual／feedback／gates 已實作；LTR/rolling/LLM 研究與 production deployment acceptance 分開，live readiness 仍未完成 |
| FL、FIM、patch | deterministic Stage-3 與 handoff 已實作；MAIN 非 native FIM；REPAIR 有獨立統一 FL→FIM pipeline；official benchmark 成功本次 Not verified |
| Tests／validation／commit | supplied tests normalization、temp-copy regression、deterministic commit text 已有；自主 LLM tests、獨立 RIPR／coverage 量測尚未實作 |

MAIN 流程：raw → structured → duplicate（positive 結束／review 暫停）→ priority → assignee → predicted FL → provided/opt-in generated diff → supplied tests／plan → temp apply／可選 before-after-regression → commit text → final JSON。

REPAIR 流程：ticket + clean base repo → fresh code index／predicted FL → `LocalizationForPatchV1` → FIM body generation → diff／gates／可選 supplied validation → artifacts → 可選生成後 reference comparison／official evaluator。REPAIR 不含 MAIN 前段分類／routing，MAIN default 不自動進 REPAIR。外層 FIM `attach_fim()` 是顯式 adapter，不能說全系統已預設採用。

## 4. 本次執行驗證

環境：Windows PowerShell，Python 3.12.14，interpreter 為 `C:/Users/User/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe`。各 suite 從自己的 root 執行，設 `PYTHONDONTWRITEBYTECODE=1`；MAIN／DUP／REPAIR 設 `PYTHONPATH=<cwd>/src`。所有命令模式為 `& $py -m unittest discover -s tests -p <下表檔名> -v`。

| cwd 代號（見總覽） | Pattern | 結果 |
|---|---|---|
| MAIN | `test_main_stage3_integration.py` | 4/4 passed；client 為 mock |
| MAIN | `test_stage3_patch_handoff.py` | 15/15 passed；context／source／gold exclusion／diff contracts |
| MAIN | `test_pipeline.py` | 38 passed、2 failed，共 40 |
| REPAIR | `test_repair_pipeline.py` | 2/2 passed；真實 FL、可控 backend、before/after/regression |
| REPAIR | `test_patch_cli_and_evaluation.py` | 10/10 passed；official harness 為 fake runner，不是 Docker 評分 |
| REPAIR | `test_run_stage4_batch.py` | 33/33 passed；sanitization、snapshot、failure/resume、統計契約 |
| FIM | `test_fim.py` | 29/29 passed；包含 local unittest validator |
| FL | `test_merged_ground_truth.py` | 17/17 passed；temporary synthetic repo／receipt fixtures，不重建真實 gold |
| DUP | `test_metrics.py` | module import error：缺少 `joblib`，測試未實際載入；Not verified |

總計已載入的 150 項測試中 148 passed、2 failed；另有 1 次 DUP suite import failure。失敗診斷的重跑不重複計入測試總數。

MAIN 失敗為 `test_pipeline_completes_only_after_fib_and_regression_pass` 和 `test_pipeline_does_not_execute_ticket_commands_without_explicit_trust`。兩者預期 completed_verified／tests_passed，實際 patch_unverified。診斷 trace 確認第一個案例在 `subprocess.run(['python3', '-c', ...])` 遭 `WinError 5`；`shutil.which('python3')` 為 None，第二個測試亦硬編碼同一 interpreter。沒有修改測試、安裝 aliases 或改 production code；此環境的 MAIN 完整 functional path 不可宣稱 verified。

未跑完整 suite、真實 Ollama、SBERT training／priority retraining、外部 issue APIs、official SWE-bench、live assignee routing。這些是 Not verified，不以既有 README 的歷史測試數替代。

## 5. 文件／程式落差與未完成能力

| 原敘述／可能誤讀 | 程式與 artifact 證據 | 本次文件處理 |
|---|---|---|
| Flow plan 各步驟都是 LLM／研究模型 | MAIN default extractor 無 LLM、duplicate 輕量 similarity、priority deterministic | 清楚分 research／default／adapter |
| MAIN README 列 patch supplied/manual，且以舊 retrieval 描述定位 | config 已有 opt-in patch LLM；BugLocalizer 已預設 deterministic Stage-3 | 記錄目前實作，不把舊 README 當完整能力上限 |
| `patch/README.md` 的入口／12-ticket pilot 代表所有 repair 工作 | REPAIR 另有統一 CLI、post-generation reference、official adapter、batch | 將 REPAIR 列獨立真實入口，未從舊 pilot 推論最新修復率 |
| REPAIR README 仍以 Fault Localization Feature 開頭；INTEGRATION 結尾說下一步加入 official evaluator | `src/fim_patch/evaluation.py`／CLI 已有 adapter，INTEGRATION 前段也提已加入 | 指出時間層疊；區分 adapter 已實作與官方執行未成功驗證 |
| 舊 FL preparation/evaluation 指令可直接套所有 gold | 獨立 FL 已改 merged policy／explicit legacy，其他 checkout 未必一致 | 在 datasets/evaluation 路由標明版本差異，未變更 definition |
| TestGenerator／RIPR／coverage 名稱表示完整研究實作 | supplied tests normalization；RIPR 同一 boolean；coverage 常數 0 | 標 Partially implemented／Not implemented 的具體能力 |
| `patch_success_rate`、`fib_success_rate` 等於功能修復／真實 FiB | patch evaluator 計 generated/provided；test evaluator 看空的 fib_passed_tests，實際執行另在 regression | 記錄現有語意，不改公式 |
| ToJson 16-field schema 表示現存標註皆有 16 欄 | 56-row manual 六欄、100-row public 四欄；MAIN 另有 priority None/bug type 差異 | 保留資料實況與 schema adapter 注意事項 |
| assignee 具部署／shadow scripts 即 deployment-ready | phase8 saved shadow 為 0 live rows／gate false | 保留 live evidence／roster／holdout 的驗收缺口 |
| 現有路徑／CI 說明可直接執行 | MAIN README 的 `../ToJson` 等不符合目前雙層目錄；PRI 有舊 macOS 絕對路徑；MAIN CI 是 docs template | 提供 workspace 相對 routing 和正確 cwd；未啟動 CI |
| MAIN 安裝後 console script 可用 | pyproject 指 `bug_tracking_llm_system.cli:main`，目前 src 為 `main.py` 與頂層 packages，未發現該 package/cli | 記錄靜態不一致；安裝後入口 Not verified，不在本次修正 |

已有 `models/`、saved reports 或 pipeline tests 不代表完整 production 部署。Local FIM／REPAIR 的 synthetic `generated/plausible` artifacts 已讀取，`official_resolved` 仍 null；官方雲端錯誤細節來自既有 INTEGRATION 記錄，本次未重現外部環境。

## 6. 自我 review 與變更保護

檢查要求：AGENTS 精簡；技術細節在對應 docs；既有研究文件只路由不複製；未把 planned/experimental 寫為 default Implemented；未把 mock/synthetic 寫為真實模型 verified；所有 templates 含六欄；沒有 read-all 要求或新 Oracle→FIM pipeline。

本次以明確 source/script/test/config/docs 範圍建立 416 個既有小型檔案 SHA-256 baseline（每檔小於 2 MB），最後比對其內容；這不是全 dataset byte-for-byte checksum。DUP／PRI／FL 的既有 dirty changes 保留；REPAIR 原先 clean。Git safe.directory 僅用單次 `git -c` 讀取，未修改 global config。

交付前檢查結果：AGENTS 為 34 行；13 份新增 Markdown 中的 125 個 local links 全部存在；6 個 templates 各有且僅有一組指定六欄；416 個既有檔案 SHA-256 全部不變。四個 nested Git checkout 的 status 與盤點時一致，REPAIR 維持 clean。沒有改 production code、dataset、ground truth、evaluation definition 或模型設定。

## 7. 下一個建議 implementation task

優先修正 **MAIN 的 Python interpreter／test subprocess 可攜性**：讓上述兩項 pipeline 測試在目前 Windows 環境真實走通，保留 before-fail/after-pass/regression 的嚴格語意；不調整成功 gate 來取得綠燈。這是已有可重現失敗、範圍小且會直接影響完整系統驗證的任務。

該任務的驗收是 MAIN 40 項 pipeline tests 通過，記錄實際子程序 interpreter、輸入 patch／test fixtures／最終 verification，且不更換模型或修改 dataset。後續再以獨立授權任務接合研究 adapters 或完成 official benchmark；本次不順便實作。
