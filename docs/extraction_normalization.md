# Extraction and normalization

## 入口與契約

| 位置 | 已有能力 | 輸入 → 輸出 |
|---|---|---|
| [JSON scripts/To_Json](../ToJson/scripts/To_Json/) | fetch／transform／sample／extract／postprocess／evaluate | issue 原始資料 → `bug_report` → `{id, predicted_json, extraction_meta...}` |
| [MAIN TicketExtractor](../bug_tracking_llm_system/bug_tracking_llm_system/src/modules/ticket_extractor.py) | normalize；可注入 LLM client | raw dict → shared structured ticket |
| [MAIN json_schema](../bug_tracking_llm_system/bug_tracking_llm_system/src/utils/json_schema.py) | 欄位 aliases、defaults、priority／bug type normalization、JSON repair | raw 或含 `predicted_json` 的 dict → normalized dict |

主系統 `build_default_orchestrator()` 只傳 extractor prompt path，沒有傳 llm_client。因此 default 是 normalization，不能宣稱會自動呼叫 JSON 子專案的 Code Llama。跨入口資料可共用，不等於推論流程已自動整合。

共用欄位包含 ticket_id、title、description、product、severity、bug_type、component、os、version、priority、error_message、steps_to_reproduce、expected_behavior、actual_behavior、logs、screenshots_text。完整欄位見 [JSON schema](../ToJson/dataset/schema.json)。注意此檔是簡易欄位型別表，不是完整 JSON Schema validator。

## 實際差異

JSON extraction 預設 `codellama:7b-instruct`、Ollama `/api/generate`、zero-shot，保留抽取 metadata。MAIN 的 `priority` default 是 `None`，JSON schema 寫 string、extractor 預設空字串；MAIN bug types 額外包含 `runtime_error`、`security`。修改 adapter 時需檢查實際 canonicalization，不可只複製 schema。

`TicketExtractor` 注入 LLM 後會 repair JSON，再與 raw 合併；repair 失敗可能回空 dict，最後仍走 normalization。輸出合法不代表抽取正確，測試須分辨 fallback。

## 狀態與驗收

Normalization：Implemented，已接主流程。獨立 LLM extraction：Implemented but not integrated（相對 MAIN default），本次模型推論 Not verified。抓取 API、模型服務及資料準確率亦未在本次重跑。

變更後以小型 raw ticket 驗證欄位、缺值、aliases、`predicted_json` 與下游 duplicate／priority 使用方式。不得把 tracker source consistency 當人工 gold accuracy；既有六欄評分和擴充欄位評分分開。指令與分母見 [evaluation](evaluation.md)；資料與標註見 [datasets](datasets.md) 和 [manual_annotation](../ToJson/docs/manual_annotation.md)。

只有維護獨立抽取流程時再讀 [ToJson README](../ToJson/README.md)。`scripts/Literature_Datasets/` 的 HumanEval／MBPP 是程式生成 benchmark，不是 ticket extraction，也不是目前 FIM 修補成功率。
