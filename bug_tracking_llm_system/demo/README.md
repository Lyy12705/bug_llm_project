# AI Bug Assistant Demo

這個目錄提供一個無外部相依的本機展示介面，用來把專題核心 pipeline 包裝成「現有 bug tracker 的 AI 輔助套件」。

## 啟動

從 repository 根目錄執行：

```bash
python3 bug_tracking_llm_system/demo/demo_app.py --port 8765
```

開啟：

```text
http://127.0.0.1:8765
```

## 建議展示流程

1. 左側文字框可以直接輸入或貼上 bug report，支援中文與英文文字。
2. 先選 `Duplicate 案例`，按 `開始分析`，展示使用者輸入文字、JSON 欄位抽取，以及 duplicate 半自動判定。系統會列出 Top-k 候選，最後由工程師確認是否真的重複。
3. 再選 `整合功能案例`，按 `開始分析`，展示使用者輸入文字、JSON 欄位抽取、duplicate 候選清單、priority prediction，以及 assignee suggestion。
4. 在 `分配負責人` 停一下，說明目前使用 Hybrid triager：結合歷史 component/product 分派、component-owner mapping、BM25 文字相似度。

## 修改案例

展示資料放在：

```text
bug_tracking_llm_system/demo/data/scenarios.json
```

每個案例包含：

- `ticket`：左側 bug tracker 顯示的原始 ticket。
- `workflow`：AI assistant 的流程步驟。
- `result`：符合目前 pipeline JSON 風格的分析結果。
- `tracker_comment`：可貼回 bug tracker 的摘要。
