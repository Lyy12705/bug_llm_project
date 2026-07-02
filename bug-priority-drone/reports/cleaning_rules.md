# Eclipse Bug Report 資料清洗規則

本文件說明 `scripts/clean_bug_reports.py` 的資料清洗與前處理規則。目標是將 Eclipse Bugzilla 原始資料整理成可供 DRONE/GRAY priority prediction 使用的標準資料表。

## 輸入與輸出

- 輸入檔案：`data/raw/eclipse_bugzilla_raw_with_dupe.csv`
- 輸出檔案：`data/processed/eclipse_bug_reports_clean_with_dupe.csv`
- 原始標籤：`priority`
- 模型標籤：`priority_num`
- `description` 來源：Eclipse Bugzilla 的第一則 comment，也就是使用者最初填寫的 bug report 描述。

## 清洗規則

| 規則 | 目的 | 實作方式 |
|---|---|---|
| 保留研究欄位 | 只保留 priority prediction 需要的欄位 | 保留 `id`, `summary`, `description`, `priority`, `severity`, `product`, `component`, `creator`, `creation_time`, `status`, `resolution`, `fetch_group`；若 raw data 有 `dupe_of` 也保留 |
| 移除缺少文字的資料 | DRONE textual factor 需要 `summary` 與第一則 `description` | 移除空白 `summary` 或空白 `description` |
| 清理文字 | 降低 URL、email、hex code、log-like token 對文字特徵的干擾 | 使用 `clean_text()` |
| 保留有效 priority | 只保留標準 P1-P5 | `priority in {"P1", "P2", "P3", "P4", "P5"}` |
| 建立序位標籤 | 將 P1-P5 轉成模型可讀的 ordinal label | `P1=1`, `P2=2`, `P3=3`, `P4=4`, `P5=5` |
| 補齊缺失類別欄位 | 避免 one-hot encoding 或 feature extraction 出現缺值錯誤 | 對 `severity`, `product`, `component`, `creator`, `status`, `resolution`, `fetch_group` 補 `unknown` |
| 整併低頻類別 | 降低類別特徵過度稀疏 | 出現次數小於 `RARE_VALUE_THRESHOLD=5` 的值改為 `other` |
| 解析時間欄位 | temporal factor 需要可靠的時間順序 | `creation_time` 轉成 datetime，無法解析者移除 |
| 建立文字合併欄位 | 讓 textual factor 可以同時使用 summary 與 description | `text = summary + " " + description` |
| 控制文字長度 | 移除太短資料，截斷極長描述 | `MIN_TEXT_LEN=10`, `MAX_TEXT_LEN=5000` |
| 排序與去重 | 避免 temporal / related-report features 使用未來資訊或重複 ticket | 依 `creation_time` 排序，依 `id` 去重 |
| 可選排除 id | 建立 holdout 或測試資料時避免資料重疊 | 使用 `--exclude-ids-from` |

## 文字正規化

`clean_text()` 會執行以下處理：

- 缺失文字轉為空字串。
- 文字轉為小寫。
- 換行、tab 等空白壓成一般空白。
- URL 替換為 `<URL>`。
- Email 替換為 `<EMAIL>`。
- Hex code 替換為 `<HEX>`。
- 移除過長的 alphanumeric / log-like token。
- 移除大量重複符號。
- 合併多餘空白。

## Priority 對應方式

| 原始 priority | 意義 | `priority_num` |
|---|---|---:|
| P1 | 最高優先級 | 1 |
| P2 | 高優先級 | 2 |
| P3 | 中等優先級 | 3 |
| P4 | 低優先級 | 4 |
| P5 | 最低優先級 | 5 |

`priority_num` 保留 priority 的序位關係，數值越小代表優先級越高。這也符合 DRONE/GRAY 將 priority 視為 ordinal label 的設定。

## 避免資料洩漏

- Fetch 階段只取第一則 comment 作為 `description`，不使用後續 comments，避免混入修復討論、狀態變更或 triage 結果。
- 清洗後依 `creation_time` 排序；後續 temporal、author、related-report features 只使用歷史 ticket。
- Evaluation 會移除與 training metadata 重疊的 bug id。
- `dupe_of` 只用於訓練 REP- related-report similarity 權重，不會直接當成 priority prediction 的 label。

## Fetch 補充說明

`scripts/fetch_eclipse_bugzilla.py` 先使用 Eclipse Bugzilla REST API 搜尋 bug metadata。若公開 REST comment endpoint 回傳空 body，fetcher 會 fallback 到 Bugzilla XML endpoint，讀取第一則 comment 作為原始 `description`。

若單筆 description 請求遇到暫時性 500 或 XML 解析失敗，fetcher 會重試；重試後仍失敗時，該列會保留 metadata、將 `description` 留空，並記錄 `description_source=missing` 與 `description_error`。清洗階段會依規則移除沒有原始 description 的資料。

若某些 priority 可用資料不足 `--target-per-priority`，fetcher 會保留實際可抓筆數，並在輸出時列出 priority distribution 與缺少 description 的筆數。
