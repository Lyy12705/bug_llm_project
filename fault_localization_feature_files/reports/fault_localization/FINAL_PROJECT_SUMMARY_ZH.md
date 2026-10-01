# 錯誤定位專題最終總結

- 更新日期：2026-09-09（第三階段決策凍結；非完整研究驗收完成日期）
- 第一階段實作版本：E8-A
- 第一階段準確率基準：E6
- 第二階段正式設定：retrieval baseline；`codellama:7b-instruct` reranker預設關閉
- 第三階段選定方法：`coverage-aware-v1` deterministic baseline；WP4探索性pilot完成，完整驗收未完成

## 第三階段定案：exploratory_due_to_G2_not_met

本次納入1票與10票smoke、46票pilot的既有結果，維持deterministic baseline，
不採用Symbol LLM為預設；本文件更新不改動執行設定。以下數字的分母為46筆
Stage-2 conditional-eligible development tickets，不是全體61筆的end-to-end指標。

| Exact Hit | Baseline | LLM | 差異（百分點） |
|---|---:|---:|---:|
| Hit@1 | 26.09% | 10.87% | -15.22 |
| Hit@3 | 47.83% | 41.30% | -6.52 |
| Hit@5 | 60.87% | 58.70% | -2.17 |

Hit@5差異的paired bootstrap 95% CI為[-15.22, +10.87]百分點，未提供改善證據。
valid 42/46（91.30%）、fallback 4/46（8.70%），未通過G3。G2 Candidate Recall@30
為68.64%，未達90%；這是不同於ticket-level Hit的指標，不能直接混用。
G4正式validation未完成；本輪不足以通過G5，工程決策為保留baseline。

尚缺L1-blend、逐票耗時與p95、全體end-to-end報告、完整重現metadata，以及4筆fallback
細分原因。這些均未因凍結而視為完成；46筆也未達原訂50筆pilot目標。
合併驗證已實際通過204/204項回歸測試，未重新執行Ollama模型實驗。
依據：[結果摘要](stage3_wp4_pilot/pilot_46tickets_summary.json)、
[配對分析](stage3_wp4_pilot/pilot_46tickets_analysis.json)、
[第三階段計畫](../../STAGE3_SYMBOL_RERANKER_IMPLEMENTATION_PLAN_ZH.md)。

## 一、最終結論

本專題正式採用「**E8實作＋E6準確率基準**」。E8完整保留E6的固定50候選、
SBERT、Symbol／API擴充與Call Graph Top-3排名流程，並新增`.pyi`與`.scala`來源
支援；E6則提供相同500筆Validation的成對比較及500筆Final Holdout結果，因此作為
主要準確率證據。

第一階段在既有12個Repository的Validation 500筆中，Recall@20由82.98%提高至
85.92%，增加2.94個百分點；Hit@20增加2.00個百分點、Top-1不變、MRR增加0.0178。
因此可以主張「Top-20正確檔案涵蓋率提升」，但不能主張所有排名能力都提升。

第二階段的Code Llama 7B流程已穩定接通，但在相同10張票的無分數洩漏配對比較中，
File Hit@1、Hit@3、Hit@5與MRR均比retrieval baseline低0.1。因此7B reranker不列為
正式預設，系統維持retrieval baseline。

## 二、最終系統架構

```text
Ticket標題、描述與錯誤訊息
  → 第一階段E8-A
      → TF-IDF固定50個候選檔案
      → SBERT語意重新排序
      → Symbol／API實作位置擴充
      → 前3名候選的一層Call Graph擴充
      → 支援.py、.pyi、.scala等已實作來源格式
      → 目標輸出Top-20唯一候選檔案
  → 第二階段正式設定
      → retrieval baseline保留Top-5檔案
      → Code Llama 7B reranker預設關閉
  → AST抽取Top-5檔案內的函式、Method與Class
  → Symbol retrieval fallback產生Symbol排名
```

## 三、E0～E8實驗總覽

| 實驗 | 主要變更 | 量化結果 | 正式決定 |
|---|---|---|---|
| E0 | TF-IDF Top-50＋SBERT基準 | Hit@20 89.80%、Recall@20 82.98%、Top-1 56.20%、MRR 0.6597 | 作為原始基準 |
| E1 | 多chunk／Symbol／Package檔案分數整合 | 最佳E1-B Recall +0.96百分點，但95% CI為-0.31～+2.27 | 僅列探索性結果 |
| E2 | 單向／雙向Import Graph | 最佳E2-B Recall +0.62百分點，但CI跨0且Top-1下降6.80百分點 | 不保留 |
| E3 | Symbol definition、API re-export與wrapper擴充 | 原始E3-C點估計為Recall 84.72%；Guarded下降0.17，Namespace差異為0 | 保留原始E3-C，不採限制版 |
| E4 | 一層Call Graph，來源限制為前3名 | 相對E3-C Recall +1.20百分點，95% CI為+0.09～+2.36；Top-1下降4.60百分點 | 通過門檻並保留 |
| E5 | 依Repository大小使用75／100候選 | Recall只增加0.19／0.25百分點，CI均跨0 | 維持固定50候選 |
| E6 | 固定50＋E3-C＋Call Graph Top-3最終組合 | Validation Recall 85.92%；Final Holdout Recall 67.17% | 凍結為準確率基準 |
| E7 | 大型Repository候選池由50增至100 | Development Recall只增加0.106百分點，95% CI為-0.224～+0.459 | 不保留E7-B |
| E8 | 在E6／E7-A排名流程加入`.pyi`與`.scala`索引 | 外部Holdout Recall 68.89%，98／100筆符合固定Top-20契約 | 採為專題實作版本；不宣稱準確率優於E6 |

## 四、第一階段結果判讀

### 4.1 E0與E6的同資料Validation比較

| 指標 | E0 | E6 | E6－E0 |
|---|---:|---:|---:|
| Hit@20 | 89.80% | 91.80% | +2.00百分點 |
| Recall@20 | 82.98% | 85.92% | +2.94百分點 |
| Top-1 | 56.20% | 56.20% | 0.00百分點 |
| MRR@20 | 0.6597 | 0.6775 | +0.0178 |
| 完整／部分／未命中 | 376／73／51 | 392／67／41 | 完整+16、未命中-10 |

這是E0與最終E6在相同Validation 500筆上的可比較結果。改善集中在Top-20涵蓋：
更多正確修改檔案進入候選集合，但第一名正確率沒有提高。E4 Call Graph是E0～E6中
唯一以單項成對實驗明確通過預設Recall門檻與信賴區間條件的功能。

### 4.2 未見Repository的一次性結果

| 實驗 | Ticket／Repository | Hit@20 | Recall@20 | Top-1 | MRR | 固定Top-20契約 |
|---|---:|---:|---:|---:|---:|---:|
| E6 Final Holdout | 500／31 | 84.68% | 67.17% | 41.13% | 0.5248 | 497／500 |
| E7-A新Holdout | 27／4 | 70.37% | 62.63% | 40.74% | 0.5031 | 20／27 |
| E8-A外部Holdout | 100／53 | 89.00% | 68.89% | 54.00% | 0.6143 | 98／100 |

三個Holdout的資料來源、Ticket與Repository不同，不能用表面百分比進行受控優劣比較。
E8的主要可採用價值是擴大來源格式與Repository涵蓋，不是已證明比E6更準。E6的
Validation Recall 85.92%下降到未見Repository Final Holdout的67.17%，顯示跨Repository
泛化仍是第一階段的主要限制。

## 五、第二階段結果判讀

正式比較使用相同10張`astropy/astropy` Ticket、相同Stage-1候選與相同Gold；LLM
prompt不包含retrieval score。File與Symbol LLM呼叫均為10／10成功、warnings為0、
fallback為0，證明JSON Schema與每批最多5個候選的管線可穩定執行。

| 檔案層級指標 | Retrieval baseline | Code Llama 7B | LLM－baseline |
|---|---:|---:|---:|
| File Hit@1 | 60% | 50% | -10百分點 |
| File Hit@3 | 70% | 60% | -10百分點 |
| File Hit@5 | 80% | 70% | -10百分點 |
| File MRR | 0.675 | 0.575 | -0.100 |

逐票結果為改善0筆、相同7筆、惡化1筆、兩者皆未命中2筆。這代表「模型服務與
格式契約成功」不等於「排名準確率提升」。因此`codellama:7b-instruct`只保留為
實驗選項，不設為正式流程預設。

這10張票的`fixed_symbols`全部為空，具有Symbol gold的票數是0；Symbol Hit@K與MRR
為N/A。Symbol LLM 10／10成功只能證明流程可執行，不能主張Symbol定位準確率。

## 六、固定Top-20輸出契約限制

目前契約要求每張Ticket恰好輸出20個不重複候選檔案，但部分小型、stub型或模板型
Repository本身不足20個可索引的唯一來源檔案：

- E6有497／500筆符合契約；3筆只有7、7、18個候選。
- E8有98／100筆符合契約；2筆只有17、4個候選。
- E6與E8的預測執行失敗均為0；候選不足不等於程式或模型呼叫失敗。
- 兩組使用不同Holdout，99.4%與98.0%只能各自描述，不能用來判定E6優於E8。
- 未來可在新的Development實驗評估`min(20, 可索引唯一檔案數)`及
  `candidate_pool_exhausted`標記，但不得事後將既有E6／E8結果改判為通過。

## 七、研究問題回答

### RQ1：E0～E8是否提升第一階段？

在相同Validation 500筆上有提升：E0至E6的Recall@20增加2.94個百分點、Hit@20增加
2.00個百分點、MRR增加0.0178，Top-1不變。可以主張候選涵蓋改善；未見Repository
Holdout結果明顯較低，因此不能主張跨Repository問題已解決。

### RQ2：E8是否比E6更好？

E8更適合作為專題實作版本，因為它保留E6排名流程並增加`.pyi`與`.scala`支援；但
兩者沒有在同一Holdout進行A/B比較，所以不能主張E8準確率顯著優於E6。E6仍是主要
準確率基準，E8提供工程涵蓋與外部資料證據。

### RQ3：第二階段Code Llama 7B是否提升結果？

沒有。相同10張票的File Hit@1、Hit@3、Hit@5與MRR均下降0.1；正式流程應維持
retrieval baseline，7B reranker預設關閉。

## 八、研究限制

1. E6、E7與E8使用不同Holdout，彼此只能描述，不能作為同資料因果比較。
2. E6 Final Holdout的Recall@20只有67.17%，跨Repository泛化仍不足。
3. 固定Top-20契約沒有處理可索引檔案少於20個的Repository。
4. 第二階段只測試10張Astropy Ticket，樣本小且Repository單一。
5. 第二階段的10票資料沒有Symbol gold；第三階段另有46票conditional探索性評估，但正式研究驗收未完成。

## 九、正式採用設定

| 項目 | 專題正式設定 |
|---|---|
| 第一階段程式 | E8-A |
| 第一階段準確率依據 | E6 Validation與Final Holdout |
| Stage-1初始候選池 | TF-IDF固定50個檔案 |
| 語意重排 | SBERT |
| 關係擴充 | E3-C Symbol／API＋Call Graph outgoing Top-3 |
| 來源格式延伸 | E8的`.pyi`與`.scala`支援 |
| 第一階段輸出 | 目標Top-20唯一檔案；不足案例列為契約限制 |
| 第二階段檔案排序 | Retrieval baseline Top-5 |
| Code Llama 7B | 實驗選項，預設關閉 |
| Symbol排序 | 選定coverage-aware-v1 deterministic baseline；已有探索性評估，完整驗收未完成 |

## 十、後續工作

目前優先工作：使用凍結的定位輸出銜接補丁生成與驗證，並向指導教授確認研究範圍調整。
以下候選／模型研究列為後續研究，不自動重啟WP3局部調參。

1. 只在新的Development資料定義小型／模板型Repository的條件式Top-K契約。
2. 若修改契約或排名方法，建立新實驗ID並使用全新的Repository-disjoint Holdout。
3. 使用具有Symbol gold的資料與更強模型，重新進行第二階段配對比較。

## 十一、主要產物

- [第一階段E0～E8計劃與結果](STAGE1_NEXT_ITERATION_IMPLEMENTATION_EXPERIMENT_PLAN_ZH.md)
- [E6 Final Holdout結果](stage1_next_iteration/final_holdout_v2/FINAL_HOLDOUT_V2_RESULTS_ZH.md)
- [E8外部Holdout結果](stage1_next_iteration/e8_holdout_v1/E8_A_SWEBENCH_LIVE_HOLDOUT_RESULTS_ZH.md)
- [第二階段整合與結果](STAGE2_INTEGRATION_RESULT_ZH.md)
- [第二階段正式配對摘要](stage2_codellama_7b_paired_10_no_score_leak/paired_summary.json)

## 十二、最終判定

本專題已完成第一、二階段銜接與量化評估。第一階段在既有Validation的Top-20涵蓋率
有可重現提升，但跨Repository泛化與固定Top-20輸出契約仍有限制；第二階段7B模型
雖能穩定執行，卻沒有改善檔案定位準確率。最終系統採用E8作為第一階段實作、E6
作為準確率基準，並以retrieval baseline取代Code Llama 7B作為第二階段正式設定。

第三階段已完成一輪WP4探索性pilot，維持coverage-aware-v1為選定方法、Symbol LLM
不採用為預設。G2／G3未過，G4與完整WP4驗收未完成；此定案保留全部限制，作為
後續補丁生成的工程基準，不構成Symbol定位已通過完整研究驗收的宣稱。
