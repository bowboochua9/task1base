# Kết quả public hiện tại của Task 1 LegalIR

Tài liệu này ghi lại mốc public tốt nhất hiện tại sau khi thử nhánh
`step6+sbertft+bgem3rerankft+LLMXGB`.

## Bảng tổng hợp

| Step | Pipeline | Dev Recall@5 | Dev Precision@5 | Public Precision | Public Recall | Ghi chú |
|---|---|---:|---:|---:|---:|---|
| Step 1 | Data validator, cleaning, chunking, split | N/A | N/A | N/A | N/A | Tạo `324,377` chunks từ `8,532` docs; split `6000/1000`; validator có `0` errors, `35` warnings duplicate question. |
| Step 2 | Chunk-BM25 baseline | 0.744167 | 0.158000 | 0.155600 | 0.727833 | BM25 chunk top300, aggregate về document, xuất top5. |
| Step 3 | Tuned chunk-BM25 | 0.758667 | 0.161000 | 0.158400 | 0.740500 | Best trial `b_0p90`. |
| Step 4 | BGE-M3 dense + BM25 RRF | 0.852833 | 0.180600 | 0.177800 | 0.830833 | Dùng `BAAI/bge-m3`, dense top300 chunks, RRF với BM25. |
| Step 5 | Fine-tuned Vietnamese Bi-Encoder + RRF | 0.874250 | 0.185600 | 0.181400 | 0.844833 | Fine-tune `bkai-foundation-models/vietnamese-bi-encoder`, MNRL, 1 epoch. |
| Step 6a | Fine-tuned PhoRanker + fusion | 0.903167 | 0.192400 | 0.187200 | 0.870667 | `itdainb/PhoRanker`, 10 semi-hard negatives, 2 epochs. |
| Step 6b | Fine-tuned AITeamVN/Vietnamese_Reranker + fusion | 0.905000 | 0.193000 | 0.192200 | 0.894000 | Baseline cũ tốt nhất. |
| Step 6 + SBERT-FT `rrf_sbert0p25` | N/A | N/A | 0.197000 | 0.913667 | Weighted RRF, Step 6 weight `1.0`, SBERT weight `0.25`. |
| Step 6 + SBERT-FT `rrf_sbert0p40` | N/A | N/A | 0.197200 | 0.916167 | Weighted RRF, Step 6 weight `1.0`, SBERT weight `0.40`. |
| Step 6 + SBERT-FT `rrf_sbert0p60` | N/A | N/A | 0.197400 | 0.917333 | Baseline cũ; weighted RRF, Step 6 weight `1.0`, SBERT weight `0.60`. |
| XGB/LLM only control | N/A | N/A | 0.199200 | 0.922667 | Dùng trực tiếp `legalir_LLM_XGB_reranker/public_submission/submission.json`. |
| Step 6 + SBERT-FT + XGB/LLM `w0p25` | N/A | N/A | **0.200400** | **0.928833** | Best hiện tại; weighted RRF, Step 6 `1.0`, SBERT `0.6`, XGB/LLM `0.25`. |
| Step 6 + SBERT-FT + XGB/LLM `w0p40` | N/A | N/A | 0.200200 | 0.927833 | Gần best; XGB/LLM weight `0.40`. |
| Step 6 + SBERT-FT + CE + XGB/LLM | N/A | N/A | 0.198600 | 0.921917 | `full_ce0p25_xgb0p40`; thấp hơn không dùng CE. |

## Chi tiết mốc chính

Step 2 public score:

```text
precision = 0.1556
recall    = 0.7278333333333332
```

Step 3 public score:

```text
precision = 0.1584
recall    = 0.7405
```

Step 4 public score:

```text
precision = 0.1778
recall    = 0.8308333333333333
```

Step 5 public score:

```text
precision = 0.18140000000000003
recall    = 0.8448333333333332
```

Step 6 public scores:

```text
phoranker_resized_finetuned:
  precision = 0.1872
  recall    = 0.8706666666666667

aiteamvn_vietnamese_reranker_finetuned:
  precision = 0.19220000000000004
  recall    = 0.894
```

Step 6 + SBERT-FT public scores:

```text
rrf_sbert0p25:
  precision = 0.19700000000000004
  recall    = 0.9136666666666666

rrf_sbert0p40:
  precision = 0.19720000000000004
  recall    = 0.9161666666666666

rrf_sbert0p60:
  precision = 0.19740000000000005
  recall    = 0.9173333333333332
```

Step 6 + SBERT-FT + BGE-M3 reranker FT + LLM/XGB public scores:

```text
xgb_llm_only_control:
  precision = 0.19920000000000004
  recall    = 0.9226666666666667

full_ce0p25_xgb0p40:
  precision = 0.19860000000000005
  recall    = 0.9219166666666666

baseline_plus_xgb_w0p25:
  precision = 0.20040000000000002
  recall    = 0.9288333333333333

baseline_plus_xgb_w0p40:
  precision = 0.20020000000000002
  recall    = 0.9278333333333333
```

## Best hiện tại

Pipeline tốt nhất hiện tại là:

```text
Step 6 AITeamVN/Vietnamese_Reranker fused ranking
+ SBERT-FT public ranking
+ LLM/XGB reranker public ranking
+ weighted RRF
```

Best candidate:

```text
candidate     = baseline_plus_xgb_w0p25
rrf_k         = 60
step6_weight  = 1.0
sbert_weight  = 0.6
xgb_weight    = 0.25
ce_weight     = 0.0
```

Submission tốt nhất hiện nằm ở:

```text
task1/pipeline/baselinecur/submission.zip
```

Nguồn copy:

```text
task1/pipeline/step6+sbertft+bgem3rerankft+LLMXGB/step6_sbertft_bgem3rerankft_llmxgb/candidates/baseline_plus_xgb_w0p25/submission.zip
```

So với Step 6b cũ:

```text
precision gain = +0.008199999999999977
recall gain    = +0.03483333333333327
```

So với baseline `rrf_sbert0p60`:

```text
precision gain = +0.002999999999999975
recall gain    = +0.011499999999999955
```

## Dev metrics của anchor Step 6

Step 6 AITeamVN vẫn là anchor có dev metric đầy đủ:

```text
Recall@5     = 0.905000
Precision@5  = 0.193000
Hit@5        = 0.922000
MRR          = 0.7434394090533815
Recall@20    = 0.9656666666666667
Recall@50    = 0.9761666666666666
Recall@90    = 0.9806666666666666
Recall@100   = 0.982000
Exist@90     = 0.987000
```

Nhánh `step6+sbertft` hiện là fusion public-ranking nhanh, không có dev score
trên split chính vì nó dùng output public đã có từ SBERT-FT experiment.

## Nguồn kiểm chứng trong workspace

- `task1/pipeline/step1/outputs/reports/validation_report.json`
- `task1/pipeline/step2/outputs/reports/run_report.json`
- `task1/pipeline/step3/outputs/reports/run_report.json`
- `task1/pipeline/step4/step4/reports/run_report.json`
- `task1/pipeline/step5/step5/reports/run_report.json`
- `task1/pipeline/step6/step6/reports/run_report.json`
- `task1/pipeline/step6/step6/reports/public_scores_manual.json`
- `task1/pipeline/step6+sbertft/outputs/run_report.json`
- `task1/pipeline/step6+sbertft+bgem3rerankft+LLMXGB/step6_sbertft_bgem3rerankft_llmxgb/run_report.json`
- Public scores mới do người dùng báo sau khi nộp Codabench.
