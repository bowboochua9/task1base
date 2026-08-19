# Step 6 + SBERT-FT + BGE-M3 reranker FT + LLM XGB

Nhánh này tạo submission thử nghiệm bằng cách fuse:

```text
Step 6 AITeamVN fused ranking
SBERT-FT public ranking
BGE-M3 reranker-v2 fine-tuned CE listwise ranking
LLM + XGBoost scored candidates
```

Toàn bộ tín hiệu mới lấy từ dataset:

```text
/kaggle/input/datasets/mphatfromuit/dsc2026-baseline/DSC26_weight_report
```

Notebook không train, không inference model, không dùng GPU. Nó chỉ đọc ranking
và scored candidates đã có, sau đó sinh nhiều `submission.zip` để nộp thử.

## Input Kaggle

```text
/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/step7/step6/rankings/aiteamvn_vietnamese_reranker_finetuned/public_rankings_step6_fused.jsonl
/kaggle/input/datasets/mphatfromuit/dsc2026-baseline/DSC26_weight_report/legalir_sbert_cl/public_submission/public_ranked_contexts.csv
/kaggle/input/datasets/mphatfromuit/dsc2026-baseline/DSC26_weight_report/legalir_ce_listwise/public_submission/public_official_bm25_sbert_ce_rankings.jsonl
/kaggle/input/datasets/mphatfromuit/dsc2026-baseline/DSC26_weight_report/legalir_LLM_XGB_reranker/public_submission/public_official_xgb_scored_candidates.parquet
/kaggle/input/datasets/mphatfromuit/dsc2026-baseline/DSC26_weight_report/legalir_LLM_XGB_reranker/public_submission/submission.json
```

Schema parquet đã thấy trong Kaggle:

```text
qid
candidate_id
xgb_llm_score
xgb_retrieval_score
```

Notebook dùng `candidate_id` làm document ID và ưu tiên `xgb_llm_score`.

## Output

```text
/kaggle/working/step6_sbertft_bgem3rerankft_llmxgb/
  candidates/<candidate>/submission.zip
  run_report.json
```

Submit ưu tiên:

```text
xgb_llm_only_control
baseline_plus_xgb_w0p25
full_ce0p25_xgb0p40
```

`xgb_llm_only_control` dùng trực tiếp ranking XGB/LLM của bạn Phát, nên là
candidate quan trọng nhất nếu đúng report public recall khoảng `0.9227`.

## Public scores đã nộp

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

Best hiện tại là `baseline_plus_xgb_w0p25`. Notebook đã đổi default candidate
sang candidate này sau khi có score Codabench.
