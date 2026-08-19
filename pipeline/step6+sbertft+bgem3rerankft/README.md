# Step 6 + SBERT-FT + BGE-M3 reranker FT

Nhánh này tạo submission thử nghiệm bằng cách fuse baseline current:

```text
Step 6 AITeamVN fused ranking + SBERT-FT, weight 0.6
```

với CE listwise trong dataset:

```text
/kaggle/input/datasets/mphatfromuit/dsc2026-baseline/DSC26_weight_report/legalir_ce_listwise/
```

Theo ghi chú của user, CE này là `BAAI/bge-reranker-v2-m3` đã fine-tune.

## Input Kaggle

Notebook dùng đúng các path sau, không fallback sang artifact khác:

```text
/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/step7/step6/rankings/aiteamvn_vietnamese_reranker_finetuned/public_rankings_step6_fused.jsonl
/kaggle/input/datasets/mphatfromuit/dsc2026-baseline/DSC26_weight_report/legalir_sbert_cl/public_submission/public_ranked_contexts.csv
/kaggle/input/datasets/mphatfromuit/dsc2026-baseline/DSC26_weight_report/legalir_ce_listwise/public_submission/public_official_bm25_sbert_ce_rankings.jsonl
```

## Output

```text
/kaggle/working/step6_sbertft_bgem3rerankft/
  candidates/<candidate>/submission.zip
  run_report.json
```

Submit trước các candidate ít phá baseline:

```text
baseline_plus_ce_w0p15
baseline_plus_ce_w0p25
baseline_plus_ce_w0p40
```

Baseline cần vượt:

```text
rrf_sbert0p60 public recall = 0.9173333333333332
```
