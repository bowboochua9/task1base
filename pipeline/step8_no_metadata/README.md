# Step 8 No-Metadata P0

Nhánh này triển khai P0 trong `task1/pipeline/plan_caithien.md`: bỏ metadata branch, replay Step 4/5 bằng artifact đã có, rồi chuẩn bị input cho Step 6 AITeamVN rerank top 50 mới.

## Local replay

Chạy local, không cần GPU:

```powershell
$env:PYTHONIOENCODING='utf-8'
python task1\pipeline\step8_no_metadata\replay_p0_no_metadata.py
```

Output chính:

```text
task1/pipeline/step8_no_metadata/rankings/dev_rankings_step5_no_metadata_fused.jsonl
task1/pipeline/step8_no_metadata/rankings/public_rankings_step5_no_metadata_fused.jsonl
task1/pipeline/step8_no_metadata/reports/run_report.json
task1/pipeline/step8_no_metadata/submission/step5_no_metadata/submission.zip
```

## Metric local hiện tại

| Stage | R@1 | R@5 | R@10 | R@20 | R@50 | R@90 | R@100 | Precision@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Step 4 current metadata | 0.502000 | 0.852833 | 0.912833 | 0.950417 | 0.970333 | 0.981167 | 0.982000 | 0.180600 |
| Step 4 no-metadata | 0.515667 | 0.855500 | 0.916833 | 0.950417 | 0.970333 | 0.981500 | 0.982500 | 0.181000 |
| Step 5 current metadata | 0.510667 | 0.874250 | 0.931083 | 0.960167 | 0.976167 | 0.980667 | 0.982000 | 0.185600 |
| Step 5 no-metadata | 0.516167 | 0.877250 | 0.931083 | 0.960167 | 0.976667 | 0.982500 | 0.982500 | 0.186200 |

## Kaggle dataset zip

`step8_no_metadata.zip` chỉ chứa data input mới cần cho Step 8:

```text
step8_no_metadata/rankings/dev_rankings_step5_no_metadata_fused.jsonl
step8_no_metadata/rankings/public_rankings_step5_no_metadata_fused.jsonl
step8_no_metadata/configs/canonical_dev_ids.json
step8_no_metadata/configs/p0_no_metadata_config.json
```

Theo log Kaggle hiện tại, upload `step8_no_metadata.zip` đang được Kaggle mount
dưới wrapper folder `step8_no_metadata/`, nên path thực tế có hai tầng
`step8_no_metadata/step8_no_metadata/`:

```text
/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/step8_no_metadata/step8_no_metadata/rankings/dev_rankings_step5_no_metadata_fused.jsonl
/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/step8_no_metadata/step8_no_metadata/rankings/public_rankings_step5_no_metadata_fused.jsonl
```

Không đóng gói notebook, `.py`, report, metric hay README vào zip dataset.
Zip phải dùng entry path dạng POSIX `/`, không dùng `\`. Tránh tạo bằng
`Compress-Archive` trên Windows nếu Kaggle báo forbidden character.

P0 GPU rerank còn cần model Step 6 AITeamVN. Upload thêm:

```text
task1/pipeline/step8_no_metadata/step8_reranker_model.zip
```

Zip này chứa:

```text
step6/models/aiteamvn_vietnamese_reranker_finetuned/config.json
step6/models/aiteamvn_vietnamese_reranker_finetuned/model.safetensors
step6/models/aiteamvn_vietnamese_reranker_finetuned/tokenizer.json
step6/models/aiteamvn_vietnamese_reranker_finetuned/tokenizer_config.json
```

Theo log Kaggle hiện tại, zip model này đang được mount dưới wrapper folder
`step8_reranker_model/`, nên path model thực tế là:

```text
/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/step8_reranker_model/step6/models/aiteamvn_vietnamese_reranker_finetuned
```

## GPU rerank Step 6

Notebook standalone:

```text
task1/pipeline/step8_no_metadata/kaggle_p0_step6_no_metadata_rerank.ipynb
```

Notebook tự chứa code rerank, không ghi script tạm vào `/kaggle/working`, và
dùng input cố định:

```text
/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/step8_no_metadata/step8_no_metadata/rankings/*.jsonl
/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/step4/chunks.jsonl
/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/step8_reranker_model/step6/models/aiteamvn_vietnamese_reranker_finetuned
```

Không có fallback path. Các path trên được lấy theo đúng `DATA_ROOT tree preview`
trong log Kaggle mới nhất.

Cấu hình rerank hiện dùng cho T4 16GB:

```text
MAX_SEQ_LENGTH = 320
EVIDENCE_MAX_CHARS = 1400
RERANK_BATCH_SIZE = 8
torch.inference_mode + FP16 autocast
```

Không tăng lại `RERANK_BATCH_SIZE = 32` trên T4, vì log Kaggle đã OOM ở batch
32 với model AITeamVN 568M.

Output GPU dự kiến:

```text
/kaggle/working/step8_no_metadata_step6/rankings/aiteamvn_vietnamese_reranker_finetuned_no_metadata/
/kaggle/working/step8_no_metadata_step6/metrics/aiteamvn_vietnamese_reranker_finetuned_no_metadata/
/kaggle/working/step8_no_metadata_step6/submission/aiteamvn_vietnamese_reranker_finetuned_no_metadata/submission.zip
/kaggle/working/step8_no_metadata_step6/reports/run_report.json
```

Chỉ nộp public nếu `dev_metrics_step6_fused.json` tăng so với Step 6 control cùng split.

## Kết quả GPU đã chạy

Output đã tải về local:

```text
task1/pipeline/step8_no_metadata/step8_no_metadata_step6/
```

Metric chính:

```text
best_fusion:
  reranker_weight  = 0.7
  retrieval_weight = 0.5

fused dev:
  Recall@5     = 0.9048333333333335
  Precision@5  = 0.192799999999998
  Recall@10    = 0.9472500000000003
  Recall@20    = 0.9675000000000001
  Recall@50    = 0.9766666666666668
  Recall@100   = 0.9825
  Hit@5        = 0.921
  MRR          = 0.7364215758203391

submission validation:
  num_public_queries     = 1000
  num_submission_queries = 1000
  answer_length          = 5 for all queries
  num_errors             = 0
  num_warnings           = 0
```

So với Step 6 control cùng split:

```text
Step 6 control Recall@5 = 0.905000
P0 no-metadata Recall@5 = 0.9048333333333335
```

P0 chưa vượt dev, nên không ưu tiên nộp public nếu còn ít lượt nộp.

Quy ước cho các lần chạy Kaggle sau: nếu artifact/output đã có local hoặc đã
có trong Kaggle input dataset, notebook không xuất lại bản trùng. Chỉ xuất file
mới cần cho bước kế tiếp, metric/report/submission mới, và manifest ngắn nếu cần.
