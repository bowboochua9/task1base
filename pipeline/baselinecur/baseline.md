# Baseline hiện tại tốt nhất

Baseline public tốt nhất hiện tại là fusion từ Step 6, SBERT-FT và LLM/XGB:

```text
Step 6 AITeamVN/Vietnamese_Reranker fused ranking
+ SBERT-FT public ranking
+ LLM/XGB reranker public ranking
+ weighted RRF
```

Đây là pipeline đang có public recall cao nhất đã ghi nhận:

```text
candidate          = baseline_plus_xgb_w0p25
step6_weight       = 1.0
sbert_weight       = 0.6
xgb_weight         = 0.25
ce_weight          = 0.0
rrf_k              = 60
Public precision   = 0.20040000000000002
Public recall      = 0.9288333333333333
```

File nộp đã copy vào folder này:

```text
task1/pipeline/baselinecur/submission.zip
task1/pipeline/baselinecur/submission.json
task1/pipeline/baselinecur/submission_validation.json
task1/pipeline/baselinecur/submission_manifest.md
```

Step 6 riêng vẫn là anchor có dev metric đầy đủ:

```text
Public precision = 0.19220000000000004
Public recall    = 0.894
Dev Recall@5     = 0.905
Dev Precision@5  = 0.193
```

Notebook tái chạy baseline:

```text
task1/pipeline/baselinecur/baseline.ipynb
```

Notebook này được copy từ:

```text
task1/pipeline/step6+sbertft+bgem3rerankft+LLMXGB/kaggle_step6_sbertft_bgem3rerankft_llmxgb_fusion.ipynb
```

## Full pipeline hiện tại

Baseline public tốt nhất hiện tại là fusion sau Step 6, SBERT-FT và LLM/XGB,
nhưng anchor Step 6 phụ thuộc vào toàn bộ artifact được chuẩn bị từ Step 1 đến
Step 5. Luồng đầy đủ hiện tại như sau.

### Step 1 - Data prep

Mục tiêu:

- Validate schema train/public/corpus.
- Kiểm tra duplicate question, missing gold IDs, context IDs.
- Clean text nhưng vẫn giữ trace về document gốc.
- Chunk corpus theo cấu trúc pháp luật và sliding window.
- Chia split cố định `6000 train / 1000 dev`, seed `42`.

Output cần giữ:

- `chunks.jsonl`
- `train_split.json`
- `dev_split.json`
- `reports/validation_report.json`

Thông số chính:

```text
num_docs      = 8,532
num_chunks    = 324,377
train queries = 6,000
dev queries   = 1,000
```

### Step 2 - Chunk-BM25 baseline

Mục tiêu:

- Build BM25 index trên `chunks.jsonl`.
- Retrieve top 300 chunks/query.
- Aggregate chunk về document bằng:

```text
max_score + 0.20 * mean_top3 + 0.05 * support_count
```

- Xuất top 100 documents/query và submission top 5.

Output quan trọng:

- `rankings/dev_rankings.jsonl`
- `rankings/public_rankings.jsonl`
- `metrics/dev_metrics.json`
- `submission/submission.zip`

Vai trò trong full pipeline:

- Là baseline lexical đầu tiên.
- Cung cấp công thức aggregate và metric implementation cho các bước sau.

### Step 3 - Tuned chunk-BM25

Mục tiêu:

- Sweep cấu hình BM25/aggregate nhỏ trên dev.
- Chọn best BM25 candidate config.

Best config hiện tại:

```text
name           = b_0p90
k1             = 1.5
b              = 0.9
top_chunks     = 300
top_docs       = 100
heading_weight = 2.0
```

Output quan trọng:

- `best_config.json`
- `rankings/train_rankings_best.jsonl`
- `rankings/dev_rankings_best.jsonl`
- `rankings/public_rankings_best.jsonl`

Vai trò trong full pipeline:

- Cung cấp BM25 branch đã tune cho Step 4.
- Cung cấp train candidates để các bước GPU sau có thể mine/evaluate ổn định.

### Step 4 - BGE-M3 dense + BM25 RRF

Mục tiêu:

- Dùng `BAAI/bge-m3` để encode toàn bộ chunks.
- Dense retrieval top 300 chunks/query bằng cosine similarity.
- Aggregate dense chunk về document.
- Fuse BM25 tuned branch và dense branch bằng Reciprocal Rank Fusion.

Config chính:

```text
model          = BAAI/bge-m3
dense_top      = 300 chunks
fused_top_docs = 100
rrf_k          = 60
bm25_weight    = 1.0
dense_weight   = 1.0
metadata_weight = 0.2
```

Output quan trọng:

- `step4/chunks.jsonl`
- `step4/train_split.json`
- `step4/dev_split.json`
- `step4/best_config.json`
- `step4/dev_rankings_best.jsonl`
- `step4/train_rankings_best.jsonl`
- `step4/public_rankings_best.jsonl`

Vai trò trong full pipeline:

- Tăng candidate recall mạnh so với BM25 thuần.
- Cung cấp retrieval branch tốt hơn cho Step 5 fine-tuned bi-encoder.

### Step 5 - Fine-tuned Vietnamese Bi-Encoder + RRF

Mục tiêu:

- Fine-tune `bkai-foundation-models/vietnamese-bi-encoder` bằng
  `MultipleNegativesRankingLoss`.
- Positive chunks được chọn từ gold documents.
- Encode toàn corpus bằng bi-encoder đã fine-tune.
- Fuse Step 4 retrieval và fine-tuned dense retrieval bằng RRF.
- Xuất fused rankings cho train/dev/public.

Config chính:

```text
model              = bkai-foundation-models/vietnamese-bi-encoder
epochs             = 1
learning_rate      = 4e-5
weight_decay       = 0.02
max_seq_length     = 256
positives_per_gold = 2
best rrf_k         = 20
step4_weight       = 1.2
ft_dense_weight    = 0.4
```

Output bắt buộc cho Step 6:

- `step6/step5/rankings/train_rankings_step5_fused.jsonl`
- `step6/step5/rankings/dev_rankings_step5_fused.jsonl`
- `step6/step5/rankings/public_rankings_step5_fused.jsonl`
- `step6/step5/reports/model_manifest.json`
- `step6/step5/reports/run_report.json`

Ghi chú layout:

```text
Trong Kaggle dataset hiện tại, Step 5 artifact đầy đủ đang nằm trong
step6/step5/ vì trước đó được đóng gói chung vào Step 6 input bundle.
```

### Step 6 - Fine-tuned reranker baseline tốt nhất

Input chính của Step 6:

- `step4/chunks.jsonl`
- `step4/train_split.json`
- `step4/dev_split.json`
- `step6/step5/rankings/train_rankings_step5_fused.jsonl`
- `step6/step5/rankings/dev_rankings_step5_fused.jsonl`
- `step6/step5/rankings/public_rankings_step5_fused.jsonl`

Luồng xử lý Step 6:

1. Load chunks và Step 5 fused candidate rankings.
2. Build training pairs cho reranker từ top candidates.
3. Với mỗi positive, mine 10 semi-hard negatives sau khi loại toàn bộ gold
   document của query.
4. Fine-tune reranker bằng BCE loss trong 2 epoch.
5. Rerank top 50 documents.
6. Fuse score reranker với retrieval score.
7. Chọn config theo dev Recall@5, Precision@5 làm tie-break.
8. Xuất `submission.zip` với đúng tối đa 5 document IDs/query.

Best candidate:

```text
slug      = aiteamvn_vietnamese_reranker_finetuned
model_id  = AITeamVN/Vietnamese_Reranker
params    = 567,755,777
max_len   = 384
epochs    = 2
lr        = 2e-5
negatives = 10 semi-hard negatives / positive
```

Best fusion:

```text
reranker_weight  = 0.8
retrieval_weight = 0.5
```

## Output tốt nhất

Best public submission đã copy vào `baselinecur`:

```text
task1/pipeline/baselinecur/submission.zip
```

Nguồn gốc:

```text
task1/pipeline/step6+sbertft+bgem3rerankft+LLMXGB/step6_sbertft_bgem3rerankft_llmxgb/candidates/baseline_plus_xgb_w0p25/submission.zip
```

Validation:

```text
num_public_queries     = 1000
num_submission_queries = 1000
answer length          = 5 for all queries
num_errors             = 0
num_warnings           = 0
```

Các artifact quan trọng:

- `task1/pipeline/step6/step6/models/aiteamvn_vietnamese_reranker_finetuned/`
- `task1/pipeline/step6/step6/rankings/aiteamvn_vietnamese_reranker_finetuned/dev_rankings_step6_fused.jsonl`
- `task1/pipeline/step6/step6/rankings/aiteamvn_vietnamese_reranker_finetuned/public_rankings_step6_fused.jsonl`
- `task1/pipeline/step6/step6/metrics/aiteamvn_vietnamese_reranker_finetuned/dev_metrics_step6_fused.json`
- `task1/pipeline/step6/step6/reports/candidate_summary_sorted.json`
- `task1/pipeline/step6/step6/reports/public_scores_manual.json`
- `task1/pipeline/step6+sbertft/outputs/candidates/rrf_sbert0p60/submission.zip`
- `task1/pipeline/step6+sbertft+bgem3rerankft+LLMXGB/step6_sbertft_bgem3rerankft_llmxgb/run_report.json`
- `task1/pipeline/step6+sbertft+bgem3rerankft+LLMXGB/step6_sbertft_bgem3rerankft_llmxgb/candidates/baseline_plus_xgb_w0p25/`

## Vì sao giữ Step 6 làm anchor

So với các mốc trước, Step 6b tăng public recall rõ nhất:

```text
Step 2 -> Step 3: +0.012667 recall
Step 3 -> Step 4: +0.090333 recall
Step 4 -> Step 5: +0.014000 recall
Step 5 -> Step 6b: +0.049167 recall
```

Step 6b cũng tốt hơn Step 6 PhoRanker:

```text
precision: +0.005000
recall:    +0.023333
```

Vì vậy, mọi thử nghiệm kế tiếp nên so với mốc:

```text
Dev Recall@5 >= 0.905
Public Recall >= 0.894
```

Nhưng nếu xét file nộp public tốt nhất đã biết, mốc cần vượt là:

```text
Public Recall >= 0.9288333333333333
Public Precision >= 0.20040000000000002
```
