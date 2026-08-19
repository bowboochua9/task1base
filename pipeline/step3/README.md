# Step 3 - Tune Chunk-BM25 To Plateau

Mục tiêu theo `plan_pipeline.md`: tối ưu Chunk-BM25 trước khi chuyển sang dense/BGE-M3.

Input tối thiểu:

```text
chunks.jsonl
train_split.json
dev_split.json
```

Chạy local:

```bash
python task1/pipeline/step3/legalir_step3_tune_bm25.py --data-root task1/pipeline/step1/outputs
```

Output local mặc định:

```text
task1/pipeline/step3/outputs/
```

Output chính:

- `metrics/ablation_summary.json`: toàn bộ trial, sort theo `Recall@5`, rồi `Precision@5`.
- `metrics/dev_metrics_best.json`: metric của cấu hình tốt nhất.
- `configs/best_config.json`: cấu hình thắng dev.
- `metrics/best_trial_summary.json`: metric chi tiết của trial thắng.
- `rankings/dev_rankings_best.jsonl`: top 100 document kèm evidence chunks cho dev.
- `predictions/dev_predictions_top5_best.json`: top 5 dev predictions.
- `reports/run_report.json`: tóm tắt run và input cần cho bước sau.

Tạo public submission chỉ khi đáng dùng lượt nộp:

```bash
python task1/pipeline/step3/legalir_step3_tune_bm25.py \
  --data-root task1/pipeline/step1/outputs \
  --best-config-file task1/pipeline/step3/outputs/configs/best_config.json \
  --public-file task1/public-official.json \
  --predict-public
```

Output public:

- `submission/submission.json`
- `submission/submission.zip`
- `submission/submission_validation.json`
- `rankings/public_rankings_best.jsonl`

Nếu Step 4 chạy Kaggle GPU riêng, cần upload các data/artifact sau:

- Từ Step 1 dataset: `chunks.jsonl`, `train_split.json`, `dev_split.json`
- Từ Step 3 output: `rankings/dev_rankings_best.jsonl`, `metrics/dev_metrics_best.json`, `configs/best_config.json`

Nếu các bước sau cần mine negative hoặc train trên toàn train, chạy thêm:

```bash
python task1/pipeline/step3/legalir_step3_tune_bm25.py \
  --data-root task1/pipeline/step1/outputs \
  --eval-train
```

Rồi upload thêm:

- `rankings/train_rankings_best.jsonl`
- `metrics/train_metrics_best.json`

Không cần upload report/config phụ nếu chúng đã được version bằng code/notebook. Chỉ upload artifact mà bước GPU sau thực sự đọc.
