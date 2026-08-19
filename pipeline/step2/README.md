# Step 2 - Chunk-BM25 Baseline

Mục tiêu: tái lập baseline `Chunk-BM25 -> aggregate chunk về document -> evaluate dev`.

Input Kaggle Dataset tối thiểu:

```text
/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/chunks.jsonl
/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/train_split.json
/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/dev_split.json
```

Chạy local:

```bash
python task1/pipeline/step2/legalir_step2_bm25.py --data-root task1/pipeline/step1/outputs
```

Output local mặc định:

```text
task1/pipeline/step2/outputs/
```

Chạy trên Kaggle bằng `kaggle_step2_chunk_bm25.ipynb`.

Output Kaggle:

```text
/kaggle/working/step2/
```

Các output chính:

- `metrics/dev_metrics.json`: Recall/Precision/Hit/MRR trên dev.
- `predictions/dev_predictions_top5.json`: top 5 document cho từng dev query.
- `rankings/dev_rankings.jsonl`: top 100 document, kèm tối đa 3 evidence chunk/document.
- `reports/run_report.json`: config BM25 và thống kê index.

Tạo submission public chỉ khi đáng dùng một lượt nộp:

```bash
python task1/pipeline/step2/legalir_step2_bm25.py ^
  --data-root task1/pipeline/step1/outputs ^
  --public-file task1/public-official.json ^
  --predict-public
```

Output submission:

- `submission/submission.json`
- `submission/submission.zip`
- `submission/submission_validation.json`

`submission.zip` chỉ chứa đúng một file `submission.json` theo format:

```json
{
  "question_id": {
    "answer": ["doc_id_1", "doc_id_2"]
  }
}
```

Không nên nộp mọi ablation. Chỉ nộp thử khi dev metric hoặc logic pipeline đổi thật sự đáng kể.

Step 3 cần upload output Step 2 nếu chạy notebook Kaggle riêng và muốn dùng lại candidate/ranking BM25:

- `rankings/dev_rankings.jsonl`
- `predictions/dev_predictions_top5.json`
- `metrics/dev_metrics.json`

Nếu Step 3 chỉ ablation BGE-M3 + RRF trên dev thì ba file trên là đủ. Nếu các bước sau cần mine candidates trên train, chạy Step 2 thêm `--eval-train` rồi upload thêm:

- `rankings/train_rankings.jsonl`
- `predictions/train_predictions_top5.json`
- `metrics/train_metrics.json`

Không cần upload BM25 config/report nếu config được version trong code/notebook.
