# Step 4 - BGE-M3 Dense Retrieval + Metadata + RRF

Mục tiêu theo `plan_pipeline.md`: thêm BGE-M3 dense chunk retrieval, metadata branch, rồi fuse với BM25 bằng RRF.

Đây là bước cần GPU. Chạy trên Kaggle T4x2.

Input Kaggle hiện tại sau khi Kaggle tự giải nén dataset:

```text
/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/best_config.json
/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/chunks.jsonl
```

Notebook ưu tiên đọc artifact phẳng trong `/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/`. Nó vẫn giữ fallback cho layout cũ `/step4/` hoặc `step4.zip` nếu sau này cần.

Nội dung `step4.zip`:

```text
chunks.jsonl
train_split.json
dev_split.json
dev_rankings_best.jsonl
public_rankings_best.jsonl
train_rankings_best.jsonl
best_config.json
dev_metrics_best.json
```

Code Step 4 nằm trực tiếp trong `kaggle_step4_bge_rrf.ipynb` theo các cell rõ ràng: Kaggle input paths, Step 2 helpers, Step 4 BGE/RRF logic và cell chạy cuối. Notebook không tự ghi script `.py` vào `/kaggle/working` nữa. Dataset zip chỉ chứa artifact.

Notebook có internet nên có thể tải model:

```text
BAAI/bge-m3
```

Model có trong `[DSC@UIT 2026] Danh sách mô hình - Sheet1.csv`; model card kiểm tra được: license MIT, embedding dimension 1024, sequence length 8192, hỗ trợ dense retrieval qua `FlagEmbedding`.

Chạy bằng:

```text
kaggle_step4_bge_rrf.ipynb
```

Output chính:

- `embeddings/chunk_embeddings_fp16.npy`: cache dense vectors, rất lớn, không cần upload nếu bước sau không tái dùng dense search.
- `rankings/dev_rankings_rrf.jsonl`: BM25 + dense + metadata RRF candidates.
- `metrics/dev_metrics_rrf.json`: dev metric.
- `predictions/dev_predictions_top5_rrf.json`: dev top 5.
- `reports/run_report.json`: config và input provenance.

Kaggle có thể tự thêm `__huggingface_repos__.json` khi notebook tải model từ Hugging Face. File này không phải artifact retrieval/submission, nhưng có ích cho compliance vì ghi `repoId` và `commitHash` của model đã tải. Không cần upload file này cho Step 5 trừ khi muốn giữ provenance model.

`public_rankings_best.jsonl` đã nằm trong `step4.zip`, nên public submission dùng được RRF đầy đủ với BM25 + dense + metadata.

Notebook mặc định bật `--predict-public`, nên mỗi lần chạy Step 4 đều tạo:

- `submission/submission.json`
- `submission/submission.zip`
- `submission/submission_validation.json`
- `rankings/public_rankings_rrf.jsonl`

Artifact cần upload cho Step 5 GPU fine-tune Bi-Encoder:

- Step 1 data: `chunks.jsonl`, `train_split.json`, `dev_split.json`
- Step 4 candidates: `rankings/dev_rankings_rrf.jsonl`
- Nếu mine negatives trên train: chạy Step 4 thêm `--eval-train` và upload `rankings/train_rankings_rrf.jsonl`.

`train_rankings_best.jsonl` cũng đã nằm trong artifact Step 4. Nếu cần train candidates cho bước sau, truyền:

```bash
--bm25-train-rankings-file /kaggle/input/datasets/bowboochua9/stnhdscduaiti26/train_rankings_best.jsonl --eval-train
```

Không cần upload `chunk_embeddings_fp16.npy` trừ khi muốn tránh encode lại trong notebook khác. File này lớn nhưng là cache vận hành, không phải data bắt buộc.
