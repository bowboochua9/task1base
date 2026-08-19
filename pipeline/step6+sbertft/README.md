# Step 6 + SBERT-FT quick fusion

Nhánh này dùng lại các artifact đã chạy xong, không fine-tune lại model:

- Baseline Step 6:
  `task1/pipeline/step6/step6/`
- SBERT contrastive fine-tuned experiment:
  `other_research/kaggle_output/legalir_sbert_cl/`

Dataset Kaggle chứa output SBERT-FT gốc:

```text
/kaggle/input/datasets/mphatfromuit/cl-sbert/kaggle_output
```

Trong đó các file public ranking cần cho fusion nằm ở:

```text
/kaggle/input/datasets/mphatfromuit/cl-sbert/kaggle_output/legalir_sbert_cl/public_submission/public_ranked_contexts.csv
/kaggle/input/datasets/mphatfromuit/cl-sbert/kaggle_output/legalir_sbert_cl/public_submission/submission.json
```

Mục tiêu là sinh nhanh `submission.zip` để nộp thử Codabench bằng cách fuse
ranking public của Step 6 với ranking public của SBERT-FT.

## Files

- `kaggle_step6_sbertft_fusion.ipynb`: notebook chạy fuse và tạo submission.
- `step6+sbertft.zip`: input artifact upload lên Kaggle Dataset.
- `step6+sbertft/`: nội dung đã đóng gói trong zip.
- `outputs/`: kết quả chạy local từ notebook/script tương đương.

## Input contract

`step6+sbertft.zip` có top-level folder `step6+sbertft/`:

```text
step6+sbertft/
|-- public-official.json
|-- step6/
|   |-- rankings/public_rankings_step6_fused.jsonl
|   |-- reports/run_report.json
|   |-- reports/public_scores_manual.json
|   `-- submission/submission.json
`-- sbertft/
    |-- public_submission/public_ranked_contexts.csv
    |-- public_submission/submission.json
    |-- reports/experiment_manifest.json
    `-- logs/
        |-- val_method_comparison.csv
        `-- test_method_comparison.csv
```

Không đóng gói SBERT model weights vì nhánh này không inference lại model; nó
chỉ dùng public rankings đã có.

## Fusion mặc định

Notebook dùng weighted RRF:

```text
score(doc) = step6_weight / (rrf_k + step6_rank)
           + sbert_weight / (rrf_k + sbert_rank)
```

Candidate mặc định sau khi có public score:

```text
rrf_sbert0p60
step6_weight = 1.0
sbert_weight = 0.6
rrf_k = 60
```

Step 6 vẫn là neo chính, SBERT-FT thêm tín hiệu phụ. Notebook cũng sinh thêm
candidate `rrf_sbert0p25`, `rrf_sbert0p40`, `rrf_equal` và `step6_then_sbert`
để audit nhanh.

## Cách chạy

Local:

```powershell
jupyter notebook task1/pipeline/step6+sbertft/kaggle_step6_sbertft_fusion.ipynb
```

Kaggle:

1. Upload `task1/pipeline/step6+sbertft/step6+sbertft.zip` vào Dataset.
2. Attach Dataset vào notebook.
3. Chạy toàn bộ `kaggle_step6_sbertft_fusion.ipynb`.
4. Tải:

```text
/kaggle/working/step6_sbertft/submission.zip
```

## Baseline cần so

Step 6 hiện tại:

```text
Dev Recall@5     = 0.905
Dev Precision@5  = 0.193
Public Recall    = 0.894
Public Precision = 0.1922
```

SBERT-FT standalone public đã được ghi nhận khoảng:

```text
Public Recall    = 0.890583
Public Precision = 0.1910
```

Kết quả public đã nộp:

| Candidate | Public Precision | Public Recall |
|---|---:|---:|
| `rrf_sbert0p25` | 0.19700000000000004 | 0.9136666666666666 |
| `rrf_sbert0p40` | 0.19720000000000004 | 0.9161666666666666 |
| `rrf_sbert0p60` | **0.19740000000000005** | **0.9173333333333332** |

`rrf_sbert0p60` hiện là baseline public tốt nhất.
