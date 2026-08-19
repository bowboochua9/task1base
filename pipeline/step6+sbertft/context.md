# Context cho nhánh `step6+sbertft`

Tài liệu này dùng để handoff cho chat mới khi tiếp tục nhánh
`task1/pipeline/step6+sbertft/`.

Mục tiêu của nhánh này: dùng thẳng baseline Step 6 tốt nhất và ranking
SBERT-FT đã fine-tune sẵn để tạo nhanh `submission.zip` nộp Codabench. Nhánh
này không train lại model và không inference lại model.

## Baseline cần giữ trong đầu

Full pipeline hiện tại trước nhánh này:

1. Step 1: validate, clean, chunk corpus, tạo split `6000 train / 1000 dev`.
2. Step 2: chunk-BM25 baseline.
3. Step 3: tuned chunk-BM25.
4. Step 4: `BAAI/bge-m3` dense retrieval + BM25 RRF.
5. Step 5: fine-tune `bkai-foundation-models/vietnamese-bi-encoder` + RRF.
6. Step 6: fine-tune reranker và fuse với retrieval.
7. Step 7: Qwen3 reranker đã thử nhưng không thay baseline Step 6.

Baseline tốt nhất hiện tại là Step 6:

```text
model_id          = AITeamVN/Vietnamese_Reranker
slug              = aiteamvn_vietnamese_reranker_finetuned
dev Recall@5      = 0.905
dev Precision@5   = 0.193
public Recall     = 0.894
public Precision  = 0.19220000000000004
best submission   = task1/pipeline/step6/step6/submission/aiteamvn_vietnamese_reranker_finetuned/submission.zip
```

SBERT-FT standalone public đã ghi nhận:

```text
public Recall     = 0.8905833333333334
public Precision  = 0.19100000000000003
```

Không được gọi nhánh `step6+sbertft` là baseline mới trước khi public score
vượt Step 6.

## Cấu trúc thư mục hiện tại

Local folder:

```text
task1/pipeline/step6+sbertft/
|-- README.md
|-- context.md
|-- kaggle_step6_sbertft_fusion.ipynb
|-- step6+sbertft.zip
|-- step6+sbertft/
|   |-- public-official.json
|   |-- step6/
|   |   |-- rankings/public_rankings_step6_fused.jsonl
|   |   |-- reports/run_report.json
|   |   |-- reports/public_scores_manual.json
|   |   `-- submission/submission.json
|   `-- sbertft/
|       |-- public_submission/public_ranked_contexts.csv
|       |-- public_submission/submission.json
|       |-- reports/experiment_manifest.json
|       `-- logs/
|           |-- val_method_comparison.csv
|           `-- test_method_comparison.csv
`-- outputs/
    |-- run_report.json
    |-- submission.zip
    `-- candidates/<candidate>/submission.zip
```

`step6+sbertft.zip` có top-level folder `step6+sbertft/`. Khi upload vào Kaggle
Dataset chung hiện tại, path đúng cần dùng là:

```text
/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/step6+sbertft
```

Ngoài ra, output SBERT-FT gốc trong:

```text
other_research/kaggle_output
```

đã được upload lên Kaggle Dataset riêng:

```text
/kaggle/input/datasets/mphatfromuit/cl-sbert/kaggle_output
```

Ví dụ:

```text
/kaggle/input/datasets/mphatfromuit/cl-sbert/kaggle_output/dsc2026.log
```

Nếu không dùng bundle `step6+sbertft.zip`, chat mới có thể lấy SBERT public
ranking trực tiếp từ dataset này:

```text
/kaggle/input/datasets/mphatfromuit/cl-sbert/kaggle_output/legalir_sbert_cl/public_submission/public_ranked_contexts.csv
/kaggle/input/datasets/mphatfromuit/cl-sbert/kaggle_output/legalir_sbert_cl/public_submission/submission.json
```

Nhánh P0 `step8_no_metadata` không dùng dataset `cl-sbert`; P0 chỉ cần
no-metadata rankings, chunks và model AITeamVN Step 6.

Các file bắt buộc dưới Kaggle:

```text
/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/step6+sbertft/public-official.json
/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/step6+sbertft/step6/rankings/public_rankings_step6_fused.jsonl
/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/step6+sbertft/step6/submission/submission.json
/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/step6+sbertft/sbertft/public_submission/public_ranked_contexts.csv
/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/step6+sbertft/sbertft/public_submission/submission.json
```

Nếu chat mới thấy path khác trong log, phải in tree rồi sửa path chính xác.
Không tự fallback sang Step 4/Step 5/Step 7 artifact khác.

## Điều chỉnh cần làm trong notebook

Notebook hiện tại:

```text
task1/pipeline/step6+sbertft/kaggle_step6_sbertft_fusion.ipynb
```

Notebook đang có hàm scan nhẹ `/kaggle/input` để tìm root. Theo quy ước pipeline
hiện tại của project, nên đổi sang path cố định để tránh chạy nhầm dataset.

Nên sửa cell `Locate input artifact` thành dạng explicit:

```python
IS_KAGGLE = Path('/kaggle/input').exists()
OUTPUT_DIR = Path('/kaggle/working/step6_sbertft') if IS_KAGGLE else Path('task1/pipeline/step6+sbertft/outputs')

if IS_KAGGLE:
    DATA_ROOT = Path('/kaggle/input/datasets/bowboochua9/stnhdscduaiti26')
    INPUT_ROOT = DATA_ROOT / 'step6+sbertft'
else:
    INPUT_ROOT = Path('task1/pipeline/step6+sbertft/step6+sbertft')

REQUIRED_RELATIVE_FILES = [
    Path('public-official.json'),
    Path('step6/rankings/public_rankings_step6_fused.jsonl'),
    Path('step6/submission/submission.json'),
    Path('sbertft/public_submission/public_ranked_contexts.csv'),
    Path('sbertft/public_submission/submission.json'),
]

missing = [str(INPUT_ROOT / rel) for rel in REQUIRED_RELATIVE_FILES if not (INPUT_ROOT / rel).exists()]
if missing:
    print('INPUT_ROOT:', INPUT_ROOT)
    print('Missing required files:')
    print('\\n'.join(missing))
    raise FileNotFoundError('Missing required step6+sbertft artifacts.')
```

Không dùng automatic fallback. Nếu dataset được upload vào slug khác, sửa
`DATA_ROOT` một lần theo path Kaggle thật rồi chạy lại.

## Notebook đang làm gì

Notebook không train, không load model, không dùng GPU.

Input:

- Step 6 public fused ranking.
- Step 6 public submission top 5.
- SBERT-FT public ranked contexts.
- SBERT-FT public submission top 5.
- Public question IDs từ `public-official.json`.

Output:

- `/kaggle/working/step6_sbertft/submission.zip`
- `/kaggle/working/step6_sbertft/candidates/<candidate>/submission.zip`
- `/kaggle/working/step6_sbertft/run_report.json`

Các candidate hiện có:

```text
rrf_sbert0p25
rrf_sbert0p40
rrf_sbert0p60
rrf_equal
step6_then_sbert
```

Fusion mặc định:

```text
method        = weighted RRF
rrf_k         = 60
step6_weight  = 1.0
sbert_weight  = 0.4
candidate     = rrf_sbert0p40
```

Sau khi nộp Codabench, default nên đổi sang:

```text
method        = weighted RRF
rrf_k         = 60
step6_weight  = 1.0
sbert_weight  = 0.6
candidate     = rrf_sbert0p60
public score  = precision 0.19740000000000005, recall 0.9173333333333332
```

`step6_then_sbert` giữ nguyên top 5 của Step 6 nên không tạo prediction mới
thật sự; nó chỉ là control để kiểm tra validator.

## Kết quả local đã có

`outputs/run_report.json` hiện nên ghi:

```text
status            = ok
default_candidate = rrf_sbert0p60
method            = quick_public_rank_fusion_no_training_no_model_inference
```

Candidate audit:

| Candidate | Changed vs Step6 | Avg top5 overlap Step6 | Avg top5 overlap SBERT |
|---|---:|---:|---:|
| rrf_sbert0p25 | 838 | 4.018 | 3.154 |
| rrf_sbert0p40 | 941 | 3.886 | 3.306 |
| rrf_sbert0p60 | 975 | 3.742 | 3.457 |
| rrf_equal | 985 | 3.574 | 3.669 |
| step6_then_sbert | 0 | 5.000 | 2.708 |

Tất cả submission validation hiện pass:

```text
num_public_queries     = 1000
num_submission_queries = 1000
answer_length          = 5 for all queries
num_errors             = 0
```

## Điều chỉnh để đồng bộ với Step 1-7 trước đó

1. Giữ Step 6 AITeamVN là anchor. Mọi fusion phải so với public Recall `0.894`.
2. Không lấy artifact từ Step 4/5/7 nếu file trong `step6+sbertft/` thiếu.
   Thiếu file thì fail-fast và yêu cầu upload lại zip đúng.
3. Không train/inference trong notebook này. Nếu cần dùng SBERT-FT như nhánh
   retrieval thật, đó là một notebook khác.
4. Không tune weight bằng public score sau nhiều lần nộp. Có thể tạo vài
   candidate để nộp thử hạn chế, nhưng phải ghi candidate nào đã nộp và score.
5. Kết quả public mới cho thấy cả ba candidate đều vượt Step 6:

```text
rrf_sbert0p25: precision 0.19700000000000004, recall 0.9136666666666666
rrf_sbert0p40: precision 0.19720000000000004, recall 0.9161666666666666
rrf_sbert0p60: precision 0.19740000000000005, recall 0.9173333333333332
```

   `rrf_sbert0p60` là best hiện tại.

6. Nếu candidate nào vượt Step 6 public Recall `0.894`, cập nhật:

```text
task1/pipeline/baselinecur/respublic.md
task1/pipeline/baselinecur/baseline.md
task1/pipeline/step6+sbertft/README.md
task1/pipeline/step6+sbertft/context.md
```

## File chat mới nên đọc trước khi làm

```text
task1/pipeline/baselinecur/respublic.md
task1/pipeline/baselinecur/baseline.md
task1/pipeline/step6+sbertft/README.md
task1/pipeline/step6+sbertft/context.md
task1/pipeline/step6+sbertft/kaggle_step6_sbertft_fusion.ipynb
task1/pipeline/step6+sbertft/outputs/run_report.json
```

## Checklist trước khi đưa submission cho người dùng

- `submission.zip` chỉ chứa đúng `submission.json`.
- Có đúng 1000 public query IDs.
- Mỗi answer có 1-5 doc IDs, hiện mặc định là 5.
- Không có duplicate doc IDs trong một answer.
- Không có ID ngoài corpus theo validator hiện có.
- Ghi rõ candidate slug, ví dụ `rrf_sbert0p25`.
- Không tuyên bố tốt hơn Step 6 nếu chưa có Codabench public score.
