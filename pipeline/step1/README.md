# Step 1 - Data Validator, Cleaning, Chunking, Metric

Chạy local:

```bash
python task1/pipeline/step1/legalir_step1.py --input-root task1 --context-dir selected-contexts/selected-contexts
```

Output local mặc định:

```text
task1/pipeline/step1/outputs/
```

Chạy trên Kaggle bằng `kaggle_step1_data_prep.ipynb`.

Input Kaggle:

```text
/kaggle/input/datasets/ttdatto/uit-dsc26/LegalIR - Public Test/
```

Output Kaggle:

```text
/kaggle/working/step1/
```

Nếu Step 2 chạy trong notebook Kaggle riêng, chỉ cần upload artifact data thật sự:

- `corpus/chunks.jsonl`
- `splits/train_split.json` hoặc bản copy `train_split.json`
- `splits/dev_split.json` hoặc bản copy `dev_split.json`

Không cần upload `chunk_manifest.json`, `validation_report.json`, `train_ids.json`, `dev_ids.json` hoặc `cleaned_contexts.jsonl` cho Step 2 nếu config/report đã được version trong code/notebook.
