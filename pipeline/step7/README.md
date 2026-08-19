# Step 7 - Qwen3 reranker, score fusion, ablation

Step 7 adds `Qwen/Qwen3-Reranker-0.6B` on top of the best Step 6 output:
`aiteamvn_vietnamese_reranker_finetuned`.

## Files

- `kaggle_step7_qwen3_reranker.ipynb`: Kaggle notebook. Code lives here, not in the dataset artifact.
- `step7.zip`: artifact-only package to upload to the Kaggle dataset.
- `step7_zip_manifest.json`: list of files included in `step7.zip`.

## Required Kaggle inputs

Existing dataset files from previous uploads:

- `/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/step4/chunks.jsonl`
- `/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/step4/dev_split.json`
- `/kaggle/input/datasets/ttdatto/uit-dsc26/LegalIR - Public Test/public-official.json`

New upload for this step:

- Upload `task1/pipeline/step7/step7.zip`.
- Kaggle should expose its contents under:
  `/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/step7/step6/...`
- Inside the zip, the top-level folder is `step6/`, not `step7/`.

## Outputs

The notebook writes to `/kaggle/working/step7`:

- `rankings/qwen3_reranker/dev_rankings_qwen3.jsonl`
- `rankings/qwen3_reranker/public_rankings_qwen3.jsonl`
- `rankings/step7_fused_best/dev_rankings_step7_fused.jsonl`
- `rankings/step7_fused_best/public_rankings_step7_fused.jsonl`
- `metrics/dev_metrics_qwen3_only.json`
- `metrics/dev_metrics_step7_fused.json`
- `metrics/fusion_trials.json`
- `submission/qwen3_only/submission.zip`
- `submission/step7_fused_best/submission.zip`
- `reports/model_manifest.json`
- `reports/run_report.json`

Submit `submission/step7_fused_best/submission.zip` first unless dev ablation
shows `qwen3_only` is better.

## Current Result

The first Kaggle run reached the session limit after finishing dev Qwen3
reranking, before public Qwen3 reranking.

Dev result:

- Step 6 baseline Recall@5: `0.905`
- Qwen3-only Recall@5: `0.7223333333333334`
- Step 7 fused Recall@5: `0.8998333333333333`

Decision: reject Qwen3 for the current configuration because it lowers dev
Recall@5. The notebook now skips public Qwen3 reranking when dev does not
improve, and writes `submission/step7_rejected_use_step6/submission.zip`.
