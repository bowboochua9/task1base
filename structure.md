

## Dataset gốc

```text
/kaggle/input/datasets/ttdatto/uit-dsc26/LegalIR - Public Test/train.json
/kaggle/input/datasets/ttdatto/uit-dsc26/LegalIR - Public Test/public-official.json
/kaggle/input/datasets/ttdatto/uit-dsc26/LegalIR - Public Test/selected-contexts
```

## Quy ước cập nhật

Sau mỗi lần upload hoặc cập nhật Kaggle Dataset, phải cập nhật lại file này
trước khi sửa notebook hoặc thay path input. Snapshot dưới đây là nguồn sự thật
cho các path `/kaggle/input/...`; notebook không được tự fallback sang path khác
nếu path trong snapshot thiếu hoặc lệch.

## Code dataset DSCuit2026

Dataset này dùng để mount code/notebook của workspace lên Kaggle. Artifact dữ
liệu đã chạy xong vẫn nằm ở dataset `stnhdscduaiti26`; notebook không được lấy
rankings, model checkpoint hoặc submission artifact từ dataset code này.

```text
===== /kaggle/input/dscuit2026 =====
dscuit2026/
  task1/
    structure.md
    context.md
    pipeline/
      plan_pipeline.md
      plan_caithien.md
      step1/
        kaggle_step1_data_prep.ipynb
        legalir_step1.py
        README.md
      step2/
        kaggle_step2_chunk_bm25.ipynb
        legalir_step2_bm25.py
        README.md
      step3/
        kaggle_step3_tune_bm25.ipynb
        legalir_step3_tune_bm25.py
        README.md
      step4/
        kaggle_step4_bge_rrf.ipynb
        legalir_step4_bge_rrf.py
        README.md
      step5/
        kaggle_step5_finetune_biencoder.ipynb
        README.md
      step6/
        kaggle_step6_phoranker.ipynb
        README.md
      step6+sbertft/
        context.md
        README.md
      step6+sbertft+bgem3rerankft/
        kaggle_step6_sbertft_bgem3rerankft_fusion.ipynb
        README.md
      step6+sbertft+bgem3rerankft+LLMXGB/
        kaggle_step6_sbertft_bgem3rerankft_llmxgb_fusion.ipynb
        README.md
      step7/
        plan_implement_step7b.md
        step7b_1_clean_sbert_cl_train.ipynb
      step8_no_metadata/
        kaggle_p0_step6_no_metadata_rerank.ipynb
      baselinecur/
        baseline.ipynb
        baseline.md
        respublic.md
```

## Artifact pipeline

Hiện có 2 Kaggle Dataset artifact chính cần phân biệt:

- `stnhdscduaiti26`: artifact pipeline của mình theo từng step, gồm split,
  chunks, rankings, model checkpoint Step 5/6, Step 7, Step 8 và các submission
  đã tạo.
- `DSC2026_baseline`: artifact/weight/report từ nhánh research bên ngoài, dùng
  để lấy tín hiệu SBERT-FT, CE BGE-M3 reranker v2 và LLM/XGB khi fusion. Đây
  không phải output do các step 1-8 của mình trực tiếp sinh ra, nên mọi kết quả
  dùng từ đây phải ghi rõ nguồn và lưu ý split train/dev khác pipeline chính.

```text
===== /kaggle/input/datasets/bowboochua9/stnhdscduaiti26 =====
stnhdscduaiti26/
  step4/
    best_config.json  [253.0B]
    chunks.jsonl  [669.4MB]
    dev_metrics_best.json  [305.2KB]
    dev_rankings_best.jsonl  [48.4MB]
    dev_split.json  [197.6KB]
    public_rankings_best.jsonl  [48.1MB]
    train_rankings_best.jsonl  [45.4MB]
    train_split.json  [1.2MB]
  p1_2_merged_input/
    split_manifest.json
    feature_contract.json
    preflight_report.json
    gpu_notebook_static_audit_report.json
    step3/
      outputs/
        rankings/
          train_rankings_best.jsonl  [289.3 MB]
  step5/
    dev_rankings_rrf.jsonl  [95.1MB]
    public_rankings_rrf.jsonl  [95.0MB]
    step4_config.json  [412.0B]
    step4_dev_metrics_rrf.json  [290.7KB]
    step4_huggingface_repos.json  [791.0B]
    step4_run_report.json  [1.9KB]
    step4_submission_validation.json  [176.0B]
  step6/
    step5/
      huggingface_repos.json  [420.0B]
      configs/
        best_fusion_config.json  [67.0B]
        runtime_training_config.json  [733.0B]
        step5_config.json  [670.0B]
      metrics/
        dev_metrics_step5_fused.json  [326.8KB]
        fusion_trials.json  [22.9KB]
      models/
        vietnamese-bi-encoder-mnrl/
          README.md  [54.1KB]
          added_tokens.json  [22.0B]
          bpe.codes  [1.1MB]
          config.json  [752.0B]
          config_sentence_transformers.json  [283.0B]
          model.safetensors  [515.0MB]
          modules.json  [277.0B]
          sentence_bert_config.json  [241.0B]
          tokenizer_config.json  [1.2KB]
          vocab.txt  [874.3KB]
          1_Pooling/
            config.json  [90.0B]
      rankings/
        dev_rankings_ft_dense.jsonl  [28.1MB]
        dev_rankings_step5_fused.jsonl  [2.3MB]
        public_rankings_ft_dense.jsonl  [28.4MB]
        public_rankings_step5_fused.jsonl  [2.3MB]
        train_rankings_step5_fused.jsonl  [7.7MB]
      reports/
        model_manifest.json  [785.0B]
        run_report.json  [2.6KB]
      submission/
        submission_validation.json  [176.0B]
  step7/
    step6/
      candidates/
        aiteamvn_vietnamese_reranker_finetuned/
          best_fusion_config.json  [56.0B]
          run_report.json  [2.2KB]
      metrics/
        aiteamvn_vietnamese_reranker_finetuned/
          dev_metrics_reranker_only.json  [326.9KB]
          dev_metrics_step6_fused.json  [326.6KB]
          fusion_trials.json  [7.1KB]
      rankings/
        aiteamvn_vietnamese_reranker_finetuned/
          dev_rankings_reranker.jsonl  [8.6MB]
          dev_rankings_step6_fused.jsonl  [1.8MB]
          public_rankings_reranker.jsonl  [8.6MB]
          public_rankings_step6_fused.jsonl  [1.8MB]
      reports/
        candidate_summary.json  [4.6KB]
        candidate_summary_sorted.json  [4.6KB]
        huggingface_repos.json  [496.0B]
        public_scores_manual.json  [448.0B]
        run_report.json  [7.9KB]
      submission/
        aiteamvn_vietnamese_reranker_finetuned/
          submission_validation.json  [176.0B]
  step8_no_metadata/
    step8_no_metadata/
      configs/
        canonical_dev_ids.json
        p0_no_metadata_config.json
      rankings/
        dev_rankings_step5_no_metadata_fused.jsonl
        public_rankings_step5_no_metadata_fused.jsonl
  step8_reranker_model/
    step6/
      models/
        aiteamvn_vietnamese_reranker_finetuned/
          config.json
          model.safetensors
          tokenizer.json
          tokenizer_config.json
```

```text
===== /kaggle/input/datasets/mphatfromuit/dsc2026-baseline/DSC26_weight_report =====
DSC26_weight_report/
  legalir_sbert_cl/
    public_submission/
      public_ranked_contexts.csv
      submission.json
    best_model/
      # SBERT fine-tuned checkpoint dùng làm tín hiệu ranking/fusion.
    logs/
    cache/
  legalir_ce_listwise/
    public_submission/
      public_official_bm25_sbert_ce_rankings.jsonl
      # `ce` trong dataset này là BGE-M3 reranker v2 fine-tuned/listwise.
  legalir_LLM_XGB_reranker/
    public_submission/
      public_official_xgb_scored_candidates.parquet
      submission.json
      # XGB/LLM scored candidates dùng cho fusion, hiện là tín hiệu giúp tăng
      # public Recall tốt nhất khi cộng vào baseline.
```

Các notebook hiện đang dùng `DSC2026_baseline`:

```text
task1/pipeline/baselinecur/baseline.ipynb
task1/pipeline/step6+sbertft+bgem3rerankft/kaggle_step6_sbertft_bgem3rerankft_fusion.ipynb
task1/pipeline/step6+sbertft+bgem3rerankft+LLMXGB/kaggle_step6_sbertft_bgem3rerankft_llmxgb_fusion.ipynb
```

## Local mirror của `DSC2026_baseline`

Dataset `DSC2026_baseline` đã được tải về local tại:

```text
other_research/DSC26_weight_report/
```

Các artifact local quan trọng:

```text
other_research/DSC26_weight_report/legalir_sbert_cl/
  experiment_manifest.json
  cache/splits.json
  best_model/model.safetensors
  public_submission/public_ranked_contexts.csv
  public_submission/submission.json

other_research/DSC26_weight_report/legalir_ce_listwise/
  best_model/config.json
  best_model/model-001.safetensors
  public_submission/public_official_bm25_sbert_ce_rankings.jsonl
  public_submission/public_submission_summary.json
  tables/ce_training_history.csv
  tables/ce_input_k_sweep_val_test.csv

other_research/DSC26_weight_report/legalir_LLM_XGB_reranker/
  public_submission/public_official_xgb_scored_candidates.parquet
  public_submission/public_submission_summary.json
  llm_features/public_official_qwen3b_awq_labels.parquet
  llm_features/meta_train_qwen3b_awq_labels.parquet
  tables/public_official_xgb_features.parquet
  tables/meta_train_xgb_features.parquet
  tables/test500_xgb_scored_candidates.parquet
  tables/xgb_retrieval_plus_llm_feature_importance.csv
  xgb_model/xgb_retrieval_plus_llm.json
```

P1 audit hiện nằm ở:

```text
task1/pipeline/step9_p1_audit/
  kaggle_p1_0_external_audit.ipynb
  kaggle_p1_1_oracle_disagreement.ipynb
  kaggle_p1_2_xgb_clean_stacker.ipynb
```

Kết quả P1.0 local mới nhất:

```text
task1/pipeline/step9_p1_audit/outputs/p1_0_local_audit_report.json
task1/pipeline/step9_p1_audit/outputs/p1_0_local_audit_summary.md
```

Kết quả P1.1 local mới nhất:

```text
task1/pipeline/step9_p1_audit/p1_1_local_oracle/p1_1_oracle_disagreement_report.json
task1/pipeline/step9_p1_audit/p1_1_local_oracle/p1_1_oracle_disagreement_rows.jsonl
task1/pipeline/step9_p1_audit/p1_1_local_oracle/p1_1_oracle_disagreement_summary.md
```

P1.1 dùng canonical dev và các ranking Step 4/5/6 trong pipeline gốc, không
dùng artifact external `DSC2026_baseline`.

Kết quả P1.2 control/preflight local mới nhất:

```text
task1/pipeline/step9_p1_audit/p1_2_xgb_clean/p1_2_split_manifest.json
task1/pipeline/step9_p1_audit/p1_2_xgb_clean/p1_2_fast_holdout_clean_score_step4_report.json
task1/pipeline/step9_p1_audit/p1_2_xgb_clean/p1_2_5fold_clean_score_step4_report.json
task1/pipeline/step9_p1_audit/p1_2_xgb_clean/p1_2_full_step4_step5_step6_preflight.json
task1/pipeline/step9_p1_audit/p1_2_xgb_clean/p1_2_clean_xgb_summary.md
task1/pipeline/step9_p1_audit/p1_2_xgb_clean/models/p1_2_fast_clean_score_step4_xgbranker.json
task1/pipeline/step9_p1_audit/p1_2_xgb_clean/models/p1_2_5fold_clean_score_step4_xgbranker.json
```

P1.2 local mới hoàn tất control/preflight, chưa hoàn tất full clean stacker.
Control sạch đã chạy bằng Step3 BM25 score/support + Step4 rank copy từ Step5
artifact; không dùng Step5/Step6 learned train feature in-sample. Full Step4/5/6
clean stacker đang blocked vì còn thiếu
`p1_2_clean_base_features/*.parquet` và `manifest.json`.

P1.2 clean base-feature prep local hiện có:

```text
task1/pipeline/step9_p1_audit/p1_2_tools.py
task1/pipeline/step9_p1_audit/kaggle_p1_2_generate_clean_base_features_gpu.ipynb
task1/pipeline/step9_p1_audit/p1_2_clean_base_features/split_manifest.json
task1/pipeline/step9_p1_audit/p1_2_clean_base_features/feature_contract.json
task1/pipeline/step9_p1_audit/p1_2_clean_base_features/preflight_report.json
task1/pipeline/step9_p1_audit/p1_2_clean_base_features/merge_validation_report.json
task1/pipeline/step9_p1_audit/p1_2_clean_base_features/gpu_notebook_static_audit_report.json
task1/pipeline/step9_p1_audit/p1_2_clean_base_features/input_zip_manifest.json
task1/pipeline/step9_p1_audit/p1_2_clean_base_features_input.zip
task1/pipeline/step9_p1_audit/p1_2_step3_train_rankings_input.zip
task1/pipeline/step9_p1_audit/p1_2_step3_train_rankings_input_manifest.json
```

`kaggle_p1_2_generate_clean_base_features_gpu.ipynb` hiện là GPU phase-run
implementation cho `fast`, `oof_fold`, `fulltrain_dev`, `merge`; notebook
`.ipynb` chứa code trực tiếp trong cell và kiểm tra đủ dataset input trước khi
tải HuggingFace model. CPU local maintenance dùng
`p1_2_tools.py`:

```text
prepare-manifest
audit-notebook
package-manifest-input
package-train-rankings
validate-merge
```

Validator local đang trả `blocked_missing_gpu_outputs` vì chưa có output chạy
từ Kaggle:

Chỉ giữ một notebook GPU chính. Nếu toàn bộ clean base-feature generation vượt
giới hạn một phiên Kaggle, chạy lại cùng notebook với phase khác nhau
(`fast`, `oof_fold` + fold 1..5, `fulltrain_dev`) và tải các parquet part về
local để merge bằng validator CPU.

Static audit local mới nhất `gpu_notebook_static_audit_report.json` có status
`ok`: notebook 14 cells, canonical train/dev 6000/1000, fast split 5000/1000,
OOF 5 fold cover đúng 6000 heldout và không phát hiện pattern external
`other_research`/`DSC26_weight_report` hoặc train Step5/6 in-sample ranking.

ZIP local chuẩn bị upload Kaggle Dataset input cho manifest/contract:

```text
task1/pipeline/step9_p1_audit/p1_2_clean_base_features_input.zip
sha256=8c73a3561345790ae258de9b554e6ff9dd089982e059a08707056c5f69ec757e
namelist=[
  split_manifest.json,
  feature_contract.json,
  preflight_report.json,
  gpu_notebook_static_audit_report.json
]
```

Đã upload/copy vào dataset chính:

```text
===== /kaggle/input/datasets/bowboochua9/stnhdscduaiti26/p1_2_merged_input =====
p1_2_merged_input/
  split_manifest.json
  feature_contract.json
  preflight_report.json
  gpu_notebook_static_audit_report.json
  step3/
    outputs/
      rankings/
        train_rankings_best.jsonl  [289.3 MB]
```

Đây là dataset artifact/input JSON, không chứa helper `.py`, ranking, model
checkpoint hoặc output parquet.

P1.2 full clean base-feature generation còn cần full canonical-train Step3/BM25
rankings để mine candidate cho Step 6 sạch:

```text
actual Kaggle path =
/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/p1_2_step3_train_rankings_input/step3/outputs/rankings/train_rankings_best.jsonl

local source =
task1/pipeline/step3/outputs/rankings/train_rankings_best.jsonl

local coverage = 6000 unique train qids
local size = 303391070 bytes
uploaded ZIP =
task1/pipeline/step9_p1_audit/p1_2_step3_train_rankings_input.zip
zip sha256 = 1cd32d6d2e8c3827328801801aa44eb0fee34067c81c3fe16537a66c67fe1eb9
zip namelist = [step3/outputs/rankings/train_rankings_best.jsonl]
```

Ghi nhớ layout upload ZIP vào dataset chính: file nằm qua một tầng thư mục ngoài
theo tên package upload (`p1_2_step3_train_rankings_input/`), không nằm thẳng
dưới root `stnhdscduaiti26/`.

**ĐÃ UPLOAD**: `p1_2_step3_train_rankings_input.zip` lên Kaggle dataset `stnhdscduaiti26`. Path thực tế trên Kaggle:
```
/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/p1_2_merged_input/step3/outputs/rankings/train_rankings_best.jsonl
```

**ĐÃ GỘP**: Tạo `p1_2_merged_input.zip` (65 MB) chứa cả `p1_2_clean_base_features_input` (manifest/contract) và `p1_2_step3_train_rankings_input` (full train rankings). Upload zip này **duy nhất** lên dataset `stnhdscduaiti26` thay cho 2 zip cũ. Sau khi upload, dataset structure:
```
/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/p1_2_merged_input/
  split_manifest.json
  feature_contract.json
  preflight_report.json
  gpu_notebook_static_audit_report.json
  step3/
    outputs/
      rankings/
        train_rankings_best.jsonl
```

**QUAN TRỌNG**: `fast_holdout_features.parquet` KHÔNG có trong zip này (đây là OUTPUT của phase `fast`, không phải input). Phase `fast` phải chạy trước để tạo ra file này + model checkpoints, sau đó mới chạy merge.

**Lịch trình 1 session Kaggle 12h (chạy tuần tự trong 1 notebook execution)**:
| Phase | Thời gian ước lượng | Output |
|-------|-------------------|--------|
| `fast` | ~3h | `fast_holdout_features.parquet` + step5/step6 models |
| `oof_fold_1` → `oof_fold_5` | ~4-5h | 5 file `oof_fold_{1..5}_features.parquet` |
| `fulltrain_dev` | ~2h | `dev_features.parquet` |
| `merge` | ~10min | 3 file final parquet + `manifest.json` |
| **Tổng** | **~9-10h** | **Trong limit 12h** |

Cách chạy: Set `P1_2_GPU_PHASE=fast` → chạy xong → đổi `P1_2_GPU_PHASE=oof_fold` + `P1_2_OOF_FOLD=1..5` lần lượt → `fulltrain_dev` → `merge`. KHÔNG cần tắt kernel giữa các phase.

Base model input cho P1.2:

```text
default = P1_2_BASE_MODEL_SOURCE=hf_download
source = HuggingFace links/model IDs trong [DSC@UIT 2026] Danh sách mô hình - Sheet1.csv
bkai-foundation-models/vietnamese-bi-encoder revision=84f9d9ada0d1a3c37557398b9ae9fcedcdf40be0
AITeamVN/Vietnamese_Reranker revision=f536976248403314225d7fdfdbc87f0e9516a54e
```

Khi Kaggle bật internet, không cần upload Dataset base model. Cache local
`legalir_base_models_input.zip` và `legalir_base_models_dataset/` đã xóa khỏi
workspace để tránh nhầm với input bắt buộc. Nếu cần chạy offline thì tạo/mount
Dataset base model snapshot riêng và set `P1_2_BASE_MODEL_SOURCE=kaggle_dataset`:

```text
===== /kaggle/input/datasets/bowboochua9/legalir-base-models =====
legalir-base-models/
  AITeamVN/
    Vietnamese_Reranker/
      config.json
      model.safetensors
      sentencepiece.bpe.model
      special_tokens_map.json
      tokenizer_config.json
  bkai-foundation-models/
    vietnamese-bi-encoder/
      1_Pooling/config.json
      README.md
      added_tokens.json
      bpe.codes
      config.json
      config_sentence_transformers.json
      model.safetensors
      modules.json
      sentence_bert_config.json
      special_tokens_map.json
      tokenizer_config.json
      vocab.txt
  legalir_base_models_manifest.json
```

```text
task1/pipeline/step9_p1_audit/p1_2_clean_base_features/parts/fast_holdout_features.parquet
task1/pipeline/step9_p1_audit/p1_2_clean_base_features/parts/oof_fold_1_features.parquet
task1/pipeline/step9_p1_audit/p1_2_clean_base_features/parts/oof_fold_2_features.parquet
task1/pipeline/step9_p1_audit/p1_2_clean_base_features/parts/oof_fold_3_features.parquet
task1/pipeline/step9_p1_audit/p1_2_clean_base_features/parts/oof_fold_4_features.parquet
task1/pipeline/step9_p1_audit/p1_2_clean_base_features/parts/oof_fold_5_features.parquet
task1/pipeline/step9_p1_audit/p1_2_clean_base_features/parts/dev_features.parquet
```

Nếu dùng Kaggle internet, notebook tải model từ HuggingFace và không cần các
path dưới. Nếu tắt internet và dùng offline cache thì cần mount:

```text
/kaggle/input/datasets/bowboochua9/legalir-base-models/bkai-foundation-models/vietnamese-bi-encoder
/kaggle/input/datasets/bowboochua9/legalir-base-models/AITeamVN/Vietnamese_Reranker
```
