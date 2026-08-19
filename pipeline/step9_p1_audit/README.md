# Step 9 P1 audit

Nhánh này triển khai các checkpoint P1 trong
`task1/pipeline/plan_caithien.md`.

## P1.0 external audit

Audit provenance, split, feature lineage và parameter budget của các artifact
external SBERT-FT, BGE CE và LLM/XGB.

Notebook:

```text
task1/pipeline/step9_p1_audit/kaggle_p1_0_external_audit.ipynb
```

Phạm vi:

- CPU-only trên Kaggle.
- Không train model.
- Không inference GPU.
- Không sinh `submission.zip`.
- Không fallback sang split, model hoặc artifact khác.

Input Kaggle cố định:

```text
/kaggle/input/datasets/bowboochua9/stnhdscduaiti26
/kaggle/input/datasets/mphatfromuit/dsc2026-baseline/DSC26_weight_report
/kaggle/input/dscuit2026/dscuit2026/[DSC@UIT 2026] Danh sách mô hình - Sheet1.csv
```

Nếu whitelist CSV chưa nằm trong code dataset, notebook sẽ fail rõ ràng. Khi
upload hoặc cập nhật dataset để thêm file này, cập nhật lại `task1/structure.md`
trước khi sửa path.

Input local cố định:

```text
[DSC@UIT 2026] Danh sách mô hình - Sheet1.csv
task1/pipeline/
other_research/DSC26_weight_report/
```

Local mirror `other_research/DSC26_weight_report/` hiện có đầy đủ public
ranking/parquet, CE weights và XGB model JSON để chạy audit CPU. Môi trường
local cần `pyarrow` để đọc các file `.parquet`.

Output Kaggle:

```text
/kaggle/working/p1_0_external_audit/p1_0_external_audit_report.json
/kaggle/working/p1_0_external_audit/p1_0_external_audit_summary.md
```

Output local sau lần audit hiện tại:

```text
task1/pipeline/step9_p1_audit/outputs/p1_0_local_audit_report.json
task1/pipeline/step9_p1_audit/outputs/p1_0_local_audit_summary.md
```

Kết luận local hiện tại:

```text
status = blocked
known_parameter_total_excluding_unknown_llm = 1,973,508,098
remaining_before_llm = 2,026,491,902
llm_model_name = Qwen/Qwen2.5-3B-Instruct-AWQ
llm_whitelist_exact_hit = false
```

P1.0 đã chạy xong nhưng bị block bởi kết quả audit không đạt: exact AWQ model
ID không có trong whitelist CSV, thiếu revision / license / parameter count /
weight checksum cho LLM, external split vẫn là `6000/500/500` khác canonical
dev, và parameter budget còn lại không đủ cho một LLM 3B-class trong pipeline
<4B.

Điều kiện pass P1.0:

- Có split IDs và seed cho external branches.
- Có model ID, revision, license, checksum và parameter count từ artifact thực
  tế hoặc manifest đáng tin cậy.
- Xác định được feature lineage của XGB/LLM, gồm cột nào phụ thuộc model nào.
- Tổng parameter count của toàn hệ thống dưới `4,000,000,000`.

Nếu thiếu một mục, nhánh external chỉ giữ làm insight nghiên cứu và không được
xem là pipeline final compliant.

## P1.1 oracle/disagreement audit

Notebook:

```text
task1/pipeline/step9_p1_audit/kaggle_p1_1_oracle_disagreement.ipynb
```

Phạm vi:

- CPU-only.
- Không train model.
- Không inference GPU.
- Không sinh `submission.zip`.
- Không dùng artifact external `DSC2026_baseline`.
- Chỉ dùng canonical dev và ranking Step 4/5/6 trong pipeline gốc.
- Thiếu input thì fail rõ theo đúng file cần bổ sung.

Input local cố định:

```text
task1/pipeline/step1/outputs/dev_split.json
task1/pipeline/step4/step4/rankings/dev_rankings_rrf.jsonl
task1/pipeline/step5/step5/rankings/dev_rankings_step5_fused.jsonl
task1/pipeline/step6/step6/rankings/aiteamvn_vietnamese_reranker_finetuned/dev_rankings_reranker.jsonl
task1/pipeline/step6/step6/rankings/aiteamvn_vietnamese_reranker_finetuned/dev_rankings_step6_fused.jsonl
```

Input Kaggle cố định:

```text
/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/step4/dev_split.json
/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/step4/dev_rankings_best.jsonl
/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/step6/step5/rankings/dev_rankings_step5_fused.jsonl
/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/step7/step6/rankings/aiteamvn_vietnamese_reranker_finetuned/dev_rankings_reranker.jsonl
/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/step7/step6/rankings/aiteamvn_vietnamese_reranker_finetuned/dev_rankings_step6_fused.jsonl
```

Output local sau lần audit hiện tại:

```text
task1/pipeline/step9_p1_audit/p1_1_local_oracle/p1_1_oracle_disagreement_report.json
task1/pipeline/step9_p1_audit/p1_1_local_oracle/p1_1_oracle_disagreement_rows.jsonl
task1/pipeline/step9_p1_audit/p1_1_local_oracle/p1_1_oracle_disagreement_summary.md
```

Kết luận local hiện tại:

```text
status = ok
canonical_dev_queries = 1000
step6_fused Recall@5 = 0.905000
oracle_union_top20 Recall@5 = 0.973333
delta_vs_step6_fused Recall@5 = 0.068333
passes_P1.2_headroom_gate = true
oracle_beats_step6_fused_queries = 82
oracle_ties_step6_fused_queries = 918
```

P1.1 đủ headroom để sang P1.2, nhưng P1.0 external compliance vẫn đang blocked
riêng; vì vậy P1.2 phải dùng feature/OOF sạch trên canonical train và giữ
canonical dev untouched.

## P1.2 clean XGBoost control/preflight

Notebook:

```text
task1/pipeline/step9_p1_audit/kaggle_p1_2_xgb_clean_stacker.ipynb
```

Phạm vi:

- CPU-only.
- Train XGBoost ranker, không train/inference neural model.
- Canonical dev chỉ dùng đúng một lần để đánh giá cuối config đã chọn trong
  train.
- Không dùng external `DSC2026_baseline`.
- Không dùng `ft_dense_doc_ids` hoặc `fused_doc_ids` từ train Step 5 để train
  stacker, vì đây là in-sample feature của model đã học canonical train.

Output local sau lần chạy hiện tại:

```text
task1/pipeline/step9_p1_audit/p1_2_xgb_clean/p1_2_split_manifest.json
task1/pipeline/step9_p1_audit/p1_2_xgb_clean/p1_2_fast_holdout_clean_score_step4_report.json
task1/pipeline/step9_p1_audit/p1_2_xgb_clean/p1_2_5fold_clean_score_step4_report.json
task1/pipeline/step9_p1_audit/p1_2_xgb_clean/p1_2_full_step4_step5_step6_preflight.json
task1/pipeline/step9_p1_audit/p1_2_xgb_clean/p1_2_clean_xgb_summary.md
```

Trạng thái chuẩn hiện tại:

```text
P1.2 = control/preflight complete
full clean stacker = blocked by missing OOF/base-feature artifacts
```

Kết quả control local hiện tại:

```text
track = clean score + Step4 rank control
local score branch = Step3 BM25 score/support
fast_holdout dev Recall@5 = 0.765667
5-fold full-train dev Recall@5 = 0.766667
Step4 dev Recall@5 = 0.852833
Step6 dev Recall@5 = 0.905000
passes_step6_gate = false
```

Control sạch này fail gate rõ ràng, đúng kỳ vọng vì feature còn nghèo. Không
sinh `submission.zip` và không xem đây là full P1.2.

Full Step4/5/6 clean stacker hiện `blocked` vì thiếu các feature artifact sạch:

```text
task1/pipeline/step9_p1_audit/p1_2_clean_base_features/fast_holdout_features.parquet
task1/pipeline/step9_p1_audit/p1_2_clean_base_features/train_oof_features.parquet
task1/pipeline/step9_p1_audit/p1_2_clean_base_features/dev_features.parquet
task1/pipeline/step9_p1_audit/p1_2_clean_base_features/manifest.json
```

Bước tiếp theo của P1.2 là tạo các artifact này bằng protocol không leakage:
learned Step 5/6 base score của mỗi train query phải đến từ checkpoint không
học query đó. Không cần chạy lại P1.1 và không cần chạy lại P1.2 control.

### P1.2 clean base-feature artifact prep

CPU local prep/validator:

```text
task1/pipeline/step9_p1_audit/p1_2_tools.py
```

Subcommands:

```text
prepare-manifest
audit-notebook
package-manifest-input
package-train-rankings
validate-merge
```

Output hiện có:

```text
task1/pipeline/step9_p1_audit/p1_2_clean_base_features/split_manifest.json
task1/pipeline/step9_p1_audit/p1_2_clean_base_features/feature_contract.json
task1/pipeline/step9_p1_audit/p1_2_clean_base_features/preflight_report.json
task1/pipeline/step9_p1_audit/p1_2_clean_base_features/merge_validation_report.json
task1/pipeline/step9_p1_audit/p1_2_clean_base_features/gpu_notebook_static_audit_report.json
task1/pipeline/step9_p1_audit/p1_2_clean_base_features/input_zip_manifest.json
task1/pipeline/step9_p1_audit/p1_2_merged_input.zip
```

`p1_2_tools.py validate-merge` đã chạy local và đang trả
`blocked_missing_gpu_outputs`, đúng kỳ vọng vì chưa có các parquet part từ GPU:

```text
parts/fast_holdout_features.parquet
parts/oof_fold_1_features.parquet
parts/oof_fold_2_features.parquet
parts/oof_fold_3_features.parquet
parts/oof_fold_4_features.parquet
parts/oof_fold_5_features.parquet
parts/dev_features.parquet
```

Notebook GPU hiện có:

```text
task1/pipeline/step9_p1_audit/kaggle_p1_2_generate_clean_base_features_gpu_v6_kaggle_resume.ipynb
```

Trạng thái notebook này: phase-run implementation đã validate được bằng
`nbformat` và `nbconvert`; notebook `.ipynb` chứa code trực tiếp trong cell để
chạy Kaggle. Notebook kiểm tra đủ dataset input trước khi tải HuggingFace model
để fail sớm nếu thiếu `split_manifest`, `feature_contract`, hoặc full Step3
train rankings.

Chỉ giữ một notebook GPU chính. Nếu một phiên Kaggle không đủ cho toàn bộ 7
base-model training phase, chạy lại cùng notebook với `P1_2_GPU_PHASE` /
`P1_2_OOF_FOLD` khác nhau, không tạo nhiều notebook clone.

Static audit local đã pass:

```text
script = task1/pipeline/step9_p1_audit/p1_2_tools.py audit-notebook
report = task1/pipeline/step9_p1_audit/p1_2_clean_base_features/gpu_notebook_static_audit_report.json
ipynb_cells = 14
canonical_train_queries = 6000
canonical_dev_queries = 1000
fast_base_queries = 5000
fast_stacker_queries = 1000
oof_folds = 5
oof_heldout_total = 6000
forbidden external/in-sample pattern hits = []
```

Manifest/contract input ZIP đã đóng gói để upload Kaggle Dataset:

```text
zip = task1/pipeline/step9_p1_audit/p1_2_merged_input.zip
current Kaggle location = /kaggle/input/datasets/bowboochua9/stnhdscduaiti26/p1_2_merged_input
```

ZIP namelist bắt buộc:

```text
split_manifest.json
feature_contract.json
preflight_report.json
gpu_notebook_static_audit_report.json
```

Notebook mặc định đọc:

```text
P1_2_MANIFEST_ROOT=/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/p1_2_merged_input
P1_2_TRAIN_RANKINGS_PATH=/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/p1_2_merged_input/step3/outputs/rankings/train_rankings_best.jsonl
```

`P1_2_TRAIN_RANKINGS_PATH` phải trỏ tới full canonical-train Step3/BM25
rankings. Local source chuẩn là
`task1/pipeline/step3/outputs/rankings/train_rankings_best.jsonl` với 6000
unique train qids. Không dùng partial
`stnhdscduaiti26/step4/train_rankings_best.jsonl` cho P1.2 full.

Các file đã được gộp chung trong `p1_2_merged_input.zip` và upload lên dataset chính.

Base model input:

```text
default = P1_2_BASE_MODEL_SOURCE=hf_download
source  = HuggingFace links/model IDs from [DSC@UIT 2026] Danh sách mô hình - Sheet1.csv
bkai-foundation-models/vietnamese-bi-encoder revision = 84f9d9ada0d1a3c37557398b9ae9fcedcdf40be0
AITeamVN/Vietnamese_Reranker revision = f536976248403314225d7fdfdbc87f0e9516a54e
```

Local `legalir_base_models_input.zip` và `legalir_base_models_dataset/` đã xóa
vì Kaggle dùng internet để tải exact HuggingFace revisions theo whitelist. Nếu
cần offline cache thì tạo/mount Dataset base model snapshot riêng và set:

```text
P1_2_BASE_MODEL_SOURCE=kaggle_dataset
mount = /kaggle/input/datasets/bowboochua9/legalir-base-models
```

Các phase cần chạy trên Kaggle T4x2:

```text
P1_2_GPU_PHASE=fast
P1_2_GPU_PHASE=oof_fold, P1_2_OOF_FOLD=1
P1_2_GPU_PHASE=oof_fold, P1_2_OOF_FOLD=2
P1_2_GPU_PHASE=oof_fold, P1_2_OOF_FOLD=3
P1_2_GPU_PHASE=oof_fold, P1_2_OOF_FOLD=4
P1_2_GPU_PHASE=oof_fold, P1_2_OOF_FOLD=5
P1_2_GPU_PHASE=fulltrain_dev
P1_2_GPU_PHASE=merge
```

Output phase GPU dự kiến:

```text
/kaggle/working/p1_2_clean_base_features/parts/fast_holdout_features.parquet
/kaggle/working/p1_2_clean_base_features/parts/oof_fold_1_features.parquet
/kaggle/working/p1_2_clean_base_features/parts/oof_fold_2_features.parquet
/kaggle/working/p1_2_clean_base_features/parts/oof_fold_3_features.parquet
/kaggle/working/p1_2_clean_base_features/parts/oof_fold_4_features.parquet
/kaggle/working/p1_2_clean_base_features/parts/oof_fold_5_features.parquet
/kaggle/working/p1_2_clean_base_features/parts/dev_features.parquet
```

Phase `merge` sẽ ghi:

```text
/kaggle/working/p1_2_clean_base_features/fast_holdout_features.parquet
/kaggle/working/p1_2_clean_base_features/train_oof_features.parquet
/kaggle/working/p1_2_clean_base_features/dev_features.parquet
/kaggle/working/p1_2_clean_base_features/manifest.json
```

Notebook sẽ tải base model từ HuggingFace khi `P1_2_BASE_MODEL_SOURCE=hf_download`.
Nếu đổi sang `kaggle_dataset`, notebook fail rõ khi thiếu base model snapshot:

```text
/kaggle/input/datasets/bowboochua9/legalir-base-models/bkai-foundation-models/vietnamese-bi-encoder
/kaggle/input/datasets/bowboochua9/legalir-base-models/AITeamVN/Vietnamese_Reranker
```

Các snapshot này phải là base model, không phải checkpoint Step 5/6 đã
fine-tune full canonical train.
