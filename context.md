# Context Task 1 LegalIR DSC@UIT 2026

File này là handoff ngắn cho chat mới. Khi tiếp tục Task 1, đọc file này trước,
sau đó mở các file được trỏ ở từng mục. Không tự suy luận lại path Kaggle nếu
`task1/structure.md` đã có snapshot.

## Trạng thái hiện tại

Task đang làm: LegalIR Task 1, nộp `submission.zip` chứa duy nhất
`submission.json`, mỗi query trả 1-5 `document_id`. Recall là mục tiêu chính,
Precision là tie-break. Nếu dự đoán quá 5 ID cho một query thì Recall/Precision
query đó bị tính 0.

Baseline public tốt nhất hiện tại:

```text
Step 6 AITeamVN/Vietnamese_Reranker fused ranking
+ SBERT-FT public ranking
+ LLM/XGB reranker public ranking
+ weighted RRF

candidate          = baseline_plus_xgb_w0p25
step6_weight       = 1.0
sbert_weight       = 0.6
xgb_weight         = 0.25
ce_weight          = 0.0
rrf_k              = 60
Public Recall@5    = 0.928833
Public Precision@5 = 0.200400
```

Submission tốt nhất đang khóa:

```text
task1/pipeline/baselinecur/submission.zip
```

Lưu ý: Step 6 + SBERT-FT chưa có dev metric hợp lệ cùng canonical split, vì
SBERT-FT dùng split khác. Không dùng public leaderboard để tune quá nhiều.

## File cần đọc trước

Plan pipeline gốc đã làm:

```text
task1/pipeline/plan_pipeline.md
```

Plan cải thiện đang làm:

```text
task1/pipeline/plan_caithien.md
```

Snapshot cấu trúc Kaggle Dataset hiện tại:

```text
task1/structure.md
```

Baseline current:

```text
task1/pipeline/baselinecur/respublic.md
task1/pipeline/baselinecur/baseline.md
task1/pipeline/baselinecur/baseline.ipynb
```

Context riêng cho nhánh Step 6 + SBERT-FT:

```text
task1/pipeline/step6+sbertft/context.md
```

Context/README nhánh P0 no-metadata đang làm:

```text
task1/pipeline/step8_no_metadata/README.md
task1/pipeline/step8_no_metadata/kaggle_p0_step6_no_metadata_rerank.ipynb
```

## Cấu trúc dataset Kaggle hiện tại

Nguồn sự thật là:

```text
task1/structure.md
```

Dataset gốc của BTC trên Kaggle:

```text
/kaggle/input/datasets/ttdatto/uit-dsc26/LegalIR - Public Test/train.json
/kaggle/input/datasets/ttdatto/uit-dsc26/LegalIR - Public Test/public-official.json
/kaggle/input/datasets/ttdatto/uit-dsc26/LegalIR - Public Test/selected-contexts
```

Dataset artifact pipeline hiện dùng:

```text
/kaggle/input/datasets/bowboochua9/stnhdscduaiti26
```

Path quan trọng theo snapshot mới nhất:

```text
/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/step4/chunks.jsonl
/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/step4/train_split.json
/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/step4/dev_split.json
/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/step6/step5/rankings/train_rankings_step5_fused.jsonl
/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/step6/step5/rankings/dev_rankings_step5_fused.jsonl
/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/step6/step5/rankings/public_rankings_step5_fused.jsonl
/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/step8_no_metadata/step8_no_metadata/rankings/dev_rankings_step5_no_metadata_fused.jsonl
/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/step8_no_metadata/step8_no_metadata/rankings/public_rankings_step5_no_metadata_fused.jsonl
/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/step8_reranker_model/step6/models/aiteamvn_vietnamese_reranker_finetuned
```

Quy ước bắt buộc: mỗi lần user upload hoặc cập nhật Kaggle Dataset, cập nhật
`task1/structure.md` ngay trước khi sửa notebook/path.

## Ghi chú P1.2 model input

Với P1.2 clean base-feature generation, Kaggle notebook được phép dùng internet
để tải base model trực tiếp từ HuggingFace theo link/model trong
`[DSC@UIT 2026] Danh sách mô hình - Sheet1.csv`, kèm revision đã audit trong
manifest Step 5/6:

```text
bkai-foundation-models/vietnamese-bi-encoder
revision = 84f9d9ada0d1a3c37557398b9ae9fcedcdf40be0

AITeamVN/Vietnamese_Reranker
revision = f536976248403314225d7fdfdbc87f0e9516a54e
```

Không cần upload base model Dataset nếu Kaggle internet bật. Cache offline
`legalir_base_models_input.zip` đã xóa khỏi workspace để tránh nhầm với input
bắt buộc; nếu cần chạy offline thì tạo/mount Dataset base snapshot riêng. Vẫn
không được dùng checkpoint Step 5/6 đã fine-tune full canonical train làm base
model cho P1.2 clean OOF.

## Quy ước chạy GPU Kaggle

- Từ nay không chạy local cho các bước cần GPU.
- Code chạy Kaggle nằm trực tiếp trong notebook đọc được, không để notebook tự
  ghi helper `.py` vào `/kaggle/working`.
- Dataset zip chỉ chứa artifact data/model thật cần cho step, không chứa code
  nếu user không yêu cầu.
- Không dùng fallback path tự động. Nếu thiếu file, in tree/snapshot rồi fail
  rõ ràng.
- Khi zip artifact trên Windows, dùng Python `zipfile` với arcname POSIX `/`,
  không dùng path `\` vì Kaggle báo forbidden character.
- Khi upload ZIP vào Kaggle Dataset hiện hữu, Kaggle/luồng upload có thể thêm
  một tầng thư mục ngoài theo tên gói ZIP. Phải ghi và dùng path thực tế sau
  upload, ví dụ:

```text
/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/p1_2_step3_train_rankings_input/step3/outputs/rankings/train_rankings_best.jsonl
```

  Tuyệt đối không tự giả định file nằm thẳng dưới root dataset nếu chưa kiểm
  tree thực tế.
- Sau mỗi step nên tạo `submission.zip` để user có thể nộp thử public nếu cần.
- Nếu artifact/output đã có ở máy hoặc đã có trong Kaggle input dataset thì
  notebook các lần sau không xuất lại bản trùng. Chỉ ghi output mới cần cho
  bước kế tiếp, metric/report/submission mới, và path manifest ngắn nếu cần.
- Với T4 16GB, reranker 568M nên dùng FP16/inference mode và batch nhỏ.
  Notebook P0 Step8 hiện đã hạ:

```text
MAX_SEQ_LENGTH = 320
EVIDENCE_MAX_CHARS = 1400
RERANK_BATCH_SIZE = 8
torch.inference_mode + FP16 autocast
```

Nếu vẫn OOM thì hạ `RERANK_BATCH_SIZE = 4`, không tăng lại 32.

## Public data nằm đâu

Local:

```text
task1/public-official.json
```

Kaggle gốc:

```text
/kaggle/input/datasets/ttdatto/uit-dsc26/LegalIR - Public Test/public-official.json
```

Trong một số bundle:

```text
task1/pipeline/step6+sbertft/step6+sbertft/public-official.json
/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/step6+sbertft/public-official.json
```

Public không có gold label. Local chỉ tính được metric dev; public Recall/Precision
chỉ có sau khi user nộp Codabench và báo kết quả.

## Artifact các step trước và ý nghĩa

Step 1 - Data validator, cleaning, chunking, split:

```text
task1/pipeline/step1/
task1/pipeline/step1/outputs/
```

Ý nghĩa: validate data, clean text, chunk corpus pháp luật, tạo split cố định
`6000 train / 1000 dev`, seed 42. Kết quả chính: `324,377` chunks từ `8,532`
docs.

Step 2 - Chunk-BM25 baseline:

```text
task1/pipeline/step2/
```

Ý nghĩa: baseline lexical đầu tiên, retrieve top 300 chunks, aggregate về
document, tạo metric/submission.

Step 3 - Tuned chunk-BM25:

```text
task1/pipeline/step3/
```

Ý nghĩa: tune BM25/aggregate, best trial `b_0p90`, tạo BM25 branch ổn hơn cho
Step 4.

Step 4 - BGE-M3 dense + BM25 RRF:

```text
task1/pipeline/step4/
```

Ý nghĩa: thêm `BAAI/bge-m3`, dense retrieval top 300 chunks, aggregate document,
RRF với BM25. Artifact quan trọng đã upload Kaggle: `chunks.jsonl`,
`train_split.json`, `dev_split.json`, `*_rankings_best.jsonl`.

Step 5 - Fine-tuned Vietnamese Bi-Encoder + RRF:

```text
task1/pipeline/step5/
```

Ý nghĩa: fine-tune `bkai-foundation-models/vietnamese-bi-encoder` bằng MNRL,
fuse với Step 4. Trong Kaggle artifact hiện tại, Step 5 đầy đủ nằm dưới:

```text
/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/step6/step5/
```

Step 6 - Fine-tuned reranker:

```text
task1/pipeline/step6/
```

Ý nghĩa: mine 10 semi-hard negatives/positive, fine-tune reranker, rerank top 50,
fuse reranker score với retrieval score. Best Step 6 riêng là
`AITeamVN/Vietnamese_Reranker`:

```text
Public Recall = 0.894
Public Precision = 0.192200
Dev Recall@5 = 0.905
```

Step 7:

```text
task1/pipeline/step7/
```

Ý nghĩa: đã thử Qwen3 reranker nhưng không vượt Step 6. Không xem là baseline
chính.

Step 7b:

```text
task1/pipeline/step7/plan_implement_step7b.md
task1/pipeline/step7/step7b_1_clean_sbert_cl_train.ipynb
```

Ý nghĩa: nhánh SBERT contrastive learning theo nghiên cứu khác. Đã có vấn đề và
tạm ngưng, không đưa vào pipeline chính lúc này.

Step 6 + SBERT-FT:

```text
task1/pipeline/step6+sbertft/
```

Ý nghĩa: không train/inference model, chỉ fuse Step 6 public ranking với ranking
SBERT-FT từ `other_research/kaggle_output`. Đây là baseline cũ `0.917333`.

Step 6 + SBERT-FT + BGE-M3 reranker FT + LLM/XGB:

```text
task1/pipeline/step6+sbertft+bgem3rerankft+LLMXGB/
```

Ý nghĩa: không train/inference model, chỉ fuse Step 6 + SBERT-FT baseline với
ranking/scored candidates từ dataset:

```text
/kaggle/input/datasets/mphatfromuit/dsc2026-baseline/DSC26_weight_report
```

Các public score đã nộp:

```text
xgb_llm_only_control:
  precision = 0.19920000000000004
  recall    = 0.9226666666666667

full_ce0p25_xgb0p40:
  precision = 0.19860000000000005
  recall    = 0.9219166666666666

baseline_plus_xgb_w0p25:
  precision = 0.20040000000000002
  recall    = 0.9288333333333333

baseline_plus_xgb_w0p40:
  precision = 0.20020000000000002
  recall    = 0.9278333333333333
```

Best hiện tại là `baseline_plus_xgb_w0p25`.

Step 8 no-metadata P0:

```text
task1/pipeline/step8_no_metadata/
```

Ý nghĩa: thực hiện P0 trong `plan_caithien.md`: bỏ metadata branch, replay Step
4/5 no-metadata, rồi rerun Step 6 AITeamVN trên candidate top 50 mới.

GPU rerank đã chạy xong, output local:

```text
task1/pipeline/step8_no_metadata/step8_no_metadata_step6/
```

Kết quả dev:

```text
fused Recall@5     = 0.9048333333333335
fused Precision@5  = 0.192799999999998
fused Recall@100   = 0.9825
best fusion        = reranker_weight 0.7, retrieval_weight 0.5
validation errors  = 0
```

So với Step 6 control cùng split `Recall@5 = 0.905000`, P0 no-metadata chưa
vượt dev. Không ưu tiên nộp public nếu còn ít lượt nộp.

Notebook đã sửa path theo Kaggle layout thật và sửa OOM:

```text
task1/pipeline/step8_no_metadata/kaggle_p0_step6_no_metadata_rerank.ipynb
```

Step 8 no-metadata + SBERT-FT fusion:

```text
task1/pipeline/step8_no_metadata_sbertft/
```

Ý nghĩa: thử lại baselinecur nhưng thay Step6 branch bằng Step8 no-metadata
Step6 branch, giữ SBERT-FT public ranking. Đây là CPU-only fusion, không train
và không inference model. Đã sinh các candidate `nm_sbert0p25` đến
`nm_sbert1p20`; nếu chỉ nộp một lượt để kiểm tra giả thuyết thì ưu tiên:

```text
task1/pipeline/step8_no_metadata_sbertft/outputs/candidates/nm_sbert0p60/submission.zip
```

Nhánh này chưa có public score. Dự đoán rủi ro cao vì Step8 no-metadata Step6
riêng đã thấp hơn Step6 thường trên public.

## Baseline current

Tổng kết baseline hiện tại:

```text
task1/pipeline/baselinecur/respublic.md
task1/pipeline/baselinecur/baseline.md
task1/pipeline/baselinecur/baseline.ipynb
```

Hai mốc cần phân biệt:

```text
Step 6 riêng:
  public recall    = 0.894
  public precision = 0.192200
  dev Recall@5     = 0.905

Step 6 + SBERT-FT weighted RRF:
  public recall    = 0.917333
  public precision = 0.197400
  trạng thái       = baseline cũ

Step 6 + SBERT-FT + LLM/XGB weighted RRF:
  public recall    = 0.928833
  public precision = 0.200400
  dev metric       = chưa hợp lệ cùng canonical split
```

Khi đánh giá hướng mới, so với Step 6 trên dev, nhưng chỉ xem là vượt best
public sau khi submission vượt `0.928833`.

## Other research

Các tài liệu/nghiên cứu phụ:

```text
other_research/DSC_research.pdf
other_research/DSC2026_Kaggle_T4_SBERT_CL_Experiment.md
other_research/DSC2026_Kaggle_T4_SBERT_CL_Experiment.ipynb
other_research/kaggle_output/
```

Dataset Kaggle chứa output SBERT-FT của nghiên cứu khác:

```text
/kaggle/input/datasets/mphatfromuit/cl-sbert/kaggle_output
```

Ví dụ:

```text
/kaggle/input/datasets/mphatfromuit/cl-sbert/kaggle_output/dsc2026.log
/kaggle/input/datasets/mphatfromuit/cl-sbert/kaggle_output/legalir_sbert_cl/public_submission/public_ranked_contexts.csv
/kaggle/input/datasets/mphatfromuit/cl-sbert/kaggle_output/legalir_sbert_cl/public_submission/submission.json
```

Ghi nhớ: `DSC_research.pdf` là tài liệu nền, không phải pipeline cuối.
ColBERT/GNN/LLM pruning/executable evidence không thuộc core hiện tại. Hướng
XGBoost/stacking trong `plan_caithien.md` đáng xem tiếp sau P0 vì nút thắt đang
là reorder top 10 -> top 5, không phải thiếu candidate top 100.

## Checklist cho chat mới

1. Đọc `task1/context.md`.
2. Đọc `task1/structure.md` và không dùng path ngoài snapshot nếu chưa được
   user xác nhận.
3. Nếu user vừa upload dataset, cập nhật `task1/structure.md` trước.
4. Nếu sửa notebook Kaggle, dùng path cố định và fail-fast; không fallback.
5. Nếu cần GPU, chuẩn bị notebook/artifact zip cho Kaggle, không chạy local.
6. Nếu tạo submission, validate đủ 1,000 query, mỗi answer 1-5 string IDs,
   không trùng, không ID ngoài corpus, zip chỉ chứa `submission.json`.
