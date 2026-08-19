# Step 5 - Fine-tune Vietnamese Bi-Encoder

Mục tiêu theo `plan_pipeline.md`: fine-tune `bkai-foundation-models/vietnamese-bi-encoder` bằng `MultipleNegativesRankingLoss`, tạo dense branch thứ hai, fuse với Step 4 RRF candidates, đánh giá dev và xuất public `submission.zip`.

Đây là bước cần GPU. Chạy trên Kaggle T4x2. Notebook khóa baseline về 1 GPU bằng `CUDA_VISIBLE_DEVICES=0` để tránh lỗi `DataParallel`.

## Input Kaggle

Dataset cần có artifact từ Step 4:

```text
/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/step4/chunks.jsonl
/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/step4/train_split.json
/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/step4/dev_split.json
/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/step4/dev_rankings_best.jsonl
/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/step4/public_rankings_best.jsonl
/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/step4/train_rankings_best.jsonl
/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/step4/best_config.json
```

Nếu Kaggle giải nén bị lồng thêm một tầng `step4/step4/...`, notebook cũng nhận layout đó. Đây chỉ là path alias của cùng artifact Step 4.

## Notebook

Chạy:

```text
kaggle_step5_finetune_biencoder.ipynb
```

Notebook chứa code trực tiếp theo cell, không tự sinh file `.py`.

Notebook hiện tại tự xuất đủ train/dev/public Step 5 fused rankings. Không còn notebook `5b` riêng. Output train dùng cho Step 6 là:

```text
/kaggle/working/step5/rankings/train_rankings_step5_fused.jsonl
```

Model dùng:

```text
bkai-foundation-models/vietnamese-bi-encoder
```

Model có trong `[DSC@UIT 2026] Danh sách mô hình - Sheet1.csv`. Notebook chạy local training/inference, không gọi hosted inference/API.

## Output Chính

- `models/vietnamese-bi-encoder-mnrl`: checkpoint bi-encoder fine-tuned.
- `training/train_pairs.jsonl`: positive pairs đã chọn.
- `rankings/train_rankings_step5_fused.jsonl`: train candidates đúng từ Step 5 để Step 6 mine semi-hard negatives.
- `rankings/dev_rankings_ft_dense.jsonl`: dense rankings từ model fine-tuned trên dev.
- `rankings/public_rankings_ft_dense.jsonl`: dense rankings từ model fine-tuned trên public.
- `rankings/dev_rankings_step5_fused.jsonl`: Step 4 RRF + fine-tuned dense RRF trên dev.
- `rankings/public_rankings_step5_fused.jsonl`: public candidates sau fusion.
- `metrics/train_metrics_step5_fused.json`: train metrics tham khảo.
- `metrics/dev_metrics_step5_fused.json`: dev metrics.
- `metrics/fusion_trials.json`: ablation trọng số fusion.
- `submission/submission.zip`: file nộp thử public.
- `reports/run_report.json`: provenance/config.
- `reports/model_manifest.json`: parameter audit.

## Ghi Chú

Mặc định notebook train nhanh 1 epoch, `max_train_examples=16000`, batch size 12. Nếu Step 5 tăng public/dev tốt, có thể chạy lại với nhiều examples/epochs hơn trước khi sang Step 6.

`chunk_embeddings_fp16.npy` sinh ra ở Step 5 là cache vận hành, lớn, không bắt buộc upload cho Step 6 nếu Step 6 chỉ dùng rankings/model checkpoint.

Nếu Kaggle gặp `CUDA error: device-side assert triggered`, restart runtime/kernel rồi chạy lại notebook. Notebook đã clamp `max_seq_length` theo `max_position_embeddings` của model để tránh lỗi position embedding.
