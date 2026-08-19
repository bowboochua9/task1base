# Step 6 - Fine-tuned Rerankers

Mục tiêu theo `plan_pipeline.md`: dùng candidates từ Step 5, mine semi-hard negatives, fine-tune reranker, tune fusion trên dev và xuất public `submission.zip` để nộp thử.

Notebook hiện fine-tune 2 model trong whitelist, chạy `AITeamVN/Vietnamese_Reranker` trước để chắc có submission fine-tuned trước khi thử PhoRanker:

```text
AITeamVN/Vietnamese_Reranker
itdainb/PhoRanker
```

`itdainb/PhoRanker` được cho phép `resize_token_embeddings` nếu tokenizer/model vocab lệch, và notebook tự clamp `max_seq_length` theo `max_position_embeddings` để tránh lỗi position id `258`. `AITeamVN/Vietnamese_Reranker` không resize nếu vocab lệch; nếu mismatch thì fail rõ.

Semi-hard negatives đã được mine từ `step5/rankings/train_rankings_step5_fused.jsonl`; cache training rows nằm ở:

```text
/kaggle/working/step6/training/phoranker_train_pairs.jsonl
```

Nếu cache có ít hơn 1000 rows, notebook bỏ cache và build lại để tránh dùng artifact lỗi/truncated.

## Input Kaggle

Dataset cần có artifact từ Step 4:

```text
/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/step4/chunks.jsonl
/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/step4/train_split.json
/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/step4/dev_split.json
```

Upload `task1/pipeline/step6/step6.zip`. File này bổ sung output Step 5 cần cho Step 6:

```text
step5/models/vietnamese-bi-encoder-mnrl/
step5/configs/best_fusion_config.json
step5/rankings/train_rankings_step5_fused.jsonl
step5/rankings/dev_rankings_step5_fused.jsonl
step5/rankings/public_rankings_step5_fused.jsonl
step5/reports/run_report.json
step5/reports/model_manifest.json
```

Không cần upload lại artifact Step 4 đã có.

## Notebook

Chạy:

```text
kaggle_step6_phoranker.ipynb
```

Notebook chạy local training/inference, không gọi hosted inference/API.

## Output Chính

Mỗi candidate pass sẽ có output riêng:

```text
submission/<slug>/submission.zip
submission/<slug>/submission_validation.json
metrics/<slug>/dev_metrics_step6_fused.json
metrics/<slug>/dev_metrics_reranker_only.json
metrics/<slug>/fusion_trials.json
rankings/<slug>/dev_rankings_step6_fused.jsonl
rankings/<slug>/public_rankings_step6_fused.jsonl
candidates/<slug>/run_report.json
models/<slug>/
```

Slug hiện tại:

```text
aiteamvn_vietnamese_reranker_finetuned
phoranker_resized_finetuned
```

Nộp thử từng file:

```text
/kaggle/working/step6/submission/aiteamvn_vietnamese_reranker_finetuned/submission.zip
/kaggle/working/step6/submission/phoranker_resized_finetuned/submission.zip
```

So sánh với mốc public hiện tại:

```text
Step5:  Recall 0.844833
Step6 AITeamVN zero-shot: Recall 0.854583
Step6 BGE reranker zero-shot: Recall 0.825333
Step6 PhoRanker fine-tuned: Precision 0.1872, Recall 0.870667
Step6 AITeamVN fine-tuned: Precision 0.1922, Recall 0.894000
```

Chỉ giữ candidate fine-tuned nếu public không làm giảm `Recall`.
