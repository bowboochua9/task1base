# Step 8 No-Metadata + SBERT-FT Fusion

Nhánh này thử đúng câu hỏi: lấy baseline public tốt nhất `baselinecur`
`Step6 + SBERT-FT`, nhưng thay Step6 branch bằng Step8 no-metadata Step6 branch.

Không train, không inference model, không cần GPU. Script chỉ đọc các artifact
đã có và sinh submission mới:

```text
task1/pipeline/step8_no_metadata/step8_no_metadata_step6/rankings/.../public_rankings_step6_fused.jsonl
task1/pipeline/step8_no_metadata/step8_no_metadata_step6/submission/.../submission.json
task1/pipeline/step6+sbertft/step6+sbertft/sbertft/public_submission/public_ranked_contexts.csv
task1/pipeline/step6+sbertft/step6+sbertft/sbertft/public_submission/submission.json
task1/pipeline/baselinecur/submission.json
```

Chạy:

```powershell
$env:PYTHONIOENCODING='utf-8'
python task1\pipeline\step8_no_metadata_sbertft\fuse_no_metadata_step6_with_sbertft.py
```

Output chỉ gồm các candidate submission/report mới:

```text
task1/pipeline/step8_no_metadata_sbertft/outputs/run_report.json
task1/pipeline/step8_no_metadata_sbertft/outputs/candidates/<candidate>/submission.zip
```

Không copy lại ranking/model/input đã có ở các step trước.

Baseline cần vượt:

```text
baselinecur rrf_sbert0p60:
  Public Recall@5    = 0.9173333333333332
  Public Precision@5 = 0.19740000000000005
```

Kết quả Step8 no-metadata Step6 riêng đã biết là thấp:

```text
Public Recall@5    = 0.8825
Public Precision@5 = 0.18940000000000004
```

Vì vậy nhánh này chỉ nên nộp 1-2 candidate nếu cần kiểm tra giả thuyết, không
xem là hướng chính cho đến khi public thực sự vượt baselinecur.

## Candidate đã sinh

Đã chạy local CPU và tạo:

```text
task1/pipeline/step8_no_metadata_sbertft/outputs/run_report.json
task1/pipeline/step8_no_metadata_sbertft/outputs/candidates/nm_sbert0p25/submission.zip
task1/pipeline/step8_no_metadata_sbertft/outputs/candidates/nm_sbert0p40/submission.zip
task1/pipeline/step8_no_metadata_sbertft/outputs/candidates/nm_sbert0p60/submission.zip
task1/pipeline/step8_no_metadata_sbertft/outputs/candidates/nm_sbert0p80/submission.zip
task1/pipeline/step8_no_metadata_sbertft/outputs/candidates/nm_sbert1p00/submission.zip
task1/pipeline/step8_no_metadata_sbertft/outputs/candidates/nm_sbert1p20/submission.zip
```

Tất cả zip chỉ chứa `submission.json`, đủ 1,000 query, mỗi query 5 IDs.

Audit overlap top 5 với baselinecur:

| Candidate | SBERT weight | Changed vs baselinecur | Avg overlap vs baselinecur |
|---|---:|---:|---:|
| `nm_sbert0p25` | 0.25 | 883 | 4.392 |
| `nm_sbert0p40` | 0.40 | 789 | 4.523 |
| `nm_sbert0p60` | 0.60 | 600 | 4.643 |
| `nm_sbert0p80` | 0.80 | 678 | 4.603 |
| `nm_sbert1p00` | 1.00 | 762 | 4.539 |
| `nm_sbert1p20` | 1.20 | 858 | 4.452 |

Nếu chỉ nộp một lượt để kiểm tra giả thuyết, ưu tiên:

```text
task1/pipeline/step8_no_metadata_sbertft/outputs/candidates/nm_sbert0p60/submission.zip
```

Lý do: cùng weight `0.60` với baselinecur best và ít lệch khỏi baselinecur nhất
trong các candidate. Tuy vậy dự đoán rủi ro cao vì branch no-metadata riêng đã
public thấp hơn Step6 thường (`0.8825` so với `0.894`).

Kết quả public đã nộp:

```text
nm_sbert0p60:
  precision = 0.19660000000000002
  recall    = 0.9146666666666666
```

Kết luận: vẫn thấp hơn baselinecur `rrf_sbert0p60` (`0.9173333333333332`), nên
không tiếp tục ưu tiên nhánh no-metadata + SBERT-FT.
