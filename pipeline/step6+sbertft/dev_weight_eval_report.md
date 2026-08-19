# Dev Weight Evaluation Report

Mục tiêu: đánh giá các hệ số `rrf_sbert*` trên dev để tìm hệ số tốt hơn
`rrf_sbert0p60`.

## Kết luận

Hiện **không có dev evaluation hợp lệ** cho các hệ số `Step6 + SBERT-FT` vì
artifact SBERT-FT hiện tại không có ranking trên canonical dev 1,000 query của
Step 1 theo một split sạch.

Canonical dev:

```text
task1/pipeline/step1/outputs/splits/dev_split.json
num_queries = 1000
```

SBERT-FT experiment dùng split riêng:

```text
other_research/kaggle_output/legalir_sbert_cl/cache/splits.json
train = 6000
val   = 500
test  = 500
```

Overlap với canonical dev:

```text
canonical dev ∩ SBERT train = 868
canonical dev ∩ SBERT val   = 68
canonical dev ∩ SBERT test  = 64
```

Vì 868/1000 query canonical dev đã nằm trong train của SBERT-FT experiment,
dùng model/ranking SBERT này để chọn hệ số trên canonical dev sẽ bị leakage và
không đáng tin. Hệ số “vàng” tìm được từ dev bẩn có rủi ro overfit cao.

## Candidate high-weight đã có

Các submission public-only đã tồn tại:

```text
task1/pipeline/step6+sbertft/outputs/candidates/rrf_sbert0p65/submission.zip
task1/pipeline/step6+sbertft/outputs/candidates/rrf_sbert0p7/submission.zip
task1/pipeline/step6+sbertft/outputs/candidates/rrf_sbert0p75/submission.zip
task1/pipeline/step6+sbertft/outputs/candidates/rrf_sbert0p8/submission.zip
task1/pipeline/step6+sbertft/outputs/candidates/rrf_sbert0p9/submission.zip
task1/pipeline/step6+sbertft/outputs/candidates/rrf_sbert1/submission.zip
task1/pipeline/step6+sbertft/outputs/candidates/rrf_sbert1p1/submission.zip
task1/pipeline/step6+sbertft/outputs/candidates/rrf_sbert1p2/submission.zip
task1/pipeline/step6+sbertft/outputs/candidates/rrf_sbert1p5/submission.zip
task1/pipeline/step6+sbertft/outputs/candidates/rrf_sbert2/submission.zip
```

High-weight sweep report:

```text
task1/pipeline/step6+sbertft/outputs/candidates/step6_sbertft_high_weight_sweep_report.json
```

## Audit overlap với baselinecur

Baselinecur là:

```text
rrf_sbert0p60
Public Recall@5    = 0.9173333333333332
Public Precision@5 = 0.19740000000000005
```

Overlap top 5 giữa các candidate high-weight và baselinecur:

| Candidate | SBERT weight | Changed vs baselinecur | Avg top5 overlap vs baselinecur |
|---|---:|---:|---:|
| `rrf_sbert0p65` | 0.65 | 108 | 4.962 |
| `rrf_sbert0p7` | 0.70 | 259 | 4.916 |
| `rrf_sbert0p75` | 0.75 | 338 | 4.875 |
| `rrf_sbert0p8` | 0.80 | 393 | 4.840 |
| `rrf_sbert0p9` | 0.90 | 476 | 4.777 |
| `rrf_sbert1` | 1.00 | 618 | 4.704 |
| `rrf_sbert1p1` | 1.10 | 780 | 4.633 |
| `rrf_sbert1p2` | 1.20 | 809 | 4.577 |
| `rrf_sbert1p5` | 1.50 | 862 | 4.467 |
| `rrf_sbert2` | 2.00 | 929 | 4.337 |

Nếu buộc phải chọn bằng rủi ro thấp nhất để nộp public, candidate hợp lý nhất là:

```text
task1/pipeline/step6+sbertft/outputs/candidates/rrf_sbert0p65/submission.zip
```

Lý do: đây là bước tăng weight nhỏ nhất sau `0.60`, chỉ đổi 108/1000 query so
với baselinecur.

## Cách làm dev evaluation đúng

Muốn tìm hệ số bằng dev thật, cần tạo lại SBERT-FT branch theo canonical split:

1. Train SBERT-FT chỉ trên `task1/pipeline/step4/step4/train_split.json` hoặc
   `task1/pipeline/step1/outputs/splits/train_split.json`.
2. Sinh SBERT ranking cho đúng canonical dev:

   ```text
   task1/pipeline/step1/outputs/splits/dev_split.json
   ```

3. Fuse canonical dev rankings:

   ```text
   Step6 dev ranking + canonical SBERT dev ranking
   ```

4. Sweep `sbert_weight` trên dev, chọn theo Recall@5, Precision@5 tie-break.
5. Chỉ sau đó sinh public submission bằng hệ số đã khóa.

Không dùng public leaderboard để chọn nhiều hệ số liên tiếp.
