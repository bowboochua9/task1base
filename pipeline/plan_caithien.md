# Kế hoạch cải thiện baseline LegalIR 0.9288

## 1. Mục tiêu và baseline được khóa

Baseline public tốt nhất hiện tại:

```text
Step 6 AITeamVN/Vietnamese_Reranker fused ranking
+ SBERT-FT public ranking
+ LLM/XGB reranker public ranking
+ weighted RRF (step6_weight=1.0, sbert_weight=0.6, xgb_weight=0.25, rrf_k=60)

Public Recall@5    = 0.928833
Public Precision@5 = 0.200400
```

Artifact submission được khóa, không ghi đè:

```text
task1/pipeline/baselinecur/submission.zip
```

Mục tiêu tiếp theo là vượt `0.928833` một cách có thể lặp lại trên dev, sau đó chỉ nộp 1-2 candidate public đã được chọn trước. Không tiếp tục tune trực tiếp bằng public leaderboard.

## 2. Recall hiện tại

`0.928833` là recall của submission top 5 trên public, tức Recall@5. Public không có nhãn nên không thể tính Recall@10/20/50/100.

Dev recall bên dưới được tính lại từ các ranking 1,000 query của từng step:

| Stage | R@1 | R@5 | R@10 | R@20 | R@50 | R@90 | R@100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Step 4: BM25 + BGE-M3 dense + metadata RRF | 0.502000 | 0.852833 | 0.912833 | 0.950417 | 0.970333 | 0.981167 | 0.982000 |
| Step 5: Step 4 + Vietnamese Bi-Encoder FT | 0.510667 | 0.874250 | 0.931083 | 0.960167 | 0.976167 | 0.980667 | 0.982000 |
| Step 6: Step 5 + AITeamVN reranker FT | 0.587083 | 0.905000 | 0.946083 | 0.965667 | 0.976167 | 0.980667 | 0.982000 |

Lưu ý quan trọng: final `Step 6 + SBERT-FT` hiện chưa có dev metric hợp lệ trên cùng một split. Step 6 dùng dev 1,000 query của Step 1, trong khi SBERT chia riêng `6,000 train / 500 val / 500 test`; hai tập held-out chỉ trùng 132 query. Phần lớn Step 1 dev đã nằm trong train của SBERT, nên không được phép dùng nó để tune final fusion.

Hiện ranking chỉ lưu top 100 document. Với mục tiêu gần, không cần mở rộng bắt buộc lên top 200/300 vì R@100 đã rất cao (`0.982`). Nếu sau này cần audit retrieval sâu hơn thì mới sinh thêm top 200/300 document như diagnostic riêng. `dense_top_chunks=300` là 300 chunk, không phải 300 document.

## 3. Chẩn đoán nút thắt

Khoảng mất recall của Step 6:

```text
R@100 -> R@50:  0.005833
R@50  -> R@20:  0.010500
R@20  -> R@10:  0.019584
R@10  -> R@5:   0.041083
```

Như vậy retrieval top 100 đã gần trần (`0.982`). Nút thắt lớn nhất là đưa gold từ top 10 vào top 5, nên ưu tiên reranking/calibration/stacking hơn là thêm một retriever tương tự.

Nhánh bỏ metadata đã được chạy đầy đủ, không còn là giả thuyết. Mô phỏng ban đầu cho thấy candidate recall không giảm, nhưng rerank và public score thực tế đều không vượt baseline:

| Stage mô phỏng không metadata | R@5 | R@10 | R@20 | R@50 | R@100 |
|---|---:|---:|---:|---:|---:|
| Step 4 no-metadata | 0.855500 | 0.916833 | 0.950417 | 0.970333 | 0.982500 |
| Step 5 no-metadata | 0.877250 | 0.931083 | 0.960167 | 0.976667 | 0.982500 |

Kết quả chính thức sau khi rerun AITeamVN trên candidate top 50 mới:

```text
Step 8 no-metadata + AITeamVN:
  Dev Recall@5    = 0.904833
  Public Recall@5 = 0.882500

Step 8 no-metadata + SBERT-FT, weight 0.6:
  Public Recall@5 = 0.914667
```

Cả hai đều thấp hơn control tương ứng. Vì vậy giữ metadata branch trong baseline hiện tại và đóng P0; không tiếp tục tune no-metadata.

## 4. Dữ liệu tham khảo mới

Kết quả do người dùng cung cấp từ bạn, sau đó đã tái sử dụng được artifact:

```text
BGE-reranker-v2-m3, 2 epochs: 0.9128 public Recall
+ LLM + XGBoost reranker:     0.9227 public Recall
```

Kết quả đã nộp lại từ notebook fusion:

```text
xgb_llm_only_control:
  public precision = 0.19920000000000004
  public recall    = 0.9226666666666667

baseline_plus_xgb_w0p25:
  public precision = 0.20040000000000002
  public recall    = 0.9288333333333333
```

Vì vậy nhánh LLM/XGB không còn là mốc tham khảo; nó là baseline public tốt nhất
hiện tại. `full_ce0p25_xgb0p40` thấp hơn `baseline_plus_xgb_w0p25`, nên CE
BGE-M3 reranker FT chưa được giữ trong baseline cuối.

So với baseline `rrf_sbert0p60` trước đó:

```text
XGB/LLM only 0.922667: cao hơn 0.917333 khoảng 0.005333
baseline + XGB/LLM 0.928833: cao hơn 0.917333 khoảng 0.011500
```

BGE reranker dùng riêng thấp hơn baseline và khi fuse vào `full_ce0p25_xgb0p40`
cũng thấp hơn không dùng CE. XGBoost/LLM là hướng đã xác nhận có ích nhất vì
mục tiêu hiện tại là học cách đưa candidate top 10 vào top 5.

### 4.1. Giới hạn diễn giải do split khác nhau

Các nhánh SBERT-FT, BGE-reranker-v2-m3 FT và LLM/XGB trong
`other_research` dùng split riêng `6,000 train / 500 val / 500 test`, không phải
canonical split `6,000 train / 1,000 dev` của pipeline này. Theo thông tin hiện
có, BGE và LLM/XGB dùng chung hệ dữ liệu/split với nhánh nghiên cứu đó.

Vì vậy kết quả public hiện tại chỉ cho phép kết luận:

- SBERT-FT và LLM/XGB tạo ranking có sai số bổ sung hữu ích cho Step 6.
- XGB/LLM có tín hiệu tốt cho bài toán đưa candidate từ top 10 vào top 5.
- BGE reranker FT có thể hữu ích như raw feature dù direct RRF chưa có gain.

Một phần disagreement có thể đến từ khác split/candidate mining chứ không chỉ
từ kiến trúc model. Khi tái lập canonical, phải giữ diversity có kiểm soát bằng
negative band, objective và feature khác nhau; không tạo diversity bằng cách cho
một branch học canonical dev.

Artifact XGB có `9,626` candidate cho `1,000` public query, tức tập trung gần
đúng top 10 thay vì rerank lại toàn top 50-100. Các feature đáng tái lập gồm
rank/score BM25-SBERT-CE, `ce_gap_prev`, `ce_gap_next`, chênh lệch rank giữa
các branch, `num_support_chunks` và nhãn LLM có cấu trúc. Đây là insight mạnh
hơn kết luận rằng cần thêm một backbone mới.

Không được kết luận checkpoint, epoch, negative mining hoặc weight của
`other_research` tốt hơn trên canonical dev. `baseline_plus_xgb_w0p25` vẫn được
khóa làm best public submission, nhưng đây là leaderboard-confirmed baseline,
không phải canonical-dev-validated config. Weight `0.25` đã được chọn sau khi
nộp nhiều public candidate nên không được tiếp tục sweep bằng public leaderboard.

Mọi phát triển tiếp theo phải tái tạo tín hiệu trên canonical split, đo trên
cùng 1,000 dev query chưa tham gia train, rồi mới sinh tối đa 1-2 submission.

### 4.2. Cổng compliance cho nhánh external

Precomputed score không làm model tạo ra score biến mất khỏi hệ thống. Các model
tạo `sbert_score`, `ce_score`, `llm_*` và `xgb_llm_score` đều phải có trong
manifest và được tính vào tổng tham số dưới 4B.

Tạm tính các thành phần đã biết của baseline hiện tại:

| Thành phần | Số tham số dùng để audit |
|---|---:|
| Step 4 `BAAI/bge-m3` | 568,000,000 |
| Step 5 BKAI Bi-Encoder FT | 134,998,272 |
| Step 6 AITeamVN reranker FT | 567,755,777 |
| External SBERT-FT, cùng kiến trúc BKAI | tạm tính 134,998,272 |
| External BGE-reranker-v2-m3 FT | tạm tính 567,755,777 |
| **Subtotal chưa gồm LLM** | **1,973,508,098** |

Như vậy chỉ còn khoảng `2,026,491,902` tham số cho LLM và mọi model mới. Nếu
LLM tạo feature có trên khoảng 2.026B tham số thì baseline hiện tại đã không
đạt giới hạn, dù notebook cuối chỉ đọc parquet và chạy XGBoost trên CPU.

Ví dụ để quyết định nhanh sau khi biết model LLM:

```text
LLM 1.5B -> tổng tạm 3.474B, còn khoảng 526M: không đủ thêm reranker 0.6B
LLM 1.7B -> tổng tạm 3.674B, còn khoảng 326M: chỉ vừa một model khoảng 0.3B
LLM 2.0B -> tổng tạm 3.974B, gần như không được thêm model có tham số
```

Trước thí nghiệm tiếp theo phải lấy đủ từ nhánh LLM/XGB:

```text
model_id, exact revision, license, exact parameter_count
split IDs và seed
checkpoint/config dùng để sinh từng feature
feature lineage: cột nào phụ thuộc model nào
checksum weights, feature parquet và XGBoost model
```

Thiếu một mục thì nhánh external chỉ được giữ làm insight nghiên cứu, chưa được
coi là pipeline cuối hợp lệ. Không có fallback sang model, split hoặc artifact
khác khi audit fail.

## 5. Kiểm tra epoch hiện tại

| Model | Vai trò | Epoch hiện tại | Kết luận |
|---|---|---:|---|
| `BAAI/bge-m3` | Step 4 dense retrieval | 0 | Đang zero-shot; không có epoch để tăng trong baseline hiện tại. |
| `bkai-foundation-models/vietnamese-bi-encoder` | Step 5 dense retriever | 1 | Chỉ sweep 2-3 epochs khi P1 audit cho thấy dense branch còn oracle headroom. |
| `AITeamVN/Vietnamese_Reranker` | Step 6 cross-encoder | 2 | Có thể thử 3-4, nhưng phải đánh giá checkpoint từng epoch. |
| SBERT-FT trong `other_research` | Dense/rerank branch | 3 | Cần train lại theo canonical split; có thể sweep đến epoch 5. |
| `BAAI/bge-reranker-v2-m3` | Cross-encoder external | Đã có 2 epoch trên split khác; canonical chưa train | Chỉ tái lập trên canonical split nếu raw logit có oracle headroom. |

Lịch sử SBERT-FT hiện có cho thấy tăng epoch đã có ích:

| Epoch | Val Recall@5 | Val MRR | Val CL loss |
|---:|---:|---:|---:|
| 1 | 0.890 | 0.750249 | 0.748512 |
| 2 | 0.890 | 0.760283 | 0.754083 |
| 3 | 0.898 | 0.760503 | 0.768949 |

Recall@5 tăng ở epoch 3 dù val contrastive loss xấu đi. Vì vậy checkpoint phải chọn bằng ranking metrics, không chọn bằng train/validation loss.

AITeamVN hiện chỉ lưu train loss:

```text
epoch 1: 0.183015
epoch 2: 0.119651
```

Loss giảm không đủ để kết luận epoch 3 sẽ tăng recall. Lần rerun tiếp theo phải lưu ranking và metrics riêng cho mỗi epoch.

## 6. Cổng kiểm định bắt buộc

| Tầng | Điều kiện giữ model/config |
|---|---|
| Retrieval | R@100 không giảm. R@200/R@300 chỉ là diagnostic optional nếu nghi ngờ thiếu candidate sâu. |
| Bi-Encoder | R@20, R@10 và R@5 đều tăng; R@100 không giảm. |
| Reranker | R@20, R@10 và R@5 tăng; R@50 phải giữ nguyên nếu chỉ reorder top 50. |
| Final ensemble | Tăng Recall@5 trên dev chung; Precision@5 là tie-break; không dùng public để chọn weight. |
| Compliance | Đủ model/revision/license/checksum/parameter count và tổng hệ thống `< 4,000,000,000` trước khi sinh submission. |

Nếu reranker top 50 làm thay đổi R@50 thì có lỗi candidate/filter/append-tail, không phải cải thiện model.

Mỗi báo cáo phải có ít nhất:

```text
R@1, R@5, R@10, R@20, R@50, R@100
Precision@5, Hit@5, MRR
delta so với input của chính stage
số query tăng/giảm/không đổi
oracle recall của union các branch
```

## 7. Kế hoạch thí nghiệm theo thứ tự

### P0 - Dựng baseline no-metadata và dev chung

Mục đích: tạo điểm xuất phát sạch trước khi train thêm.

1. Replay Step 4 bằng BM25 + BGE-M3 dense, đặt `metadata_weight=0`.
2. Replay Step 5 bằng FT dense ranking đã có; chưa cần train lại.
3. Sinh ranking top 100 document để đo R@100/50/20/10/5. Không cần top 300 cho vòng hiện tại; chỉ mở rộng lên top 200/300 nếu R@100 bắt đầu giảm hoặc cần audit retrieval sâu.
4. Rerun AITeamVN inference trên top 50 mới bằng model 2-epoch đã lưu.
5. Khóa một `canonical_dev_ids.json` duy nhất cho tất cả thí nghiệm sau.

Tài nguyên:

```text
Replay Step 4/5, metrics: local Python, không cần GPU
AITeamVN rerank top 50: Kaggle notebook, cần GPU
```

Không ghi đè Step 4-6 cũ; tạo nhanh output mới, ví dụ `step8_no_metadata/`.

Kết quả đã chạy:

```text
output local = task1/pipeline/step8_no_metadata/step8_no_metadata_step6/

P0 no-metadata + AITeamVN fused:
  Dev Recall@5     = 0.9048333333333335
  Dev Precision@5  = 0.192799999999998
  Dev Recall@100   = 0.9825
  best fusion       = reranker_weight 0.7, retrieval_weight 0.5

Public Recall@5:
  no-metadata + AITeamVN                  = 0.882500
  no-metadata + AITeamVN + SBERT-FT 0.6 = 0.914667
```

So với Step 6 control cùng split `Dev Recall@5 = 0.905000`, P0 chưa vượt
cổng dev; cả hai public submission cũng thấp hơn baseline cũ `0.917333` và
baseline hiện tại `0.928833`. P0 đã
đóng và baseline tiếp tục giữ metadata.

### P1 - Step 9: chẩn đoán và stacking top 10 -> top 5

Hướng này đã có phiên bản public-ranking fusion với LLM/XGB của bạn Phát và
đang là baseline tốt nhất (`baseline_plus_xgb_w0p25`, Recall@5 `0.928833`).
Gain public cho thấy P1 là hướng ưu tiên, nhưng phần còn lại bắt buộc tái lập
sạch trên canonical split; không dùng artifact public-only để tune tiếp.

#### P1.0. Audit provenance, split và budget

1. Hoàn thành manifest ở mục 4.2 cho external SBERT, BGE CE, LLM và XGBoost.
2. Đối chiếu train/val/test IDs của `other_research` với canonical train/dev.
3. Xác định chính xác feature XGBoost nào phụ thuộc SBERT, BGE CE và LLM.
4. Tính tổng tham số từ weights/config thực tế, không suy từ tên model.
5. Fail fast nếu model không whitelist, thiếu revision/license hoặc tổng đạt 4B.

P1.0 chạy trong Kaggle notebook CPU, không cần GPU. Audit local đã chạy xong
nhưng bị block bởi kết quả không đạt: external split khác canonical, metadata
model/revision/license/checksum chưa đủ và parameter budget còn lại không đủ
cho LLM 3B-class trong giới hạn <4B. Chưa pass P1.0 thì không thêm model mới
và không xem nhánh LLM/XGB external là final compliant pipeline.

#### P1.1. Oracle và disagreement audit

Dùng artifact dev đã có của Step 4, Step 5 và Step 6 để tạo một bảng theo cặp `(query_id, doc_id)` trên union top 20. Tối thiểu gồm:

```text
rank/in_top_k của Step 4, Step 5, Step 6 reranker và Step 6 fused
AITeamVN reranker_score và retrieval_rank
số branch đồng thuận ở top 5/10/20
gold label chỉ có ở dev
```

Đo:

```text
oracle Recall@5 của union branch
số query Step 4/5 thắng Step 6 và ngược lại
số gold đang ở rank 6-10 của Step 6 có thể được branch khác kéo lên top 5
breakdown theo số gold/query
```

Checkpoint P1.1 chạy bằng Kaggle notebook CPU, không cần GPU và không sinh
submission. Chỉ tiếp tục train stacker nếu oracle Recall@5 cao hơn Step 6 ít
nhất `0.010`; nếu headroom thấp hơn thì dừng stacking và chuyển sang tạo tín
hiệu model mới.

Kết quả local đã tái lập bằng notebook CPU:

```text
notebook = task1/pipeline/step9_p1_audit/kaggle_p1_1_oracle_disagreement.ipynb
output   = task1/pipeline/step9_p1_audit/p1_1_local_oracle/

canonical dev queries = 1000
step4_fused    Recall@5 = 0.852833
step5_fused    Recall@5 = 0.874250
step6_reranker Recall@5 = 0.854583
step6_fused    Recall@5 = 0.905000

oracle union top20 Recall@5 = 0.973333
delta vs Step6 fused        = 0.068333
P1.2 headroom gate          = pass

Step6 gold rank 6-10:
  docs    = 49
  queries = 48
  Step4 can lift to top5        = 16 docs / 16 queries
  Step5 can lift to top5        = 19 docs / 19 queries
  Step6 reranker can lift top5  = 11 docs / 11 queries
```

P1.1 hoàn tất và đủ headroom cho P1.2. Điều này không gỡ blocked của P1.0
external compliance; P1.2 vẫn phải train XGBoost sạch bằng OOF feature trên
canonical train, không học canonical dev và không dùng feature external khác
split làm final compliant branch.

#### P1.2. Stacker sạch trên canonical dev

SBERT-FT, BGE CE và LLM/XGB external hiện tại không được dùng để train hoặc
chọn stacker vì split khác và có overlap với canonical dev. Canonical dev 1,000
query phải giữ nguyên làm tập đánh giá cuối, không chia dev thành fold rồi train
stacker trên 4/5 dev.

Quy trình nhanh để sàng lọc:

1. Từ canonical train 6,000, giữ riêng 1,000 query làm `stacker_train`.
2. Train các learned base branch trên 5,000 query còn lại.
3. Sinh feature cho `stacker_train`, train XGBRanker theo group query.
4. Sinh feature và đánh giá đúng một lần trên canonical dev 1,000 chưa bị học.

Nếu pass gate, quy trình đầy đủ là 5-fold OOF trên toàn bộ 6,000 canonical
train: mỗi base-model score của một query phải được sinh bởi checkpoint không
học query đó. Sau đó train stacker trên OOF feature và đánh giá trên canonical
dev untouched. Mỗi query là một group; không trộn candidate giữa các query.

Thử grid nhỏ:

```text
objective = rank:pairwise
max_depth = {3, 5}
learning_rate = {0.03, 0.05}
n_estimators <= 500
subsample = 0.8
colsample_bytree = 0.8
seed = 42
```

So sánh với đúng Step 6 trên cùng canonical dev query:

```text
Recall@1/5/10/20, Precision@5, Hit@5, MRR
số query tăng/giảm/không đổi
Recall@5 toàn dev và bootstrap/seed stability
```

Chỉ giữ stacker nếu canonical-dev Recall@5 tăng ít nhất `+0.003`, gain ổn định
qua seed/bootstrap và candidate Recall@20 không giảm. Sau khi khóa config, không
train stacker final trên canonical dev theo cách làm mất tập kiểm định. Bản final
phải dùng OOF feature của train 7,000 để train stacker, còn base models được
train lại trên toàn bộ 7,000 rồi mới dự đoán public.

Kết quả control/preflight đã chạy local:

```text
notebook = task1/pipeline/step9_p1_audit/kaggle_p1_2_xgb_clean_stacker.ipynb
output   = task1/pipeline/step9_p1_audit/p1_2_xgb_clean/

Control sạch có thể chạy với artifact hiện có:
  score branch = Step3 BM25 score/support
  step4 branch = Step4 rank copy trong Step5 artifact
  không dùng Step5 FT/fused train feature
  không dùng Step6 train feature

fast holdout dev Recall@5 = 0.765667
5-fold full-train dev Recall@5 = 0.766667
Step4 dev Recall@5 = 0.852833
Step6 dev Recall@5 = 0.905000
gate vs Step6 = fail
```

Trạng thái chuẩn:

```text
P1.1 = complete, pass headroom gate
P1.2 = control/preflight complete
Full P1.2 clean stacker = blocked by missing OOF/base-feature artifacts
```

Full Step4/5/6 clean stacker hiện bị block vì thiếu artifact feature sạch:

```text
task1/pipeline/step9_p1_audit/p1_2_clean_base_features/fast_holdout_features.parquet
task1/pipeline/step9_p1_audit/p1_2_clean_base_features/train_oof_features.parquet
task1/pipeline/step9_p1_audit/p1_2_clean_base_features/dev_features.parquet
task1/pipeline/step9_p1_audit/p1_2_clean_base_features/manifest.json
```

Không được thay bằng `train_rankings_step5_fused.jsonl` hiện có để train
stacker, vì feature learned branch trong file này là in-sample trên canonical
train. Muốn tiếp tục P1.2 đúng nghĩa phải tạo clean base-feature artifact:
fast-holdout base models train trên 5,000 query và score 1,000 stacker-train;
sau đó 5-fold OOF base scores trên toàn bộ 6,000 train. Canonical dev vẫn chỉ
dùng để đánh giá cuối.

Không cần chạy lại P1.1. Không cần chạy lại P1.2 control/preflight. Bước kế
tiếp là tạo `p1_2_clean_base_features` theo protocol sạch, đặc biệt learned
Step5/Step6 score cho train phải là OOF hoặc fast-holdout, không lấy thẳng
train ranking in-sample hiện có.

Cập nhật prep local:

```text
task1/pipeline/step9_p1_audit/p1_2_tools.py = done
  prepare-manifest = done
  audit-notebook = done
  package-manifest-input = done
  package-train-rankings = done
  validate-merge = blocked_missing_gpu_outputs
task1/pipeline/step9_p1_audit/p1_2_clean_base_features/split_manifest.json = done
task1/pipeline/step9_p1_audit/p1_2_clean_base_features/feature_contract.json = done
task1/pipeline/step9_p1_audit/p1_2_clean_base_features/preflight_report.json = done
task1/pipeline/step9_p1_audit/p1_2_clean_base_features/merge_validation_report.json = blocked_missing_gpu_outputs
task1/pipeline/step9_p1_audit/p1_2_clean_base_features/gpu_notebook_static_audit_report.json = ok
task1/pipeline/step9_p1_audit/p1_2_clean_base_features/input_zip_manifest.json = ok
task1/pipeline/step9_p1_audit/p1_2_clean_base_features_input.zip = ok
```

Notebook GPU phase-run hiện có:

```text
task1/pipeline/step9_p1_audit/kaggle_p1_2_generate_clean_base_features_gpu.ipynb
```

Trạng thái notebook: validate JSON/nbconvert OK và đã có phase implementation
để train/score Step5/Step6 sạch. Notebook `.ipynb` chứa code trực tiếp trong
cell. Chỉ giữ một notebook GPU chính. Nếu một phiên Kaggle không đủ cho toàn bộ
phase, chạy lại cùng notebook với `P1_2_GPU_PHASE` / `P1_2_OOF_FOLD` khác nhau;
không tạo notebook clone.

Static audit local mới nhất pass:

```text
ipynb_cells = 14
canonical train/dev = 6000/1000
fast split = 5000/1000
oof folds = 5, heldout total = 6000
forbidden external/in-sample pattern hits = []
```

Manifest/contract input ZIP cho Kaggle:

```text
zip = task1/pipeline/step9_p1_audit/p1_2_clean_base_features_input.zip
sha256 = 8c73a3561345790ae258de9b554e6ff9dd089982e059a08707056c5f69ec757e
namelist = split_manifest.json, feature_contract.json, preflight_report.json, gpu_notebook_static_audit_report.json
current Kaggle location = /kaggle/input/datasets/bowboochua9/stnhdscduaiti26/p1_2_clean_base_features_input
notebook default P1_2_MANIFEST_ROOT = /kaggle/input/datasets/bowboochua9/stnhdscduaiti26/p1_2_clean_base_features_input
notebook default P1_2_TRAIN_RANKINGS_PATH = /kaggle/input/datasets/bowboochua9/stnhdscduaiti26/p1_2_step3_train_rankings_input/step3/outputs/rankings/train_rankings_best.jsonl
```

P1.2 full cần full canonical-train Step3/BM25 rankings cho Step 6 candidate
mining. Local source chuẩn:
`task1/pipeline/step3/outputs/rankings/train_rankings_best.jsonl` có 6000 qids.
Không dùng partial `stnhdscduaiti26/step4/train_rankings_best.jsonl` vì Kaggle
run đã báo thiếu 4210 qids.

Prepared update ZIP:
`task1/pipeline/step9_p1_audit/p1_2_step3_train_rankings_input.zip`,
sha256 `1cd32d6d2e8c3827328801801aa44eb0fee34067c81c3fe16537a66c67fe1eb9`,
namelist `[step3/outputs/rankings/train_rankings_best.jsonl]`.
Actual Kaggle path sau upload vào dataset chính:
`/kaggle/input/datasets/bowboochua9/stnhdscduaiti26/p1_2_step3_train_rankings_input/step3/outputs/rankings/train_rankings_best.jsonl`.

Base model input cho Kaggle:

```text
default = P1_2_BASE_MODEL_SOURCE=hf_download
source = HuggingFace links/model IDs trong [DSC@UIT 2026] Danh sách mô hình - Sheet1.csv
bkai-foundation-models/vietnamese-bi-encoder revision = 84f9d9ada0d1a3c37557398b9ae9fcedcdf40be0
AITeamVN/Vietnamese_Reranker revision = f536976248403314225d7fdfdbc87f0e9516a54e
```

Local `legalir_base_models_input.zip` và `legalir_base_models_dataset/` đã xóa
vì notebook mặc định tải exact HuggingFace revisions khi Kaggle internet bật.
Không được dùng checkpoint Step 5/6 đã fine-tune full train làm base model cho
clean OOF.

```text
fast -> parts/fast_holdout_features.parquet
oof_fold=1..5 -> parts/oof_fold_<k>_features.parquet
fulltrain_dev -> parts/dev_features.parquet
merge/validator -> fast_holdout_features.parquet, train_oof_features.parquet, dev_features.parquet, manifest.json
```

Nếu dùng offline cache thì cần Dataset base model snapshot:

```text
/kaggle/input/datasets/bowboochua9/legalir-base-models/bkai-foundation-models/vietnamese-bi-encoder
/kaggle/input/datasets/bowboochua9/legalir-base-models/AITeamVN/Vietnamese_Reranker
```

#### P1.3. Ghép với baseline public đã khóa

Stacker phải chứng minh gain riêng trên dev trước. Sau đó tạo đúng hai submission public đã chọn trước:

```text
candidate A = stacker Step 4/5/6
candidate B = weighted RRF(stacker, branch mới đã pass canonical dev)
```

External SBERT weight `0.6` chỉ là control public đã khóa, không phải weight mặc
định cho candidate mới. Candidate B chỉ tồn tại nếu canonical SBERT/BGE/LLM
hoặc model mới đã pass dev gate; weight phải được khóa trên canonical dev và
không sweep lại bằng public.

Tài nguyên:

```text
P1.0, P1.1 và XGBoost P1.2: Kaggle CPU notebook
Chỉ dùng Kaggle GPU nếu audit cho thấy thiếu raw score/model branch cần thiết
```

### P2 - Sàng lọc tín hiệu model mới theo diversity gate

Không thêm model chỉ vì nằm trong whitelist. Mỗi candidate mới phải chạy
zero-shot trước, đo standalone, disagreement và oracle union trên canonical dev;
chỉ fine-tune nếu nó cứu được lỗi mà baseline hiện tại bỏ sót.

Các model dưới đây đều có tên chính xác trong whitelist CSV. Thứ tự sàng lọc
phụ thuộc parameter budget còn lại:

| Ưu tiên | Model whitelist | Vai trò thử | Quyết định |
|---:|---|---|---|
| 1 nếu còn >=0.6B | `jinaai/jina-reranker-v3.5` | Listwise rerank top 10-20 | Phù hợp đúng bottleneck top 10 -> top 5 và khác pairwise CE; test zero-shot, chưa fine-tune. |
| 1 nếu chỉ còn 0.31-0.60B | `Alibaba-NLP/gte-multilingual-reranker-base` | Pairwise reranker khoảng 306M | Phương án nhẹ hơn; test zero-shot trên cùng evidence/candidate để đo diversity. |
| Chỉ khi retrieval oracle yêu cầu | `bqbbao6/vietnamese-legal-embedding` | Dense legal E5 branch, zero-shot top 100 | Chỉ giữ nếu union tăng R@100 hoặc kéo gold rank 6-20 rõ ràng; chưa fine-tune ngay. |

Model-card để kiểm tra lại trước khi chạy: [Jina reranker v3.5](https://huggingface.co/jinaai/jina-reranker-v3.5), [GTE multilingual reranker](https://huggingface.co/Alibaba-NLP/gte-multilingual-reranker-base), [Vietnamese legal embedding](https://huggingface.co/bqbbao6/vietnamese-legal-embedding), [ViRanker](https://huggingface.co/namdp-ptit/ViRanker). Whitelist chỉ xác nhận model được phép xem xét; license, revision và số tham số vẫn phải ghi vào manifest từ artifact thực tế.

Không ưu tiên ở vòng này:

- `AITeamVN/Vietnamese_Embedding_v2`, `namdp-ptit/ViRanker` và
  `BAAI/bge-reranker-base`: cùng họ BGE/XLM-R hoặc gần tín hiệu đã có, nguy cơ
  tương quan lỗi cao.
- Các E5 generic: ưu tiên bản Vietnamese legal trong whitelist trước.
- `Qwen/Qwen3-Reranker-0.6B`, PhoRanker: đã thử và không vượt Step 6.
- Model sinh 1.5B-3B hoặc reranker 2B: chưa được thêm khi chưa biết model LLM
  hiện tại và phần budget còn lại.

Gate cho model mới:

```text
retriever: union Recall@100 tăng >= 0.002 hoặc phục hồi >= 2 gold-equivalent/1,000 query
reranker: Recall@5 tăng >= 0.003 trên canonical dev, R@20/R@50 không giảm
diversity: oracle union top 5 tăng >= 0.005 so với các branch đã có
compliance: tổng tham số sau khi thêm vẫn < 4B
```

Mỗi vòng chỉ chạy một candidate. Nếu zero-shot fail gate thì dừng candidate,
không fine-tune và không tạo public submission.

#### P2B - Step 7b canonical SBERT, tạm ngưng

Nhánh này vẫn có giá trị để tạo dev chung hợp lệ, nhưng không còn là bước kế tiếp. `step7b_1` đã chạy checkpoint train; phần global dense/RRF chưa hoàn tất và nhánh đang tạm ngưng do chi phí GPU so với tín hiệu cải thiện chưa rõ.

Khi P1 cho thấy SBERT là feature còn thiếu hoặc stacker Step 4/5/6 không đủ headroom, mới tiếp tục:

1. Dùng đúng Step 1 train 6,000 / dev 1,000.
2. Sinh ranking dev/public top 100 cho từng checkpoint sạch.
3. Đánh giá standalone và fusion với Step 6 có metadata.
4. Chỉ sau đó mới thêm SBERT sạch vào feature stacker và tune weight trên dev.

Tài nguyên: Kaggle GPU notebook.

### P3 - Epoch sweep cho Vietnamese Bi-Encoder

Thử một biến trước:

```text
E1: epoch 1, lr 4e-5 (control hiện tại)
E2: epoch 2, lr 4e-5
E3: epoch 3, lr 4e-5
```

Chạy một job 3 epoch và lưu checkpoint E1/E2/E3. Nếu E2/E3 overfit, job thứ hai mới hạ `lr=2e-5`; không mở grid lớn ngay từ đầu.

Với mỗi checkpoint:

1. Sinh FT dense ranking top 100 docs.
2. Fuse với Step 4 có metadata đang thuộc baseline khóa.
3. Tune grid RRF nhỏ quanh best hiện tại: `step4_weight={1.0,1.2,1.4}`, `ft_weight={0.3,0.4,0.5}`, `rrf_k={10,20,40}`.
4. Áp dụng cổng: R@20/R@10/R@5 tăng và R@100 không giảm.

Tài nguyên: Kaggle GPU notebook.

### P4 - Hai nhánh cross-encoder

#### P4A. AITeamVN epoch checkpoint

Train/save checkpoint epoch 1-4 với config hiện tại. Đánh giá từng checkpoint trên cùng Step 5 có metadata candidates của baseline, sau đó tune score-fusion nhỏ quanh:

```text
retrieval_weight={0.4,0.5,0.6}
reranker_weight={0.7,0.8,0.9}
```

Không chọn checkpoint theo loss. Chỉ giữ checkpoint tăng R@5/R@10/R@20 và giữ R@50.

#### P4B. BGE-reranker-v2-m3

Checkpoint external 2 epoch là insight từ split khác, không phải reproduction
target hợp lệ. Chỉ fine-tune `BAAI/bge-reranker-v2-m3` trên canonical training
pairs/candidate pool nếu audit raw logit cho thấy oracle headroom. Lưu epoch
1/2/3 và chọn hoàn toàn bằng canonical dev.

Đánh giá ba cách:

1. BGE reranker + retrieval score.
2. Weighted RRF AITeamVN-fused + BGE-fused.
3. Union/stacking AITeamVN logit + BGE logit, để đo tính bổ sung thay vì chỉ số standalone.

Tài nguyên: Kaggle GPU notebook. Đây là model cross-encoder mới, không phải BGE-M3 dense của Step 4.

### P5 - Mở rộng stacker bằng feature mới

Phần XGBoost tối thiểu đã được chuyển lên P1. P5 chỉ mở rộng sau khi P1 pass gate bằng cách thêm score sạch từ P2/P3/P4.

Mỗi dòng feature là một cặp `(query, candidate_doc)` trong top 50-100. Feature đề xuất:

```text
BM25 rank, raw score, max/mean-top3/support chunk
BGE-M3 dense rank và score
Vietnamese Bi-Encoder rank và score
SBERT-FT rank và score
Step 4/5/6 fused ranks
AITeamVN reranker logit
BGE-reranker-v2-m3 logit
số branch đưa candidate vào top 5/10/20
RRF agreement score
query/doc length và lexical overlap cơ bản
score/rank gap với candidate trước và sau trong từng branch
chênh lệch rank BM25-SBERT-CE và margin top 5/top 10
num_support_chunks, best evidence chunk score và evidence agreement
```

XGB chỉ cần học trên union top 10-20 vì đây là vùng lỗi lớn nhất. Giữ tail đến
top 50 để tính oracle/candidate metrics nhưng không để negative dễ áp đảo loss.

Ablation bắt buộc theo thứ tự:

```text
A0 = rank/score Step 4/5/6, không LLM
A1 = A0 + per-query score normalization + gap/margin
A2 = A1 + CE raw logit + evidence support features
A3 = A2 + structured LLM labels
```

Chỉ giữ nhóm feature khi delta trên canonical dev dương. Direct RRF của external
BGE CE đã làm public score giảm, nên CE chỉ được đưa vào A2 dưới dạng feature;
không tự động cộng thêm một nhánh RRF.

Dùng `XGBRanker` với group theo query; thử `rank:pairwise` và `rank:ndcg`. Không train binary classifier ngẫu nhiên trên các pair doc mà bỏ qua query group.

Chống leakage:

- Base model score cho tập train XGBoost phải là out-of-fold score.
- Phương án đúng: 5-fold OOF trên 6,000 train, dev 1,000 giữ nguyên để chọn final.
- Phương án nhanh để sàng lọc: chia 6,000 thành 5,000 base-train + 1,000 stacker-train; chỉ nếu có gain mới chạy 5-fold đầy đủ.

Tài nguyên:

```text
Tạo feature/rank artifacts: Kaggle notebook CPU
Train và tune XGBoost: Kaggle notebook CPU
Sinh logits base model còn thiếu: Kaggle GPU notebook
```

Grid nhỏ ban đầu:

```text
max_depth={3,5}
learning_rate={0.03,0.05}
n_estimators tối đa=800
subsample=0.8
colsample_bytree=0.8
early_stopping theo dev Recall@5/MRR
```

### P6 - Tái lập LLM feature có ablation

Public gain của external LLM/XGB là tín hiệu tốt nhưng chưa phải bằng chứng sạch
do split khác. Chỉ tái lập sau khi XGBoost không LLM đã có gain trên canonical
dev và P1.0 xác nhận model LLM hợp whitelist/budget. Không đổi sang LLM mới ở
vòng này. Step 7 Qwen3 rerank trực tiếp trước đây không đạt:

```text
Step 6 dev R@5:       0.905000
Qwen3 + Step 6 fused: 0.899833
Qwen3 only:           0.722333
```

Vì vậy LLM không được quyền thay ranking trực tiếp. Nó chỉ tạo feature cho XGBoost, ví dụ relevance logit, citation match hoặc confidence margin. Bắt buộc so sánh:

```text
XGBoost không LLM
XGBoost + LLM feature
```

Chỉ giữ LLM nếu delta R@5 dương và ổn định trên canonical dev, không phá cổng
R@50/R@100, đồng thời model/revision/license/checksum xác định được và tổng hệ
thống dưới 4B. Nếu không tái lập được feature lineage thì loại LLM/XGB khỏi
pipeline cuối dù public artifact từng có gain.

## 8. Thứ tự chạy để tiết kiệm GPU

1. P0 đã đóng: no-metadata không vượt baseline.
2. P1.0 audit split, feature lineage, model manifest và tổng tham số external.
3. P1.1 chạy oracle/disagreement audit bằng Kaggle CPU từ artifact canonical.
4. P1.2 control/preflight đã chạy; full clean stacker đang blocked. Tạo
   `p1_2_clean_base_features` sạch trước: fast holdout trong train, sau đó
   5-fold OOF train; canonical dev luôn untouched.
5. Tái lập BGE/LLM feature trên canonical chỉ khi ablation chỉ ra feature đó có
   gain; không dùng external dev metric để chọn checkpoint.
6. Nếu vẫn thiếu tín hiệu, P2 chỉ thử một model mới theo diversity gate; ưu tiên
   listwise reranker nếu đủ budget, GTE reranker nếu chỉ còn khoảng 0.3B; legal
   embedding chỉ khi audit chỉ ra candidate retrieval còn headroom.
7. P3/P4 epoch sweep chỉ thực hiện khi audit chỉ ra branch tương ứng có lỗi bổ sung hữu ích.
8. P5 mở rộng stacker bằng feature mới đã pass dev; P6 thêm lại LLM qua ablation.
9. Khóa config trên dev, train final đúng protocol và sinh 1-2 public submissions đã chọn trước.

## 9. Tiêu chí dừng và mục tiêu gần

Mục tiêu dev thực tế đầu tiên:

```text
Step 6 control R@100 giữ nguyên
Step 6 control R@50  = candidate R@50
Final dev R@5 tăng ít nhất +0.003 so với Step 6 control cùng split
Không có stage nào đổi gain top 5 bằng cách làm giảm R@100
```

Mục tiêu public:

```text
Mốc 1: > 0.928833
Mốc 2: >= 0.935000
```

Không đảm bảo score public từ dev gain. Mỗi submission phải kèm `run_report.json`, metric dev, config, model/checkpoint manifest và `submission_validation.json` để truy vết chính xác.

## 10. Kết luận ưu tiên

Không nên fine-tune BGE-M3 dense của Step 4 ngay: retrieval R@100 đã `0.982`, trong khi phần mất lớn nhất nằm ở R@10 -> R@5. Thứ tự có kỳ vọng lợi cao hơn là:

```text
khóa baseline baseline_plus_xgb_w0p25
-> audit compliance và split của toàn bộ external branch
-> oracle/disagreement audit trên canonical artifacts
-> XGBoost bằng OOF train feature, canonical dev untouched
-> tái lập đúng tín hiệu SBERT/BGE/LLM có gain trên canonical split
-> chỉ thử một model whitelist mới nếu còn oracle headroom và parameter budget
-> mở rộng stacker bằng feature đã pass dev
-> khóa config rồi mới nộp public
```
