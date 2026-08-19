# Kế hoạch triển khai Step 7b - SBERT-CL và cải thiện reranking

## 1. Mục tiêu

Step 7b được thực hiện trước Step 8 để kiểm tra và tích hợp tín hiệu từ thí
nghiệm SBERT Contrastive Learning (SBERT-CL), sau đó xử lý nút thắt evidence
và reranking của pipeline hiện tại.

Mục tiêu chính:

1. Tái lập SBERT-CL trên đúng split và chunks của pipeline chính.
2. Kiểm tra SBERT-CL như một nhánh dense retrieval toàn corpus, không chỉ dùng
   để rerank BM25 top-200.
3. Hợp nhất candidate của BM25, BGE-M3, Step 5 và SBERT-CL.
4. Cải thiện AITeamVN/Vietnamese_Reranker bằng chunk-wise evidence scoring.
5. Chỉ tạo submission mới khi Recall@5 trên dev vượt mốc Step 6 hiện tại.

Step 8 full-train chưa được thực hiện cho đến khi Step 7b hoàn tất các ablation
và khóa được cấu hình tốt hơn hoặc xác nhận Step 6 vẫn là pipeline tốt nhất.

## 2. Mốc hiện tại

### Public leaderboard

| Pipeline | Precision | Recall |
|---|---:|---:|
| Step 5 - fine-tuned Bi-Encoder | 0.1814 | 0.844833 |
| Step 6 - fine-tuned PhoRanker | 0.1872 | 0.870667 |
| Step 6 - fine-tuned AITeamVN/Vietnamese_Reranker | **0.1922** | **0.8940** |
| SBERT-CL experiment | 0.1910 | 0.890583 |

SBERT-CL thấp hơn Step 6 best khoảng `0.00342` Recall nhưng đủ gần để xem như
một nhánh có khả năng bổ sung lỗi cho pipeline hiện tại.

### Dev chính của pipeline

Step 6 fine-tuned AITeamVN/Vietnamese_Reranker:

| Metric | Giá trị |
|---|---:|
| Recall@5 | 0.905000 |
| Recall@50 | 0.976167 |
| Recall@90 | 0.980667 |
| Recall@100 | 0.982000 |

Phân rã trên 1.000 query dev:

- 31 query thiếu ít nhất một gold document trong top-50.
- 81 query có đủ gold trong top-50 nhưng không đưa đủ gold vào top-5.
- Recall@5 với query có 1 gold: `0.9229`.
- Recall@5 với query có 2 gold: `0.7391`.
- Recall@5 với query có từ 3 gold: `0.4000`.

Vì vậy, dư địa từ reranking lớn hơn dư địa chỉ mở rộng candidate. Step 7b vẫn
thử SBERT-CL ở retrieval, nhưng ưu tiên cao nhất sau đó là evidence selection,
chunk-wise scoring và xử lý multi-gold.

## 3. Đánh giá thí nghiệm SBERT-CL hiện có

Nguồn:

- `other_research/DSC2026_Kaggle_T4_SBERT_CL_Experiment.md`
- `other_research/DSC2026_Kaggle_T4_SBERT_CL_Experiment.ipynb`
- Checkpoint fine-tuned do người dùng lưu trên Kaggle Dataset.

Các điểm có giá trị:

- Dùng `bkai-foundation-models/vietnamese-bi-encoder`.
- Word-segment query và chunk bằng PyVi trước khi encode.
- Chọn positive proxy trong toàn bộ gold document.
- Loại toàn bộ gold document khỏi negative pool.
- Kết hợp lexical negatives và semantic semi-hard negatives.
- Chọn checkpoint theo Recall@5 thay vì chỉ nhìn contrastive loss.

Các giới hạn cần sửa trước khi tích hợp:

1. Experiment dùng split `6000 train / 500 val / 500 test`, khác split chính
   `6000 train / 1000 dev`.
2. Checkpoint được gửi có thể đã train trên một phần query thuộc main dev. Do
   đó không được dùng main dev để khóa fusion weight cho checkpoint này.
3. Experiment tạo chunks riêng, khác 324.377 chunks từ Step 1.
4. Inference chỉ rerank BM25 top-200; chưa đo dense retrieval toàn corpus.
5. Mỗi sample chỉ có một lexical và một semantic explicit negative.
6. In-batch positives chưa được bảo vệ khỏi duplicate/false-negative.

Checkpoint được gửi vẫn hợp lệ để:

- kiểm tra khả năng load, revision, checksum và parameter count;
- tạo exploratory rankings;
- đo chi phí encode/inference;
- kiểm tra overlap với Step 6 public predictions.

Checkpoint này không được dùng làm bằng chứng chính để chọn cấu hình trên main
dev. Nhánh dùng để khóa pipeline phải train lại từ model BKAI gốc trên split
chính.

## 4. Nguyên tắc dữ liệu và artifact

### Input bắt buộc đã có trên Kaggle Dataset

- `step4/chunks.jsonl`
- `step4/train_split.json`
- `step4/dev_split.json`
- Step 5 dev/public rankings và dense artifacts đang được Step 6 sử dụng.
- Step 6 best dev/public rankings của
  `aiteamvn_vietnamese_reranker_finetuned`.
- Step 6 fine-tuned AITeamVN/Vietnamese_Reranker weights.
- `public-official.json` từ dataset gốc của BTC.

### Input mới

- Thư mục checkpoint SBERT-CL đã được gửi.
- Manifest của checkpoint, nếu dataset hiện chưa có:
  - model ID gốc;
  - revision;
  - license;
  - parameter count;
  - SHA-256 của từng weight file;
  - split và preprocessing đã dùng khi train.

Notebook phải khai báo một path chính xác cho checkpoint. Không tìm tự động,
không fallback sang BKAI base hoặc Step 5 checkpoint khi path bị thiếu.

Không đóng gói lại `chunks.jsonl`, split hoặc Step 6 artifacts vào archive mới
nếu chúng đã tồn tại trong Kaggle Dataset. Chỉ upload artifact thật sự mới.

## 5. Giai đoạn A - Tái lập sạch SBERT-CL

### A1. Baseline có kiểm soát

Train lại từ:

```text
bkai-foundation-models/vietnamese-bi-encoder
```

Sử dụng:

- đúng `step4/train_split.json` gồm 6.000 query;
- đúng `step4/dev_split.json` gồm 1.000 query;
- đúng `step4/chunks.jsonl`;
- PyVi cho cả query và chunk;
- max sequence length 256;
- cùng seed 42.

Thử nghiệm đầu tiên phải giữ recipe SBERT-CL gần bản nghiên cứu nhất để đo riêng
tác động của việc đồng bộ split/chunk. Không thay loss, số negatives và positive
selector cùng lúc.

### A2. Negative/loss ablation

Sau khi có baseline sạch, chạy lần lượt:

1. Hai explicit negatives như experiment hiện tại.
2. Mười semi-hard negatives cho mỗi positive.
3. `MultipleNegativesRankingLoss` với batch không trùng positive document.
4. `CachedMultipleNegativesRankingLoss` để tăng effective batch size trên T4.
5. Optional: `CachedGISTEmbedLoss` với guide model đã whitelist và false-negative
   filtering bằng similarity margin.

Mọi negative phải:

- không thuộc bất kỳ gold document nào của query;
- không trùng document ID;
- không phải near-duplicate rõ ràng của positive;
- được lấy trong semi-hard band, không tự động fallback sang hard-top;
- ghi rõ nguồn BM25, BGE-M3, Step 5 hoặc SBERT-CL.

Batch sampler phải tránh để hai positive cùng document bị coi là negatives của
nhau.

### A3. Positive evidence mining

So sánh ba cách chọn positive chunk trong gold document:

1. Lexical overlap hiện tại.
2. Lexical top-N rồi SBERT frozen chọn lại.
3. Union lexical + BGE-M3 + reranker teacher, giữ 1-3 chunks/gold.

Không dùng public labels hoặc public leaderboard để chọn positive strategy.

## 6. Giai đoạn B - Global dense retrieval và candidate fusion

### B1. Encode toàn corpus

Checkpoint tốt nhất từ Giai đoạn A encode toàn bộ 324.377 chunks đã
word-segment. Embeddings phải:

- normalize trước cosine/dot-product;
- lưu FP16 memmap hoặc shard để resume;
- có metadata gồm model checksum, preprocessing checksum và chunk manifest
  checksum;
- bị invalidated nếu một trong các checksum thay đổi.

### B2. Dense search

Truy xuất top-300 chunks toàn corpus cho train/dev/public, sau đó aggregate về
document bằng cấu hình đang dùng:

```text
max_score + 0.20 * mean_top3 + 0.05 * support_count
```

Giữ top-100 document và tối đa ba evidence chunks/document.

### B3. Candidate audit

Đo trên main dev:

- SBERT-CL standalone Recall@5/20/50/90/100;
- candidate Recall@50/90/100 của Step 5;
- candidate Recall@50/90/100 của union;
- số gold chỉ SBERT-CL tìm thấy;
- số gold Step 5 tìm thấy nhưng SBERT-CL bỏ lỡ;
- Jaccard và rank correlation giữa các nhánh;
- breakdown theo số gold và độ dài document.

### B4. RRF

Sweep giới hạn trên dev:

- `rrf_k`: 20, 40, 60, 80;
- SBERT-CL weight: 0.2, 0.4, 0.6, 0.8, 1.0;
- giữ nguyên nhánh Step 4/5 làm baseline.

Không dùng public score để chọn weight. Chỉ giữ SBERT-CL nếu candidate metrics
tăng hoặc final Recall@5 tăng sau cùng reranker.

## 7. Giai đoạn C - Rerank candidate union

Trước tiên dùng nguyên checkpoint fine-tuned
`AITeamVN/Vietnamese_Reranker` của Step 6, không train lại. Điều này tách riêng
gain do candidate retrieval khỏi gain do retraining reranker.

So sánh:

```text
Step 6 candidates -> Step 6 reranker
Step 7b union candidates -> cùng Step 6 reranker
```

Nếu union không tăng Recall@5 dù candidate Recall tăng, chuyển sang chunk-wise
evidence reranking.

## 8. Giai đoạn D - Chunk-wise evidence reranking

Step 6 hiện ghép tối đa ba chunks rồi cắt chuỗi ở 1.800 ký tự. Cách này có thể
làm chunk đầu chiếm gần hết input và loại mất evidence ở chunk sau.

Thay bằng việc score riêng từng cặp:

```text
(query, evidence_chunk_1)
(query, evidence_chunk_2)
(query, evidence_chunk_3)
```

Aggregate về document:

```text
document_score = max_chunk_score
               + alpha * mean_top2_chunk_score
               + beta * normalized_retrieval_score
```

Sweep nhỏ:

- evidence chunks/document: 1, 2, 3;
- `alpha`: 0.0, 0.1, 0.2;
- `beta`: 0.2, 0.3, 0.4;
- rerank depth: 30 và 50 documents.

Cache pair scores để sweep aggregate weights không phải inference lại.

Theo dõi riêng:

- 31 query candidate-miss;
- 81 query reranker-miss dù candidate đầy đủ;
- query có 1, 2 và 3+ gold;
- số query được sửa và số query bị làm hỏng so với Step 6.

## 9. Giai đoạn E - Fine-tune reranker lần hai

Chỉ thực hiện sau khi khóa candidate union và evidence selector.

Mine training groups từ candidate union mới:

- tất cả gold documents là positives;
- 10 semi-hard negatives/positive;
- ưu tiên negatives mà retriever và reranker bất đồng;
- loại toàn bộ gold, duplicate và near-duplicate;
- oversample có kiểm soát query multi-gold.

Ablation loss:

1. BCE hiện tại làm control.
2. BCE với balanced query-group sampling.
3. Pairwise margin/ranking loss.
4. Listwise softmax trên một query group.

Checkpoint selection:

1. Recall@5;
2. Precision@5 nếu Recall@5 bằng nhau;
3. Recall@5 của nhóm multi-gold;
4. paired bootstrap hoặc ba seed để tránh chọn gain do nhiễu.

## 10. Model ablation sau plateau

Chỉ thử model mới nếu SBERT-CL và chunk-wise reranking đã plateau.

Thứ tự ưu tiên:

1. `bqbbao6/vietnamese-legal-embedding`
   - khoảng 0.3B tham số;
   - fine-tuned cho Vietnamese legal retrieval;
   - dùng raw text với prefix `query:` và `passage:`.
2. `AITeamVN/Vietnamese_Embedding_v2`
   - Vietnamese-tuned từ BGE-M3;
   - có thể tương quan cao với nhánh BGE-M3 hiện tại.
3. `Qwen/Qwen3-Embedding-0.6B`
   - instruction-aware và đa ngôn ngữ;
   - chi phí encode lớn hơn, chỉ giữ khi tăng candidate recall.

Mỗi model chỉ chạy zero-shot global retrieval trước. Không fine-tune trước khi
chứng minh được tín hiệu bổ sung trên main dev.

Không quay lại Qwen3-Reranker trong Step 7b vì Step 7 đã cho kết quả thấp hơn
Step 6 trên dev.

## 11. Tiêu chí chấp nhận

### Candidate branch

Giữ nhánh mới khi thỏa ít nhất một điều kiện:

- tăng Recall@50/90/100 trên dev và không làm final Recall@5 giảm;
- tăng final Recall@5 sau cùng reranker;
- giảm đáng kể thời gian/RAM mà Recall@5 không giảm.

### Final pipeline

Mốc cần vượt:

```text
Step 6 dev Recall@5 = 0.905
Step 6 dev Precision@5 = 0.193
```

Chỉ chấp nhận Step 7b khi:

1. Recall@5 lớn hơn `0.905`; hoặc
2. Recall@5 bằng trong sai số cho phép và Precision@5 cao hơn;
3. kết quả không phụ thuộc một seed duy nhất;
4. submission validator có 0 lỗi;
5. compliance audit pass.

Không thay best submission hiện tại chỉ vì một nhánh có standalone public score
gần bằng Step 6.

## 12. Submission policy

Mỗi candidate có thể tạo `submission.zip`, nhưng chỉ đề nghị nộp Codabench khi:

- dev Recall@5 tăng rõ so với `0.905`;
- hoặc error-overlap audit chứng minh ensemble sửa được tập lỗi khác Step 6;
- output có đúng 1.000 question IDs;
- mỗi answer có 1-5 document IDs dạng chuỗi;
- không duplicate hoặc ID ngoài corpus;
- ZIP chỉ chứa `submission.json`.

Nếu Step 7b không vượt gate, output phải ghi rõ `rejected` và copy best Step 6
submission chỉ để đóng gói/audit, không gọi đó là submission cải thiện.

## 13. Output dự kiến

```text
step7b/
|-- configs/
|   |-- step7b_config.json
|   `-- best_config.json
|-- models/
|   `-- bkai_sbert_cl_clean/
|-- embeddings/
|   |-- chunk_embeddings_fp16.npy
|   `-- embedding_manifest.json
|-- rankings/
|   |-- dev_rankings_sbert_cl_global.jsonl
|   |-- public_rankings_sbert_cl_global.jsonl
|   |-- dev_rankings_union.jsonl
|   `-- public_rankings_union.jsonl
|-- evidence_scores/
|   |-- dev_chunk_pair_scores.jsonl
|   `-- public_chunk_pair_scores.jsonl
|-- metrics/
|   |-- candidate_metrics.json
|   |-- overlap_report.json
|   |-- fusion_trials.json
|   |-- chunkwise_trials.json
|   `-- error_breakdown.json
|-- reports/
|   |-- model_manifest.json
|   |-- compliance_audit.json
|   |-- split_leakage_audit.json
|   |-- run_report.json
|   `-- final_decision.json
`-- submission/
    `-- <accepted_candidate>/
        |-- submission.json
        |-- submission_validation.json
        `-- submission.zip
```

Không bắt buộc đóng gói embeddings vào dataset của bước sau nếu Step 8 chỉ cần
rankings và model weights. Artifact upload cho bước sau phải được quyết định sau
khi biết Step 7b candidate nào được chấp nhận.

## 14. Thứ tự notebook Kaggle

Để tránh vượt giới hạn session T4x2, tách thành các notebook có checkpoint:

1. `step7b_1_clean_sbert_cl_train.ipynb`
   - chuẩn hóa split/chunks;
   - PyVi cache;
   - train và chọn checkpoint.
2. `step7b_2_global_dense_and_rrf.ipynb`
   - encode toàn corpus;
   - dense retrieval;
   - candidate audit và RRF.
3. `step7b_3_chunkwise_rerank.ipynb`
   - score evidence chunks;
   - aggregate ablation;
   - tạo submission nếu pass gate.
4. `step7b_4_reranker_retrain.ipynb` chỉ khi cần
   - mine candidate union;
   - train reranker lần hai;
   - final comparison.

Mỗi notebook phải fail-fast khi thiếu đúng artifact yêu cầu. Không dùng hidden
fallback sang checkpoint, ranking hoặc candidate pool của bước khác.

## 15. Cơ sở kỹ thuật

- BKAI Vietnamese Bi-Encoder yêu cầu input được word-segment trước khi dùng với
  SentenceTransformers và model gốc được train bằng MNRL:
  <https://huggingface.co/bkai-foundation-models/vietnamese-bi-encoder>
- SentenceTransformers khuyến nghị `NO_DUPLICATES` cho loss dùng in-batch
  negatives và hỗ trợ Cached MNRL/GIST để tăng effective batch, lọc false
  negatives:
  <https://sbert.net/docs/package_reference/sentence_transformer/losses.html>
- Vietnamese legal embedding candidate:
  <https://huggingface.co/bqbbao6/vietnamese-legal-embedding>
- Qwen3 Embedding 0.6B:
  <https://huggingface.co/Qwen/Qwen3-Embedding-0.6B>
- Vietnamese semi-hard LegalIR paper:
  <https://arxiv.org/abs/2507.14619>

## 16. Quyết định cuối của Step 7b

Kết quả cuối phải thuộc một trong ba trạng thái:

1. `accepted_retrieval`: candidate union mới cải thiện final Recall@5.
2. `accepted_chunkwise_reranker`: chunk-wise reranking cải thiện final Recall@5.
3. `rejected_keep_step6`: không có cấu hình nào vượt Step 6; giữ submission
   Step 6 Recall public `0.894` và chuyển sang Step 8 audit/package thay vì
   full-train mù.
