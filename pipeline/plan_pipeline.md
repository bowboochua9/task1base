# Plan pipeline Task 1 LegalIR

## Cơ sở thiết kế

Pipeline có tham khảo trực tiếp [2507.14619v1.pdf](../baseline/2507.14619v1.pdf):

- Fine-tune Vietnamese Bi-Encoder bằng `MultipleNegativesRankingLoss`.
- Truy xuất top 90 và đánh giá bằng `Exist@90`.
- Rerank bằng PhoRanker.
- Lấy semi-hard negatives từ candidate pool sau khi loại gold.
- Cấu hình 10 semi-hard negatives đạt `MRR@10 = 0.7911`, tốt hơn hard/easy negatives.

Các điều chỉnh bắt buộc cho DSC2026:

- Paper xử lý văn bản ngắn hơn đáng kể; corpus hiện tại phải chunk theo cấu trúc pháp luật rồi aggregate về document.
- Paper trả top 10, nhưng DSC2026 chỉ cho tối đa 5 document.
- Paper tối ưu `MRR@10`; pipeline này phải tối ưu Recall trước, Precision làm tie-break.
- Với câu hỏi multi-gold, phải loại toàn bộ gold document khỏi negative pool.
- EDA cho thấy full-document BM25 sai granularity vì context rất dài; chunk là đơn vị retrieval, document ID chỉ dùng để aggregate và submit.
- Chunk-BM25 + aggregate trên sample EDA tăng mạnh so với full-document BM25, nên phải đạt trần lexical trước khi thêm các nhánh dense/reranker phức tạp.

Pipeline cũng tham khảo [DSC_research.pdf](../../other_research/DSC_research.pdf) như tài liệu nền, không xem là pipeline cuối:

- SBERT, ANCE, RocketQA và RocketQAv2 củng cố thiết kế dense retrieval có precomputed embeddings, ANN/hard-negative mining và khả năng dùng reranker làm teacher sau khi reranker đủ ổn định.
- ColBERT/late interaction là hướng ablation sau plateau, không thuộc core vì chưa có model ColBERT rõ trong danh sách cho phép và chi phí lưu trữ/search cao hơn single-vector bi-encoder.
- Các hướng GNN, graph contrastive learning, LLM pruning và executable evidence solving chỉ được giữ ở backlog nghiên cứu vì chưa chắc phù hợp whitelist model, ngân sách compute và mục tiêu submission document-ID `Recall@5`.

## Kiểm soát quy định mô hình

Tạo manifest cho mọi thành phần có tham số, gồm model ID, revision, license, số tham số thực tế, vai trò và checksum.

Cấu hình dự kiến:

| Thành phần | Model | Tham số xấp xỉ |
|---|---|---:|
| Dense zero-shot | `BAAI/bge-m3` | 568M |
| Dense fine-tune | `bkai-foundation-models/vietnamese-bi-encoder` | khoảng 135M |
| Reranker zero-shot | `Qwen/Qwen3-Reranker-0.6B` | khoảng 600M |
| Reranker fine-tune | `itdainb/PhoRanker` | khoảng 135M |
| Lexical retrieval | BM25S | 0 |

Tổng dự kiến dưới 1.5B tham số. Số chính thức phải được đọc từ config/weights, không suy ra từ tên model.

Pipeline phải dừng ngay nếu:

- Tổng tham số các model đang hoạt động đạt hoặc vượt `4,000,000,000`.
- Model không nằm trong danh sách cho phép.
- Không xác định được license hoặc revision.
- Một thành phần gọi hosted inference, commercial API hoặc third-party model service.

LoRA, quantization, AWQ, GPTQ hoặc GGUF chỉ được xem là tối ưu vận hành; parameter audit vẫn dùng số tham số đầy đủ. Tất cả inference và training chạy từ weights local. Model được đóng gói thành Kaggle Dataset để không phụ thuộc dịch vụ bên ngoài.

## Pipeline dữ liệu

- Kiểm tra schema, ID trùng, gold ID thiếu và câu hỏi trùng.
- Chia cố định `6000 train / 1000 dev`, seed `42`, phân tầng theo số gold document.
- Làm sạch header, footer, chữ ký, dòng phân cách và từ bị ngắt nhưng giữ bản gốc để truy vết.
- Tách theo `Chương -> Mục -> Điều -> Khoản -> Điểm`.
- Đoạn dài chia thành cửa sổ 320 từ, overlap 60 từ; đoạn trên 900 từ hoặc không có heading dùng sliding window.
- Prepend tên văn bản và heading pháp luật; nếu context thiếu `name`, fallback từ heading, link hoặc phần mở đầu của passage.
- Sinh deterministic `chunk_id`, metadata cấu trúc và checksum.
- Trích xuất metadata loại văn bản, số hiệu, năm ban hành và tín hiệu sửa đổi/bổ sung/thay thế/bãi bỏ để dùng như field riêng khi truy vấn/ranking.
- Cache corpus đã làm sạch, chunk manifest, embeddings và indexes để resume được trên Kaggle.
- Audit số chunk quá lớn theo document; ngưỡng cảnh báo ban đầu là 420 chunks/context theo khuyến nghị EDA.

## Retrieval và reranking

1. BM25S lấy top 300 chunk, có score riêng cho `name`, heading và chunk body.
2. Tối ưu nhánh Chunk-BM25 trước: chunk size/overlap, stopword sau bỏ dấu, boost số hiệu, năm, `Điều/Khoản`, loại văn bản và heading.
3. Chỉ chuyển pha khi 2-3 ablation lexical liên tiếp tăng dưới 1-2 điểm Recall.
4. BGE-M3 lấy top 300 dense chunk bằng cosine similarity và FAISS.
5. Fine-tuned Vietnamese Bi-Encoder tạo nhánh dense thứ hai.
6. Thêm metadata branch cho loại văn bản, năm, số hiệu và tín hiệu sửa đổi khi dev chứng minh có ích.
7. Aggregate chunk về document bằng:

   ```text
   max_score + 0.20 * mean_top3 + 0.05 * support_count
   ```

8. Hợp nhất các nhánh bằng Reciprocal Rank Fusion, giữ top 90-100 document.
9. Giữ tối đa 3 evidence chunk cho mỗi candidate document.
10. Rerank top 50 document bằng Qwen3-Reranker và PhoRanker ở cấp evidence; ưu tiên phân biệt `support`, `insufficient` và `irrelevant`.
11. Điểm ban đầu:

   ```text
   0.70 * normalized_reranker + 0.30 * normalized_retrieval
   ```

12. Tối ưu retrieval depth, RRF constant và trọng số score bằng dev set.
13. Chỉ ensemble hai reranker nếu cải thiện `Recall@5` ổn định trên 3 seed.

## Huấn luyện

- Positive chunk của mỗi gold document là 1-3 chunk khớp query tốt nhất.
- Bi-encoder dùng `MultipleNegativesRankingLoss`, khởi đầu với learning rate `4e-5`, weight decay `0.02`.
- Mine lại candidates sau mỗi phiên bản bi-encoder được chấp nhận.
- Với mỗi positive, chọn ngẫu nhiên 10 semi-hard negatives trong top candidates sau khi loại tất cả gold ID.
- Thực hiện bước denoise negative pool trước khi ghi training sample: loại toàn bộ gold của query, loại candidate trùng ID và loại near-duplicate rõ ràng nếu checksum/text overlap cho thấy có nguy cơ false negative.
- Không ưu tiên hard-top sai nhất làm mặc định; hard negatives chỉ dùng trong ablation vì có rủi ro nhiễu cao hơn semi-hard negatives.
- PhoRanker dùng `BCEWithLogitsLoss`, 2 epoch, learning rate `2e-5`.
- Sau khi PhoRanker hoặc Qwen3-Reranker ổn định, có thể thử distill score/listwise ranking từ reranker sang Bi-Encoder, nhưng chỉ như ablation và chỉ giữ nếu tăng `Recall@5`.
- T4x2 chạy FP16; sử dụng DDP, gradient accumulation và gradient checkpointing khi cần.
- Chọn checkpoint theo `Recall@5`; nếu bằng nhau thì chọn `Precision@5` cao hơn.

## Đầu ra và đánh giá

Theo dõi:

- Candidate stage: `Exist@90`, `Recall@90` và `Recall@100`.
- Final stage: Recall, Precision, `Hit@1/5`, MRR và `Recall@1/5/20`.
- Breakdown theo số gold, intent, độ dài văn bản và độ phổ biến document.

Mặc định xuất top 5 để ưu tiên Recall. Adaptive top-k chỉ được dùng nếu Precision tăng và Recall giảm không quá `0.001` trên dev.

Submission validator phải kiểm tra:

- Đủ toàn bộ query public.
- Mỗi answer gồm 1-5 ID dạng chuỗi.
- Không có ID trùng hoặc ID ngoài corpus.
- ZIP chỉ chứa `submission.json`.
- Metric implementation gán Recall và Precision bằng 0 khi dự đoán quá 5 ID.

## Thứ tự triển khai

1. Data validator, cleaning, chunking và official metric.
2. Tái lập chunk-BM25 baseline.
3. Tối ưu Chunk-BM25 đến plateau trên đủ 7.000 train: field score, metadata boost, citation boost, chunk cap và depth `Recall@20/50/100`.
4. Thêm BGE-M3, metadata branch và RRF.
5. Fine-tune Vietnamese Bi-Encoder, đánh giá `Exist@90`.
6. Mine semi-hard negatives và fine-tune PhoRanker.
7. Thêm Qwen3-Reranker, score fusion và ablation.
8. Khóa cấu hình trên dev, train lại bằng toàn bộ 7.000 câu.
9. Predict public set, chạy compliance audit và đóng gói submission.

Mỗi thành phần chỉ được giữ trong pipeline cuối khi cải thiện metric hoặc giảm đáng kể chi phí mà không làm giảm Recall. Cross-reference graph và LLM verifier không thuộc pipeline chính cho đến khi retrieval-reranking đạt plateau.

## Nghiên cứu bổ sung và quyết định phạm vi

- `Synthetic query generation`, `LLM judge/pruning`, `GNN/graph contrastive learning`, `local executable evidence solving` và ColBERT-style late interaction không thuộc pipeline chính.
- Chỉ xem xét các nhánh trên sau khi retrieval-reranking plateau và phải pass đủ: model nằm trong whitelist, tổng tham số dưới 4B, toàn bộ chạy local, không dùng API và không làm giảm `Recall@5`.
- Với Task 1, evidence pruning chỉ dùng để chọn output top-k nếu tăng Precision và Recall giảm không quá `0.001` trên dev; không dùng generator-aligned pruning làm quyết định chính cho submission.
- Khi ghi cơ sở nghiên cứu, dùng tên paper/arXiv đã kiểm tra: `2405.11791` hiện là LEXA/graph contrastive learning kế thừa CaseGNN, không ghi chắc là `CaseGNN++` trong pipeline chính.
