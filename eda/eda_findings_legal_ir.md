# 📖 UIT-DSC 2026: LegalIR – Kết quả Phân tích Dữ liệu (EDA) & Đánh giá

Dựa trên quá trình Khám phá Dữ liệu (EDA) dành riêng cho bài toán **LegalIR (Information Retrieval)**, dưới đây là phân tích chi tiết về tập dữ liệu, làm nổi bật những đặc trưng, độ khó và các gợi ý để xây dựng mô hình.

---

## 1. Dữ liệu ngữ cảnh (Context Corpus): Khối lượng văn bản pháp luật đồ sộ

**Thách thức về độ dài văn bản**
Các tài liệu ngữ cảnh được cung cấp cho bài toán LegalIR không phải là những đoạn văn ngắn; chúng là toàn bộ các văn bản luật (Luật, Nghị định, Thông tư và Quy chuẩn Kỹ thuật).
* **Độ dài trung vị (median)** là khoảng 4.800 từ.
* **Độ dài lớn nhất** lên tới con số khủng khiếp là **~1,2 triệu từ** (gần 6 triệu ký tự). Những tài liệu khổng lồ này thường là các Quy chuẩn Kỹ thuật (QCVN / TCVN) chứa rất nhiều bảng biểu và thông số.

**Mê cung "Sửa đổi, Bổ sung"**
Có tới **89,2%** tài liệu chứa các từ khóa liên quan đến việc sửa đổi (ví dụ: *sửa đổi, bổ sung, thay thế, bãi bỏ*). Hệ thống pháp luật Việt Nam có tính liên kết rất cao, với các nghị định mới liên tục sửa đổi các nghị định cũ.

## 2. Các cụm chủ đề: Những lĩnh vực của Pháp luật Việt Nam
Khi áp dụng thuật toán K-Means clustering (K=10) trên các vector TF-IDF, chúng ta thấy rõ các lĩnh vực khác nhau trong tập dữ liệu. Hệ thống sẽ cần phải xử lý các câu hỏi thuộc nhiều chủ đề đa dạng như:
* **Tổ chức bộ máy & Hành chính** (Chính phủ, cán bộ, nghĩa vụ)
* **Giáo dục & Đào tạo** (Trường học, tuyển sinh)
* **Tài chính, Kế toán & Ngân hàng** (Thuế, tín dụng, vốn)
* **Tiêu chuẩn Kỹ thuật** (TCVN/QCVN - Các quy chuẩn, tiêu chuẩn)
* **Đất đai & Bất động sản** (Sổ đỏ, quy hoạch, nhà ở)

---

## 3. Phân tích Câu hỏi

Phân tích 8.000 câu hỏi LegalIR cho thấy có thể phân loại câu hỏi theo hai hướng: **Intent** (mục đích hỏi là gì) và **Domain** (thuộc lĩnh vực luật nào).

**Các Intent phổ biến nhất (Hỏi về cái gì?)**
1. **Thủ tục / Trình tự** - "Thủ tục xin cấp giấy phép xây dựng thế nào?"
2. **Hồ sơ / Giấy tờ** - "Hồ sơ đăng ký kết hôn bao gồm những gì?"
3. **Thẩm quyền / Chủ thể** - "Cơ quan nào có thẩm quyền cấp sổ đỏ?"
4. **Điều kiện / Nguyên tắc**
5. **Thời gian / Hạn**

*Nhận xét*: Tỷ lệ câu hỏi về "Thủ tục" và "Hồ sơ" rất cao. Điều này có nghĩa là mô hình truy xuất (retrieval) phải rất giỏi trong việc tìm kiếm các danh sách từng bước hoặc các điều kiện được liệt kê trong văn bản luật.

## 4. Thách thức Truy xuất: Quy tắc 80/20 & Câu hỏi có nhiều đáp án

Mục tiêu cốt lõi của LegalIR là bài toán phân loại/xếp hạng: tìm các tài liệu ngữ cảnh trong đống "tàng thư" gồm 8.532 tài liệu để trả lời câu hỏi.

**Ràng buộc về nhiều đáp án:**
Một điểm rất quan trọng là **~7,9% câu hỏi trong tập train có nhiều hơn 1 ID ngữ cảnh đúng (ground truth)** (dao động từ 2 đến 5 đáp án).
* *Nhận xét về Metric*: Cuộc thi cho phép bạn đưa ra tối đa 5 ID ngữ cảnh. Vì một số câu hỏi *thực sự* có tới 5 đáp án đúng, việc chỉ trả về 1 đáp án duy nhất sẽ làm giảm điểm Recall của bạn một cách oan uổng ở những câu hỏi đó.

Tuy nhiên, sự phân bố của các câu trả lời lại cực kỳ lệch:
* Nằm ở nhóm đầu, **1.540 tài liệu (18% kho dữ liệu) trả lời cho 80% toàn bộ câu hỏi train**.
* Ngược lại, **5.592 tài liệu (65% kho dữ liệu)** *chưa bao giờ* được dùng làm câu trả lời trong tập train.
* *Nhận xét*: Tập dữ liệu có những bộ luật "trụ cột" (như Bộ luật Dân sự, Luật Doanh nghiệp, Bộ luật Lao động) đóng vai trò là những trung tâm giải đáp câu hỏi khổng lồ. Thực tế, có sự tương quan thuận ($r = 0,465$) giữa độ dài tài liệu và tần suất được trích dẫn. Luật càng dài thì càng bao quát nhiều vấn đề và trả lời được nhiều câu hỏi hơn.

## 5. Mô phỏng Baseline: Tỷ lệ trúng đích bằng từ vựng (Hit Rate của BM25 / TF-IDF)

Để thấy rõ độ khó của bài toán IR, chúng tôi đã mô phỏng một hệ thống truy xuất dựa trên từ vựng (Lexical TF-IDF) trên một mẫu 1.000 câu hỏi. Chúng tôi đã tính "Hit Rate" (tỷ lệ có *ít nhất một* ID đúng nằm trong top K kết quả):
* **Hit@1:** 28,7%
* **Hit@2:** 40,5%
* **Hit@5:** 58,3%
* **Hit@10:** 69,8%

*Nhận xét về Chiến lược*: Cuộc thi cho phép đưa ra tối đa 5 ID ngữ cảnh, nhưng đưa ra nhiều hơn 1 ID sẽ làm giảm Precision nếu đó là các kết quả sai (false positives). Vì phương pháp khớp từ vựng thuần túy chỉ đạt 40,5% ở Hit@2 nhưng tăng vọt lên 58,3% ở Hit@5, **bạn không thể chỉ dựa vào BM25**. Bạn bắt buộc phải huấn luyện một mô hình xếp hạng lại theo ngữ nghĩa (Cross-Encoder) có khả năng đưa ra ngưỡng độ tự tin (confidence threshold). Nếu có nhiều tài liệu vượt qua ngưỡng tự tin cao này, hãy xuất tất cả chúng (tối đa 5) để tối đa hóa Recall, nếu không thì chỉ xuất 1 tài liệu để bảo vệ điểm Precision.

---

## 💡 Các chiến lược để xây dựng mô hình

1. **Bắt buộc phải chia nhỏ văn bản (Chunking):** Các mô hình dense retriever tiêu chuẩn (Bi-Encoders) thường có giới hạn ngữ cảnh từ 512 đến 8k token. Việc đẩy một tiêu chuẩn kỹ thuật dài 1,2 triệu từ vào mô hình dense sẽ làm sập mô hình hoặc bị cắt xén nghiêm trọng. Chúng ta **bắt buộc** phải chia nhỏ tài liệu.
2. **Metadata là chìa khóa:** Vì 89% tài liệu có liên quan đến việc sửa đổi, nên việc chỉ dựa vào độ tương đồng văn bản là rất nguy hiểm. Chúng ta cần trích xuất các siêu dữ liệu (metadata) (ví dụ: Năm ban hành, Loại tài liệu) để giúp xếp hạng các luật còn hiệu lực cao hơn các luật đã bị bãi bỏ.
3. **Tìm kiếm từ vựng vẫn cực kỳ quan trọng:** Các truy vấn pháp lý thường chứa các cụm từ chính xác (ví dụ: "Khoản 2 Điều 15 Nghị định 123"). Truy xuất thưa (Sparse retrieval như BM25) cực kỳ mạnh mẽ trong việc khớp chính xác từ vựng và không nên bị bỏ qua để chạy theo mỗi dense retrieval.

---

## 🚀 Các bước tiếp theo

### Xây dựng Baseline cho IR (BM25 + Cross-Encoder)
1. **Phân tích phân cấp & Chia nhỏ (Chunking):** Chia nhỏ 8.532 tài liệu theo ranh giới ngữ nghĩa (ví dụ: theo *Điều*) hoặc cửa sổ 500 từ gối lên nhau. Cần lưu lại ID tài liệu gốc của mỗi đoạn nhỏ đó.
2. **Index Từ vựng:** Xây dựng chỉ mục BM25 trên các đoạn đã chia nhỏ. BM25 nổi tiếng là rất khó bị đánh bại trong truy xuất văn bản pháp luật nếu mô hình dense không được fine-tune kỹ.
3. **Truy xuất & Tổng hợp:** Đối với mỗi câu hỏi, truy xuất top 50 đoạn nhỏ bằng BM25, ánh xạ các đoạn đó về ID tài liệu gốc của chúng, và lấy điểm số cao nhất (max score) làm điểm cho tài liệu gốc đó.
4. **Xếp hạng lại (Reranking):** Xếp hạng lại top 5-10 tài liệu gốc bằng một mô hình cross-encoder tiếng Việt (ví dụ: `vinai/phobert-base` được fine-tune cho bài toán sequence classification).
