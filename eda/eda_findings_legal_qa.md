# 📖 UIT-DSC 2026: LegalQA – Kết quả Phân tích Dữ liệu (EDA) & Đánh giá

Dựa trên quá trình Khám phá Dữ liệu (EDA) dành riêng cho bài toán **LegalQA (Reading Comprehension)**, dưới đây là phân tích chi tiết về tập dữ liệu, làm nổi bật những đặc trưng, độ khó và các gợi ý để xây dựng mô hình.

---

## 1. Dữ liệu ngữ cảnh (Context Corpus): Khối lượng văn bản pháp luật đồ sộ

**Thách thức về độ dài văn bản**
Các tài liệu ngữ cảnh được cung cấp cho bài toán LegalQA không phải là những đoạn văn ngắn; chúng là toàn bộ các văn bản luật (Luật, Nghị định, Thông tư và Quy chuẩn Kỹ thuật).
* **Độ dài trung vị (median)** là khoảng 4.800 từ.
* **Độ dài lớn nhất** lên tới con số khủng khiếp là **~1,2 triệu từ** (gần 6 triệu ký tự). Những tài liệu khổng lồ này thường là các Quy chuẩn Kỹ thuật (QCVN / TCVN) chứa rất nhiều bảng biểu và thông số.

**Mê cung "Sửa đổi, Bổ sung"**
Có tới **89,2%** tài liệu chứa các từ khóa liên quan đến việc sửa đổi (ví dụ: *sửa đổi, bổ sung, thay thế, bãi bỏ*). Hệ thống pháp luật Việt Nam có tính liên kết rất cao, với các nghị định mới liên tục sửa đổi các nghị định cũ.

## 2. Các cụm chủ đề: Những lĩnh vực của Pháp luật Việt Nam
Khi áp dụng thuật toán K-Means clustering (K=10) trên các vector TF-IDF, chúng ta thấy rõ các lĩnh vực khác nhau trong tập dữ liệu. Hệ thống sẽ cần phải đọc hiểu các câu hỏi thuộc nhiều chủ đề đa dạng như:
* **Tổ chức bộ máy & Hành chính** (Chính phủ, cán bộ, nghĩa vụ)
* **Giáo dục & Đào tạo** (Trường học, tuyển sinh)
* **Tài chính, Kế toán & Ngân hàng** (Thuế, tín dụng, vốn)
* **Tiêu chuẩn Kỹ thuật** (TCVN/QCVN - Các quy chuẩn, tiêu chuẩn)
* **Đất đai & Bất động sản** (Sổ đỏ, quy hoạch, nhà ở)

---

## 3. Phân tích Câu hỏi

Phân tích 8.000 câu hỏi LegalQA cho thấy có thể phân loại câu hỏi theo hai hướng: **Intent** (mục đích hỏi là gì) và **Domain** (thuộc lĩnh vực luật nào).

**Các Intent phổ biến nhất (Hỏi về cái gì?)**
1. **Thủ tục / Trình tự** - "Thủ tục xin cấp giấy phép xây dựng thế nào?"
2. **Hồ sơ / Giấy tờ** - "Hồ sơ đăng ký kết hôn bao gồm những gì?"
3. **Thẩm quyền / Chủ thể** - "Cơ quan nào có thẩm quyền cấp sổ đỏ?"
4. **Điều kiện / Nguyên tắc**
5. **Thời gian / Hạn**

*Nhận xét*: Tỷ lệ câu hỏi về "Thủ tục" và "Hồ sơ" rất cao. Điều này có nghĩa là mô hình Reading Comprehension phải thực sự giỏi trong việc trích xuất các danh sách từng bước hoặc các điều kiện được liệt kê.

## 4. Thách thức Đọc hiểu: Phân tích Câu trả lời

**Tính trích xuất & Câu trả lời dài**
Câu trả lời trong LegalQA không phải là những câu ngắn gọn.
* **Độ dài trung vị của câu trả lời là 312 từ**, và câu dài nhất lên tới hơn 2.400 từ!
* Văn bản sử dụng rất nhiều từ khóa pháp lý (*Điều, Khoản, Nghị định, Luật*).

*Nhận xét*: Câu trả lời chủ yếu là trích xuất trực tiếp hoặc ghép nối các điều khoản luật cụ thể từ ngữ cảnh. Mô hình không cần "sáng tác" văn bản; nó cần hoạt động như một công cụ trích xuất (extractive summarizer) có độ chính xác cao.

## 5. Phong cách trả lời & Tối ưu hóa METEOR

Để đạt điểm cao nhất cho metric **METEOR** (và ROUGE-L), câu trả lời do mô hình sinh ra phải khớp chính xác với câu từ và văn phong của dữ liệu gốc (ground truth). Chúng tôi đã trích xuất các cụm từ mở đầu và dấu câu kết thúc của cả 7.000 câu trả lời để tìm ra quy tắc ngầm của người gán nhãn.

**Top 5 Cụm từ mở đầu (Prefixes):**
1. `Căn cứ theo quy định` (471 lần)
2. `Theo quy định tại Điều` (195 lần)
3. `Căn cứ khoản 1 Điều` (182 lần)
4. `Theo quy định tại khoản` (158 lần)
5. `Căn cứ khoản 2 Điều` (142 lần)

**Top Dấu câu kết thúc:**
1. `.` (Dấu chấm) - 4.478 câu (64%)
2. `)` (Dấu ngoặc đơn đóng) - 1.330 câu (19%)

*Nhận xét về Metric*: Các câu trả lời có tính rập khuôn rất cao! Hầu như tất cả đều bắt đầu bằng việc trích dẫn rõ ràng điều khoản (ví dụ: "Căn cứ theo..."). Nếu bạn sử dụng mô hình Generative LLM, bạn phải prompt để nó định dạng đầu ra bắt đầu bằng những cụm từ này và kết thúc bằng dấu chấm, nếu không điểm METEOR của bạn sẽ tụt thê thảm do sai lệch văn phong.

---

## 💡 Các chiến lược để xây dựng mô hình

1. **Bắt buộc phải chia nhỏ văn bản (Chunking):** Việc đẩy một tiêu chuẩn kỹ thuật dài 1,2 triệu từ vào LLM hay các mô hình dense retriever sẽ gây ra lỗi tràn bộ nhớ hoặc bị cắt xén dữ liệu. Chúng ta **bắt buộc** phải chia nhỏ tài liệu theo các phần logic (ví dụ: chia theo `Điều`).
2. **Sinh văn bản (Generative) vs Trích xuất (Extractive):** Vì các câu trả lời hầu hết là trích xuất nguyên văn luật, một mô hình Extractive QA truyền thống (dự đoán vị trí bắt đầu/kết thúc) có thể gặp khó khăn nếu phải ghép nhiều đoạn. Phương pháp Retrieval-Augmented Generation (RAG) sử dụng LLM với prompt *"trích xuất và giữ nguyên văn"* là một lựa chọn tối ưu.

---

## 🚀 Các bước tiếp theo

### Xây dựng Baseline cho QA (RAG)
1. **Load Ngữ cảnh:** Sử dụng tài liệu ngữ cảnh (ground-truth) đã được cung cấp.
2. **Chia nhỏ (Chunking):** Cắt tài liệu khổng lồ thành các đoạn nhỏ dựa trên ranh giới ngữ nghĩa (ví dụ: `^Điều \d+:`) hoặc cửa sổ 500 từ.
3. **Truy xuất cục bộ (Local Retrieval):** Sử dụng BM25 hoặc cross-encoder để tìm 2-3 đoạn liên quan nhất *bên trong* tài liệu đó dựa trên câu hỏi.
4. **Sinh câu trả lời (Generation):** Dùng prompt cho LLM (ví dụ: Llama 3, Gemini): *"Dựa hoàn toàn vào các điều khoản pháp luật được cung cấp, hãy trả lời câu hỏi. Trích dẫn nguyên văn luật nếu có thể."*
