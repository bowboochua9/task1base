1. **Noise Artifacts Quantification**:
   - `\r\n\n` / `\r\n` linebreaks split across 99.87% of context files.
   - Divider lines (`-------`): 94.20%
   - Metadata headers (`Số: ...`, `Hà Nội, ngày ...`): 93.80%
   - National motto (`CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM`): 90.00%
   - Recipient/Signature footers (`Nơi nhận:...`): 80.53%

2. **Empirical Impact**:
   - 28.21% average text noise reduction.
   - Fixes broken Vietnamese words (e.g., `Giao\n thông` -> `Giao thông`).
   - Reclaims ~144 tokens per 512-token chunk.

3. **Multi-Stage Cleaning Pipeline**:
   - Stage 1: Linebreak & Broken Word Repair (`\b[word]\n\s*[word]\b`).
   - Stage 2: Administrative Header Cleaning (strip motto, agency headers while keeping document title).
   - Stage 3: Signature & Recipient Footer Removal (`Nơi nhận:`).
   - Stage 4: Structural Section & Article Boundary Normalization (`Điều X.`).
   - Stage 5: Legal Amendment Cross-Referencing ("Sửa đổi", "Bổ sung").
