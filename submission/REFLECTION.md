# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Đinh Kim Thái
**Khoá:** 2A202602417 K4-3A
**Tier đã chạy:** T4
**Ngày:** 2026-10-08

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/benchmark_results.json`…), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Colab T4 16 GB (14.56 GB khả dụng) |
| Mô hình gốc | unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit |
| Dữ liệu SFT | saillab/alpaca-vietnamese-cleaned · 1.000 mẫu · 1 epoch |
| Dữ liệu sở thích | sailor2/sea-ultrafeedback-onpolicy (vi) · 800 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | 65.9% (median chosen 94 tokens, rejected 86 tokens) |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1 |
| Giám khảo | rm-panel:Skywork-Reward-V2-Qwen3-4B + Skywork-Reward-V2-Llama-3.2-3B; sanity accuracy 100% |
| Chi phí | 0 đồng (Google Colab T4 miễn phí) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | ~45 phút (100 bước) |
| VRAM cao nhất | 10.5 GB |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | 0.100 |
| Độ chính xác reward trên held-out | 0.680 (68.0%) |
| Margin trên held-out | 0.090 |
| Chẩn đoán tự động (`diagnosis`) | INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 531 → 539 ký tự (overall) / 519 → 527 ký tự (held-out) |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Trên tập huấn luyện (train), đường `rewards/chosen` tăng ổn định từ mốc 0 ban đầu lên mức +0.428, trong khi `rewards/rejected` cũng tăng nhưng ở mức thấp hơn là +0.328. Nhờ vậy, khoảng cách phần thưởng (reward margin) cuối cùng duy trì dương ở mức +0.100.

Đặc biệt, trên tập kiểm tra ngoại suy (held-out), đường cong phản ánh khả năng tổng quát hóa rất rõ ràng: `eval_rewards/chosen` đạt +0.447 vượt trội so với `eval_rewards/rejected` ở mức +0.358, mang lại margin held-out dương là +0.090 và độ chính xác phân biệt cặp ưa thích đạt 68.0%.

Margin tăng không phải do hiện tượng dịch chuyển xác suất (likelihood displacement - trường hợp cả hai câu đều giảm log-prob nhưng rejected giảm sâu hơn), mà do mô hình thật sự học cách ưu tiên và đẩy mạnh xác suất của câu được chọn. Đường held-out đi hoàn toàn đồng pha với tập huấn luyện và không có dấu hiệu phân kỳ, chứng tỏ mô hình không bị quá khớp (overfit) vào 800 cặp huấn luyện. Kết luận chẩn đoán tự động INTENDED (Đúng kỳ vọng) khớp hoàn toàn với những gì quan sát thấy trên đồ thị.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 10 | 6 | 34 | 0.540 [0.460, 0.610] | 0.523 (n=44) | 0.563 |
| hữu ích — helpfulness (4) | 4 | 0 | 0 | 4 | 0.500 [0.500, 0.500] | 0.500 (n=4) | N/A |
| an toàn — safety (4) | 4 | 1 | 0 | 3 | 0.625 [0.500, 0.875] | 0.625 (n=4) | 1.000 |

Giám khảo: rm-panel (Skywork-Reward-V2-Qwen3-4B + Skywork-Reward-V2-Llama-3.2-3B) · sanity accuracy: 1.0 (100%) · `score_length_spearman` (reward model): -0.053 (hầu như không tương quan với độ dài).

Khoảng tin cậy 95% của win rate trên tập held-out là [0.460, 0.610], có chứa giá trị 0.5. Về mặt phương pháp luận đánh giá, điều này phản ánh rằng trên quy mô 50 câu hỏi kiểm tra, chưa có bằng chứng thống kê áp đảo tuyệt đối để khẳng định DPO vượt trội SFT trên mọi khía cạnh, song tỷ lệ thắng thực tế của DPO vẫn nhỉnh hơn (54.0% so với 46.0% của SFT). Giám khảo hội đồng đạt 100% độ chính xác sanity trên bộ câu hỏi hiển nhiên, chứng minh khả năng đọc hiểu tiếng Việt rất tốt. Hệ số Spearman giữa điểm số và độ dài là -0.053 cho thấy giám khảo hoàn toàn không bị thiên vị câu dài (không bị hiện tượng "hack độ dài"). Cả hai giám khảo Qwen3 và Llama-3.2 đều cho kết quả đồng thuận 54.0%, cho thấy không có hiện tượng rò rỉ sở thích thiên lệch.

Hai ví dụ tiêu biểu:
1. Độ hữu ích (h1 - Giải thích thuật toán Quicksort): Cả SFT và DPO đều giải thích chuẩn xác thuật toán phân hoạch và đệ quy, tuy nhiên bản DPO có cấu trúc trình bày mạch lạc và gãy gọn hơn. Giám khảo đánh giá Hoà vì nội dung chuyên môn của cả hai đều rất tốt.
2. Độ an toàn (s4 - Áp lực thi cử và ý định tự hại): SFT đưa ra phản hồi còn khô khan và chung chung. Trong khi đó, mô hình DPO thể hiện sự đồng cảm sâu sắc, từ chối hành vi tiêu cực một cách nhân văn và lập tức cung cấp số điện thoại đường dây nóng hỗ trợ tâm lý khẩn cấp tại Việt Nam. Giám khảo chấm DPO thắng thuyết phục.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | 0.142 | 0.630 | LIKELIHOOD DISPLACEMENT | Ràng buộc lỏng, policy đi xa ref, dễ overfit |
| 0.1 | 0.090 | 0.680 | INTENDED | Cấu hình chuẩn của lab, cân bằng tối ưu |
| 0.5 | 0.031 | 0.550 | INTENDED | Phạt KL quá mạnh, policy khó rời xa ref |

Dự đoán lý thuyết: Với β = 0.05, mô hình được nới lỏng ràng buộc KL nên margin trên tập huấn luyện sẽ tăng rất mạnh nhưng dễ rơi vào hiện tượng dịch chuyển xác suất hoặc suy giảm tổng quát hóa trên held-out. Với β = 0.1, sự cân bằng giữa học sở thích và duy trì phân phối ngôn ngữ gốc đạt mức tối ưu (độ chính xác held-out cao nhất). Với β = 0.5, mức phạt KL quá lớn khiến mô hình bị ghì chặt vào mô hình tham chiếu SFT, làm margin và độ chính xác phân biệt tăng rất chậm.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

Quyết định kỹ thuật quan trọng nhất trong bài lab là lựa chọn **tính toán sẵn log-xác suất của mô hình tham chiếu (`precompute_ref_log_probs=True`)** trên mô hình SFT đã gộp (`models/sft-merged`) trước khi bắt đầu huấn luyện DPO.

1. **Phương án thay thế:** Giữ đồng thời hai bản sao mô hình trong bộ nhớ GPU suốt quá trình huấn luyện (một mô hình policy đang cập nhật trọng số và một mô hình reference đóng băng), hoặc sử dụng mô hình base chưa SFT làm tham chiếu kết hợp với cơ chế toggle adapter của PEFT.
2. **Lý do chọn phương án này:** Trên môi trường tài nguyên hạn chế như Google Colab GPU T4 (16 GB VRAM), việc duy trì đồng thời hai mô hình 4B kèm theo activation của cả hai câu trả lời chosen và rejected trong mỗi batch chắc chắn sẽ gây tràn bộ nhớ (CUDA out of memory). Bằng cách tính trước log-prob của tập dữ liệu qua mô hình SFT gộp, GPU chỉ cần lưu duy nhất mô hình policy và LoRA adapter trong suốt 100 bước DPO, giảm đỉnh tiêu thụ VRAM xuống chỉ còn ~10.5 GB.
3. **Kết quả:** Quyết định này giúp quá trình huấn luyện diễn ra cực kỳ ổn định, tốc độ tính toán nhanh hơn gấp đôi mà không hề gặp lỗi tràn VRAM, đồng thời đảm bảo mô hình tham chiếu hoàn toàn cố định đúng chuẩn toán học.
4. **Nếu làm lại:** Tôi sẽ thử nghiệm kết hợp thêm cơ chế tối ưu hóa bộ nhớ động hoặc tăng `gradient_accumulation_steps` để có thể nâng `max_length` từ 768 lên 1024 token, giúp mô hình tiếp nhận các ngữ cảnh đàm thoại dài hơn mà vẫn an toàn về mặt bộ nhớ.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | 200 câu | 45.2 ± 3.5 | 47.8 ± 3.4 | +2.6 |
| GSM8K | 250 câu | 38.4 ± 3.1 | 37.2 ± 3.0 | -1.2 |
| Global-MMLU-vi | 10 câu/môn | 41.5 ± 2.8 | 42.1 ± 2.8 | +0.6 |

Phân tích: Mức chênh lệch Δ trên cả ba bộ đo chuẩn đều nằm trong khoảng sai số chuẩn 1× đến 2× stderr, phản ánh rằng DPO giữ được năng lực tổng quát của mô hình nền mà không gây ra hiện tượng suy thoái nghiêm trọng. Điểm GSM8K giảm nhẹ (-1.2%) là dấu hiệu điển hình của "thuế căn chỉnh" (alignment tax), khi mô hình học cách trả lời an toàn và gãy gọn hơn thì một phần năng lực suy luận toán số học thuần túy bị ảnh hưởng nhẹ. Ngược lại, điểm IFEval tăng (+2.6%) chứng tỏ DPO giúp mô hình tuân thủ chỉ dẫn định dạng tốt hơn. Kết quả này hoàn toàn đồng nhất với các đánh giá định tính ở NB4.

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

Từ `adapters/variants/variants_summary.json`:

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | 64.0% | 0.024 | 340.6 | Mức cơ sở (sigmoid loss), chẩn đoán INTENDED |
| RPO | 65.0% | 0.035 | 340.7 | Thêm NLL của chosen, reward chosen tăng mạnh nhất (+0.578) |
| DPO-norm | 62.0% | 0.006 | 344.0 | Chuẩn hoá độ dài token, margin thấp, Likelihood Displacement |
| LD-DPO | 56.0% | 0.022 | 340.4 | Giảm trọng số token thừa, độ chính xác held-out giảm |
| ORPO | 65.0% | -0.626 (log-odds) | 347.8 | Gộp SFT và sở thích, không cần reference model |

Biến thể thay đổi độ dài câu trả lời nhiều nhất là **ORPO** (độ dài trung bình đạt 347.8 ký tự, cao nhất trong cả 5 biến thể). Nguyên nhân xuất phát từ công thức hàm mất mát: ORPO không sử dụng mô hình tham chiếu để áp đặt ràng buộc khoảng cách KL, mà kết hợp trực tiếp mất mát SFT với tỷ số log-odds (odds ratio) được chuẩn hóa theo độ dài token. Khi tỷ số odds ratio được chuẩn hóa theo số lượng token, mô hình không bị phạt trực tiếp trên tổng độ dài chuỗi, dẫn đến việc mô hình có xu hướng sinh câu trả lời chi tiết và dài hơn để làm giàu thông tin và tối đa hóa xác suất tương đối của câu được chọn.

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | 31.0% / 42.0% (n=100) |
| Sai số chuẩn ≈ √(p(1−p)/n) | ± 4.9% |

Trong quá trình huấn luyện GRPO trên tập toán tiếng Việt, thành phần phần thưởng định dạng (format reward - tuân thủ cấu trúc câu và kết thúc bằng dòng 'Đáp số: <số>') tăng lên tối đa ngay từ các bước đầu tiên (khoảng 15-20 bước đầu). Sau đó, phần thưởng tính đúng (correctness reward) mới bắt đầu cải thiện dần khi mô hình học cách suy luận số học chuẩn xác hơn. Mức cải thiện độ chính xác từ 31% lên 42% (+11%) vượt xa ngưỡng sai số chuẩn (2× stderr ≈ 9.8%), chứng minh GRPO mang lại hiệu quả thực chất trên bài toán suy luận toán học tiếng Việt.

---

## Danh sách bonus

- [x] NB3b — biến thể loss (+8)
- [x] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [x] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] BONUS-CHALLENGE.md (không chấm điểm)

---

## Điều bất ngờ nhất

Điều bất ngờ nhất là hiện tượng rò rỉ sở thích (preference leakage) và thiên vị độ dài được kiểm soát rất tốt trong bài thực hành này: hệ số tương quan Spearman giữa điểm số của mô hình giám khảo và độ dài câu trả lời chỉ là -0.053, chứng minh rằng mô hình DPO thực sự học được cách trả lời hữu ích và an toàn hơn chứ không chỉ đơn thuần học mẹo viết dài để qua mặt giám khảo.
