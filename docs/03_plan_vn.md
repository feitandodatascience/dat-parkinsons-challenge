# 03 — Kế hoạch làm việc: 15 ngày, vừa học vừa thi

**Hôm nay: 2026-09-01. Hạn chót: 2026-09-16, 23:59 UTC.** Tức là khoảng hai tuần, khá gấp nhưng
vẫn làm được — 801 người đã đăng ký và phần lớn sẽ không nộp gì cả. Một bài nộp hoàn chỉnh, được
hiệu chuẩn, được kiểm định trung thực sẽ đưa bạn vượt lên trên đa số.

Kế hoạch này được xây trên một nguyên tắc: **đưa được một bài nộp hợp lệ lên bảng xếp hạng ngay
trong ngày đầu tiên**, rồi mới cải thiện nó. Một model tuyệt vời nhưng chưa đóng gói xong trước
ngày 16 thì được 0 điểm.

Việc học được đan xen vào, không dồn hết lên đầu. Bạn học một kiến thức chuyên ngành đúng vào
lúc nó làm thay đổi một quyết định bạn đang phải đưa ra — đó là lúc kiến thức đó bám lại.

---

## Giai đoạn 0 — Thiết lập (ngày 1, buổi sáng, ~2 tiếng)

| | Việc cần làm |
|---|---|
| ☐ | **Cài Python 3.12.** Bạn đang có 3.14; runtime chỉ chạy 3.12 và một số wheel đã ghim phiên bản sẽ không tồn tại cho 3.14. `py -3.12 -m venv .venv` rồi `.venv\Scripts\activate` |
| ☐ | `pip install -e ".[dev]"` |
| ☐ | `pytest -q` — 68 test phải pass trên dữ liệu tổng hợp, không cần dữ liệu cuộc thi |
| ☐ | Chấp nhận quy định cuộc thi và **tự tay tải dữ liệu về** vào `data/raw/` |
| ☐ | Cài **Docker Desktop** — cần để kiểm thử bài nộp tại máy |
| ☐ | Clone [repo runtime](https://github.com/drivendataorg/competition-sfmn-parkinsons-runtime) thành một thư mục ngang hàng, chạy `just pull` (cần ~21 GB trống) |
| ☐ | Xác nhận GPU: `python -c "import torch; print(torch.cuda.is_available())"` |

**Không có GPU tại máy?** Hãy quyết định ngay bây giờ, đừng để tới ngày thứ 10. Colab Pro hoặc
thuê một A100/A10 trong vài ngày là rẻ so với hai tuần thời gian của bạn. Chạy CNN 3D trên vài
nghìn khối ảnh không phải là việc dành cho CPU.

**Đọc trong lúc chờ tải (~30 phút):** [docs/01](01_domain_primer_dat_scans_vn.md) §§1–4. Đủ để
biết thể vân hình dấu phẩy là gì và vì sao nó biến thành một chấm.

---

## Giai đoạn 1 — Nhìn thấy dữ liệu, nộp bài trọn vẹn (ngày 1–2)

Mục tiêu của hai ngày này là một con số trên bảng xếp hạng. Bất kỳ con số nào.

| | Việc cần làm |
|---|---|
| ☐ | `python scripts/01_inspect_data.py` |
| ☐ | **Tự bạn** đọc `reports/eda/raw/SHARED_DRAFT.md`. Xác nhận không có giá trị của từng mẫu. |
| ☐ | Chép nó sang `reports/eda/shared/dataset_profile.md` → giờ Claude mới có thể hỗ trợ |
| ☐ | Mở vài ảnh trong một trình xem (ITK-SNAP, 3D Slicer, hoặc `nibabel` + matplotlib trong notebook). **Hãy nhìn chúng.** Tìm ra hình dấu phẩy. Tìm một ảnh đã thành cái chấm. |
| ☐ | `python scripts/02_make_folds.py` |
| ☐ | Huấn luyện baseline: `python scripts/03_train.py --config conf/baseline.yaml` |
| ☐ | `python scripts/04_evaluate.py` → OOF log loss, AUROC, ECE. So với baseline tiên nghiệm. |
| ☐ | `python scripts/05_package_submission.py --temperature <giá trị đã fit> --fallback <tỷ lệ dương>` |
| ☐ | Kiểm thử trong Docker: `just pack-submission && just test-submission` |
| ☐ | **Nộp smoke test.** Rồi mới nộp bài đầy đủ. |

> Việc tự bạn nhìn vào ảnh không phải là tuỳ chọn và cũng không phải thủ tục hình thức. Đó là
> phần duy nhất của dự án này mà Claude về bản chất không thể làm thay bạn, và đó là nơi trực
> giác của bạn về việc model cần phát hiện cái gì được hình thành. Hãy dành ra một tiếng.

**Các mốc học tập (khi chúng xuất hiện):**
- Khi bạn nhìn thấy bảng shape/spacing → [docs/02](02_cv_for_3d_medical_imaging_vn.md) Delta 1.
- Khi bạn lần đầu thấy OOF log loss → [docs/00](00_competition_brief_vn.md) mục "Chỉ số đánh
  giá", và đọc docstring đầu module của [calibrate.py](../src/datpd/calibrate.py). Hãy tính
  baseline tiên nghiệm và ghi nó thành dòng số không của nhật ký thí nghiệm.
- Nếu smoke test thất bại → lỗi gần như chắc chắn mang tính cấu trúc (import, đường dẫn, vị trí
  file đầu ra). Đó chính xác là lý do chúng ta làm việc này vào ngày 2 chứ không phải ngày 15.

**Tiêu chí hoàn thành: có một điểm số trên bảng xếp hạng.** Đừng đi tiếp cho tới khi bạn có nó.

---

## Giai đoạn 2 — Làm cho việc kiểm định đáng tin cậy (ngày 3–5)

Một con số CV mà bạn không tin còn tệ hơn là không có con số nào, bởi vì bạn sẽ tối ưu hoá theo
nó.

| | Việc cần làm |
|---|---|
| ☐ | Chạy `conf/grouped_cv.yaml`. So sánh log loss của `fold_strat` với `fold_group`. |
| ☐ | Chấm điểm theo từng chữ ký thu nhận ảnh. Có phải một trung tâm đang gánh toàn bộ sai số? |
| ☐ | So log loss CV của bạn với điểm trên bảng xếp hạng. Ghi lại khoảng cách và dấu của nó. |
| ☐ | Fit hiệu chuẩn temperature và isotonic trên OOF. Ghi lại cả ba (`choose_calibrator`). |
| ☐ | Tinh chỉnh ngưỡng clip dự đoán trên OOF. |
| ☐ | Thử bỏ bớt (ablate) cách chuẩn hoá: `reference_region` vs `zscore_brain` vs `max`. |
| ☐ | Thử bỏ bớt cách crop: `center` vs `com`. |

Tới cuối giai đoạn này bạn phải nói được rằng: *"CV của tôi đáng tin trong sai số X, và bảng xếp
hạng bám theo nó."* Mọi quyết định về sau đều dựa vào điều đó.

**Học:** [docs/01](01_domain_primer_dat_scans_vn.md) §5 (SBR — vì sao `reference_region` là lựa
chọn mặc định có nguyên tắc) và §6 (vì sao sự mơ hồ là không thể loại bỏ và hiệu chuẩn là câu
trả lời). Rồi tới [docs/02](02_cv_for_3d_medical_imaging_vn.md) Delta 6 và 7.

---

## Giai đoạn 3 — Cải thiện model (ngày 6–11)

Bây giờ, và chỉ bây giờ, mới đuổi theo điểm số. Mỗi lần một thay đổi, mỗi thay đổi được phán xét
bằng OOF log loss, mỗi thay đổi là một dòng trong
[reports/experiments/log.md](../reports/experiments/log.md).

Sắp xếp theo giá trị kỳ vọng trên mỗi đơn vị công sức:

1. **Độ phân giải.** 96³ @ 2 mm → 128³ @ 1,5 mm. Thường xuyên là cú thắng lớn nhất trong ảnh y
   tế 3D, bởi vì các cấu trúc là nhỏ so với trường nhìn.
2. **Quét augmentation.** Biên độ xoay/tịnh tiến/co giãn. Rồi một phép kiểm chứng có kiểm soát
   cho `flip_lr` — giờ bạn đã biết vì sao nó gây tranh cãi (docs/01 §4).
3. **Backbone 3D đã pretrain.** `monai.networks.nets.DenseNet121(spatial_dims=3)` và họ hàng.
   Kiểm tra giấy phép của trọng số; ghi lại trong [EXTERNAL_ASSETS.md](EXTERNAL_ASSETS.md).
4. **Một model 2.5D.** Các lát cắt đi qua một backbone ImageNet từ `timm` kèm attention pooling.
   Đủ khác biệt so với CNN 3D để ensemble tốt (docs/02 Delta 2).
5. **Label smoothing** (0,05) — rẻ, thường giúp trực tiếp cho hiệu chuẩn.
6. **Ensemble.** Trung bình các model fold trong không gian **logit**; rồi trung bình qua các
   kiến trúc. Hãy hiệu chuẩn lại cho ensemble — mức hiệu chuẩn của một ensemble khác với của
   từng thành viên.
7. **Cắt theo trọng tâm khối / đăng ký về template.** Nếu EDA cho thấy các bộ não không được căn
   giữa một cách nhất quán.
8. **Các đặc trưng kiểu SBR tường minh** (tỷ số putamen/caudate, chỉ số bất đối xứng) đưa vào
   một model gradient boosting nhỏ, rồi trộn với CNN. Diễn giải được, và đa dạng theo một cách
   hữu ích.

**Quy tắc cứng cho giai đoạn này:** nếu một thay đổi không cải thiện OOF log loss, hãy hoàn tác
nó. Giữ nhật ký trung thực, bao gồm cả các thất bại — chính các thất bại mới là thứ ngăn bạn thử
lại đúng ý tưởng đó vào 2 giờ sáng ngày 14.

**Học:** tới lúc này bạn đọc các bài báo với những câu hỏi thật sự. Tìm "deep learning DaT SPECT"
trên PubMed. Đọc một hai bài và hỏi cụ thể: tiền xử lý thế nào? đơn trung tâm hay đa trung tâm?
kiểm định ra sao? Tổng quát hoá đa trung tâm là phần khó, và phần lớn công trình đã công bố né
tránh nó.

---

## Giai đoạn 4 — Chốt sổ và nộp bài (ngày 12–15)

Ngừng thêm ý tưởng mới. Giai đoạn này là về việc không đánh mất những gì bạn đang có.

| | Việc cần làm |
|---|---|
| ☐ | Đóng băng kiến trúc. Huấn luyện lại các fold cuối cùng với một seed cố định. |
| ☐ | Fit lại hiệu chuẩn trên dự đoán OOF cuối cùng. Nướng giá trị temperature vào `main.py`. |
| ☐ | Đặt `FALLBACK_PROB` bằng tỷ lệ dương tính của tập huấn luyện. |
| ☐ | Chạy test Docker đầy đủ tại máy trên dữ liệu demo. Kiểm tra thời gian — ngoại suy ra quy mô tập test đầy đủ và xác nhận bạn nằm trong 3 tiếng với biên an toàn. |
| ☐ | Đọc lại các quy định về tính độc lập ([docs/00](00_competition_brief_vn.md)). Rà soát `main.py` từng dòng một. |
| ☐ | Nộp bài với **ít nhất 48 tiếng dự phòng.** Giới hạn số lần nộp tính theo ngày; hãy chừa chỗ để sửa nếu hỏng. |
| ☐ | Viết `docs/SOLUTION.md` — bắt buộc với người thắng cuộc, và nó buộc bạn phải tự xem xét lại các lựa chọn của mình. |
| ☐ | Xác nhận repo được cấp phép MIT và tái lập được. |

**Ngày 15 là dự phòng, một cách có chủ đích.** Sẽ có thứ gì đó hỏng. Các kiểu hỏng thường gặp:
một checkpoint không nhét vừa file zip, một import chỉ tồn tại ở máy bạn, một lần chạy suy luận
vượt quá giới hạn thời gian khi chạy ở quy mô tập test đầy đủ.

---

## Lộ trình học, dạng độc lập

Nếu bạn muốn một khung xương sống độc lập với đồng hồ đếm ngược của cuộc thi:

**Kiến thức chuyên ngành (~3 tiếng tổng cộng)**
1. [docs/01](01_domain_primer_dat_scans_vn.md) từ đầu đến cuối — 40 phút.
2. Radiopaedia "DaT scan" — 20 phút, chủ yếu để xem hình.
3. Hướng dẫn thực hành EANM/SNMMI về DaT SPECT, đọc lướt — 45 phút. Tập trung vào tiêu chí thu
   nhận ảnh và tiêu chí diễn giải.
4. Một bài báo PubMed về deep learning trên DaT SPECT — 45 phút.
5. Tài liệu về bộ dữ liệu PPMI — 30 phút. Bối cảnh cho gần như mọi công trình đã công bố. Chỉ
   đọc như y văn: dữ liệu PPMI **không** được phép dùng trong cuộc thi này
   ([EXTERNAL_ASSETS.md](EXTERNAL_ASSETS.md)).

**Computer vision cho ảnh y tế 3D (~6 tiếng)**
1. [docs/02](02_cv_for_3d_medical_imaging_vn.md) từ đầu đến cuối — 45 phút.
2. Repo tutorial của MONAI, các notebook `3d_classification` — 2 tiếng. Đây là thứ gần nhất với
   bài toán này trong số các tài nguyên công khai.
3. Tài liệu TorchIO về các phép transform — 1 tiếng. Bộ từ vựng augmentation 3D.
4. Về hiệu chuẩn: đọc [calibrate.py](../src/datpd/calibrate.py) cùng các test của nó, rồi tìm
   *"On Calibration of Modern Neural Networks"* (Guo và cộng sự, 2017) — bài báo về temperature
   scaling. Ngắn và áp dụng được trực tiếp.
5. Bất kỳ bài báo/tài liệu nào về nnU-Net — 1 tiếng. Đó là bài toán phân đoạn chứ không phải
   phân loại, nhưng triết lý tiền xử lý của nó (resample về một spacing chung, chuẩn hoá theo
   từng ca, để dữ liệu tự quyết định) chính xác là thứ áp dụng được ở đây.

**Thói quen đáng xây dựng:** mỗi lần bạn gặp một thuật ngữ không biết — *putamen*, *SBR*,
*OSEM*, *SWEDD* — hãy thêm một dòng vào bảng tra cứu trong
[docs/01](01_domain_primer_dat_scans_vn.md) §7. Hai tuần làm vậy sẽ cho bạn một vốn từ chuyên
ngành thực sự dùng được.

---

## Cách chạy các phiên làm việc với Claude Code

Xem [docs/04](04_working_with_claude_code_vn.md) để biết chi tiết. Bản rút gọn:

- **Claude không thể nhìn thấy dữ liệu của bạn. Không bao giờ.** Điều đó được cưỡng chế bằng cơ
  chế kỹ thuật, chứ không chỉ là một lời đề nghị. Hãy làm việc theo lối:
  *Claude viết script → bạn chạy nó → bạn duyệt và chuyển các số liệu tổng hợp lên → Claude đọc
  những thứ đó.*
- **Bắt đầu mỗi phiên bằng cách nói bạn đang ở giai đoạn nào** và dán vài dòng cuối của
  [reports/experiments/log.md](../reports/experiments/log.md). Đó là bối cảnh khiến lời khuyên
  trở nên cụ thể thay vì chung chung.
- **Hãy hỏi lý lẽ, không chỉ hỏi code.** "Vì sao lại dùng GroupNorm ở đây?" là câu hỏi có giá
  trị cao nhất bạn có thể đặt ra trong lúc học, và nó không tốn gì cả.
- **Nhờ Claude rà soát `main.py` của bạn đối chiếu với các quy định về tính độc lập** trước mỗi
  lần nộp bài. Đó là một phép kiểm tra máy móc, đúng kiểu việc rất dễ làm sai lúc 2 giờ sáng
  ngày thứ 15.
