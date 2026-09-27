# 00 — Tóm tắt cuộc thi

Toàn bộ những điều mang tính ràng buộc của DaT Parkinson's Challenge, gom về một chỗ. Đọc tài
liệu này trước khi viết bất kỳ dòng code nào. Các trang nguồn được liệt kê ở cuối.

---

## Yêu cầu, tóm trong một câu

Cho một ảnh chụp não DaT SPECT 3D, hãy xuất ra **xác suất** rằng ảnh đó là bệnh lý (bất thường).

## Các thông tin cốt lõi

| | |
|---|---|
| Nền tảng tổ chức | DrivenData, cuộc thi #311 |
| Đơn vị chủ trì | Hội Y học Hạt nhân Pháp (SFMN), Health Data Hub, GaelO |
| Nguồn tài trợ | Kế hoạch France 2030 |
| Tổng giải thưởng | 25.000 € — 12.500 € / 7.500 € / 5.000 € (chi trả bằng USD theo tỷ giá ngày 2026-07-22) |
| **Hạn chót** | **2026-09-16, 23:59 UTC** |
| Số người tham gia | ~801 tại thời điểm viết tài liệu |
| Cột nhãn mục tiêu | `is_pathologic` — `0.0` bình thường, `1.0` bất thường |
| **Chỉ số đánh giá** | **Log loss.** Càng thấp càng tốt. |
| Chỉ số phụ | AUROC, chỉ hiển thị để tham khảo, **không ảnh hưởng tới xếp hạng** |
| Hình thức nộp bài | Code execution (bạn nộp code, không nộp dự đoán) |

Người thắng cuộc được mời tới một buổi lễ tại Pháp, và có thể được mời tham gia một hợp tác
tiếp nối (không bắt buộc) về việc dự đoán **tuổi và giới tính** từ ảnh DaT — một gợi ý rằng
những tín hiệu đó có tồn tại trong ảnh, điều đáng ghi nhớ khi bạn suy nghĩ về các yếu tố gây
nhiễu (confounder).

---

## Dữ liệu

- **Định dạng:** NIfTI nén, đuôi `.nii.gz`, mỗi file là một khối 3D đã tái tạo.
- **Voxel:** số nguyên không dấu 16-bit (số đếm photon SPECT thô, chưa hiệu chuẩn giữa các máy).
- **Tên file:** `<uid>` + đuôi NIfTI, trong đó `uid` là khoá để nối với bảng nhãn.
- **Nhãn:** file CSV nhãn huấn luyện, gồm các cột `uid` và `is_pathologic`.
- **Nguồn gốc:** mười trung tâm bệnh viện tại Pháp. Đây là một trong những bộ dữ liệu DaT đa
  trung tâm lớn nhất từng được tập hợp cho machine learning. Được gán nhãn bởi **11 chuyên gia
  y học hạt nhân**.

### Đặc tính định hình bài toán: tính không đồng nhất (heterogeneity)

Các khối ảnh khác nhau **cả về** kích thước mảng **lẫn** kích thước vật lý của voxel:

| Ví dụ shape | Ví dụ spacing |
|---|---|
| 142 x 142 x 112 | 2,46 mm đẳng hướng |
| 128 x 128 x 128 | 3,895 mm đẳng hướng |

Đây không phải nhiễu để bỏ qua — đây chính là bài toán kỹ thuật trung tâm. Một khối rộng 128
voxel ở 3,895 mm bao phủ ~499 mm; một khối rộng 142 voxel ở 2,46 mm bao phủ ~349 mm. Nếu đưa cả
hai vào CNN mà không resample, thì cùng một kernel tích chập đang nhìn vào giải phẫu ở những tỷ
lệ vật lý khác nhau, và mạng sẽ lãng phí năng lực học chỉ để bù trừ sự lệch pha đó.

**Hãy resample mọi thứ về một lưới milimet chung.** Xem
[src/datpd/data/preprocess.py](../src/datpd/data/preprocess.py).

Shape và spacing còn đóng vai trò như một **dấu vân tay của máy chụp** — một proxy dùng được cho
trung tâm bệnh viện, vốn không được cung cấp trực tiếp. Hãy dùng nó cho cross-validation theo
nhóm ([src/datpd/data/splits.py](../src/datpd/data/splits.py)).

---

## Chỉ số đánh giá: log loss

```
L(y, p) = -( y·log(p) + (1-y)·log(1-p) )
```

lấy trung bình trên toàn bộ mẫu. Ba hệ quả cần định hình mọi quyết định của bạn:

1. **Bạn bị chấm điểm trên mức độ hiệu chuẩn (calibration), không chỉ trên thứ tự xếp hạng.**
   Một model có AUROC xuất sắc nhưng xác suất quá tự tin có thể thua một model xếp hạng kém hơn
   nhưng được hiệu chuẩn tốt. Hãy fit một bộ hiệu chuẩn trên các dự đoán out-of-fold (OOF) — xem
   [src/datpd/calibrate.py](../src/datpd/calibrate.py).
2. **Sai lầm với độ tự tin cao gây thảm hoạ, và không bị chặn trên.** Một dự đoán `p=1e-6` trên
   một ca dương tính đóng góp ~13,8 vào tổng, so với ~0,1 cho một dự đoán tử tế. Hãy clip đầu ra
   của bạn. Tinh chỉnh ngưỡng clip trên dữ liệu OOF.
3. **Biết sàn của mình ở đâu.** Dự đoán tỷ lệ lớp tiên nghiệm (class prior) cho mọi mẫu sẽ cho
   ra một giá trị log loss cụ thể. Hãy tính nó trước tiên
   (`datpd.utils.metrics.baseline_logloss`) và ghi lại như thí nghiệm số không. Bất kỳ thứ gì
   không vượt được nó thì chưa phải là một model.

---

## Nộp bài: đây là cuộc thi dạng code-execution

Bạn **không** upload dự đoán. Bạn upload code, và DrivenData sẽ chạy nó trên một tập test mà bạn
không bao giờ được nhìn thấy.

### Cấu trúc file zip

```
submission.zip
├── main.py            (bắt buộc, đặt ở gốc)
├── my_module/...      (mọi code hỗ trợ)
└── assets/...         (trọng số model — bắt buộc kèm theo, vì không có mạng)
```

### Môi trường thực thi

| | |
|---|---|
| Python | **chỉ 3.12** |
| Phần cứng | 1x NVIDIA A100 80GB, 24 vCPU, 220 GB RAM |
| Giới hạn thời gian | **3 tiếng** cho lần chạy đầy đủ, **6 phút** cho smoke test |
| Mạng | **Không có** |
| Logging | ≤ 300 dòng, ≤ 300 ký tự/dòng |

### Hệ thống file lúc chạy

```
/code_execution/
├── data/                     (chỉ đọc)
│   ├── niftis/<uid>.nii.gz
│   └── submission_format.csv
└── src/                      (file bạn đã giải nén; thư mục làm việc)
```

`main.py` phải ghi ra `submission.csv` trong thư mục làm việc:

```
uid,is_pathologic
xaji0y6d,0.04
pbhsahxt,0.12
```

### Thư viện phụ thuộc

Chỉ được dùng những gì đã có sẵn trong image. Những cái đáng chú ý: `torch==2.12.1+cu129`,
`torchvision`, `torchio==1.2.1`, `monai`, `nibabel`, `simpleitk`, `antspyx`,
`pytorch-lightning`, `timm`, `transformers`, `scikit-learn`, `scipy`, `numpy`, `pandas`,
`polars`, `xgboost`, `lightgbm`, `opencv-python-headless`, `einops`, `matplotlib`, `loguru`,
`tensorflow`. Danh sách đầy đủ nằm trong `CLAUDE.md §3`. Một import bị thiếu đồng
nghĩa với một bài nộp thất bại.

Thư viện mới đòi hỏi phải gửi PR vào repo runtime — với thời hạn như hiện tại, hãy coi như đó
không phải một lựa chọn.

### Kiểm thử tại máy

Clone <https://github.com/drivendataorg/competition-sfmn-parkinsons-runtime>, rồi chạy:

```bash
just pull              # tải image chính thức (cần ~21 GB trống)
just pack-submission   # nén submission_src/
just test-submission   # chạy trong container, đúng như nền tảng sẽ chạy
```

Sau đó hãy nộp một **smoke test** (6 phút, mẫu nhỏ) trước lần chạy đầy đủ. Làm việc này sớm; bài
nộp đầu tiên của bất kỳ cuộc thi nào cũng thường hỏng vì một lý do cấu trúc, chứ không phải vì
model.

---

## Các quy định ràng buộc lời giải của bạn

Đây không phải hướng dẫn tuỳ nghi. Vi phạm sẽ làm bài nộp mất hiệu lực.

- **Tính độc lập giữa từng mẫu.** "Xử lý từng mẫu dữ liệu test một cách độc lập trong quá trình
  suy luận, không sử dụng thông tin từ các ca khác trong tập test." Điều này cấm: chuẩn hoá bằng
  thống kê tính trên tập test, pseudo-labelling trên tập test, adaptive BatchNorm, mọi hình thức
  TTA hay ensemble có gộp thông tin giữa các mẫu. Hãy dùng chuẩn hoá theo từng mẫu và các lớp
  norm không phụ thuộc batch (GroupNorm), và luôn luôn gọi `model.eval()`.
- **Không huấn luyện lại lúc suy luận.** Trọng số phải nằm sẵn trong file zip.
- **Không gán nhãn thủ công cho dữ liệu test.**
- **Dữ liệu ngoài và trọng số pretrained được phép** nếu bạn nắm giữ đầy đủ quyền và giấy phép
  mà không cần xin phê duyệt từ bên thứ ba, và có thể chứng minh khi được yêu cầu. Chúng phải
  được đóng gói sẵn vào file zip. Hãy ghi lại nguồn gốc trong
  [EXTERNAL_ASSETS.md](EXTERNAL_ASSETS.md).
  **PPMI bị loại trừ một cách cụ thể** — Thoả thuận Sử dụng Dữ liệu của nó không bao trùm việc
  dùng cho thi đấu, nên dữ liệu PPMI và trọng số có nguồn gốc từ PPMI sẽ khiến lời giải không đủ
  điều kiện nhận giải. Một DUA đã kiểm soát quyền truy cập thì thường cũng hạn chế mục đích sử
  dụng; "công khai" không đồng nghĩa với "được cấp phép cho việc này".
- **Không chia sẻ code riêng tư** ra ngoài đội của bạn. Chia sẻ công khai thì được phép, và mặc
  nhiên được cấp phép theo giấy phép MIT.
- **Một tài khoản cho mỗi người.** Nhiều tài khoản đồng nghĩa với bị loại.
- **Người thắng cuộc phải mã nguồn mở theo giấy phép MIT**, kèm tài liệu theo mẫu của DrivenData
  và mô tả các tài nguyên cần thiết để build và chạy lời giải. Hãy giữ repo ở trạng thái tái lập
  được ngay từ đầu; chắp vá việc này vào phút chót là một cực hình.

### Xử lý dữ liệu (vì sao có CLAUDE.md §0)

Quy định cho phép "đọc và học từ Dữ liệu, phân tích Dữ liệu, chỉnh sửa Dữ liệu và nói chung là
chuẩn bị Bài nộp của bạn", đồng thời cấm truyền tải, sao chép, công bố, phân phối lại hoặc bằng
cách khác làm cho Dữ liệu tiếp cận được với bất kỳ bên nào không tham gia cuộc thi — **trong đó
nêu đích danh các dịch vụ AI của bên thứ ba**. Bạn cũng không được có bất kỳ nỗ lực nào nhằm
định danh cá nhân người bệnh.

Claude Code là một dịch vụ AI của bên thứ ba. Đó chính là toàn bộ lý do tồn tại của các hàng rào
bảo vệ trong `CLAUDE.md` §0, các quy tắc deny trong `.claude/settings.json`, và
hook trong `.claude/hooks/guard_data.py`. Claude viết code; bạn chạy code; Claude chỉ đọc những
số liệu tổng hợp mà bạn đã duyệt và chuyển lên.

---

## Nguồn tham khảo

- [Tổng quan](https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/page/987/)
- [Về dữ liệu](https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/page/988/)
- [Mô tả bài toán](https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/page/990/)
- [Định dạng nộp code](https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/page/989/)
- [Quy định chính thức](https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/rules/)
- [Repo runtime](https://github.com/drivendataorg/competition-sfmn-parkinsons-runtime)
