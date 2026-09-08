# 01 — Nhập môn lĩnh vực: ảnh DaT scan thực chất là gì

Viết cho người bắt đầu từ con số không về y học hạt nhân. Mục tiêu: sau khoảng 40 phút, bạn nên
hiểu được ảnh cho thấy điều gì, "bất thường" nghĩa là gì, vì sao khoảng 1 trong 5 ca là ca khó,
và trong số những dữ kiện đó thì cái nào nên làm thay đổi model của bạn. Không phần nào ở đây
đòi hỏi kiến thức sinh học từ trước.

---

## 1. Vấn đề lâm sàng

Các hội chứng Parkinson — bệnh Parkinson và những bệnh họ hàng của nó — gây ra run, cứng cơ và
chậm vận động. Rắc rối nằm ở chỗ **nhiều bệnh rất khác nhau lại trông giống nhau khi khám lâm
sàng**, trong khi cách điều trị và tiên lượng thì khác nhau rõ rệt:

| Trông giống Parkinson, VÀ là thoái hoá thần kinh | Trông giống Parkinson, nhưng KHÔNG phải |
|---|---|
| Bệnh Parkinson (PD) | Run vô căn (essential tremor) |
| Teo đa hệ thống (MSA) | Parkinson do thuốc (ví dụ do thuốc chống loạn thần) |
| Liệt trên nhân tiến triển (PSP) | Parkinson do mạch máu |
| Thoái hoá vỏ não – hạch nền (CBD) | Rối loạn vận động chức năng / tâm căn |
| Sa sút trí tuệ thể Lewy (DLB) | |

Chẩn đoán sai đồng nghĩa với nhiều năm dùng sai thuốc, hoặc một tiên lượng sai. Chỉ dựa vào khám
lâm sàng thì một tỷ lệ đáng kể các ca bị phân loại nhầm, đặc biệt ở giai đoạn sớm khi các dấu
hiệu còn kín đáo.

**Đây chính là chỗ ảnh DaT khẳng định giá trị của nó.** Nó trả lời tốt đúng một câu hỏi cụ thể:
*hệ dopamine có đang thực sự thoái hoá hay không?*

---

## 2. Sinh học, gói trong năm câu

1. Một nhóm nơron ở **chất đen (substantia nigra)** thuộc trung não gửi các sợi trục giải phóng
   dopamine lên một cấu trúc gọi là **thể vân (striatum)**.
2. Trên đầu mút của những sợi đó có một protein tên là **chất vận chuyển dopamine (DAT)**, có
   nhiệm vụ hút dopamine trở ngược ra khỏi khe synap.
3. Trong hội chứng Parkinson thoái hoá, các nơron ấy chết đi, nên các sợi trục co rút lại, nên
   **lượng DAT trong thể vân giảm xuống**.
4. Ảnh DaT tiêm vào một chất đánh dấu phóng xạ (tracer) **bám vào DAT**, do đó cường độ tín hiệu
   trong thể vân là một proxy cho việc còn bao nhiêu đầu mút dopamine sống sót.
5. Các bệnh ở cột bên phải của bảng trên không làm tổn thương các đầu mút này, nên ảnh của họ
   trông **bình thường** — và đó chính xác là khả năng phân biệt mà ảnh này mang lại.

Một giới hạn quan trọng, đáng khắc cốt ghi tâm: **ảnh chỉ nói "có thoái hoá / không thoái hoá".
Nó không nói "PD hay MSA hay PSP"** — tất cả những bệnh đó đều cho ảnh bất thường. Nhãn nhị phân
của chúng ta là `is_pathologic`, tương ứng đúng với điều mà ảnh thực sự có thể chứng minh.

---

## 3. Ảnh được tạo ra như thế nào

- **Tracer:** ioflupane gắn iod-123, thường viết là **I-123 FP-CIT**. Tên thương mại
  **DaTscan** chính là nguồn gốc tên của cuộc thi.
- **Chuẩn bị:** bệnh nhân uống kali iodua để ngăn tuyến giáp hấp thu iod tự do.
- **Chờ:** chụp ảnh diễn ra **3–6 tiếng** sau khi tiêm, khi tracer đã gắn kết và phần tracer
  chưa gắn đã được thải trừ.
- **Thu nhận:** **SPECT** — Single Photon Emission Computed Tomography (chụp cắt lớp phát xạ đơn
  photon). Một camera gamma xoay quanh đầu và đếm từng photon một.
- **Tái tạo:** các hình chiếu được chuyển thành khối 3D (tái tạo lặp OSEM, hoặc filtered
  back-projection kiểu cũ), có thể kèm hiệu chỉnh suy giảm và tán xạ.

### Điều này có ý nghĩa gì với bạn với tư cách người xây model

| Thuộc tính | Giá trị | Hệ quả |
|---|---|---|
| Độ phân giải không gian | ~8–12 mm FWHM | Rất nhoè. Kết cấu chi tiết phần lớn là nhiễu; đừng kỳ vọng có cạnh sắc. |
| Thống kê đếm photon | Thấp | Nhiễu Poisson rất đáng kể. Khử nhiễu quá tay sẽ xoá mất tín hiệu thật. |
| Đơn vị voxel | Số đếm thô, uint16 | **Không được hiệu chuẩn giữa các máy chụp.** Cường độ tuyệt đối là vô nghĩa khi so giữa các trung tâm. |
| Tái tạo ảnh | Khác nhau tuỳ trung tâm | Các mức làm mượt và hiệu chỉnh khác nhau tạo ra kết cấu khác nhau cho cùng một giải phẫu. |

Hàng thứ ba là lý do vì sao chuẩn hoá cường độ theo từng ảnh không phải là tuỳ chọn. Nó cũng là
lý do vì sao bác sĩ lâm sàng đọc **tỷ số**, chứ không đọc giá trị tuyệt đối (§5).

Hàng "tái tạo ảnh" là lý do bộ dữ liệu này khó và là lý do cuộc thi này tồn tại: một model rất
dễ học được *bệnh viện nào đã chụp ảnh này* thay vì *bệnh nhân có bệnh hay không*.

---

## 4. Ảnh bình thường và bất thường trông như thế nào

Thể vân có hai phần, và tên của chúng xuất hiện liên tục trong y văn:

- **Nhân đuôi (caudate nucleus)** — phần phía trước.
- **Nhân bèo sẫm (putamen)** — phần phía sau. Còn được chia nhỏ thành putamen trước và putamen
  sau.

Trên ảnh bình thường, mỗi bên bán cầu hiện lên một hình **dấu phẩy** hoặc **lưỡi liềm** sáng và
cân đối: caudate là phần đầu mập của dấu phẩy, putamen là cái đuôi quét về phía sau.

Quá trình thoái hoá diễn ra theo một khuôn mẫu không gian rất định hình:

```
                  PHÍA TRƯỚC (phía trán)
                          |
   BÌNH THƯỜNG            |          BẤT THƯỜNG
                          |
   caudate  ████          |          caudate  ███       <- teo sau cùng, còn được giữ một phần
      ▲     ████          |             ▲     ███
      |     ███           |             |     ██
   putamen  ███           |          putamen  ▒         <- teo TRƯỚC TIÊN, tổn thương nặng nhất
      ▼     ██            |             ▼     ·
                          |
   "dấu phẩy"/"lưỡi liềm" |          "dấu chấm"
                  PHÍA SAU (phía gáy)
```

Ba khuôn mẫu cần thuộc tên, bởi vì chúng chính là tín hiệu thật sự:

1. **Gradient sau-ra-trước.** Putamen sau mất tín hiệu đầu tiên, rồi tới putamen trước, rồi mới
   tới caudate. Dấu phẩy rụng mất cái đuôi và biến thành một chấm.
2. **Tính bất đối xứng.** Tổn thương thường nặng hơn ở một bên — bên đối diện với chi có triệu
   chứng nhiều hơn của bệnh nhân. Mất tín hiệu đối xứng hai bên gợi ý nhiều hơn tới các hội
   chứng không điển hình.
3. **Nền sáng lên tương đối so với thể vân.** Khi phần gắn kết đặc hiệu giảm đi, thể vân nổi bật
   kém hơn so với mô xung quanh, và toàn ảnh trông "phẳng" hơn.

> **Hệ quả cho việc xây model (điểm này quan trọng).**
> Lật trái–phải là một phép augmentation phản xạ trong computer vision ảnh tự nhiên. Ở đây nó có
> thể phá huỷ tín hiệu, bởi vì **tính bất đối xứng mang giá trị chẩn đoán** và bên trái/phải có
> ý nghĩa thật. Nó bị tắt mặc định trong [conf/baseline.yaml](../conf/baseline.yaml). Hãy chạy
> đúng một thí nghiệm có kiểm soát trước khi bật nó lên, và đánh giá bằng OOF log loss.

---

## 5. Bác sĩ lâm sàng thực sự định lượng nó ra sao: chỉ số SBR

Bác sĩ chẩn đoán hình ảnh không đọc số đếm thô. Họ tính một **tỷ số gắn kết đặc hiệu (specific
binding ratio, SBR)**:

```
SBR = (số đếm trung bình ở vùng thể vân − số đếm trung bình ở vùng tham chiếu)
      ────────────────────────────────────────────────────────────────────────
                    số đếm trung bình ở vùng tham chiếu
```

**Vùng tham chiếu** là một phần não về cơ bản không có DAT — thường là vỏ não chẩm, đôi khi là
toàn não hoặc tiểu não. Tín hiệu ở đó là phần gắn kết "không đặc hiệu", tức mức nền của tracer
chỉ đơn giản là đang có mặt chứ không gắn vào cái gì cả.

Việc chia cho vùng tham chiếu triệt tiêu được độ khuếch đại của máy chụp, liều tiêm và kích
thước bệnh nhân. Đó chính xác là lý do `normalize_intensity(method="reference_region")` trong
[preprocess.py](../src/datpd/data/preprocess.py) là lựa chọn mặc định: nó là một xấp xỉ rẻ tiền
của chính đại lượng mà bác sĩ tin dùng, và nó bất biến theo tỷ lệ một cách có cấu trúc (có hẳn
một test khẳng định đúng điều đó).

Những đại lượng liên quan mà bạn sẽ gặp trong các bài báo: **SBR của putamen**, **SBR của
caudate**, **tỷ số putamen/caudate** (nắm bắt cái gradient), và **chỉ số bất đối xứng**
`|trái − phải| / (trái + phải)`. Cả bốn đều là những đặc trưng thủ công mạnh và diễn giải được.
Nếu bạn đăng ký (register) các ảnh về một template chung, việc tính chúng một cách tường minh là
một baseline không-deep-learning hoàn toàn đáng tin — và là một phép kiểm tra tỉnh táo hữu ích
đối với bất cứ thứ gì CNN của bạn học được.

---

## 6. Vì sao khoảng 1 trong 5 ca là ca khó

Ban tổ chức cho biết khoảng một phần năm số ảnh DaT là khó diễn giải. Biết được lý do sẽ cho bạn
biết log loss của bạn sẽ bị tiêu tốn ở đâu:

- **Bệnh giai đoạn sớm.** Tín hiệu mất đi từ từ. Một ảnh chụp ở sát ngưỡng thì thực sự là ranh
  giới, và hai chuyên gia có thể bất đồng.
- **Tuổi tác.** Mật độ DAT giảm tự nhiên theo tuổi, khoảng 5–8% mỗi thập kỷ. Một ảnh giảm nhẹ ở
  người 80 tuổi có thể là bình thường so với tuổi của họ; cũng ảnh đó ở người 45 tuổi thì không.
  Điều này là có thật, và là lý do ban tổ chức đề cập tới việc dự đoán tuổi/giới tính như một
  hướng nghiên cứu tiếp nối.
- **SWEDD** — "Scans Without Evidence of Dopaminergic Deficit" (ảnh không có bằng chứng thiếu
  hụt dopamine): bệnh nhân có chẩn đoán lâm sàng là hội chứng Parkinson nhưng ảnh lại bình
  thường.
- **Khác biệt về máy chụp và cách tái tạo ảnh.** Một bộ lọc tái tạo khác có thể làm dịch chuyển
  độ tương phản biểu kiến đủ nhiều để đẩy một ca ranh giới sang phía bên kia.
- **Nhiễu ảnh do cử động và do suy giảm tín hiệu.**

> **Hệ quả cho việc xây model.** Sự mơ hồ là không thể loại bỏ, và **log loss thưởng cho bạn vì
> đã thừa nhận điều đó**. Một model xuất ra 0,5 cho những ảnh thực sự ở ranh giới và 0,02 / 0,98
> cho những ảnh rõ ràng sẽ thắng một model tự tin ở mọi nơi. Đừng đuổi theo accuracy; hãy đuổi
> theo xác suất trung thực. Đây là chuyển dịch tư duy lớn nhất so với MNIST, nơi nhãn là hoàn
> toàn rõ ràng.

---

## 7. Bảng tra cứu nhanh về giải phẫu

| Thuật ngữ | Nghĩa nôm na |
|---|---|
| Striatum (thể vân) | Cấu trúc đích. Caudate + putamen. Nơi tracer tập trung. |
| Caudate nucleus (nhân đuôi) | Phần trước của thể vân. Phần đầu của "dấu phẩy". Được giữ lại lâu nhất. |
| Putamen (nhân bèo sẫm) | Phần sau. Cái đuôi của "dấu phẩy". Thoái hoá đầu tiên, đầu sau nặng nhất. |
| Substantia nigra (chất đen) | Nguồn gốc các nơron dopamine ở trung não. Quá nhỏ để thấy trên SPECT. |
| Vỏ não chẩm (occipital cortex) | Phía sau não. Dùng làm vùng tham chiếu không có DAT. |
| Cùng bên / đối bên | Ipsilateral / contralateral. Tổn thương não nằm đối bên với triệu chứng. |
| Trước / sau | Anterior / posterior. |
| Trên / dưới | Superior / inferior. |
| Trái / phải | Theo quy ước ảnh X-quang, ảnh có thể bị đảo gương trái–phải — hãy kiểm tra mã định hướng, đừng phỏng đoán. |
| SBR | Tỷ số gắn kết đặc hiệu (§5). |
| SPECT | Phương thức chụp ảnh. Đếm photon, nhoè, nhiễu. |
| I-123 FP-CIT / ioflupane | Chất đánh dấu phóng xạ. |
| SWEDD | Lâm sàng có hội chứng Parkinson nhưng ảnh bình thường. |
| RAS / LPS | Mã định hướng trục. `nibabel.as_closest_canonical` chuẩn hoá về RAS. |

---

## 8. Năm điều cần mang theo khi xây model

1. **Resample về đơn vị milimet.** Tỷ lệ vật lý phải nhất quán, nếu không phép tích chập sẽ
   không mạch lạc giữa các trung tâm. ([preprocess.py](../src/datpd/data/preprocess.py))
2. **Chuẩn hoá theo từng ảnh, theo kiểu tỷ số.** Số đếm tuyệt đối phụ thuộc máy chụp và vô nghĩa
   khi so giữa các cơ sở; tỷ số mới là thứ mà cả sinh học lẫn bác sĩ đều dùng.
3. **Đừng lật trái–phải theo mặc định.** Bất đối xứng là tín hiệu, không phải phiền nhiễu.
4. **Trung tâm bệnh viện là một yếu tố gây nhiễu.** Mười bệnh viện, máy chụp khác nhau. Hãy dùng
   grouped CV ([splits.py](../src/datpd/data/splits.py)), nếu không bạn sẽ tự lừa chính mình.
5. **Hãy hiệu chuẩn.** Một trong năm ca là thực sự mơ hồ, và log loss trả công cho bạn vì đã nói
   ra điều đó. ([calibrate.py](../src/datpd/calibrate.py))

---

## 9. Đọc thêm ở đâu

Xếp gần đúng theo mức hữu ích trên mỗi phút bỏ ra. Không tài liệu nào là bắt buộc phải đọc trước
khi bắt đầu viết code.

- **Radiopaedia — "DaT scan"** và **"Dopamine transporter imaging"**: ngắn, nhiều hình, lý tưởng
  để xây trực giác thị giác về quá trình từ dấu phẩy thành chấm. Bắt đầu từ đây.
- **Hướng dẫn thực hành EANM/SNMMI về chụp hình chất vận chuyển dopamine** (tìm: *EANM DaT SPECT
  guideline*): cách ảnh được thu nhận và đọc, bao gồm cả quy ước về vùng tham chiếu.
- **PPMI (Parkinson's Progression Markers Initiative)**, <https://www.ppmi-info.org>: bộ dữ liệu
  DaT công khai lớn. Các bài báo và quy ước DICOM của nó là nền y văn cho hầu hết công trình
  deep learning về DaT đã công bố, và việc đọc chúng là đáng giá.
  **Nhưng bản thân dữ liệu thì không được dùng trong cuộc thi này** — ban tổ chức đã ra phán
  quyết rằng Thoả thuận Sử dụng Dữ liệu (DUA) của PPMI không bao trùm việc dùng cho thi đấu, nên
  các model huấn luyện trên nó không đủ điều kiện nhận giải. Điều này áp dụng cả cho trọng số
  pretrain có nguồn gốc từ PPMI. Xem [EXTERNAL_ASSETS.md](EXTERNAL_ASSETS.md) để biết phán quyết
  và lý lẽ của nó. Hãy đọc y văn; đừng tải ảnh về.
- Tìm **"deep learning DaT SPECT classification"** trên PubMed để xem các công trình đi trước.
  Hãy chuẩn bị tinh thần gặp những nghiên cứu nhỏ, đơn trung tâm, báo cáo accuracy rất cao — hãy
  coi các con số đó là lạc quan; toàn bộ điểm mấu chốt của cuộc thi này là khả năng tổng quát
  hoá đa trung tâm khó hơn nhiều.
