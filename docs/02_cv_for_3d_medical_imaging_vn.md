# 02 — Từ MNIST tới ảnh y tế 3D

Bạn đã từng huấn luyện một CNN trên MNIST, với `ReduceLROnPlateau` và vòng lặp Keras/PyTorch
quen thuộc. Kiến thức đó chuyển giao được xa hơn bạn tưởng. Tài liệu này nói về những phần
**không** chuyển giao được, viết dưới dạng một tập hợp các điểm khác biệt (delta) thay vì một
bài hướng dẫn lại từ đầu.

---

## Bảy điểm khác biệt

| | MNIST | Cuộc thi này |
|---|---|---|
| 1. Đầu vào | 28x28x1, đồng nhất | ~142x142x112 **hoặc** 128x128x128, tỷ lệ **vật lý** thay đổi |
| 2. Phép toán | `Conv2d` | `Conv3d` — tính toán và bộ nhớ tăng theo luỹ thừa ba |
| 3. Lớp chuẩn hoá | `BatchNorm` | `GroupNorm` (batch quá nhỏ, và BN có nguy cơ vi phạm quy định) |
| 4. Lượng dữ liệu | 60.000 mẫu sạch | Vài nghìn, nhiễu, đắt đỏ |
| 5. Chất lượng nhãn | Rõ ràng tuyệt đối | ~1 trong 5 ca thực sự mơ hồ, chuyên gia bất đồng là chuyện có thật |
| 6. Chỉ số đánh giá | Accuracy | **Log loss** — mức độ hiệu chuẩn cũng bị chấm điểm |
| 7. Cách chia dữ liệu | Chia ngẫu nhiên là đủ | **Chia theo nhóm** — mười trung tâm, rò rỉ dữ liệu diễn ra âm thầm |

Hãy đi qua chúng theo thứ tự; mỗi điểm đều là một chỗ mà bản năng từ MNIST sẽ chủ động dẫn bạn
đi sai đường.

---

## Delta 1 — Shape thay đổi *và* tỷ lệ vật lý cũng thay đổi

Ý tưởng thực sự mới: trong ảnh y tế, một voxel có **kích thước tính bằng milimet**, và kích
thước đó khác nhau giữa các lần chụp.

```
Volume A: 142 x 142 x 112 @ 2.46 mm   ->  349 x 349 x 276 mm phần đầu
Volume B: 128 x 128 x 128 @ 3.895 mm  ->  499 x 499 x 499 mm phần đầu
```

Nếu bạn resize cả hai về 96³ bằng một lệnh `zoom` ngây thơ, thể vân sẽ chiếm số voxel khác nhau
ở mỗi ảnh — nên một kernel 3x3x3 sẽ trải trên những lượng giải phẫu khác nhau, và mạng sẽ đốt
năng lực học chỉ để gỡ rối chuyện đó.

**Cách khắc phục, theo đúng thứ tự này:**

1. **Định hướng lại (reorient)** về hệ quy chiếu chuẩn RAS (`nibabel.as_closest_canonical`) để
   các trục mang cùng một ý nghĩa.
2. **Resample** về một spacing cố định, ví dụ 2,0 mm đẳng hướng. Giờ thì một voxel = một kích
   thước vật lý duy nhất.
3. **Cắt hoặc đệm (crop/pad)** về một shape mảng cố định, ví dụ 96³. Giờ thì chỉ còn một shape
   tensor duy nhất.

Bước 2 và bước 3 là hai phép toán khác nhau, và gộp chúng làm một là lỗi kinh điển của người mới.
Được cài đặt trong [preprocess.py](../src/datpd/data/preprocess.py); test
`test_full_pipeline_yields_fixed_shape` khẳng định rằng các đầu vào không đồng nhất cho ra một
shape đầu ra duy nhất.

**Hãy cache kết quả.** Resample hàng nghìn khối ảnh ở mỗi epoch sẽ chiếm phần lớn thời gian chạy
của bạn. Tiền xử lý một lần vào `data/processed/cache/`, sau đó quá trình huấn luyện chỉ đọc các
mảng có kích thước cố định.

---

## Delta 2 — Tích chập 3D và bức tường bộ nhớ

Một `Conv3d` với kernel bằng 3 có 27 trọng số cho mỗi cặp kênh, so với 9 của `Conv2d`. Tệ hơn,
các tensor kích hoạt tăng theo luỹ thừa ba:

```
96 x 96 x 96 x 32 kênh x 4 byte  =  113 MB   cho MỘT mẫu, MỘT lớp
```

Các hệ quả thực tế:

- **Batch size phải nhỏ** (4–16), chứ không phải 128.
- **Giảm kích thước sớm.** Phần stem trong [resnet3d.py](../src/datpd/models/resnet3d.py) dùng
  `stride=2` ngay lập tức; giảm một nửa mỗi chiều sẽ cắt bộ nhớ kích hoạt đi 8 lần.
- **Số kênh giữ ở mức khiêm tốn** — (16, 32, 64, 128), không phải (64, 128, 256, 512).
- **Dùng AMP** (`torch.autocast` + `GradScaler`). Xấp xỉ giảm một nửa bộ nhớ, nhanh hơn ~1,5–2
  lần trên A100.
- **Gradient checkpointing** nếu bạn cần một model lớn hơn mức bộ nhớ cho phép.

### Lựa chọn thay thế: cách tiếp cận 2.5D

Thay vì một CNN 3D, hãy lấy các lát cắt 2D (mặt phẳng ngang, hoặc vài lát quanh thể vân), đưa
chúng vào một **backbone 2D đã pretrain** (ResNet/EfficientNet qua `timm` — có sẵn trong
runtime), rồi tổng hợp lại theo từng ảnh (mean, max, hoặc attention pooling — đây chính là
multiple-instance learning).

Vì sao nên cân nhắc:
- **Trọng số pretrain trên ImageNet có sẵn ở 2D và phần lớn không có ở 3D.** Với chỉ vài nghìn
  mẫu huấn luyện, việc chuyển giao đó có thể còn quan trọng hơn cả chiều không gian thứ ba.
- Rẻ hơn nhiều, nên bạn lặp thí nghiệm nhanh hơn.

Vì sao không nên:
- Vứt bỏ ngữ cảnh xuyên qua mặt phẳng lát cắt, vốn là thứ mã hoá gradient sau-ra-trước
  (docs/01 §4).

Với thời hạn hiện tại, một kế hoạch hợp lý là: CNN 3D làm hướng chính, 2.5D làm thành viên tạo
đa dạng cho ensemble cuối cùng. Ensemble các model khác nhau về kiến trúc luôn giúp cải thiện
log loss một cách đáng tin cậy.

---

## Delta 3 — GroupNorm, không phải BatchNorm

BatchNorm chuẩn hoá bằng thống kê tính **trên toàn batch**. Ở đây có hai vấn đề:

1. **Về mặt thống kê.** Ở batch size 4–8, các thống kê đó rất nhiễu, và các giá trị running mean
   dùng lúc suy luận là ước lượng tồi. Đây là một chế độ hỏng nổi tiếng trong ảnh y tế 3D.
2. **Về mặt quy định.** Nếu một lớp BatchNorm còn ở chế độ `train()` lúc suy luận, thì dự đoán
   của một mẫu sẽ phụ thuộc vào các mẫu khác trong cùng batch. Điều đó vi phạm trực tiếp quy
   định về tính độc lập giữa từng mẫu của cuộc thi
   ([00_competition_brief_vn](00_competition_brief_vn.md)).

**GroupNorm** chuẩn hoá trong từng nhóm kênh của một mẫu đơn lẻ. Nó độc lập với batch một cách
có cấu trúc, điều này khiến tính độc lập trở thành đúng về mặt cấu trúc chứ không chỉ là một lời
hứa mà bạn phải nhớ giữ. Test `test_eval_mode_is_batch_size_independent` trong bộ test kiểm
chứng điều đó.

---

## Delta 4 — Dữ liệu ít

Bạn không thể dùng sức mạnh thô để cày vài nghìn khối ảnh 3D theo cách bạn cày 60.000 chữ số
MNIST.

**Augmentation tôn trọng giải phẫu** (xem [AugConfig](../src/datpd/config.py)):

| Phép augmentation | Kết luận |
|---|---|
| Xoay nhỏ (±10°) | Nên — tư thế đặt đầu thực sự có thay đổi |
| Tịnh tiến (±5 mm) | Nên — việc căn giữa có thay đổi |
| Phóng to/thu nhỏ (0,9–1,1) | Nên — kích thước đầu có thay đổi |
| Nhiễu cường độ / nhiễu Gauss | Nên — mô phỏng biến thiên của thống kê đếm photon |
| **Lật trái–phải** | **Mặc định tắt.** Bất đối xứng có giá trị chẩn đoán (docs/01 §4). Hãy kiểm chứng nó như một thí nghiệm. |
| Biến dạng đàn hồi | Thận trọng — có thể làm méo chính cái hình dạng mà bạn đang phân loại |
| Cắt xén mạnh tay | Không — bạn có thể cắt mất chính cấu trúc cần quan tâm |
| MixUp | Đáng thử một thí nghiệm; nó cũng có xu hướng cải thiện mức độ hiệu chuẩn |

`torchio` có trong runtime và được thiết kế riêng cho augmentation ảnh y tế 3D.
`monai.transforms` cũng vậy. Hãy ưu tiên chúng hơn là tự viết tay.

**Điều chuẩn hoá (regularisation):** dropout trước lớp head, weight decay, early stopping dựa
trên **log loss** trên tập validation (không phải accuracy, và cũng không phải validation loss
nếu bạn dùng một hàm loss huấn luyện khác).

---

## Delta 5 — Nhãn mơ hồ

Nhãn MNIST là chắc chắn. Ở đây, 11 chuyên gia gán nhãn thủ công và khoảng 20% số ca là ca khó.
Một số nhãn thực chất là tung đồng xu.

- **Đừng đuổi theo 100% accuracy trên tập huấn luyện.** Bạn sẽ chỉ đang học thuộc nhiễu nhãn.
- **Label smoothing** (nhãn đích 0,05/0,95 thay vì 0/1) rất phù hợp một cách tự nhiên và giúp
  cải thiện hiệu chuẩn một cách trực tiếp.
- **Đừng bao giờ coi một mẫu huấn luyện bị phân loại sai với độ tự tin cao là một lỗi cần sửa.**
  Nó có thể đơn giản chỉ là một ca khó.

---

## Delta 6 — Log loss thay đổi toàn bộ vòng lặp của bạn

Đây là chuyển dịch thực tiễn lớn nhất. Cụ thể:

**Huấn luyện bằng `BCEWithLogitsLoss`**, không phải sigmoid rồi mới BCE. Nó gộp các phép toán
lại và ổn định về mặt số học — model xuất ra một logit, đó là lý do
[resnet3d.py](../src/datpd/models/resnet3d.py) trả về logit.

**Early-stop dựa trên log loss của tập validation.** Không phải accuracy, không phải AUROC.

**Luôn luôn hiệu chuẩn.** Quy trình chuẩn:

```
huấn luyện K fold  ->  thu thập dự đoán out-of-fold  ->  fit temperature trên OOF
                   ->  nướng hằng số đó vào main.py  ->  nộp bài
```

`TemperatureScaler` trong [calibrate.py](../src/datpd/calibrate.py) chỉ có một tham số, chạy mất
vài giây, và thường mang lại một cải thiện log loss thật sự mà gần như miễn phí. Hãy fit nó
**chỉ trên dự đoán OOF** — fit trên dự đoán của tập huấn luyện sẽ cho ra temperature xấp xỉ 1 và
chẳng có tác dụng gì.

**Hãy clip đầu ra.** Một dự đoán 1e-6 nằm sai phía sẽ tốn ~13,8. Tinh chỉnh ngưỡng clip trên OOF
(`DEFAULT_CLIP` trong [metrics.py](../src/datpd/utils/metrics.py)).

**Báo cáo cả hai chỉ số, mọi lần.** AUROC tăng + log loss tăng = một vấn đề về hiệu chuẩn, không
phải một thất bại về mô hình hoá. Hãy hiệu chuẩn lại trước khi kết luận bất cứ điều gì.

### Dịch các thói quen MNIST của bạn sang đây

Notebook MNIST của bạn dùng `ReduceLROnPlateau`. Nó vẫn dùng được ở đây — chỉ cần trỏ nó vào
**log loss** trên tập validation. Với các lần chạy khoảng 30 epoch, cosine annealing kèm một
giai đoạn warmup ngắn thường là lựa chọn mặc định mạnh hơn và không cần tinh chỉnh tham số
patience; cả hai đều được hỗ trợ trong [config.py](../src/datpd/config.py). Cứ giữ
`ReduceLROnPlateau` nếu bạn muốn ít thứ phải chỉnh — đó là một lựa chọn hợp lý, không phải một
sai lầm.

---

## Delta 7 — Cross-validation theo nhóm

Chia ngẫu nhiên là ổn với MNIST. Ở đây, mười trung tâm nghĩa là một phép chia ngẫu nhiên có thể
đặt các ảnh từ cùng một máy chụp vào cả hai phía, và model có thể khai thác các nhiễu ảnh do tái
tạo. Kết quả CV trông tuyệt vời, còn bảng xếp hạng thì không.

Trung tâm không được cung cấp, nhưng cặp `(shape, spacing)` gần như là một dấu vân tay của máy
chụp. Hãy tính cả hai cách chia ([splits.py](../src/datpd/data/splits.py)):

- `fold_strat` — phân tầng (stratified). Phương sai thấp hơn, hơi lạc quan.
- `fold_group` — nhóm theo chữ ký thu nhận ảnh. Bi quan, nhưng trung thực hơn.

**Hãy theo dõi cả hai.** Một khoảng cách lớn nghĩa là hiệu ứng trung tâm là có thật; khi đó hãy
tin con số grouped và ưu tiên tính bền vững (augmentation mạnh hơn, chuẩn hoá theo từng ảnh) hơn
là vắt kiệt điểm CV.

---

## Một kiến trúc khởi đầu cụ thể

Hãy bắt đầu với [`ResNet3D`](../src/datpd/models/resnet3d.py):

```
input  (B, 1, 96, 96, 96)
  stem       Conv3d k5 s2 -> GroupNorm -> SiLU        (B, 16, 48, 48, 48)
  stage 1    ConvBlock s2, ConvBlock                  (B, 32, 24, 24, 24)
  stage 2    ConvBlock s2, ConvBlock                  (B, 64, 12, 12, 12)
  stage 3    ConvBlock s2, ConvBlock                  (B, 128, 6, 6, 6)
  pool       AdaptiveAvgPool3d(1)                     (B, 128)
  head       Dropout -> Linear                        (B,) logits
```

Khoảng 1–2 triệu tham số. Nhỏ một cách có chủ đích: với vài nghìn mẫu, một model lớn hơn chủ yếu
chỉ mua thêm overfitting. Hãy làm cho nó chạy thông suốt từ đầu đến cuối, nộp bài, *rồi mới*
phóng to.

**Khi bạn muốn nhiều hơn:** `monai.networks.nets` cung cấp DenseNet121 / ResNet / SEResNet phiên
bản 3D, một số có trọng số pretrain từ các kho dữ liệu kiểu `MedicalNet`. MONAI có trong runtime.
Hãy kiểm tra giấy phép của mọi trọng số pretrain và ghi lại nó trong
[EXTERNAL_ASSETS.md](EXTERNAL_ASSETS.md).

---

## Gỡ lỗi một pipeline 3D (theo đúng thứ tự này)

1. **Overfit 10 mẫu.** Nếu loss không tiến về ~0 thì lỗi nằm ở dữ liệu hoặc vòng lặp, không phải
   ở model. Đây là tín hiệu nhanh nhất bạn có thể có được.
2. **Kiểm tra shape và spacing sau tiền xử lý** — dùng `assert`, đừng dùng `print`. Shape và
   dtype thì an toàn để ghi log; giá trị voxel thì không (CLAUDE.md §0).
3. **Xác minh rằng tiền xử lý lúc huấn luyện và lúc suy luận là hoàn toàn giống nhau.** Đây là
   nguyên nhân số một của tình trạng "CV thì tuyệt vời, bảng xếp hạng thì thảm hoạ".
   `05_package_submission.py` đóng gói chính các module đó lại với nhau đúng để ngăn sự phân ly
   này.
4. **Kiểm tra định hướng.** Nếu một số khối ảnh bị lật, model sẽ nhìn thấy hai quần thể đối
   gương nhau. `as_closest_canonical` xử lý việc này; hãy xác nhận rằng các mã định hướng là
   đồng nhất trong bản tóm tắt EDA của bạn.
5. **So sánh với baseline tiên nghiệm.** Nếu bạn không thắng nổi `baseline_logloss`, hãy dừng
   việc tinh chỉnh lại và quay về bước 1.
