# 04 — Làm việc với Claude Code trong dự án này

Ràng buộc định hình mọi thứ: **Claude không bao giờ được nhìn thấy dữ liệu cuộc thi.** Tài liệu
này là hướng dẫn thực hành để làm việc hiệu quả bên trong ràng buộc đó.

---

## Vì sao ràng buộc này tồn tại

Quy định của DrivenData cấm làm cho Dữ liệu tiếp cận được với bất kỳ bên nào không tham gia cuộc
thi, và nêu đích danh các dịch vụ AI của bên thứ ba. Claude Code là một trong số đó. Dữ liệu đi
vào ngữ cảnh của Claude là một vi phạm quy định, với hệ quả là bị loại — và dữ liệu ở đây là ảnh
y tế của bệnh nhân từ mười bệnh viện Pháp, nên lý lẽ đạo đức trùng khớp với lý lẽ pháp lý.

## Nó được cưỡng chế như thế nào

Ba lớp độc lập, để không một sai sót đơn lẻ nào là đủ để phá vỡ:

| Lớp | File | Chặn được gì |
|---|---|---|
| Quy tắc deny quyền truy cập | `.claude/settings.json` | `Read`/`Grep`/`Glob` vào `data/`, `reports/eda/raw/`, `models/`, và theo phần mở rộng file (`*.nii.gz`, `*.pt`, ...) |
| Hook PreToolUse | `.claude/hooks/guard_data.py` | Các đường vòng qua Bash mà quy tắc theo đường dẫn không thấy được: `cat`, `head`, `strings`, `xxd`, `zcat`, `python -c "print(df.head())"` |
| Quy tắc hành vi | `CLAUDE.md` §0 | Những gì công cụ không thấy được — bạn dán dữ liệu vào cuộc trò chuyện, hay một subagent được nhờ liếc nhìn hộ |
| Git ignore | `.gitignore` | Việc commit dữ liệu vào một repo rồi sẽ trở thành công khai theo giấy phép MIT |

Hook này đã được kiểm thử. `tests/test_guard_hook.py` khẳng định 47 trường hợp — mọi đường vòng
liệt kê ở trên đều nằm trong đó. Chạy `pytest tests/test_guard_hook.py -q` sau khi bạn động vào
hook.

Nếu Claude bị chặn, đó là hệ thống đang hoạt động đúng. Đừng vô hiệu hoá nó, và đừng đi vòng
bằng cách dán nội dung vào cuộc trò chuyện — làm thế là phá vỡ cả ba lớp cùng một lúc.

---

## Quy trình thay thế cho câu "để tôi xem qua dữ liệu của bạn"

```
   ┌──────────┐  viết ra  ┌──────────────────┐
   │  Claude  │──────────▶│ scripts/*.py     │   chỉ có code — không thấy dữ liệu
   └──────────┘           └──────────────────┘
                                   │ bạn chạy nó
                                   ▼
                          ┌──────────────────┐
                          │ reports/eda/raw/ │   VÙNG CÁCH LY — Claude bị chặn
                          └──────────────────┘
                                   │ bạn đọc + thẩm định
                                   ▼
                          ┌────────────────────┐
   ┌──────────┐    đọc    │reports/eda/shared/ │   chỉ số liệu tổng hợp
   │  Claude  │◀──────────└────────────────────┘
   └──────────┘
```

**Bạn chính là cái khoang đệm khí (airlock).** Bước thẩm định là một phán đoán của con người và
không thể tự động hoá — đó là lý do `01_inspect_data.py` ghi bản nháp của nó vào `raw/` và bắt
bạn phải tự tay chuyển nó đi, thay vì ghi thẳng vào `shared/`.

### Cái gì được phép nằm trong `shared/`

✅ Số đếm, cân bằng lớp, tỷ lệ dương tính
✅ Phân phối trên toàn bộ tập dữ liệu — phân vị, min/max, histogram
✅ Các giá trị shape/spacing riêng biệt và tần suất của chúng
✅ Số liệu tổng hợp theo nhóm khi nhóm đủ lớn (ví dụ tỷ lệ dương tính theo từng chữ ký thu nhận ảnh)
✅ Giá trị các chỉ số, đường cong loss, thời gian chạy

❌ Bất kỳ `uid` nào
❌ Bất kỳ dòng, giá trị hay voxel nào của một mẫu đơn lẻ
❌ Ảnh hoặc kết xuất hình ảnh của các bản chụp
❌ Một nhóm nhỏ tới mức một thống kê có thể định danh được một cá nhân

Phép thử cần áp dụng: *liệu thứ này có thể bị đảo ngược để khôi phục lại ảnh chụp hoặc nhãn của
một bệnh nhân cụ thể không?* Nếu có, hoặc nếu bạn không chắc, thì nó ở lại trong `raw/`.

---

## Các khuôn mẫu phiên làm việc hiệu quả

### Bắt đầu một phiên

Hãy cho Claude cái trạng thái mà nó không thể tự khám phá ra:

> Tôi đang ở Giai đoạn 2. OOF log loss của baseline là 0,42 (stratified), 0,51 (grouped),
> AUROC 0,89, baseline tiên nghiệm 0,61. Bảng xếp hạng 0,49. Đây là 5 dòng cuối của nhật ký
> thí nghiệm: [...] Tôi muốn hiểu vì sao grouped CV lại tệ hơn nhiều đến vậy.

Đó là câu hỏi Claude trả lời tốt. Còn "hãy nhìn dữ liệu của tôi và bảo tôi phải làm gì" là câu
hỏi Claude hoàn toàn không thể trả lời.

### Bốn việc Claude thực sự giỏi ở đây

1. **Viết và tái cấu trúc code pipeline.** Tiền xử lý, vòng lặp huấn luyện, đường ống cấu hình,
   đóng gói. Không phụ thuộc dữ liệu, và là phần lớn khối lượng công việc.
2. **Rà soát `main.py` đối chiếu với quy định cuộc thi** trước mỗi lần nộp. Việc máy móc, rủi ro
   cao, và rất dễ sai với một con người đang mệt. Hãy yêu cầu tường minh:
   *"rà soát submission_src/main.py đối chiếu với CLAUDE.md §1.1 và §3."*
3. **Giải thích lĩnh vực chuyên môn và phương pháp.** "Vì sao dùng GroupNorm?", "tỷ số
   putamen/caudate nghĩa là gì và tôi tính nó thế nào?", "vì sao AUROC của tôi tăng mà log loss
   lại tệ đi?" — hãy hỏi những câu này liên tục. Đây là nửa phần học tập của dự án và nó không
   tốn gì cả.
4. **Diễn giải kết quả tổng hợp.** Cho các bản tóm tắt trong `shared/` và các nhật ký chỉ số,
   Claude có thể suy luận về hiệu ứng trung tâm, về hiệu chuẩn, và về việc nên thử gì tiếp theo.

### Ba việc tuyệt đối đừng bao giờ yêu cầu

1. "Đọc file CSV nhãn trong `data/raw/` và cho tôi biết cân bằng lớp." → thay vào đó hãy chạy
   `scripts/01_inspect_data.py`.
2. "Đây là traceback của tôi" — kèm theo một mảng hoặc một dòng dataframe đã được in ra.
   **Hãy che giá trị trước khi dán.** Shape và dtype thì được; voxel và nhãn thì không.
3. "Chỉ cần kiểm tra giúp tôi một ảnh thôi mà." Quy tắc này không có ngưỡng. Một dòng cũng là
   một vi phạm.

### Gỡ lỗi mà không để lộ dữ liệu

Hãy thêm assertion và log chỉ về cấu trúc, thay vì dùng print:

```python
# an toàn: chỉ có cấu trúc
logger.info(f"batch={x.shape} dtype={x.dtype} device={x.device}")
assert x.shape[1:] == (1, 96, 96, 96), f"shape không như mong đợi {x.shape}"
assert torch.isfinite(x).all(), "có giá trị không hữu hạn sau khi tiền xử lý"

# KHÔNG an toàn: giá trị
logger.info(f"giá trị của batch: {x[0, 0, 48]}")   # <- không bao giờ
print(df.head())                                   # <- không bao giờ
```

Rồi hãy dán lỗi assertion đó. Dòng `AssertionError: shape không như mong đợi torch.Size([8, 1,
96, 96, 112])` cho Claude mọi thứ nó cần, và không tiết lộ gì về bệnh nhân.

---

## Các prompt hữu ích, dán là dùng được

```
Rà soát submission_src/main.py đối chiếu với CLAUDE.md §1.1 (tính độc lập giữa từng mẫu)
và §3 (thư viện của runtime). Liệt kê mọi thứ sẽ thất bại trên nền tảng của ban tổ chức.
```

```
Đây là reports/eda/shared/dataset_profile.md. Với phân phối chữ ký thu nhận ảnh như vậy,
grouped CV có khả thi không, hay có nhóm nào quá nhỏ? Mỗi fold tôi sẽ có n bằng bao nhiêu?
```

```
OOF của tôi: logloss 0,44, AUROC 0,91, ECE 0,08. Baseline tiên nghiệm 0,61. Giải thích
ECE đó hàm ý điều gì và tôi nên làm gì trước khi thử một model lớn hơn.
```

```
Viết một script đo <X> về tập dữ liệu và chỉ ghi các số liệu tổng hợp vào
reports/eda/raw/. Tôi sẽ chạy nó và chuyển bản tóm tắt lên.
```

```
Dạy tôi kiến thức nền chuyên ngành đằng sau <thuật ngữ tôi gặp hôm nay>, và cho tôi biết
liệu nó có nên làm thay đổi điều gì trong khâu tiền xử lý của tôi không.
```

---

## Giữ cho các hàng rào bảo vệ luôn trung thực

- Sau khi sửa `.claude/hooks/guard_data.py`, hãy chạy `pytest tests/test_guard_hook.py -q`.
- Nếu bạn thêm một vị trí chứa dữ liệu mới, hãy thêm nó vào **cả bốn** lớp: `GUARDED_DIRS` trong
  hook, `permissions.deny` trong settings, `.gitignore`, và CLAUDE.md §0.
- Nếu Claude bao giờ tạo ra output có vẻ chứa giá trị dữ liệu, hãy dừng lại và kiểm tra xem lớp
  nào đã thất bại. Thêm một test case cho tình huống đó.
- `CLAUDE.md` và `.claude/` được giữ ở máy cục bộ và không được công bố trong repo công khai.
  Tài liệu này mô tả nội dung của chúng.
