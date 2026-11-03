# Dự đoán thời gian chuyến taxi New York – Đội UNETI18
Olympic Trí tuệ nhân tạo UNETI 2026 – Vòng 2

## Cấu trúc

```
uneti18_taxi/
├── colab_run.ipynb      huấn luyện trên Google Colab
├── train.py             EDA, tiền xử lý, 4 mô hình, chọn mô hình, file nộp
├── predict.py           dự đoán 1 hoặc nhiều file test bằng mô hình đã lưu
├── predictor.py         nạp mô hình (dùng chung cho predict.py và app.py)
├── app.py               ứng dụng Streamlit
├── make_report.py       sinh báo cáo PDF 3 trang
├── src/features.py      tiền xử lý + tạo đặc trưng (dùng chung Train/Val/Test/App)
├── src/metrics.py       RMSLE, kiểm tra file nộp
├── data/                UNETI18_TEST_3000.csv, test/ (các bộ test khác), train.csv
├── team.json            tên và phân công 3 thành viên
├── setup.bat            tạo môi trường ảo + cài thư viện (Windows)
├── run_app.bat          chạy ứng dụng (Windows)
└── requirements.txt
```

Sau khi huấn luyện sẽ có thêm `artifacts/` (mô hình), `artifacts_val/` và
`data/pseudo_tests/` (đánh giá), `outputs/` (biểu đồ, metrics, file nộp, báo cáo).

## A. Huấn luyện trên Google Colab

1. Tải thư mục `uneti18_taxi` lên Google Drive (`MyDrive/uneti18_taxi`).
2. Đặt file Train vào `data/train.csv`, các bộ test khác vào `data/test/`.
3. Mở `colab_run.ipynb` bằng Colab, chạy lần lượt các ô (khoảng 20 phút).
4. Ô cuối tải về `ket_qua.zip` gồm `artifacts/`, `artifacts_val/`, `outputs/`, `data/pseudo_tests/`.

## B. Chạy trên máy cá nhân (Windows, cmd)

Yêu cầu: Python 3.11 hoặc 3.12 (khi cài nhớ tick **Add python.exe to PATH**).

**Bước 1 – Giải nén** `ket_qua.zip` vào thư mục project, sao cho có:
```
uneti18_taxi\artifacts\meta.json
uneti18_taxi\artifacts\encoder.npz
uneti18_taxi\artifacts\lgb_seed42.txt ...
```

**Bước 2 – Tạo môi trường và cài thư viện** (chỉ làm 1 lần). Mở cmd:
```bat
cd /d D:\uneti18_taxi
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
(Hoặc nháy đúp `setup.bat`.)

**Bước 3 – Kiểm tra mô hình chạy được:**
```bat
python predict.py --test data\UNETI18_TEST_3000.csv --out outputs\submission_UNETI18.csv
```
Kết quả đúng: `[OK] UNETI18_TEST_3000.csv  3000 dòng -> outputs\submission_UNETI18.csv`.

**Bước 4 – Chạy ứng dụng:**
```bat
streamlit run app.py
```
Trình duyệt tự mở `http://localhost:8501` (hoặc nháy đúp `run_app.bat`).
Nếu lần đầu Streamlit hỏi `Email:` thì nhấn Enter. Dừng ứng dụng: `Ctrl + C`.

Các lệnh khác (đều chạy sau khi đã `activate`):
```bat
:: dự đoán mọi file trong data\test -> outputs\predictions\
python predict.py --test data\test

:: chấm RMSLE trên các bộ test giả có nhãn
python predict.py --test data\pseudo_tests --artifacts artifacts_val --out-dir outputs\pseudo_eval

:: sinh lại báo cáo PDF (sau khi sửa team.json)
python make_report.py
```

Lần sau chỉ cần:
```bat
cd /d D:\uneti18_taxi
.venv\Scripts\activate
streamlit run app.py
```

## Kịch bản video 2–3 phút

| Thời gian | Nội dung |
|---|---|
| 0:00–0:20 | Mở cmd, `activate`, `streamlit run app.py`; giới thiệu sidebar (mô hình, RMSLE kiểm định). |
| 0:20–1:00 | Trường hợp 1: Times Square → Sân bay JFK, Thứ Sáu 17:00. Bấm Dự đoán, giải thích kết quả (giây, phút), bản đồ, biểu đồ 24 giờ. |
| 1:00–1:40 | Trường hợp 2: Wall Street → Upper East Side, Chủ Nhật 03:00. So sánh với trường hợp 1. |
| 1:40–2:10 | Trường hợp 3: tự nhập tọa độ, đổi số khách, dự đoán; bảng lịch sử. |
| 2:10–2:40 | Tab "Dự đoán theo file CSV": tải `data\UNETI18_TEST_3000.csv`, tải kết quả về. |

## Lỗi thường gặp

| Lỗi | Cách xử lý |
|---|---|
| `'python' is not recognized` | Cài lại Python, tick *Add python.exe to PATH*; hoặc dùng `py` thay cho `python`. |
| `running scripts is disabled` | Đang dùng PowerShell. Chuyển sang cmd, hoặc chạy `.venv\Scripts\activate.bat`. |
| App báo "Chưa có mô hình" | Chưa giải nén `ket_qua.zip` đúng chỗ (xem Bước 1), hoặc cmd chưa `cd` vào thư mục project. |
| `No module named ...` | Chưa `activate` môi trường, hoặc chưa chạy `pip install -r requirements.txt`. |
| Bản đồ không hiện | Bản đồ cần Internet; dự đoán vẫn hoạt động bình thường. |

## Trước khi nộp
- [ ] `outputs/submission_UNETI18.csv` hợp lệ (3000 dòng, 2 cột `trip_id,trip_duration`).
- [ ] Đã sửa `team.json`, chạy `make_report.py`, kiểm tra PDF đủ 3 trang.
- [ ] Video 2–3 phút, ít nhất 2 đầu vào khác nhau.
- [ ] Nén mã nguồn kèm `artifacts/` (không cần file Train).
