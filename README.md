# Project 2 - Machine Learning Model Development

## 📋 Mô Tả Dự Án

Project 2 là một dự án phát triển mô hình học máy (Machine Learning) nhằm xây dựng, huấn luyện và triển khai các mô hình dự đoán hiệu suất cao. Dự án này cung cấp một nền tảng hoàn chỉnh cho quá trình xử lý dữ liệu, xây dựng mô hình, đánh giá hiệu suất và lưu trữ các mô hình đã huấn luyện.

## 🎯 Mục Đích

- **Phát triển mô hình**: Xây dựng các mô hình học máy từ dữ liệu thô
- **Tối ưu hóa hiệu suất**: Tuning siêu tham số và cải thiện độ chính xác
- **Tái sử dụng**: Lưu trữ các mô hình đã huấn luyện để sử dụng sau này
- **Tự động hóa**: Tạo các script tự động để huấn luyện và đánh giá

## 📁 Cấu Trúc Dự Án

```
Project2/
├── README.md                 # Tài liệu dự án
├── best_model.pth           # Mô hình tốt nhất đã huấn luyện
├── .venv/                   # Môi trường ảo Python
├── data/                    # Thư mục chứa dữ liệu
│   ├── raw/                 # Dữ liệu thô chưa xử lý
│   ├── processed/           # Dữ liệu đã xử lý
│   └── splits/              # Dữ liệu chia thành train/test
├── scripts/                 # Các script Python
│   ├── data_preprocessing.py    # Xử lý và chuẩn bị dữ liệu
│   ├── model_training.py        # Huấn luyện mô hình
│   ├── model_evaluation.py      # Đánh giá hiệu suất
│   ├── inference.py             # Sử dụng mô hình cho dự đoán
│   └── utils.py                 # Các hàm tiện ích
├── requirements.txt         # Các thư viện Python cần thiết
└── .gitignore              # Tệp git ignore
```

## 🚀 Các Tính Năng Chính

### 1. **Xử Lý Dữ Liệu**

- Làm sạch và chuẩn hóa dữ liệu thô
- Xử lý các giá trị bị thiếu
- Tách biệt tập huấn luyện, xác thực và kiểm thử
- Chuẩn hóa đặc trưng (feature scaling)

### 2. **Xây Dựng & Huấn Luyện Mô Hình**

- Hỗ trợ nhiều thuật toán học máy khác nhau
- Tuning siêu tham số tự động
- Cross-validation để đánh giá chéo
- Lưu trữ mô hình tốt nhất

### 3. **Đánh Giá & Phân Tích**

- Tính toán các chỉ số hiệu suất (accuracy, precision, recall, F1-score)
- Vẽ biểu đồ đánh giá
- Phân tích lỗi (error analysis)
- Visualize kết quả

### 4. **Suy Luận (Inference)**

- Sử dụng mô hình đã huấn luyện để dự đoán trên dữ liệu mới
- Batch prediction
- Real-time inference

## 💻 Yêu Cầu Hệ Thống

- **Python 3.8+**
- **Các thư viện chính:**
  - `pandas` - Xử lý dữ liệu
  - `numpy` - Tính toán số học
  - `scikit-learn` - Các thuật toán ML
  - `pytorch` hoặc `tensorflow` - Deep Learning (nếu có)
  - `matplotlib` / `seaborn` - Trực quan hóa dữ liệu

## 📦 Cài Đặt

### 1. **Tạo môi trường ảo**

```bash
python -m venv .venv
```

### 2. **Kích hoạt môi trường ảo**

**Trên Windows:**

```bash
.venv\Scripts\activate
```

**Trên Linux/Mac:**

```bash
source .venv/bin/activate
```

### 3. **Cài đặt các thư viện**

```bash
pip install -r requirements.txt
```

## 🔄 Quy Trình Làm Việc

### Bước 1: Chuẩn Bị Dữ Liệu

```bash
python scripts/data_preprocessing.py
```

- Tải dữ liệu từ thư mục `data/raw/`
- Xử lý và làm sạch
- Lưu vào `data/processed/`

### Bước 2: Huấn Luyện Mô Hình

```bash
python scripts/model_training.py
```

- Chia dữ liệu thành train/test
- Huấn luyện mô hình
- Lưu mô hình tốt nhất vào `best_model.pth`

### Bước 3: Đánh Giá Mô Hình

```bash
python scripts/model_evaluation.py
```

- Tính toán các chỉ số hiệu suất
- Tạo báo cáo chi tiết
- Hiển thị visualizations

### Bước 4: Sử Dụng Mô Hình để Dự Đoán

```bash
python scripts/inference.py --input data/test_new.csv
```

- Tải mô hình từ `best_model.pth`
- Dự đoán trên dữ liệu mới
- Xuất kết quả

## 📊 Kết Quả Mô Hình

Mô hình tốt nhất được lưu trong file `best_model.pth`. Các chỉ số hiệu suất:

| Chỉ Số    | Giá Trị |
| --------- | ------- |
| Accuracy  | XX%     |
| Precision | XX%     |
| Recall    | XX%     |
| F1-Score  | XX%     |

## 🔧 Cấu Hình

Các tham số chính có thể cấu hình:

```python
# data_preprocessing.py
TEST_SIZE = 0.2          # Kích thước tập test
RANDOM_STATE = 42        # Seed cho tái tạo kết quả

# model_training.py
LEARNING_RATE = 0.001    # Tốc độ học
EPOCHS = 100             # Số epoch huấn luyện
BATCH_SIZE = 32          # Kích thước batch
```

## 🐛 Troubleshooting

### Lỗi: Module not found

```bash
pip install -r requirements.txt
```

### Lỗi: Dữ liệu không tìm thấy

- Kiểm tra dữ liệu có trong thư mục `data/raw/` không
- Đảm bảo đường dẫn file đúng

### Lỗi: Mô hình không tìm thấy

- Chạy `model_training.py` để huấn luyện mô hình trước
- Kiểm tra `best_model.pth` có tồn tại không

## 📝 Ghi Chú

- Luôn lưu seed (RANDOM_STATE) để đảm bảo kết quả có thể tái tạo
- Kiểm tra dữ liệu trước khi huấn luyện
- Sử dụng cross-validation để đánh giá chéo
- Theo dõi quá trình huấn luyện để phát hiện overfitting

## 👨‍💼 Tác Giả

Project 2 - Machine Learning Development

## 📄 Giấy Phép

MIT License

---

**Cập nhật lần cuối:** 2026-05-17
