# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | Ngô Kỳ Anh |
| MSSV | 2A202602916 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/glacerjust/K4-L3L4-Track2-Day21-NgoKyAnh-2A202602916-CI-CD-for-AI-Systems |
| Ngày nộp | 07/10/2026 |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

Dưới đây là kết quả thực nghiệm thực tế thu thập từ giao diện MLflow UI ở Bước 1:

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 200 | 0.1 | 5 | 0.7149 | 0.8740 |
| 2 | 50 | 0.05 | 2 | 0.6051 | 0.8460 |
| 3 | 100 | 0.1 | 3 | 0.7109 | 0.8780 |

**Bộ siêu tham số đã chọn:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=5`.

**Lý do:** Bộ siêu tham số được chọn đạt điểm F1 cao nhất trong cả 3 lần thử nghiệm (0.7149), vượt qua ngưỡng chất lượng tối thiểu 0.65 của bài lab. Điểm mấu chốt là lần chạy 3 có accuracy cao nhất (0.8780) nhưng F1 lại thấp hơn lần 1 (0.7109 so với 0.7149). Điều này chứng minh rằng việc tối ưu accuracy không đồng nghĩa với khả năng bắt trúng lớp thu nhập cao, do accuracy bị chi phối bởi việc đoán đúng lớp đa số. Khi tăng số lượng cây lên 200 kết hợp độ sâu cây bằng 5, mô hình Gradient Boosting có đủ dung lượng để học các quan hệ phi tuyến phức tạp trong dữ liệu điều tra dân số, dù phải đánh đổi một chút thời gian huấn luyện (9.5 giây so với 6.7 giây).

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Tập dữ liệu Adult Census Income có sự mất cân bằng lớp nghiêm trọng khi chỉ có khoảng 24.8% số mẫu thuộc lớp thu nhập cao (>50K USD/năm). Nếu một mô hình ngây thơ luôn dự đoán nhãn là "thu nhập thấp" cho toàn bộ mẫu dữ liệu, mô hình đó vẫn đạt độ chính xác (accuracy) lên tới 75.2%, nhưng trên thực tế mô hình hoàn toàn vô dụng do không phát hiện được bất kỳ cá nhân thu nhập cao nào (Recall của lớp dương bằng 0). 

Chỉ số F1 trên lớp dương là trung bình điều hòa giữa Precision và Recall, phản ánh chính xác khả năng mô hình vừa phân loại đúng đối tượng thu nhập cao, vừa hạn chế tối đa việc đoán sai lớp thiểu số này. Chúng ta không sử dụng `average="weighted"` hay `average="macro"` khi tính F1 vì trọng số của lớp đa số chiếm tới 75.2% sẽ kéo giá trị trung bình lên cao, làm mất đi tính cảnh báo thực sự của Quality Gate trong pipeline CI/CD.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| Lỗi cài đặt thư viện do `pkg_resources` và `FallbackAsyncAdaptedQueuePool` bị thiếu. | Bản `setuptools >= 82` xóa `pkg_resources` và `SQLAlchemy 2.1` bỏ connection pool cũ gây xung đột với `mlflow==2.13.0`. | Ghim phiên bản `setuptools<82` và `sqlalchemy<2.1` trong file `requirements.txt`. |
| Lỗi unpickle mô hình khi khởi động service trên EC2 (`AttributeError: CyHalfBinomialLoss`). | Môi trường GitHub Actions dùng Python 3.10 sinh ra model không tương thích với `scikit-learn 1.9.1` trên EC2 (Python 3.14). | Nâng runner GitHub Actions lên Python 3.11 và ghim `scikit-learn==1.9.1` để đồng nhất phiên bản giữa CI và Production. |
| Lệnh `curl` trên Windows PowerShell báo lỗi globbing URL và sai định dạng header. | PowerShell tự gán alias `curl` sang `Invoke-WebRequest` và ký tự `[` bị nhận diện nhầm là URL globbing. | Sử dụng trực tiếp `curl.exe` cùng tùy chọn `-d "@payload.json"` để truyền body JSON an toàn. |

---

## 4. So Sánh Bước 2 và Bước 3

Dưới đây là số liệu trích xuất từ file báo cáo của pipeline qua hai giai đoạn:

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1` - 22.361 mẫu) | 0.7149 | 0.8740 |
| Bước 3 (thêm `train_batch2` - 44.722 mẫu) | 0.7354 | 0.8820 |

**Nhận xét:** Khi bổ sung thêm 22.361 mẫu dữ liệu mới ở Bước 3, cả hai chỉ số đánh giá đều có sự cải thiện tích cực: `f1_score` tăng từ 0.7149 lên 0.7354 (+0.0205) và `accuracy` tăng từ 0.8740 lên 0.8820 (+0.0080). Khối lượng dữ liệu gấp đôi đã giúp mô hình Gradient Boosting khái quát hóa tốt hơn các đặc trưng phân bố của đối tượng có thu nhập cao. Điều quan trọng nhất là toàn bộ quy trình Continuous Training đã vận hành hoàn toàn tự động: chỉ từ một commit cập nhật dữ liệu trên DVC, GitHub Actions đã tự động huấn luyện lại, vượt qua Quality Gate và kích hoạt cập nhật phiên bản model mới trên máy chủ AWS EC2 mà không cần bất kỳ sự can thiệp thủ công nào.

---

## 5. Phần Bonus Đã Thực Hiện

- [x] **Bonus 2 - Điều chỉnh ngưỡng quyết định**: Sử dụng `predict_proba` quét ngưỡng xác suất từ 0.1 đến 0.9 (bước 0.05), xác định ngưỡng tối ưu nâng F1 cao hơn ngưỡng mặc định 0.5; ghi `best_threshold` và `best_f1_score` vào `report.json` và MLflow.
- [x] **Bonus 3 - Báo cáo precision / recall tự động**: Xuất Confusion Matrix và Precision/Recall từng lớp vào `outputs/detail.txt` lưu thành CI artifact. Đối với bài toán này, sai lầm bỏ sót người thu nhập cao (False Negative - Recall thấp) tốn kém hơn việc gán nhầm người thu nhập thấp (False Positive - Precision thấp) do mục tiêu bài toán kinh doanh là tối đa hóa tỷ lệ tiếp cận tệp khách hàng tiềm năng.
- [x] **Bonus 5 - Cảnh báo lệch lạc dữ liệu**: Kiểm tra tỷ lệ lớp dương trong tập huấn luyện (24.8%, không lệch quá 5% so với phân phối tham chiếu); ghi nhận cảnh báo và lưu `positive_class_ratio` vào `outputs/report.json`.
