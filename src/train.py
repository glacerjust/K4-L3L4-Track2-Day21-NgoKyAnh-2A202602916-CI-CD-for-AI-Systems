import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

F1_THRESHOLD = 0.65

if not os.environ.get("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    # 1. Đọc dữ liệu
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # 2. Tách đặc trưng và nhãn
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # Bonus 5: Kiểm tra phân phối dữ liệu (Data Drift Check)
    pos_ratio = float(y_train.mean())
    ref_ratio = 0.248
    drift_diff = abs(pos_ratio - ref_ratio)
    if drift_diff > 0.05:
        print(f"WARNING: Data drift detected! Ty le lop duong la {pos_ratio:.1%}, lech {drift_diff:.1%} so voi tham chieu {ref_ratio:.1%}!")
    else:
        print(f"Phan phoi du lieu on dinh: Ty le lop duong la {pos_ratio:.1%}")

    with mlflow.start_run():
        # 3. Ghi nhận siêu tham số
        mlflow.log_params(params)

        # 4. Khởi tạo và huấn luyện mô hình
        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # 5. Dự đoán và tính chỉ số (chú ý F1 tính cho lớp 1, không dùng average)
        preds = model.predict(X_eval)
        f1 = float(f1_score(y_eval, preds))
        acc = float(accuracy_score(y_eval, preds))

        # Bonus 2: Điều chỉnh ngưỡng quyết định (Threshold Tuning)
        import numpy as np
        probs = model.predict_proba(X_eval)[:, 1]
        best_thresh = 0.5
        best_f1 = f1
        for t in np.arange(0.1, 0.95, 0.05):
            t = round(float(t), 2)
            t_preds = (probs >= t).astype(int)
            t_f1 = float(f1_score(y_eval, t_preds))
            if t_f1 > best_f1:
                best_f1 = t_f1
                best_thresh = t

        print(f"Threshold tuning: Nguong mac dinh 0.5 (F1={f1:.4f}) -> Nguong tot nhat {best_thresh} (F1={best_f1:.4f})")
        mlflow.log_metric("best_threshold", best_thresh)
        mlflow.log_metric("best_f1_score", best_f1)

        # Bonus 3: Báo cáo Precision / Recall và Confusion Matrix
        from sklearn.metrics import classification_report, confusion_matrix
        cm = confusion_matrix(y_eval, preds)
        cr = classification_report(y_eval, preds, target_names=["thu_nhap_thap (<=50K)", "thu_nhap_cao (>50K)"])
        detail_text = f"=== CONFUSION MATRIX ===\n{cm}\n\n=== CLASSIFICATION REPORT ===\n{cr}\n"

        os.makedirs("outputs", exist_ok=True)
        with open("outputs/detail.txt", "w", encoding="utf-8") as f:
            f.write(detail_text)

        # 6. Ghi nhận metric và log model vào MLflow
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("positive_class_ratio", pos_ratio)
        mlflow.sklearn.log_model(model, "model")

        # 7. In kết quả
        print(f"F1: {f1:.4f} | Accuracy: {acc:.4f}")

        # 8. Lưu kết quả ra outputs/report.json cho CI/CD
        report_data = {
            "f1_score": f1,
            "accuracy": acc,
            "best_threshold": best_thresh,
            "best_f1_score": best_f1,
            "positive_class_ratio": pos_ratio,
        }
        with open("outputs/report.json", "w") as f:
            json.dump(report_data, f, indent=2)

        # 9. Lưu mô hình ra models/model.joblib
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")

    # 10. Trả về f1
    return f1


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
