"""Thước đo RMSLE và kiểm tra định dạng file nộp."""
import numpy as np
import pandas as pd


def rmsle(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def validate_submission(sub: pd.DataFrame, test: pd.DataFrame) -> list:
    """Trả về danh sách lỗi (rỗng = hợp lệ) theo đúng yêu cầu 4 của đề."""
    err = []
    if list(sub.columns) != ["trip_id", "trip_duration"]:
        err.append(f"Sai cột: {list(sub.columns)}")
    if len(sub) != len(test):
        err.append(f"Sai số dòng: {len(sub)} != {len(test)}")
    if sub["trip_id"].duplicated().any():
        err.append("Có trip_id trùng")
    if set(sub["trip_id"]) != set(test["trip_id"]):
        err.append("Tập trip_id không khớp file Test")
    v = pd.to_numeric(sub["trip_duration"], errors="coerce")
    if v.isna().any():
        err.append("Có giá trị trống")
    if not np.isfinite(v.fillna(0)).all():
        err.append("Có giá trị vô hạn")
    if (v <= 0).any():
        err.append("Có giá trị không dương")
    return err
