"""Nạp mô hình đã huấn luyện và dự đoán (dùng chung cho train.py, predict.py, app.py)."""
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from features import TargetEncoder, make_features  # noqa: E402


class Predictor:
    def __init__(self, artifact_dir: str = "artifacts"):
        with open(os.path.join(artifact_dir, "meta.json"), encoding="utf-8") as fh:
            self.meta = json.load(fh)
        self.encoder = self._load_encoder(artifact_dir)
        # Đọc model dạng chuỗi: tránh lỗi đường dẫn có dấu tiếng Việt trên Windows
        self.models = []
        for f in self.meta["model_files"]:
            with open(os.path.join(artifact_dir, f), encoding="utf-8") as fh:
                self.models.append(lgb.Booster(model_str=fh.read()))
        self.features = self.meta["feature_names"]
        self.lo, self.hi = self.meta["clip_seconds"]

    @staticmethod
    def _load_encoder(artifact_dir):
        npz = os.path.join(artifact_dir, "encoder.npz")
        if os.path.exists(npz):
            return TargetEncoder.load(npz)
        import joblib  # định dạng cũ, cần cùng phiên bản pandas lúc lưu
        return joblib.load(os.path.join(artifact_dir, "encoder.joblib"))

    def predict_df(self, df: pd.DataFrame) -> np.ndarray:
        """Thời gian dự đoán (giây) cho từng dòng."""
        X = make_features(df.reset_index(drop=True), self.encoder)[self.features]
        log_pred = np.mean([m.predict(X) for m in self.models], axis=0)
        return np.clip(np.expm1(log_pred), self.lo, self.hi)

    def predict_one(self, **trip) -> float:
        return float(self.predict_df(pd.DataFrame([trip]))[0])
