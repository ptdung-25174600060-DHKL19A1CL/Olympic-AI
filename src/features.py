"""
features.py - Pipeline tiền xử lý và tạo đặc trưng DÙNG CHUNG cho
Train / Validation / Test / ứng dụng Streamlit.

Nguyên tắc: mọi phép biến đổi đều đi qua đúng các hàm trong file này,
nên dữ liệu ở cả 4 nơi được xử lý giống hệt nhau (yêu cầu 2 của đề).

Không dùng trip_id và không dùng bất kỳ thông tin nào chỉ có sau khi
chuyến đi kết thúc (đúng lưu ý kỹ thuật của đề).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Hằng số
# ----------------------------------------------------------------------------
INPUT_COLS = [
    "vendor_id", "pickup_month", "pickup_dayofweek", "pickup_hour",
    "pickup_minute_bucket", "passenger_count",
    "pickup_lon_approx", "pickup_lat_approx",
    "dropoff_lon_approx", "dropoff_lat_approx",
    "straight_line_km_approx", "store_and_fwd_flag",
]
TARGET = "trip_duration"
ID = "trip_id"

# Vùng hợp lệ của New York (dùng để phát hiện tọa độ bất thường)
NYC_LON = (-74.30, -73.65)
NYC_LAT = (40.48, 40.95)
# Tâm Manhattan để chiếu tọa độ, và các sân bay
CENTER = (-73.985, 40.758)
AIRPORTS = {
    "jfk": (-73.785, 40.645),
    "lga": (-73.872, 40.774),
    "ewr": (-74.175, 40.690),
}
# Góc nghiêng lưới phố Manhattan so với hướng Bắc (~29 độ)
MANHATTAN_ANGLE = np.deg2rad(29.0)

# Bảng target encoding: (tên, danh sách cột khóa)
TE_KEYS = [
    ("pu_cell", ["pu_cell"]),
    ("do_cell", ["do_cell"]),
    ("pair", ["pu_cell", "do_cell"]),
    ("pair_period", ["pu_cell", "do_cell", "period"]),
    ("pair_weekend", ["pu_cell", "do_cell", "is_weekend"]),
    ("pu_hour", ["pu_cell", "pickup_hour"]),
    ("do_hour", ["do_cell", "pickup_hour"]),
    ("dist_slot", ["dist_bin", "time_slot", "is_weekend"]),
    ("dist_dow_hour", ["dist_bin", "pickup_dayofweek", "pickup_hour"]),
    ("pu_zone_do_zone_hour", ["pu_zone", "do_zone", "pickup_hour"]),
]
TE_SMOOTHING = 20.0  # số mẫu "ảo" kéo trung bình nhóm nhỏ về giá trị chung


# ----------------------------------------------------------------------------
# Hàm tiện ích
# ----------------------------------------------------------------------------
def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def bearing_deg(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    y = np.sin(lon2 - lon1) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    return np.degrees(np.arctan2(y, x))


def to_km_xy(lon, lat):
    """Đổi kinh/vĩ độ thành tọa độ phẳng (km) quanh tâm Manhattan."""
    x = (lon - CENTER[0]) * 111.32 * np.cos(np.radians(CENTER[1]))
    y = (lat - CENTER[1]) * 110.57
    return x, y


def cell_id(lon, lat):
    """Mã ô lưới 0,01 độ (đúng độ phân giải BTC đã làm tròn)."""
    ix = np.round((lon + 75.0) * 100).astype("int64")
    iy = np.round((lat - 40.0) * 100).astype("int64")
    return ix * 1000 + iy


def zone_id(lon, lat):
    """Ô lưới thô hơn 0,03 độ (~3 km) để thống kê ổn định hơn."""
    ix = np.floor((lon + 75.0) / 0.03).astype("int64")
    iy = np.floor((lat - 40.0) / 0.03).astype("int64")
    return ix * 1000 + iy


# ----------------------------------------------------------------------------
# Làm sạch đầu vào (áp dụng cho mọi tập)
# ----------------------------------------------------------------------------
def clean_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa kiểu dữ liệu, điền giá trị thiếu, chặn giá trị vô lý.
    Không xóa dòng nào (Test phải dự đoán đủ 3.000 dòng)."""
    d = df.copy()
    defaults = {
        "vendor_id": 2, "pickup_month": 3, "pickup_dayofweek": 3,
        "pickup_hour": 12, "pickup_minute_bucket": 0, "passenger_count": 1,
        "pickup_lon_approx": CENTER[0], "pickup_lat_approx": CENTER[1],
        "dropoff_lon_approx": CENTER[0], "dropoff_lat_approx": CENTER[1],
        "store_and_fwd_flag": "N",
    }
    for c, v in defaults.items():
        if c not in d.columns:
            d[c] = v
        if c != "store_and_fwd_flag":
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d[c] = d[c].fillna(v)

    d["store_and_fwd_flag"] = (d["store_and_fwd_flag"].astype(str).str.upper()
                               .str.strip().eq("Y").astype("int8"))
    d["passenger_count"] = d["passenger_count"].clip(1, 6)
    d["pickup_hour"] = d["pickup_hour"].clip(0, 23)
    d["pickup_dayofweek"] = d["pickup_dayofweek"].clip(0, 6)
    d["pickup_month"] = d["pickup_month"].clip(1, 12)
    d["pickup_minute_bucket"] = (d["pickup_minute_bucket"] // 15 * 15).clip(0, 45)

    # Tọa độ ngoài NYC -> kéo về biên (không xóa dòng)
    # Làm tròn 0,01 độ đúng như bộ Test của BTC: nếu Train có độ chính xác
    # cao hơn Test thì mô hình sẽ học trên dữ liệu "khác" dữ liệu bị chấm.
    for c in ("pickup_lon_approx", "dropoff_lon_approx"):
        d[c] = d[c].clip(*NYC_LON).round(2)
    for c in ("pickup_lat_approx", "dropoff_lat_approx"):
        d[c] = d[c].clip(*NYC_LAT).round(2)

    # Khoảng cách thiếu -> tính lại từ tọa độ
    hv = haversine_km(d["pickup_lon_approx"], d["pickup_lat_approx"],
                      d["dropoff_lon_approx"], d["dropoff_lat_approx"])
    if "straight_line_km_approx" not in d.columns:
        d["straight_line_km_approx"] = np.nan
    d["straight_line_km_approx"] = pd.to_numeric(d["straight_line_km_approx"],
                                                 errors="coerce")
    d["straight_line_km_approx"] = d["straight_line_km_approx"].fillna(
        np.round(hv * 2) / 2).clip(0, 60)
    # Bộ Test dùng bậc 0,5 km -> đưa mọi tập về cùng bậc (không đổi gì nếu đã đúng)
    d["straight_line_km_approx"] = np.round(d["straight_line_km_approx"] * 2) / 2
    return d


# ----------------------------------------------------------------------------
# Đặc trưng cơ bản (không cần nhãn)
# ----------------------------------------------------------------------------
def base_features(df: pd.DataFrame) -> pd.DataFrame:
    d = clean_inputs(df)
    plon, plat = d["pickup_lon_approx"].values, d["pickup_lat_approx"].values
    dlon, dlat = d["dropoff_lon_approx"].values, d["dropoff_lat_approx"].values

    f = pd.DataFrame(index=d.index)
    f["vendor_id"] = d["vendor_id"].astype("int8")
    f["passenger_count"] = d["passenger_count"].astype("int8")
    f["store_and_fwd_flag"] = d["store_and_fwd_flag"]

    # ---- Thời gian
    f["pickup_month"] = d["pickup_month"].astype("int8")
    f["pickup_dayofweek"] = d["pickup_dayofweek"].astype("int8")
    f["pickup_hour"] = d["pickup_hour"].astype("int8")
    f["pickup_minute_bucket"] = d["pickup_minute_bucket"].astype("int8")
    f["time_slot"] = (d["pickup_hour"] * 4 + d["pickup_minute_bucket"] // 15).astype("int16")
    f["week_slot"] = (d["pickup_dayofweek"] * 96 + f["time_slot"]).astype("int16")
    f["is_weekend"] = (d["pickup_dayofweek"] >= 5).astype("int8")
    f["is_rush"] = ((f["is_weekend"] == 0)
                    & (d["pickup_hour"].between(7, 9) | d["pickup_hour"].between(16, 19))
                    ).astype("int8")
    f["is_night"] = ((d["pickup_hour"] <= 5) | (d["pickup_hour"] >= 23)).astype("int8")
    f["period"] = pd.cut(d["pickup_hour"], [-1, 5, 9, 15, 19, 23],
                         labels=False).astype("int8")  # 5 khung giờ
    hour_frac = d["pickup_hour"] + d["pickup_minute_bucket"] / 60
    f["hour_sin"] = np.sin(2 * np.pi * hour_frac / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour_frac / 24)
    f["dow_sin"] = np.sin(2 * np.pi * d["pickup_dayofweek"] / 7)
    f["dow_cos"] = np.cos(2 * np.pi * d["pickup_dayofweek"] / 7)

    # ---- Vị trí
    f["pickup_lon"], f["pickup_lat"] = plon, plat
    f["dropoff_lon"], f["dropoff_lat"] = dlon, dlat
    px, py = to_km_xy(plon, plat)
    qx, qy = to_km_xy(dlon, dlat)
    f["dx_km"], f["dy_km"] = qx - px, qy - py
    # Xoay hệ trục theo lưới phố Manhattan
    c, s = np.cos(MANHATTAN_ANGLE), np.sin(MANHATTAN_ANGLE)
    f["pu_rot_x"], f["pu_rot_y"] = px * c - py * s, px * s + py * c
    f["do_rot_x"], f["do_rot_y"] = qx * c - qy * s, qx * s + qy * c
    f["center_x"], f["center_y"] = (px + qx) / 2, (py + qy) / 2
    f["pu_dist_center"] = np.hypot(px, py)
    f["do_dist_center"] = np.hypot(qx, qy)

    # ---- Khoảng cách
    f["dist_btc_km"] = d["straight_line_km_approx"].values
    f["dist_hav_km"] = haversine_km(plon, plat, dlon, dlat)
    f["dist_manh_km"] = np.abs(f["dx_km"]) + np.abs(f["dy_km"])
    rdx = f["do_rot_x"] - f["pu_rot_x"]
    rdy = f["do_rot_y"] - f["pu_rot_y"]
    f["dist_manh_rot_km"] = np.abs(rdx) + np.abs(rdy)  # đi theo ô phố
    f["log_dist"] = np.log1p(f["dist_btc_km"])
    f["bearing"] = bearing_deg(plon, plat, dlon, dlat)
    f["same_cell"] = ((np.round(plon, 2) == np.round(dlon, 2))
                      & (np.round(plat, 2) == np.round(dlat, 2))).astype("int8")
    f["dist_bin"] = np.minimum(np.round(f["dist_btc_km"] * 2), 60).astype("int16")

    # ---- Sân bay
    for name, (alon, alat) in AIRPORTS.items():
        dp = haversine_km(plon, plat, alon, alat)
        dd = haversine_km(dlon, dlat, alon, alat)
        f[f"pu_{name}"] = (dp < 2.5).astype("int8")
        f[f"do_{name}"] = (dd < 2.5).astype("int8")
    f["airport_trip"] = (f[[c for c in f.columns
                           if c.startswith(("pu_jfk", "do_jfk", "pu_lga",
                                            "do_lga", "pu_ewr", "do_ewr"))]]
                         .max(axis=1)).astype("int8")

    # ---- Khóa nhóm cho target encoding
    f["pu_cell"] = cell_id(plon, plat)
    f["do_cell"] = cell_id(dlon, dlat)
    f["pu_zone"] = zone_id(plon, plat)
    f["do_zone"] = zone_id(dlon, dlat)
    return f


# ----------------------------------------------------------------------------
# Target encoding (thống kê lịch sử từ Train)
# ----------------------------------------------------------------------------
def _key(f: pd.DataFrame, cols):
    k = f[cols[0]].astype("int64").values.copy()
    for c in cols[1:]:
        k = k * 100003 + f[c].astype("int64").values
    return k


class TargetEncoder:
    """Lưu trung bình log(thời gian) và số chuyến cho từng nhóm khóa.
    Nhóm ít mẫu được làm trơn về trung bình chung để tránh quá khớp."""

    def __init__(self, keys=TE_KEYS, smoothing=TE_SMOOTHING):
        self.keys = keys
        self.m = smoothing
        self.tables: dict = {}
        self.global_mean = None

    def fit(self, f: pd.DataFrame, y_log: np.ndarray):
        self.global_mean = float(np.mean(y_log))
        for name, cols in self.keys:
            g = (pd.DataFrame({"k": _key(f, cols), "y": y_log})
                 .groupby("k")["y"].agg(["sum", "count"]))
            self.tables[name] = g
        return self

    def transform(self, f: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=f.index)
        for name, cols in self.keys:
            g = self.tables[name]
            k = pd.Series(_key(f, cols), index=f.index)
            s = k.map(g["sum"]).fillna(0.0).values
            n = k.map(g["count"]).fillna(0.0).values
            out[f"te_{name}"] = (s + self.m * self.global_mean) / (n + self.m)
            out[f"cnt_{name}"] = np.log1p(n)
        return out

    # Lưu dạng mảng numpy (.npz) thay vì pickle để nạp được trên mọi
    # phiên bản pandas (Colab và máy cá nhân thường khác phiên bản).
    def save(self, path: str):
        arrs = {"global_mean": np.array([self.global_mean]), "m": np.array([self.m])}
        for name, _ in self.keys:
            g = self.tables[name]
            arrs[f"{name}__k"] = g.index.values.astype("int64")
            arrs[f"{name}__sum"] = g["sum"].values.astype("float64")
            arrs[f"{name}__count"] = g["count"].values.astype("float64")
        np.savez_compressed(path, **arrs)

    @classmethod
    def load(cls, path: str) -> "TargetEncoder":
        z = np.load(path)
        enc = cls(smoothing=float(z["m"][0]))
        enc.global_mean = float(z["global_mean"][0])
        for name, _ in enc.keys:
            enc.tables[name] = pd.DataFrame(
                {"sum": z[f"{name}__sum"], "count": z[f"{name}__count"]},
                index=pd.Index(z[f"{name}__k"], name="k"))
        return enc


def oof_target_encode(f: pd.DataFrame, y_log: np.ndarray, n_folds=5, seed=42):
    """Target encoding kiểu out-of-fold cho chính tập huấn luyện:
    mỗi dòng chỉ được mã hóa bằng thống kê từ các fold KHÁC,
    nên mô hình không 'nhìn thấy' nhãn của chính nó (chống rò rỉ)."""
    rng = np.random.RandomState(seed)
    fold = rng.randint(0, n_folds, len(f))
    parts = []
    for k in range(n_folds):
        tr, va = fold != k, fold == k
        enc = TargetEncoder().fit(f[tr], y_log[tr])
        parts.append(enc.transform(f[va]))
    return pd.concat(parts).loc[f.index]


# Cột khóa chỉ dùng để nhóm, không đưa vào mô hình
DROP_FROM_MODEL = ["pu_cell", "do_cell", "pu_zone", "do_zone"]


def model_matrix(base: pd.DataFrame, te: pd.DataFrame) -> pd.DataFrame:
    X = pd.concat([base.drop(columns=DROP_FROM_MODEL), te], axis=1)
    return X.astype("float32")


def make_features(df: pd.DataFrame, encoder: TargetEncoder) -> pd.DataFrame:
    """Hàm DUY NHẤT dùng khi suy luận (Validation, Test, ứng dụng)."""
    b = base_features(df)
    return model_matrix(b, encoder.transform(b))
