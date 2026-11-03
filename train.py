"""
Huấn luyện mô hình dự đoán trip_duration (yêu cầu 1-4).

    python train.py --train data/train.csv --test data/UNETI18_TEST_3000.csv

Sinh ra: outputs/ (EDA, metrics, file nộp), artifacts/ (mô hình cuối),
artifacts_val/ + data/pseudo_tests/ (đánh giá trên các bộ test giả có nhãn).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from features import (ID, INPUT_COLS, NYC_LAT, NYC_LON, TARGET, TargetEncoder,  # noqa: E402
                      base_features, clean_inputs, model_matrix, oof_target_encode)
from metrics import rmsle, validate_submission  # noqa: E402

SEED = 42
plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREY = "#2563eb", "#ea580c", "#6b7280"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =============================================================================
# YÊU CẦU 1 - EDA
# =============================================================================
def run_eda(train: pd.DataFrame, test: pd.DataFrame, out: str) -> dict:
    os.makedirs(out, exist_ok=True)
    y = train[TARGET]
    days = ["T2", "T3", "T4", "T5", "T6", "T7", "CN"]

    summ = {
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "dtypes": {c: str(t) for c, t in train.dtypes.items()},
        "missing_train": {c: int(v) for c, v in train.isna().sum().items() if v},
        "missing_test": {c: int(v) for c, v in test.isna().sum().items() if v},
        "missing_train_total": int(train.isna().sum().sum()),
        "missing_test_total": int(test.isna().sum().sum()),
        "dup_trip_id_train": int(train[ID].duplicated().sum()),
        "overlap_trip_id": int(len(set(train[ID]) & set(test[ID]))),
        "duration": {
            "mean": float(y.mean()), "median": float(y.median()),
            "std": float(y.std()), "min": float(y.min()), "max": float(y.max()),
            "skew_raw": float(y.skew()), "skew_log": float(np.log1p(y).skew()),
            "q01": float(y.quantile(0.01)), "q99": float(y.quantile(0.99)),
            "n_lt_60s": int((y < 60).sum()), "n_gt_6h": int((y > 6 * 3600).sum()),
            "n_gt_22h": int((y > 22 * 3600).sum()),
        },
        "describe": train.describe().round(4).to_dict(),
    }

    # --- Hình 1: phân phối thời gian (gốc và log)
    fig, ax = plt.subplots(1, 2, figsize=(8, 2.8))
    ax[0].hist(y.clip(upper=y.quantile(0.995)), bins=80, color=BLUE)
    ax[0].set(title="trip_duration (giây, cắt ở phân vị 99,5%)",
              xlabel="giây", ylabel="số chuyến")
    ax[1].hist(np.log1p(y), bins=80, color=ORANGE)
    ax[1].set(title="log(1 + trip_duration)", xlabel="log giây")
    fig.tight_layout(); fig.savefig(f"{out}/fig1_duration_dist.png"); plt.close(fig)

    # --- Hình 2: heatmap thời gian trung vị theo thứ × giờ
    pv = train.pivot_table(index="pickup_dayofweek", columns="pickup_hour",
                           values=TARGET, aggfunc="median") / 60
    fig, ax = plt.subplots(figsize=(8, 2.8))
    im = ax.imshow(pv.values, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(len(pv.index)))
    ax.set_yticklabels([days[int(i)] if int(i) < 7 else str(i) for i in pv.index])
    ax.set_xticks(range(0, 24, 2)); ax.set_xticklabels(range(0, 24, 2))
    ax.set(title="Thời gian chuyến trung vị (phút) theo thứ và giờ đón",
           xlabel="giờ đón")
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout(); fig.savefig(f"{out}/fig2_heatmap_dow_hour.png"); plt.close(fig)

    # --- Hình 3: khoảng cách vs thời gian (hexbin, thang log)
    s = train.sample(min(len(train), 300_000), random_state=SEED)
    s = s[s[TARGET] < 6 * 3600]
    fig, ax = plt.subplots(figsize=(4, 3))
    hb = ax.hexbin(s["straight_line_km_approx"], s[TARGET] / 60, gridsize=45,
                   bins="log", cmap="Blues", extent=(0, 25, 0, 120))
    med = s.groupby("straight_line_km_approx")[TARGET].median() / 60
    med = med[med.index <= 25]
    ax.plot(med.index, med.values, color=ORANGE, lw=1.6, label="trung vị")
    ax.set(title="Khoảng cách vs thời gian", xlabel="khoảng cách chim bay (km)",
           ylabel="phút", ylim=(0, 120))
    ax.legend(loc="upper left", frameon=False)
    fig.colorbar(hb, ax=ax, fraction=0.04)
    fig.tight_layout(); fig.savefig(f"{out}/fig3_distance_vs_duration.png"); plt.close(fig)

    # --- Hình 4: tốc độ trung bình theo giờ (ngày thường vs cuối tuần)
    t = train[(train["straight_line_km_approx"] > 0) & (train[TARGET] > 60)
              & (train[TARGET] < 6 * 3600)].copy()
    t["speed"] = t["straight_line_km_approx"] / (t[TARGET] / 3600)
    t["weekend"] = t["pickup_dayofweek"] >= 5
    sp = t.groupby(["weekend", "pickup_hour"])["speed"].median().unstack(0)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(sp.index, sp[False], color=BLUE, marker="o", ms=3, label="ngày thường")
    if True in sp.columns:
        ax.plot(sp.index, sp[True], color=ORANGE, marker="o", ms=3, label="cuối tuần")
    ax.set(title="Tốc độ chim bay trung vị", xlabel="giờ đón", ylabel="km/h")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(f"{out}/fig4_speed_by_hour.png"); plt.close(fig)

    # --- Hình 5: mật độ điểm đón
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.hexbin(s["pickup_lon_approx"], s["pickup_lat_approx"], gridsize=60,
              bins="log", cmap="viridis", extent=(-74.05, -73.75, 40.6, 40.9))
    ax.set(title="Mật độ điểm đón", xlabel="kinh độ", ylabel="vĩ độ")
    fig.tight_layout(); fig.savefig(f"{out}/fig5_pickup_density.png"); plt.close(fig)

    # --- Nhận xét rút ra tự động từ số liệu
    by_hour = train.groupby("pickup_hour")[TARGET].median()
    wk = t[~t["weekend"]]["speed"].median()
    we = t[t["weekend"]]["speed"].median()
    pk = train[["pickup_lon_approx", "pickup_lat_approx"]].round(2)
    manh = ((pk["pickup_lon_approx"].between(-74.02, -73.93))
            & (pk["pickup_lat_approx"].between(40.70, 40.88))).mean()
    corr = float(np.corrcoef(train["straight_line_km_approx"], np.log1p(y))[0, 1])
    summ["insights"] = {
        "slowest_hour": int(by_hour.idxmax()),
        "slowest_hour_median_min": float(by_hour.max() / 60),
        "fastest_hour": int(by_hour.idxmin()),
        "fastest_hour_median_min": float(by_hour.min() / 60),
        "speed_weekday_kmh": float(wk), "speed_weekend_kmh": float(we),
        "share_pickup_manhattan": float(manh),
        "corr_dist_logdur": corr,
        "speed_min_hour": int(sp[False].idxmin()),
        "speed_min_kmh": float(sp[False].min()),
        "speed_max_hour": int(sp[False].idxmax()),
        "speed_max_kmh": float(sp[False].max()),
    }
    with open(f"{out}/eda_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summ, fh, ensure_ascii=False, indent=2)
    return summ


# =============================================================================
# YÊU CẦU 2 - xử lý ngoại lai trên tập HUẤN LUYỆN
# =============================================================================
def filter_train_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Chỉ loại các dòng chắc chắn là lỗi ghi nhận. Không áp dụng cho
    Validation/Test (Test phải dự đoán đủ, và Validation phải phản ánh
    đúng dữ liệu mà BTC sẽ chấm)."""
    y = df[TARGET]
    dist = df["straight_line_km_approx"]
    speed = dist / (y / 3600).clip(lower=1e-6)
    rules = {
        "thieu_nhan_hoac_khong_duong": y.isna() | (y <= 0),
        "qua_22_gio": y > 22 * 3600,
        "duoi_10_giay": y < 10,
        "toa_do_ngoai_NYC": ~(df["pickup_lon_approx"].between(*NYC_LON)
                              & df["dropoff_lon_approx"].between(*NYC_LON)
                              & df["pickup_lat_approx"].between(*NYC_LAT)
                              & df["dropoff_lat_approx"].between(*NYC_LAT)),
        "toc_do_tren_120kmh": (dist >= 1) & (speed > 120),
        "duoi_1km_ma_tren_3_gio":(dist <= 1) & (y > 3 * 3600),
    }
    mask = np.zeros(len(df), bool)
    report = {}
    for k, m in rules.items():
        m = m.fillna(False).values
        report[k] = int((m & ~mask).sum())
        mask |= m
    report["tong_loai"] = int(mask.sum())
    report["ty_le_loai"] = float(mask.mean())
    return df.loc[~mask].reset_index(drop=True), report


# =============================================================================
# YÊU CẦU 3 - Mô hình
# =============================================================================
LGB_PARAMS = dict(
    objective="regression", metric="rmse", learning_rate=0.05,
    num_leaves=255, min_data_in_leaf=100, feature_fraction=0.7,
    bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0,
    max_bin=255, verbose=-1, num_threads=0,
)


class GroupMedianBaseline:
    """Mô hình cơ sở: trung vị log(thời gian) theo (khoảng cách, khung giờ)."""

    def fit(self, b, y_log):
        self.g = pd.DataFrame({"d": b["dist_bin"].values, "p": b["period"].values,
                               "y": y_log}).groupby(["d", "p"])["y"].median()
        self.gd = pd.DataFrame({"d": b["dist_bin"].values, "y": y_log}
                               ).groupby("d")["y"].median()
        self.m = float(np.median(y_log))
        return self

    def predict(self, b):
        idx = pd.MultiIndex.from_arrays([b["dist_bin"].values, b["period"].values])
        p = pd.Series(self.g.reindex(idx).values)
        p = p.fillna(pd.Series(self.gd.reindex(b["dist_bin"].values).values))
        return p.fillna(self.m).values


def train_lgb(Xtr, ytr, Xva, yva, params, rounds, seed):
    p = dict(params, seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
    dtr = lgb.Dataset(Xtr, ytr, free_raw_data=False)
    dva = lgb.Dataset(Xva, yva, reference=dtr)
    hist = {}
    m = lgb.train(p, dtr, num_boost_round=rounds, valid_sets=[dva],
                  valid_names=["val"],
                  callbacks=[lgb.early_stopping(100, verbose=False),
                             lgb.log_evaluation(200), lgb.record_evaluation(hist)])
    return m, hist["val"]["rmse"]


def noise_floor(train: pd.DataFrame) -> float:
    """Ước lượng sai số không thể giảm: các chuyến có đầu vào giống hệt nhau
    vẫn có thời gian khác nhau. RMSLE của mọi mô hình không thể thấp hơn mức này."""
    d = clean_inputs(train)
    d["y"] = np.log1p(train[TARGET].values)
    keys = ["pickup_lon_approx", "pickup_lat_approx", "dropoff_lon_approx",
            "dropoff_lat_approx", "straight_line_km_approx", "pickup_dayofweek",
            "pickup_hour", "pickup_minute_bucket"]
    g = d.groupby(keys)["y"].agg(["var", "size"])
    g = g[g["size"] >= 2]
    return float(np.sqrt((g["var"] * (g["size"] - 1)).sum() / (g["size"] - 1).sum()))


# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--team", default="UNETI18")
    ap.add_argument("--fast", action="store_true",
                    help="học nhanh hơn (lr 0,1) để thử nghiệm")
    ap.add_argument("--seeds", type=int, default=3,
                    help="số seed LightGBM lấy trung bình ở mô hình cuối")
    ap.add_argument("--val-size", type=float, default=0.2)
    ap.add_argument("--skip-ablation", action="store_true",
                    help="bỏ mô hình LightGBM không target-encoding")
    ap.add_argument("--n-pseudo", type=int, default=10,
                    help="số bộ Test giả 3.000 dòng (có đáp án) cắt từ validation")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs("outputs/eda", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    params = dict(LGB_PARAMS)
    rounds = 5000
    if args.fast:
        params["learning_rate"] = 0.1
        rounds = 2000

    log("Đọc dữ liệu")
    train = pd.read_csv(args.train, dtype={ID: str})
    test = pd.read_csv(args.test, dtype={ID: str})
    miss = [c for c in INPUT_COLS + [TARGET] if c not in train.columns]
    assert not miss, f"Train thiếu cột {miss}"
    log(f"Train {train.shape}, Test {test.shape}")

    # ---------------- Yêu cầu 1
    log("EDA")
    eda = run_eda(train, test, "outputs/eda")

    # ---------------- Tách Validation TRƯỚC khi làm sạch
    tr_raw, va = train_test_split(train, test_size=args.val_size,
                                  random_state=SEED)
    tr_raw, va = tr_raw.reset_index(drop=True), va.reset_index(drop=True)
    tr, outlier_report = filter_train_outliers(tr_raw)
    log(f"Loại ngoại lai ở phần huấn luyện: {outlier_report}")

    y_tr = np.log1p(tr[TARGET].values)
    y_va_raw = va[TARGET].values
    y_va = np.log1p(y_va_raw)

    # ---------------- Yêu cầu 2: đặc trưng
    log("Tạo đặc trưng cơ bản")
    b_tr, b_va = base_features(tr), base_features(va)
    log("Target encoding out-of-fold")
    te_tr = oof_target_encode(b_tr, y_tr)
    enc_tr = TargetEncoder().fit(b_tr, y_tr)
    X_tr = model_matrix(b_tr, te_tr)
    X_va = model_matrix(b_va, enc_tr.transform(b_va))
    base_cols = [c for c in X_tr.columns if not c.startswith(("te_", "cnt_"))]

    results = []

    def add(name, desc, pred_log, extra=None):
        pred = np.expm1(pred_log).clip(1, None)
        r = {"model": name, "mo_ta": desc, "rmsle_val": rmsle(y_va_raw, pred),
             "diem_AI_uoc_tinh": 70 / (1 + rmsle(y_va_raw, pred))}
        r.update(extra or {})
        results.append(r)
        log(f"  {name}: RMSLE = {r['rmsle_val']:.5f}")

    log("M0 - Baseline trung vị nhóm")
    m0 = GroupMedianBaseline().fit(b_tr, y_tr)
    add("M0 Baseline trung vị", "Trung vị log(thời gian) theo bậc khoảng cách × khung giờ",
        m0.predict(b_va))

    log("M1 - Ridge regression (mô hình tuyến tính)")
    m1 = make_pipeline(StandardScaler(), Ridge(alpha=10.0, solver='lsqr'))
    m1.fit(X_tr[base_cols], y_tr)
    add("M1 Ridge", "Hồi quy tuyến tính Ridge trên đặc trưng cơ bản",
        m1.predict(X_va[base_cols]))

    if not args.skip_ablation:
        log("M2 - LightGBM chỉ đặc trưng cơ bản")
        m2, curve2 = train_lgb(X_tr[base_cols], y_tr, X_va[base_cols], y_va,
                               params, rounds, SEED)
        p2 = m2.predict(X_va[base_cols], num_iteration=m2.best_iteration)
        add("M2 LightGBM (cơ bản)", "LightGBM, không dùng target encoding",
            p2, {"best_iter": int(m2.best_iteration)})

    log("M3 - LightGBM đầy đủ (cơ bản + target encoding)")
    m3, curve3 = train_lgb(X_tr, y_tr, X_va, y_va, params, rounds, SEED)
    p3 = m3.predict(X_va, num_iteration=m3.best_iteration)
    add("M3 LightGBM (đầy đủ)", "LightGBM + target encoding theo ô lưới/khung giờ",
        p3, {"best_iter": int(m3.best_iteration)})

    best = min(results, key=lambda r: r["rmsle_val"])
    log(f"Mô hình tốt nhất: {best['model']}")
    # Mô hình cuối: LightGBM với bộ đặc trưng thắng trên validation
    lgb_res = [r for r in results if "LightGBM" in r["model"]]
    final_res = min(lgb_res, key=lambda r: r["rmsle_val"])
    use_te = "đầy đủ" in final_res["model"]
    final_cols = list(X_tr.columns) if use_te else base_cols
    best_iter = final_res["best_iter"]
    log(f"Cấu hình mô hình cuối: {final_res['model']} ({len(final_cols)} đặc trưng)")
    val_model, p_fin, curve = (m3, p3, curve3) if use_te else (m2, p2, curve2)
    short = final_res["model"].split(" ")[0]

    # Bộ test giả: phần validation (mô hình chưa học) cắt thành file 3.000 dòng có nhãn,
    # đi kèm mô hình chỉ học 80% Train -> chấm RMSLE trung thực bằng predict.py
    os.makedirs("artifacts_val", exist_ok=True)
    val_model.save_model("artifacts_val/lgb_val.txt",
                         num_iteration=val_model.best_iteration)
    enc_tr.save("artifacts_val/encoder.npz")
    with open("artifacts_val/meta.json", "w", encoding="utf-8") as fh:
        json.dump({"team": args.team, "feature_names": final_cols,
                   "model_files": ["lgb_val.txt"],
                   "clip_seconds": [max(float(np.quantile(tr[TARGET], 0.0005)), 1.0),
                                    float(np.quantile(tr[TARGET], 0.9995))],
                   "val_rmsle": final_res["rmsle_val"], "final_config": final_res["model"],
                   "trained_on": "train_80pct_only"}, fh, ensure_ascii=False, indent=2)
    os.makedirs("data/pseudo_tests", exist_ok=True)
    shuffled = va.sample(frac=1, random_state=SEED).reset_index(drop=True)
    n_files = min(args.n_pseudo, len(shuffled) // len(test))
    for i in range(n_files):
        part = shuffled.iloc[i * len(test):(i + 1) * len(test)]
        part[[ID] + INPUT_COLS + [TARGET]].to_csv(
            f"data/pseudo_tests/PSEUDO_TEST_{i + 1:02d}.csv", index=False)
    log(f"Đã lưu {n_files} bộ test giả vào data/pseudo_tests/")

    # Phân tích lỗi theo nhóm khoảng cách (phần hạn chế trong báo cáo)
    err = pd.DataFrame({"dist": va["straight_line_km_approx"],
                        "se": (p_fin - y_va) ** 2,
                        "air": b_va["airport_trip"].values,
                        "same": b_va["same_cell"].values})
    err["dist_grp"] = pd.cut(err["dist"], [-0.1, 0.4, 2, 5, 10, 100],
                             labels=["0 km", "0,5-2", "2,5-5", "5,5-10", ">10"])
    err_by = (err.groupby("dist_grp", observed=True)["se"]
              .agg(lambda s: float(np.sqrt(s.mean()))).to_dict())
    error_analysis = {
        "rmsle_by_dist": {str(k): v for k, v in err_by.items()},
        "rmsle_airport": float(np.sqrt(err.loc[err.air == 1, "se"].mean()))
        if (err.air == 1).any() else None,
        "rmsle_same_cell": float(np.sqrt(err.loc[err.same == 1, "se"].mean()))
        if (err.same == 1).any() else None,
        "share_val_gt_6h": float((y_va_raw > 6 * 3600).mean()),
        "rmsle_val_without_gt6h": rmsle(y_va_raw[y_va_raw <= 6 * 3600],
                                        np.expm1(p_fin[y_va_raw <= 6 * 3600])),
        "noise_floor": noise_floor(train),
    }
    log(f"Giới hạn nhiễu ước tính: RMSLE ~ {error_analysis['noise_floor']:.4f}")

    # Biểu đồ đường học và feature importance
    fig, ax = plt.subplots(figsize=(4, 2.6))
    ax.plot(curve, color=BLUE)
    ax.set(title=f"Đường học {short} (RMSE log trên validation)",
           xlabel="số cây", ylabel="RMSLE")
    fig.tight_layout(); fig.savefig("outputs/eda/fig7_learning_curve.png"); plt.close(fig)

    imp = pd.Series(val_model.feature_importance("gain"), index=final_cols)
    metrics_imp = (imp / imp.sum()).sort_values(ascending=False).head(15)
    imp = (imp / imp.sum()).sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(4, 3.3))
    ax.barh(imp.index, imp.values * 100, color=BLUE)
    ax.set(title=f"Top 15 đặc trưng ({short}, % gain)", xlabel="%")
    fig.tight_layout(); fig.savefig("outputs/eda/fig6_feature_importance.png"); plt.close(fig)

    # Biểu đồ so sánh mô hình
    fig, ax = plt.subplots(figsize=(4, 2.6))
    names = [r["model"].split(" ")[0] for r in results]
    vals = [r["rmsle_val"] for r in results]
    ax.bar(names, vals, color=[ORANGE if r is best else GREY for r in results])
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set(title="RMSLE kiểm định (thấp hơn = tốt hơn)", ylabel="RMSLE")
    fig.tight_layout(); fig.savefig("outputs/eda/fig8_model_compare.png"); plt.close(fig)

    # ---------------- Huấn luyện lại trên TOÀN BỘ Train cho mô hình cuối
    log("Huấn luyện lại mô hình cuối trên toàn bộ Train")
    full, outlier_full = filter_train_outliers(train)
    y_full = np.log1p(full[TARGET].values)
    b_full = base_features(full)
    X_full = model_matrix(b_full, oof_target_encode(b_full, y_full))[final_cols]
    encoder = TargetEncoder().fit(b_full, y_full)
    n_iter = int(best_iter * 1.1)  # nhiều dữ liệu hơn -> thêm ~10% số cây

    model_files = []
    for i in range(args.seeds):
        s = SEED + i
        p = dict(params, seed=s, bagging_seed=s, feature_fraction_seed=s)
        log(f"  seed {s}: {n_iter} cây")
        mdl = lgb.train(p, lgb.Dataset(X_full, y_full), num_boost_round=n_iter)
        fn = f"artifacts/lgb_seed{s}.txt"
        mdl.save_model(fn)
        model_files.append(os.path.basename(fn))

    lo = float(np.quantile(full[TARGET], 0.0005))
    hi = float(np.quantile(full[TARGET], 0.9995))
    encoder.save("artifacts/encoder.npz")
    meta = {
        "team": args.team, "feature_names": list(X_full.columns),
        "model_files": model_files, "n_iter": n_iter,
        "clip_seconds": [max(lo, 1.0), hi],
        "versions": {"python": platform.python_version(), "lightgbm": lgb.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__,
                     "numpy": np.__version__},
        "val_rmsle": final_res["rmsle_val"], "final_config": final_res["model"],
        "trained_on": "full_train",
    }
    with open("artifacts/meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    # ---------------- Yêu cầu 4: dự đoán Test
    log("Dự đoán bộ Test")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from predictor import Predictor  # dùng đúng lớp mà ứng dụng dùng
    pred = Predictor("artifacts").predict_df(test)
    sub = pd.DataFrame({"trip_id": test[ID].values, "trip_duration": np.round(pred, 2)})
    errs = validate_submission(sub, test)
    if errs:
        raise SystemExit("FILE NỘP KHÔNG HỢP LỆ: " + "; ".join(errs))
    sub_path = f"outputs/submission_{args.team}.csv"
    sub.to_csv(sub_path, index=False)
    log(f"Đã ghi {sub_path} - hợp lệ ({len(sub)} dòng)")

    # Kiểm tra chéo: phân phối dự đoán Test so với Train
    fig, ax = plt.subplots(figsize=(4, 2.6))
    ax.hist(np.log1p(train[TARGET]), bins=60, density=True, alpha=.5,
            color=GREY, label="Train thực tế")
    ax.hist(np.log1p(pred), bins=60, density=True, alpha=.6, color=ORANGE,
            label="Test dự đoán")
    ax.set(title="Phân phối log thời gian", xlabel="log(1+giây)")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig("outputs/eda/fig9_test_pred_dist.png"); plt.close(fig)

    metrics = {
        "team": args.team, "results": results, "best_model": best["model"], "final_model": final_res["model"],
        "outliers_trainpart": outlier_report, "outliers_full": outlier_full,
        "n_train_rows_used": int(len(full)), "n_val": int(len(va)),
        "n_features": int(X_full.shape[1]), "n_seeds": args.seeds,
        "params": params, "error_analysis": error_analysis,
        "feature_importance_top15": {k: float(v) for k, v in metrics_imp.items()},
        "test_pred_stats": {"median": float(np.median(pred)), "mean": float(pred.mean()),
                            "min": float(pred.min()), "max": float(pred.max())},
        "runtime_min": (time.time() - t0) / 60,
        "submission": sub_path,
    }
    with open("outputs/metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)
    log(f"XONG sau {(time.time() - t0) / 60:.1f} phút. "
        f"Tiếp theo: python make_report.py  rồi  streamlit run app.py")


if __name__ == "__main__":
    main()
