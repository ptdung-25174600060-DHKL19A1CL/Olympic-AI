"""
Dự đoán một hoặc nhiều file Test bằng mô hình đã lưu.

    python predict.py --test data/UNETI18_TEST_3000.csv --out outputs/submission_UNETI18.csv
    python predict.py --test data/test                      # cả thư mục (kể cả thư mục con)
    python predict.py --test data/pseudo_tests --artifacts artifacts_val

File có cột trip_duration sẽ được chấm RMSLE. File sai định dạng bị bỏ qua.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

from predictor import Predictor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from features import INPUT_COLS  # noqa: E402
from metrics import rmsle, validate_submission  # noqa: E402

SKIP = ("_pred.csv", "tong_hop.csv")


def list_csv(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            found = glob.glob(os.path.join(p, "**", "*"), recursive=True)
            files += sorted(f for f in found if f.lower().endswith(".csv"))
        else:
            files += sorted(glob.glob(p)) or [p]
    files = [f for f in files if os.path.isfile(f)
             and not os.path.basename(f).lower().endswith(SKIP)]
    return list(dict.fromkeys(files))


def predict_file(model, path):
    df = pd.read_csv(path, dtype={"trip_id": str})
    df.columns = df.columns.str.strip()
    missing = [c for c in INPUT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"thiếu cột {missing[:3]}")
    if "trip_id" not in df.columns:
        df.insert(0, "trip_id", [f"row_{i}" for i in range(len(df))])
    y = df.pop("trip_duration") if "trip_duration" in df.columns else None
    pred = model.predict_df(df)
    sub = pd.DataFrame({"trip_id": df["trip_id"].values, "trip_duration": np.round(pred, 2)})
    return sub, validate_submission(sub, df), y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--out", help="tên file kết quả khi chỉ có 1 file")
    ap.add_argument("--out-dir", default="outputs/predictions")
    ap.add_argument("--artifacts", default="artifacts")
    a = ap.parse_args()

    files = list_csv(a.test)
    if not files:
        sys.exit(f"Không tìm thấy file CSV nào trong: {' '.join(a.test)}")
    if not os.path.exists(os.path.join(a.artifacts, "meta.json")):
        sys.exit(f"Chưa có mô hình trong {a.artifacts}/ - chạy train.py trước.")

    model = Predictor(a.artifacts)
    os.makedirs(a.out_dir, exist_ok=True)
    root = os.path.commonpath([os.path.dirname(os.path.abspath(f)) for f in files])
    print(f"Mô hình: {a.artifacts} | {len(files)} file")

    rows = []
    for f in files:
        name = os.path.relpath(os.path.abspath(f), root)
        r = {"file": name, "so_dong": None, "trang_thai": "", "file_ket_qua": "",
             "trung_vi_s": None, "rmsle": None, "diem_AI": None}
        try:
            sub, errs, y = predict_file(model, f)
            out = a.out if (a.out and len(files) == 1) else os.path.join(
                a.out_dir, os.path.splitext(name)[0].replace(os.sep, "__") + "_pred.csv")
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            sub.to_csv(out, index=False)
            r.update(so_dong=len(sub), file_ket_qua=out,
                     trang_thai="OK" if not errs else "; ".join(errs),
                     trung_vi_s=round(float(sub.trip_duration.median())))
            line = f"  [OK] {name:<32} {len(sub):>6} dòng -> {out}"
            if y is not None:
                r["rmsle"] = round(rmsle(y.values, sub.trip_duration.values), 5)
                r["diem_AI"] = round(70 / (1 + r["rmsle"]), 2)
                line += f"  RMSLE={r['rmsle']:.4f}"
            if errs:
                line += f"  (!) {'; '.join(errs)}"
        except Exception as e:
            r["trang_thai"] = f"bỏ qua: {e}"
            line = f"  [--] {name:<32} bỏ qua: {e}"
        rows.append(r)
        print(line)

    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(a.out_dir, "tong_hop.csv"), index=False)
    scored = tab.dropna(subset=["rmsle"])
    if len(scored):
        m = scored.rmsle.mean()
        print(f"RMSLE trung bình ({len(scored)} file): {m:.4f} | "
              f"min {scored.rmsle.min():.4f} | max {scored.rmsle.max():.4f} | "
              f"điểm AI ~{70 / (1 + m):.2f}/70")
        if model.meta.get("trained_on") == "full_train":
            print("Lưu ý: artifacts/ đã học toàn bộ Train - nếu dữ liệu lấy từ Train thì "
                  "điểm bị ảo, hãy dùng --artifacts artifacts_val.")
    print(f"Tổng hợp: {os.path.join(a.out_dir, 'tong_hop.csv')}")


if __name__ == "__main__":
    main()
