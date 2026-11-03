"""
app.py - Ứng dụng AI mini dự đoán thời gian chuyến taxi New York.
Chạy:  streamlit run app.py

Ứng dụng nạp MÔ HÌNH ĐÃ HUẤN LUYỆN trong thư mục artifacts/ và gọi đúng
pipeline đặc trưng dùng để tạo file nộp (predictor.py -> src/features.py).
Không có kết quả nào được tính sẵn.
"""
import json
import os

import numpy as np
import pandas as pd
import streamlit as st

from predictor import Predictor
from features import haversine_km  # noqa: E402 (đường dẫn src đã được predictor thêm)

st.set_page_config(page_title="Dự đoán thời gian taxi NYC", page_icon="🚕",
                   layout="wide")

PLACES = {
    "Times Square (Midtown)": (-73.9855, 40.7580),
    "Penn Station": (-73.9935, 40.7506),
    "Grand Central": (-73.9772, 40.7527),
    "Central Park South": (-73.9760, 40.7650),
    "Upper East Side (E 86th St)": (-73.9540, 40.7790),
    "Upper West Side (W 86th St)": (-73.9760, 40.7880),
    "Harlem (125th St)": (-73.9450, 40.8080),
    "Greenwich Village": (-74.0000, 40.7330),
    "SoHo": (-74.0000, 40.7230),
    "Wall Street (Financial District)": (-74.0090, 40.7060),
    "Brooklyn - Williamsburg": (-73.9570, 40.7140),
    "Brooklyn - Downtown": (-73.9900, 40.6930),
    "Queens - Long Island City": (-73.9490, 40.7440),
    "Sân bay LaGuardia (LGA)": (-73.8740, 40.7740),
    "Sân bay JFK": (-73.7850, 40.6450),
    "Tự nhập tọa độ...": None,
}
DAYS = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]


@st.cache_resource
def load_model():
    return Predictor("artifacts")


def fmt_duration(sec: float) -> str:
    sec = int(round(sec))
    h, m, s = sec // 3600, sec % 3600 // 60, sec % 60
    return f"{h} giờ {m:02d} phút {s:02d} giây" if h else f"{m} phút {s:02d} giây"


def place_input(label: str, default: str, key: str):
    name = st.selectbox(label, list(PLACES), index=list(PLACES).index(default), key=key)
    if PLACES[name] is None:
        c1, c2 = st.columns(2)
        lon = c1.number_input("Kinh độ", -74.30, -73.65, -73.98, 0.01,
                              format="%.4f", key=key + "lon")
        lat = c2.number_input("Vĩ độ", 40.48, 40.95, 40.75, 0.01,
                              format="%.4f", key=key + "lat")
        return name, lon, lat
    return name, *PLACES[name]


def build_trip(plon, plat, dlon, dlat, month, dow, hour, minute, pax, vendor, flag):
    km = float(haversine_km(plon, plat, dlon, dlat))
    return dict(
        vendor_id=vendor, pickup_month=month, pickup_dayofweek=dow,
        pickup_hour=hour, pickup_minute_bucket=minute, passenger_count=pax,
        # làm tròn giống cách BTC chuẩn hóa dữ liệu
        pickup_lon_approx=round(plon, 2), pickup_lat_approx=round(plat, 2),
        dropoff_lon_approx=round(dlon, 2), dropoff_lat_approx=round(dlat, 2),
        straight_line_km_approx=round(km * 2) / 2,
        store_and_fwd_flag=flag,
    )


if not os.path.exists("artifacts/meta.json"):
    st.error("Chưa có mô hình. Hãy chạy `python train.py --train ... --test ...` trước.")
    st.stop()

model = load_model()
meta = model.meta

# ---------------------------------------------------------------- Sidebar
with st.sidebar:
    st.header("🧠 Mô hình đang dùng")
    st.write(f"**{meta.get('final_config', 'LightGBM')}**")
    st.write(f"Trung bình {len(model.models)} mô hình LightGBM, "
             f"{len(model.features)} đặc trưng")
    st.metric("RMSLE kiểm định", f"{meta['val_rmsle']:.4f}")
    st.caption(f"Điểm AI ước tính: {70 / (1 + meta['val_rmsle']):.2f}/70")
    st.caption("Đơn vị kết quả: giây (kèm quy đổi phút).")
    st.divider()
    st.caption("Olympic AI UNETI 2026 - Vòng 2 - Đội " + meta.get("team", ""))

st.title("🚕 Dự đoán thời gian chuyến taxi New York")
tab1, tab2 = st.tabs(["Dự đoán một chuyến", "Dự đoán theo file CSV"])

# ---------------------------------------------------------------- Tab 1
with tab1:
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.subheader("1. Lộ trình")
        pname, plon, plat = place_input("Điểm đón", "Times Square (Midtown)", "pu")
        dname, dlon, dlat = place_input("Điểm đến", "Sân bay JFK", "do")

        st.subheader("2. Thời điểm đón")
        c1, c2 = st.columns(2)
        month = c1.selectbox("Tháng", list(range(1, 13)), index=3,
                             help="Dữ liệu huấn luyện chủ yếu gồm tháng 1-6")
        dow = c2.selectbox("Thứ", range(7), index=4, format_func=lambda i: DAYS[i])
        c3, c4 = st.columns(2)
        hour = c3.slider("Giờ", 0, 23, 17)
        minute = c4.select_slider("Phút", [0, 15, 30, 45], value=0)

        st.subheader("3. Thông tin khác")
        c5, c6, c7 = st.columns(3)
        pax = c5.number_input("Số khách", 1, 6, 1)
        vendor = c6.selectbox("Hãng xe", [1, 2], index=1)
        flag = c7.selectbox("Cờ lưu/đồng bộ", ["N", "Y"])
        go = st.button("🔮 Dự đoán", type="primary", width="stretch")

    with right:
        trip = build_trip(plon, plat, dlon, dlat, month, dow, hour, minute,
                          pax, vendor, flag)
        if go:
            sec = model.predict_one(**trip)
            st.session_state.setdefault("history", []).insert(0, {
                "Điểm đón": pname, "Điểm đến": dname,
                "Thời điểm": f"{DAYS[dow]}, {hour:02d}:{minute:02d}, tháng {month}",
                "Khoảng cách (km)": trip["straight_line_km_approx"],
                "Dự đoán (giây)": round(sec), "Dự đoán (phút)": round(sec / 60, 1)})
            st.session_state["last"] = (trip, sec)

        if "last" in st.session_state:
            trip_l, sec = st.session_state["last"]
            st.subheader("Kết quả")
            m1, m2, m3 = st.columns(3)
            m1.metric("Thời gian dự đoán", f"{sec:,.0f} giây".replace(",", "."))
            m2.metric("Quy đổi", fmt_duration(sec))
            km = trip_l["straight_line_km_approx"]
            m3.metric("Tốc độ chim bay TB",
                      f"{km / (sec / 3600):.1f} km/h" if km > 0 else "-")
            st.caption(f"Khoảng cách đường chim bay ước tính: {km} km")

            pts = pd.DataFrame({
                "lat": [trip_l["pickup_lat_approx"], trip_l["dropoff_lat_approx"]],
                "lon": [trip_l["pickup_lon_approx"], trip_l["dropoff_lon_approx"]],
                "color": ["#16a34a", "#dc2626"]})
            st.map(pts, latitude="lat", longitude="lon", color="color", size=120)
            st.caption("🟢 điểm đón · 🔴 điểm đến")

            # Mô hình dự đoán lại cùng lộ trình cho 24 giờ trong ngày
            rows = [dict(trip_l, pickup_hour=h, pickup_minute_bucket=0)
                    for h in range(24)]
            curve = model.predict_df(pd.DataFrame(rows)) / 60
            st.markdown("**Cùng lộ trình, nếu đón vào giờ khác trong ngày (phút)**")
            st.bar_chart(pd.DataFrame({"phút": curve}, index=range(24)),
                         x_label="giờ đón", y_label="phút")
        else:
            st.info("Chọn thông tin chuyến đi bên trái rồi bấm **Dự đoán**.")

    if st.session_state.get("history"):
        st.subheader("Lịch sử dự đoán")
        st.dataframe(pd.DataFrame(st.session_state["history"]),
                     width="stretch", hide_index=True)

# ---------------------------------------------------------------- Tab 2
with tab2:
    st.write("Tải lên file CSV có đủ 13 cột như bộ Test của BTC; "
             "ứng dụng trả về file `trip_id,trip_duration`.")
    up = st.file_uploader("File CSV", type="csv")
    if up is not None:
        df = pd.read_csv(up, dtype={"trip_id": str})
        pred = model.predict_df(df)
        out = pd.DataFrame({"trip_id": df.get("trip_id", pd.Series(range(len(df)))),
                            "trip_duration": np.round(pred, 2)})
        c1, c2, c3 = st.columns(3)
        c1.metric("Số chuyến", len(out))
        c2.metric("Trung vị (giây)", f"{np.median(pred):.0f}")
        c3.metric("Trung bình (phút)", f"{pred.mean() / 60:.1f}")
        st.dataframe(out.head(20), hide_index=True)
        st.download_button("⬇️ Tải file kết quả", out.to_csv(index=False),
                           "submission.csv", "text/csv")
