"""
make_report.py - Sinh báo cáo PDF (tối đa 3 trang A4) từ kết quả của train.py.
Mọi con số trong báo cáo được lấy tự động từ outputs/metrics.json và
outputs/eda/eda_summary.json, nên luôn khớp với lần chạy thật.

    python make_report.py           ->  outputs/BaoCao_UNETI18.pdf
Sửa tên thành viên và phân công trong team.json trước khi chạy.
"""
import json
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

HERE = os.path.dirname(os.path.abspath(__file__))
F = os.path.join(HERE, "assets", "fonts")
pdfmetrics.registerFont(TTFont("DV", os.path.join(F, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DVB", os.path.join(F, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DVI", os.path.join(F, "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DVB", italic="DVI")

M = json.load(open("outputs/metrics.json", encoding="utf-8"))
E = json.load(open("outputs/eda/eda_summary.json", encoding="utf-8"))
T = json.load(open("team.json", encoding="utf-8"))
team = T.get("team", M["team"])
ins, dur = E["insights"], E["duration"]

BLUE = colors.HexColor("#1d4ed8")
base = ParagraphStyle("b", fontName="DV", fontSize=8.4, leading=10.8, spaceAfter=2.5)
small = ParagraphStyle("s", parent=base, fontSize=7.4, leading=9.2)
h1 = ParagraphStyle("h1", parent=base, fontName="DVB", fontSize=13, leading=16,
                    alignment=TA_CENTER, textColor=BLUE, spaceAfter=1)
sub = ParagraphStyle("sub", parent=base, alignment=TA_CENTER, fontSize=8.5,
                     textColor=colors.HexColor("#374151"), spaceAfter=5)
h2 = ParagraphStyle("h2", parent=base, fontName="DVB", fontSize=10, leading=13,
                    textColor=BLUE, spaceBefore=5, spaceAfter=2.5)
cap = ParagraphStyle("cap", parent=small, alignment=TA_CENTER,
                     textColor=colors.HexColor("#4b5563"))


def n(x, d=0):
    """Định dạng số kiểu Việt Nam: 1.234,5"""
    s = f"{x:,.{d}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def P(t, st=base):
    return Paragraph(t, st)


def bullets(items, st=base):
    return [Paragraph(f"• {t}", ParagraphStyle("li", parent=st, leftIndent=8,
                                                firstLineIndent=-7)) for t in items]


def img(name, w):
    p = f"outputs/eda/{name}"
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(p).getSize()
    return Image(p, width=w, height=w * ih / iw)


def table(data, widths, header=True, hl_row=None):
    data = [[P(str(c), small) for c in r] for r in data]
    t = Table(data, colWidths=widths, hAlign="LEFT")
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")))
    if hl_row is not None:
        st.append(("BACKGROUND", (0, hl_row), (-1, hl_row), colors.HexColor("#fef3c7")))
    t.setStyle(TableStyle(st))
    return t


W = A4[0] - 3.0 * cm
story = []

# =========================================================== TRANG 1
story += [P("BÁO CÁO KẾT QUẢ – DỰ ĐOÁN THỜI GIAN CHUYẾN TAXI NEW YORK", h1),
          P(f"Olympic Trí tuệ nhân tạo UNETI 2026 – Vòng 2 · Đội {team} · "
            f"Bộ Test {team}_TEST_3000", sub)]

story.append(P("1. Mô tả dữ liệu và khám phá (EDA)", h2))
tr, te = E["train_shape"], E["test_shape"]
story.append(P(
    f"Bộ Train gồm <b>{n(tr[0])} bản ghi × {tr[1]} cột</b>; bộ Test của đội gồm "
    f"<b>{n(te[0])} bản ghi × {te[1]} cột</b> (không có <i>trip_duration</i>). "
    f"Có 11 cột số (int/float) và 2 cột chuỗi (<i>trip_id</i>, <i>store_and_fwd_flag</i>). "
    f"Số ô thiếu: Train {n(E['missing_train_total'])}, Test {n(E['missing_test_total'])}; "
    f"trip_id trùng giữa Train và Test: {E['overlap_trip_id']}. Bộ Test làm tròn tọa độ 0,01° "
    f"(~1 km) và khoảng cách theo bậc 0,5 km, trong khi Train chi tiết hơn; đội đưa mọi tập về "
    f"cùng định dạng của Test, nên mỗi điểm đón/đến là một <b>ô lưới</b>. Biến mục tiêu có trung vị {n(dur['median'])} giây "
    f"(~{n(dur['median'] / 60, 1)} phút), trung bình {n(dur['mean'])} giây, nhỏ nhất "
    f"{n(dur['min'])} giây, lớn nhất {n(dur['max'])} giây; có {n(dur['n_lt_60s'])} chuyến "
    f"dưới 60 giây và {n(dur['n_gt_22h'])} chuyến trên 22 giờ."))

story.append(Table([[img("fig1_duration_dist.png", W * 0.62),
                     img("fig3_distance_vs_duration.png", W * 0.36)]],
                   colWidths=[W * 0.63, W * 0.37],
                   style=[("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                          ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
story.append(P("Hình 1 – Phân phối thời gian chuyến (gốc và log) · "
               "Hình 2 – Khoảng cách chim bay và thời gian", cap))
story.append(Table([[img("fig2_heatmap_dow_hour.png", W * 0.62),
                     img("fig4_speed_by_hour.png", W * 0.36)]],
                   colWidths=[W * 0.63, W * 0.37],
                   style=[("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                          ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
story.append(P("Hình 3 – Thời gian trung vị theo thứ × giờ · "
               "Hình 4 – Tốc độ trung vị theo giờ", cap))

story.append(P("<b>Nhận xét rút ra:</b>"))
story += bullets([
    f"<b>Phân phối lệch phải mạnh</b> (hệ số lệch {n(dur['skew_raw'], 1)}); sau biến đổi "
    f"log(1+x) chỉ còn {n(dur['skew_log'], 2)} – gần chuẩn (Hình 1). Vì RMSLE chính là RMSE trên "
    f"thang log, đội huấn luyện mô hình trực tiếp trên log(1+trip_duration).",
    f"<b>Khoảng cách là yếu tố mạnh nhất</b>: tương quan giữa khoảng cách và log(thời gian) là "
    f"{n(ins['corr_dist_logdur'], 2)} (Hình 2), nhưng độ phân tán rất lớn ở cùng một khoảng cách "
    f"→ cần thêm thông tin vị trí và thời điểm.",
    f"<b>Giao thông theo giờ rõ rệt</b>: giờ chậm nhất là {ins['slowest_hour']}h (trung vị "
    f"{n(ins['slowest_hour_median_min'], 1)} phút), nhanh nhất {ins['fastest_hour']}h "
    f"({n(ins['fastest_hour_median_min'], 1)} phút). Tốc độ ngày thường thấp nhất lúc "
    f"{ins['speed_min_hour']}h ({n(ins['speed_min_kmh'], 1)} km/h) và cao nhất lúc "
    f"{ins['speed_max_hour']}h ({n(ins['speed_max_kmh'], 1)} km/h); cuối tuần nhanh hơn "
    f"({n(ins['speed_weekend_kmh'], 1)} so với {n(ins['speed_weekday_kmh'], 1)} km/h) (Hình 3, 4).",
    f"<b>Tập trung không gian</b>: {n(ins['share_pickup_manhattan'] * 100, 1)}% điểm đón nằm "
    f"trong khung Manhattan → thống kê theo ô lưới có đủ dữ liệu để học tốc độ từng khu vực.",
])

story.append(PageBreak())

# =========================================================== TRANG 2
story.append(P("2. Tiền xử lý và xây dựng đặc trưng", h2))
o = M["outliers_full"]
story.append(P(
    "Toàn bộ biến đổi nằm trong <b>một module duy nhất</b> (<i>src/features.py</i>) và được "
    "gọi chung cho Train, Validation, Test và ứng dụng, bảo đảm nhất quán."))
story += bullets([
    "<b>Dữ liệu thiếu / sai miền</b>: điền giá trị mặc định (khoảng cách thiếu → tính lại "
    "bằng haversine), chặn số khách về 1–6, tọa độ ngoài NYC kéo về biên. Không xóa dòng "
    "nào của Test.",
    f"<b>Ngoại lai</b> (chỉ loại khỏi phần <i>huấn luyện</i>): trên 22 giờ "
    f"({n(o['qua_22_gio'])}), dưới 10 giây ({n(o['duoi_10_giay'])}), tọa độ ngoài NYC "
    f"({n(o['toa_do_ngoai_NYC'])}), tốc độ chim bay trên 120 km/h ({n(o['toc_do_tren_120kmh'])}), "
    f"dưới 1 km nhưng trên 3 giờ ({n(o['duoi_1km_ma_tren_3_gio'])}) – tổng "
    f"{n(o['tong_loai'])} dòng ({n(o['ty_le_loai'] * 100, 2)}%). Dữ liệu đã được BTC giới hạn "
    f"thời gian trong khoảng {n(dur['min'])}–{n(dur['max'])} giây nên rất ít dòng bị loại. "
    f"Tập Validation <b>giữ nguyên</b> để RMSLE kiểm định phản ánh đúng dữ liệu BTC chấm.",
    "<b>Đồng bộ định dạng</b>: làm tròn tọa độ của mọi tập về 0,01° và khoảng cách về bậc "
    "0,5 km như bộ Test, để mô hình học đúng loại dữ liệu sẽ bị chấm.",
    "<b>Mã hóa</b>: store_and_fwd_flag Y/N → 1/0; vendor, thứ, giờ giữ dạng số nguyên "
    "(mô hình cây tự chia nhánh); trip_id <b>không</b> dùng làm đầu vào.",
    "<b>Đặc trưng thời gian</b>: time_slot (giờ×4 + mốc 15 phút), week_slot, cuối tuần, giờ "
    "cao điểm, đêm, 5 khung giờ, sin/cos của giờ và thứ.",
    "<b>Đặc trưng hình học</b>: khoảng cách haversine, Manhattan, Manhattan theo trục phố "
    "xoay 29°, log khoảng cách, hướng di chuyển, độ lệch x/y (km), tọa độ xoay, cờ cùng ô lưới, "
    "cờ đón/trả tại JFK, LaGuardia, Newark.",
    "<b>Target encoding</b>: trung bình log(thời gian) và số chuyến theo "
    "ô đón, ô đến, <i>cặp ô</i>, cặp ô × khung giờ, cặp ô × cuối tuần, ô × giờ, khoảng cách × "
    "time_slot, vùng 3 km × giờ…; làm trơn với 20 mẫu ảo. Trên tập huấn luyện dùng "
    "<b>out-of-fold 5 phần</b> để không rò rỉ nhãn; Validation/Test dùng bảng thống kê học từ Train.",
])
USE_TE = "đầy đủ" in M.get("final_model", "")
story.append(P(f"Tổng cộng <b>{M['n_features']} đặc trưng</b> đầu vào cho mô hình cuối"
               + ("" if USE_TE else " (cấu hình thắng không dùng nhóm target encoding, "
                  "xem mục 3)") + "."))

story.append(P("3. Xây dựng và đánh giá mô hình", h2))
story.append(P(
    f"Tách ngẫu nhiên <b>80% huấn luyện / 20% kiểm định</b> ({n(M['n_val'])} chuyến) từ Train "
    f"(seed 42). Tất cả mô hình học trên y = log(1+trip_duration), dự đoán xong đổi ngược bằng "
    f"exp(x)−1, và được đánh giá bằng <b>RMSLE</b> trên tập kiểm định."))
rows = [["Mô hình", "Mô tả", "RMSLE kiểm định", "Điểm AI quy đổi"]]
hl = None
for i, r in enumerate(M["results"], 1):
    rows.append([r["model"], r["mo_ta"], n(r["rmsle_val"], 4), n(r["diem_AI_uoc_tinh"], 2)])
    if r["model"] == M.get("final_model", M["best_model"]):
        hl = i
story.append(table(rows, [W * 0.22, W * 0.46, W * 0.16, W * 0.16], hl_row=hl))
story.append(P("Điểm AI quy đổi = 70 / (1 + RMSLE). Dòng tô vàng là cấu hình được chọn.", cap))
story.append(Spacer(1, 3))
p = M["params"]
story.append(P(
    f"Siêu tham số LightGBM: learning_rate {n(p['learning_rate'], 2)}, num_leaves "
    f"{p['num_leaves']}, min_data_in_leaf {p['min_data_in_leaf']}, feature_fraction "
    f"{n(p['feature_fraction'], 1)}, bagging {n(p['bagging_fraction'], 1)}, L2 "
    f"{n(p['lambda_l2'], 1)}; số cây chọn bằng early stopping (dừng khi 100 cây liên tiếp "
    f"không cải thiện)."))
story.append(Table([[img("fig8_model_compare.png", W * 0.48),
                     img("fig7_learning_curve.png", W * 0.48)]],
                   colWidths=[W * 0.5, W * 0.5],
                   style=[("LEFTPADDING", (0, 0), (-1, -1), 0)]))
story.append(P("Hình 5 – So sánh RMSLE các mô hình · Hình 6 – Đường học của LightGBM", cap))

story.append(PageBreak())

# =========================================================== TRANG 3
res = {r["model"]: r for r in M["results"]}
m0 = M["results"][0]["rmsle_val"]
fin = res[M.get("final_model", M["best_model"])]
story.append(P("4. Lựa chọn mô hình cuối cùng", h2))
story.append(P(
    f"Đội chọn <b>{fin['model']}</b> với RMSLE kiểm định <b>{n(fin['rmsle_val'], 4)}</b>, giảm "
    f"{n((1 - fin['rmsle_val'] / m0) * 100, 1)}% so với mô hình cơ sở "
    f"(tương đương {n(fin['diem_AI_uoc_tinh'], 2)}/70 điểm AI). Lý do:"))
te_note = []
if "M2 LightGBM (cơ bản)" in res and "M3 LightGBM (đầy đủ)" in res and not USE_TE:
    te_note = [f"Thêm target encoding (M3: {n(res['M3 LightGBM (đầy đủ)']['rmsle_val'], 4)}) "
               f"không tốt hơn M2 ({n(res['M2 LightGBM (cơ bản)']['rmsle_val'], 4)}): cây quyết định "
               f"đã tự học được tốc độ theo vị trí từ tọa độ, nên đội chọn cấu hình <b>đơn giản hơn</b>."]
story += bullets(te_note + [
    "Quan hệ giữa thời gian với khoảng cách, giờ và vị trí là <b>phi tuyến và có tương tác</b> "
    "(cùng 5 km nhưng giờ cao điểm ở Midtown chậm hơn nhiều so với ban đêm) – mô hình cây "
    "gradient boosting học được, còn Ridge thì không.",
    "LightGBM huấn luyện nhanh trên CPU với hơn 1 triệu dòng, có early stopping chống quá "
    "khớp, và tối ưu trực tiếp RMSE trên log, tức đúng thước đo RMSLE của BTC.",
    f"Mô hình cuối được <b>huấn luyện lại trên toàn bộ Train</b> (số cây = 1,1 × số cây tối ưu "
    f"khi kiểm định) và lấy trung bình {M['n_seeds']} mô hình khác seed để giảm phương sai. "
    f"Dự đoán được chặn trong khoảng hợp lý quan sát được ở Train.",
    f"File nộp được kiểm tra tự động: đúng 2 cột, đủ {n(E['test_shape'][0])} dòng, "
    f"trip_id khớp bộ Test, không trùng, không trống, mọi giá trị dương và hữu hạn.",
])
story.append(Table([[img("fig6_feature_importance.png", W * 0.46),
                     img("fig9_test_pred_dist.png", W * 0.5)]],
                   colWidths=[W * 0.48, W * 0.52],
                   style=[("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                          ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
story.append(P("Hình 7 – Đặc trưng quan trọng nhất · Hình 8 – Phân phối dự đoán trên Test "
               "so với Train (khớp nhau, không lệch hệ thống)", cap))

ea = M["error_analysis"]
d = ea["rmsle_by_dist"]
story.append(P("5. Hạn chế và hướng cải thiện", h2))
story += bullets([
    "<b>Tọa độ bị làm tròn ~1 km và khoảng cách 0,5 km</b>: hai chuyến trong cùng một ô có thể "
    "khác hẳn nhau. Lỗi cao nhất ở chuyến rất ngắn – RMSLE nhóm 0 km là "
    f"{n(d.get('0 km', 0), 3)}, trong khi nhóm 2,5–5 km chỉ {n(d.get('2,5-5', 0), 3)}.",
    (f"<b>Sai số không thể giảm</b>: các chuyến có đầu vào giống hệt nhau vẫn có thời gian "
     f"khác nhau; ước lượng từ các nhóm trùng đầu vào cho RMSLE tối thiểu khoảng "
     f"<b>{n(ea['noise_floor'], 3)}</b>; mô hình đạt {n(fin['rmsle_val'], 3)}, "
     + ("tức đã xấp xỉ mức này (ước lượng chỉ gần đúng), nên khó cải thiện thêm bằng mô hình."
        if fin['rmsle_val'] < ea['noise_floor'] + 0.01 else
        f"còn cách khoảng {n(fin['rmsle_val'] - ea['noise_floor'], 3)} – đây là dư địa tối đa "
        f"có thể cải thiện.")
     if ea.get("noise_floor") else
     f"<b>Ngoại lai</b>: {n(ea['share_val_gt_6h'] * 100, 2)}% chuyến kiểm định dài trên 6 giờ."),
    "<b>Thiếu thông tin ngoại cảnh</b>: không có dữ liệu thời tiết, sự kiện, tai nạn, tuyến "
    "đường thực tế (OSRM) hay ngày lễ cụ thể (chỉ biết tháng và thứ).",
    "<b>Hướng cải thiện</b>: thêm CatBoost/XGBoost và lấy trung bình có trọng số, tối ưu siêu "
    "tham số bằng Optuna, dùng K-fold cho mô hình cuối, thêm khoảng cách đường bộ nếu được phép.",
])

story.append(P("6. Ứng dụng AI mini và phân công", h2))
story.append(P(
    "Ứng dụng Streamlit (<i>app.py</i>) nạp mô hình đã lưu và gọi đúng pipeline tạo file nộp: "
    "người dùng chọn điểm đón/đến (15 địa điểm có sẵn hoặc tự nhập tọa độ), tháng, thứ, giờ, "
    "phút, số khách; ứng dụng tự tính khoảng cách, hiển thị thời gian dự đoán bằng <b>giây</b> "
    "kèm quy đổi phút, bản đồ lộ trình, biểu đồ dự đoán theo 24 giờ và cho phép dự đoán cả "
    "file CSV."))
rows = [["Thành viên", "Nhiệm vụ"]] + [[m["name"], m["role"]] for m in T["members"]]
story.append(KeepTogether(table(rows, [W * 0.32, W * 0.68])))

doc = SimpleDocTemplate(f"outputs/BaoCao_{team}.pdf", pagesize=A4,
                        leftMargin=1.5 * cm, rightMargin=1.5 * cm,
                        topMargin=1.2 * cm, bottomMargin=1.2 * cm,
                        title=f"Báo cáo Vòng 2 - {team}", author=team)


def footer(c, d_):
    c.saveState()
    c.setFont("DV", 7)
    c.setFillColor(colors.HexColor("#6b7280"))
    c.drawRightString(A4[0] - 1.5 * cm, 0.7 * cm, f"Đội {team} · Trang {d_.page}/3")
    c.restoreState()


doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(f"Đã tạo outputs/BaoCao_{team}.pdf")
