"""Folium 인터랙티브 맵 — 정리판

레이어 그룹: [배경] / [행정·도로] / [학교] / [도시개발] / [우선순위]
"""
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from folium.plugins import MiniMap, Fullscreen, MeasureControl, HeatMap
from src.config import (
    DATA_PROCESSED, DATA_EXTERNAL, DATA_GEOJSON, OUTPUT_MAPS,
    DAEJEON_CENTER, DAEJEON_ZOOM, GU_COLORS, ESTIMATED_STUDENT_RATES,
    KDE_HEATMAP_RADIUS, KDE_HEATMAP_BLUR, KDE_HEATMAP_MIN_OPACITY,
)
from src.coords_data import load_redev_projects


# ===== 전역 CSS =====
GLOBAL_CSS = """
<style>
.leaflet-interactive, .leaflet-marker-icon, .leaflet-container, .leaflet-clickable {
    cursor: default !important;
}
.leaflet-grab { cursor: grab !important; }
.leaflet-dragging .leaflet-grab { cursor: grabbing !important; }

.leaflet-popup-content {
    font-size: 13.5px !important;
    line-height: 1.55 !important;
    min-width: 280px !important;
    max-width: 420px !important;
    margin: 12px 14px !important;
}
.leaflet-popup-content .popup-title {
    font-size: 16px; font-weight: 700; margin-bottom: 5px;
}
.leaflet-popup-content small { font-size: 12px; color: #555; }
.leaflet-popup-content hr { margin: 7px 0; border: none; border-top: 1px solid #ddd; }

.leaflet-control-layers, .leaflet-control-layers-expanded {
    font-size: 13px !important;
    line-height: 1.65 !important;
    padding: 8px 10px !important;
    max-height: 80vh;
    overflow-y: auto;
}

/* 상단 제목 박스 */
.map-title-box {
    position: fixed; top: 14px; left: 50%; transform: translateX(-50%);
    z-index: 9999; background: rgba(255,255,255,0.96);
    padding: 10px 18px; border-radius: 8px;
    border: 1px solid #bbb; box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    font-family: 'Malgun Gothic', sans-serif;
    text-align: center; min-width: 380px;
}
.map-title-box .title { font-size: 16px; font-weight: 700; color: #222; }
.map-title-box .meta { font-size: 11px; color: #666; margin-top: 3px; }

/* 범례 (접힘/펼침, 폰트 12px, 폭 280px — 한 줄 줄바꿈 없음) */
.legend-box {
    position: fixed; bottom: 20px; right: 20px; z-index: 9999;
    background: white; border: 1px solid #999; border-radius: 6px;
    font-family: 'Malgun Gothic', sans-serif; font-size: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.18);
    width: 280px;
}
.legend-section { border-bottom: 1px solid #eee; }
.legend-section:last-child { border-bottom: none; }
.legend-section summary {
    cursor: pointer; padding: 9px 12px;
    font-weight: 700; font-size: 13px; color: #333;
    list-style: none; user-select: none;
}
.legend-section summary::before {
    content: '▸ '; display: inline-block; transition: transform 0.15s;
}
.legend-section[open] summary::before { content: '▾ '; }
.legend-body { padding: 4px 14px 11px; line-height: 1.75; white-space: nowrap; }
.legend-body .dot {
    display: inline-block; width: 12px; height: 12px;
    border-radius: 50%; vertical-align: middle; margin-right: 5px;
}
.legend-body .ring {
    display: inline-block; width: 12px; height: 12px;
    border-radius: 50%; vertical-align: middle; margin-right: 5px;
    background: transparent; border: 2px solid #555;
}
</style>
"""


# 임박도 그룹 — 모두 기본 OFF (적격성 중심 화면 단순화)
IMMINENCE_GROUPS = [
    {"key": "임박", "label": "🔴 공사중",      "values": ["1_공사중"],
     "hex": "#C0392B", "marker_color": "red",       "show": False},
    {"key": "관리", "label": "🟠 관리처분",    "values": ["2_관리처분"],
     "hex": "#E74C3C", "marker_color": "lightred",  "show": False},
    {"key": "시행", "label": "🟡 사업시행",    "values": ["3_사업시행"],
     "hex": "#E67E22", "marker_color": "orange",    "show": False},
    {"key": "조합", "label": "🟢 조합·추진위", "values": ["4_조합설립", "5_초기"],
     "hex": "#F1C40F", "marker_color": "beige",     "show": False},
    {"key": "입안", "label": "⚪ 입안·미정",   "values": ["6_입안", "9_미정"],
     "hex": "#95A5A6", "marker_color": "lightgray", "show": False},
]
VALUE_TO_GROUP = {v: g for g in IMMINENCE_GROUPS for v in g["values"]}


# ===== Base map =====
def create_base_map():
    """OSM/Gray/Black 3종 base TileLayer 객체 반환 (커스텀 패널이 사용)."""
    m = folium.Map(
        location=DAEJEON_CENTER, zoom_start=DAEJEON_ZOOM,
        tiles=None, control_scale=True, prefer_canvas=True,
    )
    tile_osm = folium.TileLayer("OpenStreetMap", name="OSM",
                                control=False, overlay=False)
    tile_gray = folium.TileLayer("CartoDB positron", name="Gray",
                                 control=False, overlay=False)
    tile_black = folium.TileLayer("CartoDB dark_matter", name="Black",
                                  control=False, overlay=False)
    # 모두 add_to → JS 변수 생성. 페이지 로드 시 setBg('osm')으로 정렬.
    tile_osm.add_to(m)
    tile_gray.add_to(m)
    tile_black.add_to(m)
    return m, {"osm": tile_osm, "gray": tile_gray, "black": tile_black}


# ===== 행정 =====
def add_admin_boundary(m):
    admin_dir = DATA_EXTERNAL / "admin"
    fg_sigungu = folium.FeatureGroup(name="시군구 경계 (5개 구)", show=True)
    fg_dong = folium.FeatureGroup(name="행정동 경계 (82개)", show=False)

    sigungu = admin_dir / "daejeon_signungu.shp"
    if sigungu.exists():
        gdf = gpd.read_file(sigungu).to_crs("EPSG:4326")
        gdf_line = gdf.copy(); gdf_line["geometry"] = gdf.geometry.boundary
        folium.GeoJson(
            gdf_line,
            style_function=lambda x: {
                "color": "#222", "weight": 2.5, "dashArray": "6,4", "opacity": 0.85
            },
            tooltip=folium.GeoJsonTooltip(fields=["SIGUNGU_NM"], aliases=["구:"]),
            popup=folium.GeoJsonPopup(fields=["SIGUNGU_NM"], aliases=["대전광역시"]),
        ).add_to(fg_sigungu)

    dong = admin_dir / "daejeon_dong.shp"
    if dong.exists():
        gdf = gpd.read_file(dong).to_crs("EPSG:4326")
        gdf_line = gdf.copy(); gdf_line["geometry"] = gdf.geometry.boundary
        folium.GeoJson(
            gdf_line,
            style_function=lambda x: {"color": "#5a5a5a", "weight": 1.2, "opacity": 0.65},
            tooltip=folium.GeoJsonTooltip(fields=["ADM_NM"], aliases=["행정동:"]),
            popup=folium.GeoJsonPopup(fields=["ADM_NM"], aliases=["행정동"]),
        ).add_to(fg_dong)

    fg_sigungu.add_to(m); fg_dong.add_to(m)
    return [fg_sigungu, fg_dong]


def add_arterial_roads(m):
    geo_path = DATA_GEOJSON / "daejeon_arterial.geojson"
    fg = folium.FeatureGroup(name="OSM 주간선도로", show=False)
    if geo_path.exists():
        gdf = gpd.read_file(geo_path)
        folium.GeoJson(
            gdf,
            style_function=lambda x: {"color": "#1976D2", "weight": 1.3, "opacity": 0.5},
        ).add_to(fg)
    fg.add_to(m)
    return [fg]


# ===== 학교 — 단일 회색, 초=원/중=사각 =====
SCHOOL_COLOR = "#555555"

# 임박도 → 색 (팝업 영향사업 한 줄에 사용)
_IMMINENCE_COLOR = {
    "1_공사중":   "#C0392B",
    "2_관리처분": "#E74C3C",
    "3_사업시행": "#E67E22",
    "4_조합설립": "#F1C40F",
    "5_초기":     "#F1C40F",
    "6_입안":     "#95A5A6",
    "9_미정":     "#95A5A6",
}


def _load_priority_lookup():
    """integ_df 결과(미래시나리오 CSV)를 학교명 → dict 매핑으로 로드.
    팝업에서 우선순위·영향사업·현행운영 정보 조회용."""
    from src.config import OUTPUT_TABLES, DATA_EXTERNAL
    path = OUTPUT_TABLES / "통학버스운용_미래시나리오.csv"
    if not path.exists():
        return {}, {}, 0
    df = pd.read_csv(path)
    total_schools = len(df)
    lookup = {row["학교명"]: row.to_dict() for _, row in df.iterrows()}

    # 현행 14교 상세
    bus_path = DATA_EXTERNAL / "bus" / "대전_현행통학차량_14개교.csv"
    bus_lookup = {}
    if bus_path.exists():
        bus_df = pd.read_csv(bus_path, encoding="utf-8-sig")
        from src.integrated_priority import match_bus_to_schools
        try:
            schools_min = pd.DataFrame({"학교명": list(lookup.keys())})
            matched, _ = match_bus_to_schools(bus_df, schools_min)
            for _, r in matched.iterrows():
                bus_lookup[r["정식학교명"]] = r.to_dict()
        except Exception as e:
            print(f"  ⚠️  bus lookup 실패: {e}")
    return lookup, bus_lookup, total_schools


def _rank_strong(rank, top_n=30):
    """순위 표기 — 상위 30위 이내면 ★ + 색."""
    if pd.isna(rank):
        return '<span style="color:#999;">미산정</span>'
    rank = int(rank)
    if rank <= 30:
        color = _rank_color(rank)
        return (f'<b style="color:{color};">★ {rank}위</b>')
    return f"{rank}위"


def _build_impact_projects_html(영향사업목록_str, schools_df, school_row):
    """학교의 1km 영향사업 목록을 HTML 한 줄씩 렌더링.

    영향사업목록 컬럼은 "사업명(임박도); 사업명(임박도); ..." 형식.
    더 자세한 정보(거리·세대수·추진현황)는 schools_with_impact의 최근접_사업과
    load_redev_projects()를 조합해서 조회.
    """
    if pd.isna(영향사업목록_str) or not str(영향사업목록_str).strip():
        return '<div style="color:#888;font-size:11.5px;">1km 영향권 내 사업 없음</div>'

    # 사업별 상세 lookup (load_redev_projects)
    from src.coords_data import load_redev_projects
    project_lookup = {p["사업명"]: p for p in load_redev_projects(only_active=True)}

    items = []
    total_inflow_seda = 0
    parts = [p.strip() for p in str(영향사업목록_str).split(";") if p.strip()]
    for part in parts:
        # "사업명(임박도)" 파싱
        if "(" in part and part.endswith(")"):
            name, imm = part.rsplit("(", 1)
            name = name.strip()
            imm = imm.rstrip(")").strip()
        else:
            name, imm = part, ""
        color = _IMMINENCE_COLOR.get(imm, "#888")
        info = project_lookup.get(name, {})
        seda = info.get("세대수")
        seda_str = f"{int(seda):,}세대" if pd.notna(seda) else "세대미정"
        if pd.notna(seda):
            total_inflow_seda += seda
        추진 = info.get("추진현황", "")
        일자 = info.get("추진일자", "")
        구분 = info.get("구분", "")
        items.append(
            f'<div style="margin:3px 0;font-size:11.5px;">'
            f'<span style="color:{color};font-size:14px;line-height:11px;vertical-align:middle;">●</span> '
            f'<b>{name}</b> '
            f'<span style="color:#888;">({구분}·{추진}·{일자})</span><br>'
            f'<span style="color:#666;margin-left:14px;">{seda_str}</span>'
            f'</div>'
        )
    n = len(parts)
    # 학교급별 발생률
    level = school_row.get("학교급", "초")
    rate = ESTIMATED_STUDENT_RATES.get(level, 0)
    inflow = int(round(total_inflow_seda * rate)) if total_inflow_seda else 0
    header = (
        f'<div style="font-size:12px;margin-bottom:3px;">'
        f'<b>1km 영향권 내 사업 {n}건</b> · 예상 학생 유입 약 '
        f'<b style="color:#C0392B;">{inflow:,}명</b>'
        f'</div>'
    )
    return header + "".join(items)


def _build_bus_section(bus_info):
    """현행 14교 운영 정보 HTML."""
    if not bus_info:
        return '<div style="color:#888;font-size:11.5px;">🚌 현행 통학차량 미운영</div>'
    students = int(bus_info.get("이용학생수", 0) or 0)
    cars = int(bus_info.get("차량대수", 0) or 0)
    cost = int(bus_info.get("총비용_천원", 0) or 0)
    warn = ""
    if students == 0:
        warn = ' <span style="color:#C62828;">⚠️ 미가동</span>'
    return (
        f'<div style="font-size:11.5px;">'
        f'<b style="color:#1ABC9C;">🚌 현행 운영</b>{warn}<br>'
        f'<span style="color:#666;">'
        f'이용학생 <b>{students}명</b> · 차량 <b>{cars}대</b> · '
        f'총비용 <b>{cost:,}천원/년</b></span>'
        f'</div>'
    )


def _square_icon_html(size, color):
    return (
        f'<div style="width:{size}px;height:{size}px;'
        f'background:{color};border:1.5px solid white;'
        f'box-shadow:0 1px 2px rgba(0,0,0,0.4);"></div>'
    )


def add_schools(m, schools_df):
    """학교 마커 — popup_builder.build_school_popup 통일 팝업."""
    from src.popup_builder import build_school_popup, build_popup_context
    ctx = build_popup_context(schools_df)
    pri_lookup = ctx["priority_lookup"]
    bus_lookup = ctx["bus_lookup"]
    total_schools = ctx["total_schools"]

    # 카운트 (LayerControl name에 동적 표시)
    n_elem = (schools_df["학교급"] == "초").sum()
    n_mid = (schools_df["학교급"] == "중").sum()
    fg_elem = folium.FeatureGroup(name=f"● 초등학교 ({n_elem}교)", show=True)
    fg_mid = folium.FeatureGroup(name=f"■ 중학교 ({n_mid}교)", show=True)

    for _, s in schools_df.iterrows():
        if pd.isna(s.get("lat")):
            continue
        students = max(int(s.get("학생수합계", 1)), 1)
        radius = 3 + np.sqrt(students) / 12

        popup_html = build_school_popup(s, ctx)

        if s["학교급"] == "초":
            folium.CircleMarker(
                location=[s["lat"], s["lon"]], radius=radius,
                color="white", weight=1.2,
                fill=True, fillColor=SCHOOL_COLOR, fillOpacity=0.85,
                tooltip=f"{s['학교명']} · {students:,}명",
                popup=folium.Popup(popup_html, max_width=400),
            ).add_to(fg_elem)
        else:
            side = int(radius * 2)
            folium.Marker(
                location=[s["lat"], s["lon"]],
                icon=folium.DivIcon(
                    html=_square_icon_html(side, SCHOOL_COLOR),
                    icon_size=(side, side),
                    icon_anchor=(side // 2, side // 2),
                    class_name="empty",
                ),
                tooltip=f"{s['학교명']} · {students:,}명",
                popup=folium.Popup(popup_html, max_width=400),
            ).add_to(fg_mid)

    fg_elem.add_to(m); fg_mid.add_to(m)
    return [fg_elem, fg_mid]


# ===== 재개발 =====
def _redev_popup_html(p, in_1km_count, in_1km_students, in_15km_count, in_15km_students):
    seda = p.get("세대수", None)
    est_elem = int(round(seda * ESTIMATED_STUDENT_RATES["초"])) if pd.notna(seda) else None
    est_mid = int(round(seda * ESTIMATED_STUDENT_RATES["중"])) if pd.notna(seda) else None
    seda_disp = f"{int(seda):,}세대" if pd.notna(seda) else "<i style='color:#888;'>미정</i>"
    est_disp = (
        f"초 약 <b>{est_elem:,}명</b> · 중 약 <b>{est_mid:,}명</b>"
        if est_elem is not None else "<i style='color:#888;'>세대수 미정</i>"
    )
    chokjin = p.get("촉진지구", False)
    chokjin_str = ""
    if chokjin in (True, "True", "true", 1, "1"):
        chokjin_str = "<span style='background:#FFE082;padding:1px 5px;border-radius:3px;'>촉진지구</span> "
    chujin = p.get("추진지구", "")
    if pd.notna(chujin) and str(chujin).strip() and str(chujin) != "nan":
        chokjin_str += f"<small>({chujin})</small>"

    return f"""
    <div class="popup-title" style="color:#C0392B;">{p['사업명']}</div>
    <small>{p.get('구','')} {p.get('동','') or ''} · {p.get('구분','')} · {p.get('추진현황','')}</small>
    <hr>
    <table style="font-size:13px;line-height:1.6;">
      <tr><td style="color:#666;padding-right:10px;">규모</td>
          <td><b style="font-size:15px;">{seda_disp}</b></td></tr>
      <tr><td style="color:#666;">추진일자</td><td>{p.get('추진일자','-')}</td></tr>
      <tr><td style="color:#666;">임박도</td><td><b>{p.get('통학영향_임박도','-')}</b></td></tr>
      <tr><td style="color:#666;">지구</td><td>{chokjin_str or '-'}</td></tr>
    </table>
    <hr>
    <b>예상 학생 발생</b><br>
    <div style="font-size:13px;">{est_disp}</div>
    <hr>
    <b>1km 영향권 기존 학교</b>
    <div style="font-size:13px;">{in_1km_count}교 / 재학생 <b>{in_1km_students:,}명</b></div>
    <b>1.5km 영향권 기존 학교</b>
    <div style="font-size:13px;">{in_15km_count}교 / 재학생 <b>{in_15km_students:,}명</b></div>
    """


def add_redev_projects(m, projects=None, schools_df=None):
    """임박도 5개 그룹 = 단일 FeatureGroup에 (마커 + 1km + 1.5km) 통합 add.

    LayerControl에서 임박도 5개 항목만 노출, 토글 시 마커·영향권 모두 함께 켜짐/꺼짐.
    Returns:
        [(FeatureGroup, n_markers), ...] — 마커 개수 검증용
    """
    if projects is None:
        projects = load_redev_projects(only_active=True)

    fgs = {}
    marker_counts = {g["key"]: 0 for g in IMMINENCE_GROUPS}
    for g in IMMINENCE_GROUPS:
        fgs[g["key"]] = folium.FeatureGroup(name=g["label"], show=g["show"])

    for p in projects:
        lat, lon = p["좌표"]
        imm = p.get("통학영향_임박도", "")
        g = VALUE_TO_GROUP.get(imm)
        if g is None:
            continue
        fg = fgs[g["key"]]

        in_1km_count = in_1km_students = 0
        in_15km_count = in_15km_students = 0
        if schools_df is not None:
            in_1km = schools_df[
                (schools_df["최근접_사업"] == p["사업명"]) &
                (schools_df["최근접_거리m"] <= 1000)
            ]
            in_15km = schools_df[
                (schools_df["최근접_사업"] == p["사업명"]) &
                (schools_df["최근접_거리m"] <= 1500)
            ]
            in_1km_count = len(in_1km); in_1km_students = int(in_1km["학생수합계"].sum())
            in_15km_count = len(in_15km); in_15km_students = int(in_15km["학생수합계"].sum())

        # 같은 FG에 1.5km → 1km → 마커 순서로 add (마커가 가장 위)
        folium.Circle(
            location=[lat, lon], radius=1500,
            color=g["hex"], weight=1.5, dashArray="6,4",
            fill=False,
            tooltip=f"{p['사업명']} · 1.5km",
        ).add_to(fg)
        folium.Circle(
            location=[lat, lon], radius=1000,
            color=g["hex"], weight=1.2,
            fill=True, fillColor=g["hex"], fillOpacity=0.08,
            tooltip=f"{p['사업명']} · 1.0km",
        ).add_to(fg)
        seda = p.get("세대수")
        seda_disp = f"{int(seda):,}" if pd.notna(seda) else "미정"
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(
                _redev_popup_html(p, in_1km_count, in_1km_students,
                                  in_15km_count, in_15km_students),
                max_width=460,
            ),
            tooltip=f"{p['사업명']} ({seda_disp}세대)",
            icon=folium.Icon(color=g["marker_color"], icon="building", prefix="fa"),
        ).add_to(fg)
        marker_counts[g["key"]] += 1

    out = []
    for g in IMMINENCE_GROUPS:
        fgs[g["key"]].add_to(m)
        out.append(fgs[g["key"]])

    # 임박도별 FG 5개를 key dict로도 반환 (커스텀 패널에서 개별 참조)
    fg_by_key = {g["key"]: fgs[g["key"]] for g in IMMINENCE_GROUPS}
    return out, marker_counts, fg_by_key


# ===== 우선순위 3종 + 현행 14교 =====
def _star_icon(color, size=34):
    return folium.DivIcon(
        html=(
            f'<div style="display:flex;align-items:center;justify-content:center;'
            f'width:{size}px;height:{size}px;background:{color};'
            f'border:2.5px solid white;border-radius:50%;'
            f'box-shadow:0 2px 6px rgba(0,0,0,0.5);'
            f'color:white;font-size:{int(size*0.55)}px;font-weight:900;">★</div>'
        ),
        icon_size=(size, size), icon_anchor=(size // 2, size // 2),
        class_name="empty",
    )


def _diamond_icon(color, size=30):
    return folium.DivIcon(
        html=(
            f'<div style="display:flex;align-items:center;justify-content:center;'
            f'width:{size}px;height:{size}px;background:{color};'
            f'border:2px solid white;transform:rotate(45deg);'
            f'box-shadow:0 2px 5px rgba(0,0,0,0.5);">'
            f'<span style="transform:rotate(-45deg);color:white;'
            f'font-size:{int(size*0.42)}px;font-weight:800;">◆</span></div>'
        ),
        icon_size=(size, size), icon_anchor=(size // 2, size // 2),
        class_name="empty",
    )


def _hexagon_icon(color, size=32):
    """⬢ 육각형 (clip-path)"""
    return folium.DivIcon(
        html=(
            f'<div style="display:flex;align-items:center;justify-content:center;'
            f'width:{size}px;height:{size}px;background:{color};'
            f'clip-path:polygon(50% 0%, 95% 25%, 95% 75%, 50% 100%, 5% 75%, 5% 25%);'
            f'-webkit-clip-path:polygon(50% 0%, 95% 25%, 95% 75%, 50% 100%, 5% 75%, 5% 25%);'
            f'box-shadow:0 2px 5px rgba(0,0,0,0.5);'
            f'color:white;font-size:{int(size*0.45)}px;font-weight:900;">⬢</div>'
        ),
        icon_size=(size, size), icon_anchor=(size // 2, size // 2),
        class_name="empty",
    )


def _rank_color(rank):
    """공통 색 스킴: 1~5 빨강 / 6~15 주황 / 16~30 노랑"""
    if rank <= 5: return "#C0392B"
    if rank <= 15: return "#E67E22"
    return "#F1C40F"


def _bus_icon(size=28):
    return folium.DivIcon(
        html=(
            f'<div style="display:flex;align-items:center;justify-content:center;'
            f'width:{size}px;height:{size}px;background:#1ABC9C;'
            f'border:2px solid white;border-radius:6px;'
            f'box-shadow:0 2px 5px rgba(0,0,0,0.45);'
            f'color:white;font-size:{int(size*0.55)}px;">🚌</div>'
        ),
        icon_size=(size, size), icon_anchor=(size // 2, size // 2),
        class_name="empty",
    )


def _common_priority_popup(r, score_col, rank_col, label, color):
    """우선순위 마커 공통 팝업 HTML"""
    return f"""
    <div class="popup-title" style="color:{color};">
      <span style="font-size:20px;font-weight:900;">#{int(r[rank_col])}</span>
      &nbsp;{r['학교명']}
    </div>
    <small>{r['구']} {r.get('동','') or ''} · {r['학교급']}등학교 ·
        학생수 {int(r['학생수']):,}명</small>
    <hr>
    <table style="font-size:13px;line-height:1.65;">
      <tr><td style="color:#666;padding-right:10px;">{label}</td>
          <td><b style="font-size:16px;color:{color};">{r[score_col]:.4f}</b></td></tr>
      <tr><td style="color:#666;">도시개발영향 점수/순위</td>
          <td>{r['도시개발영향점수']:.3f} · #{int(r['도시개발영향순위'])}</td></tr>
      <tr><td style="color:#666;">적격성 점수/순위</td>
          <td>{r['적격성점수']:.3f} · #{int(r['적격성순위'])}</td></tr>
      <tr><td style="color:#666;">미래시나리오 점수/순위</td>
          <td>{r['미래시나리오점수']:.3f} · #{int(r['미래시나리오순위'])}</td></tr>
      <tr><td style="color:#666;">현행 운영</td>
          <td><b style="color:{'#1ABC9C' if r.get('현행운영여부')=='Y' else '#999'};">
              {'Y (현행 14교)' if r.get('현행운영여부')=='Y' else 'N'}</b></td></tr>
    </table>
    <hr>
    <b>영향사업 ({int(r.get('영향사업수',0))}건)</b>
    <div style="font-size:11.5px;color:#555;max-height:100px;overflow:auto;">
      {r.get('영향사업목록','') or '-'}
    </div>
    """


def add_eligibility_top30(m, csv_path=None, schools_df=None):
    """★ 신규 검토 대상 30교 (현행 14교 제외, 적격성 상위, 기본 ON).
    통일 팝업(build_school_popup) 사용."""
    from src.config import OUTPUT_TABLES
    from src.popup_builder import build_school_popup, build_popup_context
    csv_path = csv_path or OUTPUT_TABLES / "신규검토대상_상위30교.csv"
    if not csv_path.exists():
        print(f"⚠️  신규 검토 대상 CSV 없음 → 레이어 스킵")
        return []
    df = pd.read_csv(csv_path)
    if schools_df is None:
        schools_df = pd.read_csv(DATA_PROCESSED / "schools_with_impact.csv")
    ctx = build_popup_context(schools_df)
    fg = folium.FeatureGroup(
        name=f"★ 신규 검토 대상 30교 (적격성 상위·미운영) ({len(df)}교)", show=True
    )
    for _, r in df.iterrows():
        rank = int(r["미운영순위"])
        color = _rank_color(rank)
        # 학교명으로 schools_df에서 풀데이터 가져와 팝업 빌드
        full_row = schools_df[schools_df["학교명"] == r["학교명"]]
        if len(full_row) > 0:
            sch = full_row.iloc[0].to_dict()
        else:
            sch = r.to_dict()
            sch["학생수합계"] = sch.get("학생수합계", r.get("학생수합계", 0))
        popup = build_school_popup(sch, ctx)
        folium.Marker(
            location=[r["lat"], r["lon"]],
            icon=_star_icon(color, size=34),
            tooltip=f"★ #{rank} 신규 {r['학교명']} · {r['적격성점수']:.3f}",
            popup=folium.Popup(popup, max_width=400),
        ).add_to(fg)
    fg.add_to(m)
    return [fg]


def add_devimpact_top30(m, csv_path=None):
    """🏗️ 도시개발 영향 상위 30교 (보조, 기본 OFF)
    아이콘 ◆ 다이아몬드. 색상: 1~5 빨강 / 6~15 주황 / 16~30 노랑"""
    from src.config import OUTPUT_TABLES
    csv_path = csv_path or OUTPUT_TABLES / "도시개발영향_상위30교.csv"
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    fg = folium.FeatureGroup(name=f"◆ 도시개발 영향 상위 30교 ({len(df)}교)", show=False)
    for _, r in df.iterrows():
        rank = int(r["도시개발영향순위"])
        color = _rank_color(rank)
        popup = _common_priority_popup(r, "도시개발영향점수", "도시개발영향순위",
                                        "도시개발영향 점수", color)
        folium.Marker(
            location=[r["lat"], r["lon"]],
            icon=_diamond_icon(color, size=30),
            tooltip=f"◆ #{rank} 도시개발영향 {r['학교명']} · {r['도시개발영향점수']:.3f}",
            popup=folium.Popup(popup, max_width=460),
        ).add_to(fg)
    fg.add_to(m)
    return [fg]


def add_integrated_top30(m, csv_path=None):
    """⬢ 미래 시나리오 적격성 상위 30교 (보조, 기본 OFF)
    아이콘 ⬢ 육각형. 색상: 1~5 빨강 / 6~15 주황 / 16~30 노랑"""
    from src.config import OUTPUT_TABLES
    csv_path = csv_path or OUTPUT_TABLES / "통학버스운용_미래시나리오_상위30교.csv"
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    fg = folium.FeatureGroup(name="⬢ 미래 시나리오 적격성 상위 30교", show=False)
    for _, r in df.iterrows():
        rank = int(r["미래시나리오순위"])
        color = _rank_color(rank)
        popup = _common_priority_popup(r, "미래시나리오점수", "미래시나리오순위",
                                        "미래 시나리오 적격성 점수", color)
        folium.Marker(
            location=[r["lat"], r["lon"]],
            icon=_hexagon_icon(color, size=32),
            tooltip=f"⬢ #{rank} 미래 {r['학교명']} · {r['미래시나리오점수']:.3f}",
            popup=folium.Popup(popup, max_width=460),
        ).add_to(fg)
    fg.add_to(m)
    return [fg]


def add_current_bus14(m, schools_df):
    """🚌 현행 통학차량 운영 (15교). 통일 팝업 사용."""
    from src.integrated_priority import _load_bus14, match_bus_to_schools
    from src.popup_builder import build_school_popup, build_popup_context
    bus_df = _load_bus14()
    if bus_df is None:
        return []
    matched, unmatched = match_bus_to_schools(bus_df, schools_df)
    if unmatched:
        print(f"  ⚠️  현행 매칭 실패: {unmatched}")
    ctx = build_popup_context(schools_df)
    fg = folium.FeatureGroup(name=f"🚌 현행 통학차량 운영 ({len(matched)}교)", show=True)
    for _, r in matched.iterrows():
        if pd.isna(r["lat"]):
            continue
        full_row = schools_df[schools_df["학교명"] == r["정식학교명"]]
        if len(full_row) > 0:
            sch = full_row.iloc[0].to_dict()
        else:
            continue
        popup = build_school_popup(sch, ctx)
        students = int(r["이용학생수"]) if pd.notna(r["이용학생수"]) else 0
        folium.Marker(
            location=[r["lat"], r["lon"]],
            icon=_bus_icon(size=30),
            tooltip=f"🚌 {r['정식학교명']} · 이용 {students}명",
            popup=folium.Popup(popup, max_width=400),
        ).add_to(fg)
    fg.add_to(m)
    return [fg]


# ===== KDE 히트맵 (탐색용, 점수 산식 미반영) =====
# 노랑→오렌지→빨강 5단계 — 학생/도시개발 두 KDE 동일 적용 (가중 의미만 다름)
KDE_GRADIENT = {
    0.2: "#FFF3B0",
    0.4: "#FECF7B",
    0.6: "#FB8C00",
    0.8: "#E64A19",
    1.0: "#B71C1C",
}


def build_kde_data():
    """folium.plugins.HeatMap에 넣을 데이터 2종 생성

    - kde_students: 학생수 가중 (학생 분포 밀도)
    - kde_redev: 재개발 진행 사업의 세대수 가중 (도시개발 압력)
    """
    schools = pd.read_csv(DATA_PROCESSED / "schools_geocoded.csv")
    kde_students = [
        [row["lat"], row["lon"], float(row["학생수합계"])]
        for _, row in schools.iterrows()
        if pd.notna(row["lat"]) and row["학생수합계"] > 0
    ]

    redev = pd.read_csv(DATA_PROCESSED / "redev_projects_geocoded.csv")
    redev_active = redev[
        (redev["상태"] == "진행")
        & (redev["세대수"].notna())
        & (redev["lat"].notna())
    ]
    kde_redev = [
        [row["lat"], row["lon"], float(row["세대수"])]
        for _, row in redev_active.iterrows()
    ]
    return kde_students, kde_redev


def add_kde_layers(m):
    """KDE 히트맵 2종 추가 — 모두 기본 OFF, 탐색용.
    학생/도시개발 동일 RED_GRADIENT (가중 데이터만 다름)."""
    kde_students, kde_redev = build_kde_data()

    fg_kde_students = folium.FeatureGroup(
        name="🌡️ 학생 분포 (학생수 가중)", overlay=True, show=False
    )
    heatmap_students = HeatMap(
        kde_students,
        radius=KDE_HEATMAP_RADIUS,
        blur=KDE_HEATMAP_BLUR,
        min_opacity=KDE_HEATMAP_MIN_OPACITY,
        gradient=KDE_GRADIENT,
    )
    heatmap_students.add_to(fg_kde_students)
    m.add_child(fg_kde_students)

    fg_kde_redev = folium.FeatureGroup(
        name="🏗️ 도시개발 압력 (재개발 세대수)", overlay=True, show=False
    )
    heatmap_redev = HeatMap(
        kde_redev,
        radius=KDE_HEATMAP_RADIUS,
        blur=KDE_HEATMAP_BLUR,
        min_opacity=KDE_HEATMAP_MIN_OPACITY,
        gradient=KDE_GRADIENT,
    )
    heatmap_redev.add_to(fg_kde_redev)
    m.add_child(fg_kde_redev)

    return [fg_kde_students, fg_kde_redev], [heatmap_students, heatmap_redev]


# ===== 경사도 음영 ImageOverlay =====
def add_slope_overlay(m):
    """대전 경사도 음영 ImageOverlay (4326 PNG, base64 임베드).

    src/slope_overlay.py가 미리 생성한 PNG + bounds JSON 사용.
    base64로 임베드해 단일 HTML 자족 (GitHub Pages 호환).
    """
    import base64
    import json

    png_path = DATA_PROCESSED / "대전_slope_overlay.png"
    bounds_path = DATA_PROCESSED / "대전_slope_overlay_bounds.json"

    if not png_path.exists() or not bounds_path.exists():
        print("   ⚠️ slope_overlay PNG/bounds 없음. 'python -m src.slope_overlay' 먼저 실행.")
        return None

    b = json.loads(bounds_path.read_text(encoding="utf-8"))
    png_bytes = png_path.read_bytes()
    b64 = base64.b64encode(png_bytes).decode("ascii")
    data_uri = f"data:image/png;base64,{b64}"

    fg = folium.FeatureGroup(
        name="🏔️ 경사도 음영", overlay=True, show=False
    )
    folium.raster_layers.ImageOverlay(
        image=data_uri,
        bounds=[[b["south"], b["west"]], [b["north"], b["east"]]],
        opacity=1.0,         # PNG 자체 alpha 활용
        interactive=False,
        cross_origin=False,
        zindex=400,          # 베이스 위, 마커 아래
    ).add_to(fg)
    m.add_child(fg)
    return fg


# ===== 제목 박스 =====
def add_title_box(m, schools_df, projects=None):
    """제목 박스. 사업 건수는 동적으로 '진행 N건' 표기."""
    n_schools = len(schools_df) if schools_df is not None else 0
    if projects is None:
        projects = load_redev_projects(only_active=True)
    n_redev = len(projects)
    html = f"""
    <div class="map-title-box">
      <div class="title">대전광역시교육청 통학지원 운영방안 분석</div>
      <div class="meta">데이터 기준일자: 2026.3.31 ·
          학교 <b>{n_schools}</b>교 · 도시정비사업 진행 <b>{n_redev}</b>건</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(html))
    return m


# ===== 범례 (접힘/펼침) =====
def add_legend(m):
    legend = """
    <div class="legend-box">
      <details class="legend-section" open>
        <summary>🎯 신규 검토 대상 30교 (적격성 상위, 미운영)</summary>
        <div class="legend-body">
          <span style="display:inline-block;width:14px;height:14px;background:#C0392B;border-radius:50%;border:1.5px solid white;box-shadow:0 1px 2px rgba(0,0,0,0.3);vertical-align:middle;margin-right:8px;text-align:center;font-size:9px;line-height:14px;color:white;">★</span>진한 빨강 — 상위 1~5교<br>
          <span style="display:inline-block;width:14px;height:14px;background:#E67E22;border-radius:50%;border:1.5px solid white;box-shadow:0 1px 2px rgba(0,0,0,0.3);vertical-align:middle;margin-right:8px;text-align:center;font-size:9px;line-height:14px;color:white;">★</span>주황 — 상위 6~15교<br>
          <span style="display:inline-block;width:14px;height:14px;background:#F1C40F;border-radius:50%;border:1.5px solid white;box-shadow:0 1px 2px rgba(0,0,0,0.3);vertical-align:middle;margin-right:8px;text-align:center;font-size:9px;line-height:14px;color:white;">★</span>노랑 — 상위 16~30교<br>
          <span style="color:#777;font-size:11px;">현행 운영 14교 제외 · 신규 통학지원 검토 대상</span>
        </div>
      </details>
      <details class="legend-section" open>
        <summary>🚌 현행 통학차량 운영 14교</summary>
        <div class="legend-body">
          <span style="display:inline-block;width:14px;height:14px;background:#1ABC9C;border-radius:50%;border:1.5px solid white;vertical-align:middle;margin-right:8px;"></span>진한 청록 — 운영 중
        </div>
      </details>
      <details class="legend-section" open>
        <summary>🏫 학교</summary>
        <div class="legend-body">
          <span style="display:inline-block;width:12px;height:12px;background:#555;border-radius:50%;vertical-align:middle;margin-right:8px;"></span>초등학교
          <span style="display:inline-block;width:12px;height:12px;background:#555;vertical-align:middle;margin-left:14px;margin-right:8px;"></span>중학교<br>
          <span style="color:#777;font-size:11px;">크기 ∝ √(학생수)</span>
        </div>
      </details>
      <details class="legend-section">
        <summary>🏗️ 도시개발 (보조)</summary>
        <div class="legend-body">
          <div style="margin-bottom:5px;">
            <span style="display:inline-block;width:12px;height:12px;background:#C0392B;transform:rotate(45deg);vertical-align:middle;margin-right:10px;"></span>도시개발 영향 상위 30교<br>
            <span style="display:inline-block;width:12px;height:12px;background:#C0392B;clip-path:polygon(50% 0%, 95% 25%, 95% 75%, 50% 100%, 5% 75%, 5% 25%);vertical-align:middle;margin-right:10px;"></span>미래 시나리오 상위 30교
          </div>
          <div style="color:#777;font-size:11px;margin-top:5px;">[임박도]</div>
          <span class="dot" style="background:#C0392B;"></span>공사중<br>
          <span class="dot" style="background:#E74C3C;"></span>관리처분<br>
          <span class="dot" style="background:#E67E22;"></span>사업시행<br>
          <span class="dot" style="background:#F1C40F;"></span>조합·추진위<br>
          <span class="dot" style="background:#95A5A6;"></span>입안·미정<br>
          <span style="color:#777;font-size:11px;">영향권: 1km 채움 / 1.5km 점선</span>
        </div>
      </details>
      <details class="legend-section">
        <summary>📍 베이스 (옵션)</summary>
        <div class="legend-body" style="font-size:10px;line-height:1.55;">
          <span style="display:inline-block;width:60px;height:8px;vertical-align:middle;margin-right:6px;background:linear-gradient(to right,#2D8B43,#91C266,#D4B36A,#B85C2A,#8B1A1A);"></span>
          🏔️ 경사도 음영 (옵션): 녹색=평지 → 갈색=중 → 빨강=20°+
        </div>
      </details>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend))
    return m


# ===== 그룹화된 LayerControl =====
def add_grouped_layer_control(m, groups):
    """folium.plugins.GroupedLayerControl 단독 사용.

    GroupedLayerControl은 baseLayer(타일)는 자동으로 별도 처리하므로
    일반 LayerControl을 추가하지 않음 (중복 컨트롤 제거).
    """
    from folium.plugins import GroupedLayerControl
    # base tile toggle을 위해 LayerControl이 먼저 필요 (folium 0.15 요구사항)
    folium.LayerControl(collapsed=False, position="topright").add_to(m)
    # GroupedLayerControl은 overlay 그룹만 처리
    GroupedLayerControl(
        groups=groups,
        exclusive_groups=False,
        collapsed=False,
        position="topright",
    ).add_to(m)
    return m


# ===== Build =====
def build_map(schools_df, output_filename="대전_외부환경분석_도시개발.html",
              include_tram=False, verify=True):
    """커스텀 HTML+JS 레이어 패널 통합 단일 맵."""
    projects = load_redev_projects(only_active=True)
    m, base_tiles = create_base_map()

    m.get_root().html.add_child(folium.Element(GLOBAL_CSS))

    admin_fgs = add_admin_boundary(m)
    # 도로 레이어 제거 (CSV/GeoJSON 로드 안 함)
    redev_fgs, marker_counts, redev_by_key = add_redev_projects(
        m, projects=projects, schools_df=schools_df
    )
    school_fgs = add_schools(m, schools_df)
    integ_fgs = add_integrated_top30(m)
    dev_fgs = add_devimpact_top30(m)
    elig_fgs = add_eligibility_top30(m, schools_df=schools_df)
    bus_fgs = add_current_bus14(m, schools_df)
    kde_fgs, kde_heatmaps = add_kde_layers(m)
    fg_slope_overlay = add_slope_overlay(m)

    if include_tram:
        try:
            from src.viz_map_tram import add_tram_layer
            add_tram_layer(m, schools_df=schools_df, show_buffers=True)
        except ImportError:
            pass

    # 커스텀 패널 (LayerControl 폐기)
    _add_custom_panel(
        m, base_tiles,
        admin_fgs=admin_fgs, school_fgs=school_fgs,
        elig_fgs=elig_fgs, bus_fgs=bus_fgs, integ_fgs=integ_fgs,
        dev_fgs=dev_fgs, redev_by_key=redev_by_key,
        kde_fgs=kde_fgs, kde_heatmaps=kde_heatmaps,
        slope_overlay_fg=fg_slope_overlay,
        schools_df=schools_df,
    )

    MiniMap(toggle_display=True, position="bottomleft").add_to(m)
    Fullscreen(position="topleft").add_to(m)
    MeasureControl(position="topleft", primary_length_unit="meters").add_to(m)

    add_title_box(m, schools_df, projects)
    add_legend(m)

    if verify:
        priority_fgs_all = integ_fgs + dev_fgs + elig_fgs + bus_fgs
        _verify_counts(projects, redev_fgs, marker_counts, priority_fgs_all)

    out = OUTPUT_MAPS / output_filename
    m.save(str(out))
    return out


def _add_custom_panel(m, base_tiles, admin_fgs, school_fgs,
                       elig_fgs, bus_fgs, integ_fgs, dev_fgs,
                       redev_by_key, schools_df, kde_fgs=None,
                       kde_heatmaps=None, slope_overlay_fg=None):
    """커스텀 HTML+CSS+JS 레이어 패널 + window 노출 + 초기 ON/OFF 정렬."""
    fg_sigungu = admin_fgs[0] if len(admin_fgs) > 0 else None
    fg_dong = admin_fgs[1] if len(admin_fgs) > 1 else None
    fg_elem = school_fgs[0] if len(school_fgs) > 0 else None
    fg_mid = school_fgs[1] if len(school_fgs) > 1 else None
    fg_elig = elig_fgs[0] if elig_fgs else None
    fg_bus14 = bus_fgs[0] if bus_fgs else None
    fg_integ = integ_fgs[0] if integ_fgs else None
    fg_dev = dev_fgs[0] if dev_fgs else None
    fg_gongsa = redev_by_key.get("임박")
    fg_gwanli = redev_by_key.get("관리")
    fg_siheng = redev_by_key.get("시행")
    fg_johap = redev_by_key.get("조합")
    fg_ipan = redev_by_key.get("입안")
    fg_kde_students = kde_fgs[0] if kde_fgs and len(kde_fgs) > 0 else None
    fg_kde_redev = kde_fgs[1] if kde_fgs and len(kde_fgs) > 1 else None
    fg_slope = slope_overlay_fg
    hm_students = kde_heatmaps[0] if kde_heatmaps and len(kde_heatmaps) > 0 else None
    hm_redev = kde_heatmaps[1] if kde_heatmaps and len(kde_heatmaps) > 1 else None
    hm_students_v = hm_students.get_name() if hm_students else None
    hm_redev_v = hm_redev.get_name() if hm_redev else None

    # 카운트 동적
    n_elem = int((schools_df["학교급"] == "초").sum())
    n_mid = int((schools_df["학교급"] == "중").sum())

    # 14교 카운트 (실제 매칭된 마커 수)
    n_bus14 = 0
    try:
        from src.integrated_priority import match_bus_to_schools, _load_bus14
        bus_df = _load_bus14()
        if bus_df is not None:
            matched, _ = match_bus_to_schools(bus_df, schools_df)
            n_bus14 = len(matched)
    except Exception:
        pass

    # FG 객체 변수명 (folium이 JS에 노출, get_name() = 'feature_group_xxxx')
    def n(fg):
        return fg.get_name() if fg is not None else "null"

    map_var = m.get_name()
    osm_v = base_tiles["osm"].get_name()
    gray_v = base_tiles["gray"].get_name()
    black_v = base_tiles["black"].get_name()

    # === folium의 var name 들을 window 글로벌로 명시 노출 ===
    # (folium은 IIFE 안에서 var 선언하므로 외부 JS에서 직접 접근 불가)
    all_fg_names = [
        n(fg) for fg in [fg_sigungu, fg_dong, fg_elem, fg_mid,
                          fg_elig, fg_bus14, fg_integ, fg_dev,
                          fg_gongsa, fg_gwanli, fg_siheng, fg_johap, fg_ipan,
                          fg_kde_students, fg_kde_redev, fg_slope]
        if fg is not None
    ]
    hm_names = [v for v in [hm_students_v, hm_redev_v] if v]
    expose_lines = "\n".join(
        f"  try {{ window['{nm}'] = {nm}; }} catch(e) {{}}"
        for nm in [osm_v, gray_v, black_v] + all_fg_names + hm_names
    )
    m.get_root().script.add_child(folium.Element(
        f"// expose layer vars to window\n{expose_lines}"
    ))

    # KDE 슬라이더용 명시 노출 — folium IIFE 종료 후 setTimeout 500ms로 보장
    fg_kde_students_v = n(fg_kde_students)
    fg_kde_redev_v = n(fg_kde_redev)
    if hm_students_v and hm_redev_v:
        m.get_root().script.add_child(folium.Element(f"""
  // KDE HeatLayer / FeatureGroup 명시 노출 (페이지 로드 후 500ms)
  setTimeout(function() {{
    try {{
      window.heatLayer_students = {hm_students_v};
      window.heatLayer_redev = {hm_redev_v};
      window.fg_kde_students_ref = {fg_kde_students_v};
      window.fg_kde_redev_ref = {fg_kde_redev_v};
      console.log("[KDE] HeatLayers exposed:",
                  !!window.heatLayer_students,
                  !!window.heatLayer_redev);
    }} catch(e) {{
      console.error("[KDE] HeatLayer 노출 실패:", e);
    }}
  }}, 500);
"""))

    # 초기 ON/OFF 정렬 — show=False FG들은 page-load 시 removeLayer
    # (folium은 add_to만 하면 visible 시작이라 OFF FG는 명시적으로 끔)
    off_layers = []
    if fg_dong: off_layers.append(n(fg_dong))
    if fg_integ: off_layers.append(n(fg_integ))
    if fg_dev: off_layers.append(n(fg_dev))
    for fg in [fg_gongsa, fg_gwanli, fg_siheng, fg_johap, fg_ipan]:
        if fg: off_layers.append(n(fg))
    if fg_kde_students: off_layers.append(n(fg_kde_students))
    if fg_kde_redev: off_layers.append(n(fg_kde_redev))
    if fg_slope: off_layers.append(n(fg_slope))

    # 베이스 그룹 내 경사도 음영 체크박스 (fg_slope가 있을 때만)
    slope_chk_html = (
        f'<label><input type="checkbox" onchange="lpLayer(this,\'{n(fg_slope)}\')"> '
        f'🏔️ 경사도 (음영)</label>'
        if fg_slope is not None else ''
    )

    panel_html = f"""
<div id="layer-panel">
  <div class="lp-group">
    <div class="lp-header" onclick="lpToggle(this)">📍 베이스 <span class="arrow">▼</span></div>
    <div class="lp-body">
      <label><input type="checkbox" checked onchange="lpLayer(this,'{n(fg_sigungu)}')"> 시군구 경계</label>
      <label><input type="checkbox" onchange="lpLayer(this,'{n(fg_dong)}')"> 행정동 경계</label>
      {slope_chk_html}
      <div class="lp-bg">배경:
        <label><input type="radio" name="bg" checked onchange="lpBg('osm')"> OSM</label>
        <label><input type="radio" name="bg" onchange="lpBg('gray')"> Gray</label>
        <label><input type="radio" name="bg" onchange="lpBg('black')"> Black</label>
        <label><input type="radio" name="bg" onchange="lpBg('off')"> 끄기</label>
      </div>
    </div>
  </div>
  <div class="lp-group">
    <div class="lp-header" onclick="lpToggle(this)">🏫 학교 <span class="arrow">▼</span></div>
    <div class="lp-body">
      <label><input type="checkbox" checked onchange="lpLayer(this,'{n(fg_elem)}')"> ● 초등학교 ({n_elem}교)</label>
      <label><input type="checkbox" checked onchange="lpLayer(this,'{n(fg_mid)}')"> ■ 중학교 ({n_mid}교)</label>
    </div>
  </div>
  <div class="lp-group">
    <div class="lp-header" onclick="lpToggle(this)">🎯 통학지원 분석 <span class="arrow">▼</span></div>
    <div class="lp-body">
      <label><input type="checkbox" checked onchange="lpLayer(this,'{n(fg_elig)}')"> ★ 적격성 상위 30교</label>
      <label><input type="checkbox" checked onchange="lpLayer(this,'{n(fg_bus14)}')"> 🚌 현행 통학차량 ({n_bus14}교)</label>
      <label><input type="checkbox" onchange="lpLayer(this,'{n(fg_integ)}')"> ⬢ 미래 시나리오 적격성 상위 30교</label>
    </div>
  </div>
  <div class="lp-group">
    <div class="lp-header" onclick="lpToggle(this)">🏗️ 도시개발 (보조) <span class="arrow">▼</span></div>
    <div class="lp-body">
      <label><input type="checkbox" onchange="lpLayer(this,'{n(fg_dev)}')"> ◆ 도시개발 영향 상위 30교</label>
      <label><input type="checkbox" onchange="lpLayer(this,'{n(fg_gongsa)}')"> 🔴 공사중</label>
      <label><input type="checkbox" onchange="lpLayer(this,'{n(fg_gwanli)}')"> 🟠 관리처분</label>
      <label><input type="checkbox" onchange="lpLayer(this,'{n(fg_siheng)}')"> 🟡 사업시행</label>
      <label><input type="checkbox" onchange="lpLayer(this,'{n(fg_johap)}')"> 🟢 조합·추진위</label>
      <label><input type="checkbox" onchange="lpLayer(this,'{n(fg_ipan)}')"> ⚪ 입안·미정</label>
    </div>
  </div>
  <div class="lp-group">
    <div class="lp-header" onclick="lpToggle(this)">🔥 밀도 분석 <span class="arrow">▼</span></div>
    <div class="lp-body">
      <label><input type="checkbox" onchange="lpLayer(this,'{n(fg_kde_students)}')"> 🌡️ 학생 분포 (학생수 가중)</label>
      <label><input type="checkbox" onchange="lpLayer(this,'{n(fg_kde_redev)}')"> 🏗️ 도시개발 압력 (재개발 세대수)</label>
      <div class="lp-kde-slider">
        <div class="lp-kde-row">
          <span>대역폭 (radius)</span>
          <span><strong id="kde-radius-val">{KDE_HEATMAP_RADIUS}</strong> px</span>
        </div>
        <input type="range" min="10" max="60" value="{KDE_HEATMAP_RADIUS}" step="2"
               id="kde-radius-slider"
               oninput="updateKdeRadius(this.value)">
        <div class="lp-kde-ticks">
          <span>좁게</span>
          <span>넓게</span>
        </div>
      </div>
    </div>
  </div>
</div>
<style>
#layer-panel {{
  position: absolute; top: 12px; right: 12px; z-index: 9999;
  background: white; padding: 8px 10px; border: 1px solid #888;
  border-radius: 6px; font-family: 'Malgun Gothic', sans-serif; font-size: 12px;
  min-width: 240px; max-width: 280px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}}
.lp-group {{ border-bottom: 1px solid #ddd; padding: 4px 0; }}
.lp-group:last-child {{ border-bottom: none; }}
.lp-header {{
  font-weight: 700; cursor: pointer; user-select: none;
  padding: 4px 2px; font-size: 12.5px;
}}
.lp-header:hover {{ background: #f5f5f5; }}
.lp-body {{ padding: 3px 0 5px 6px; display: block; }}
.lp-body label {{
  display: block; cursor: pointer; line-height: 1.7;
  font-size: 12px;
}}
.lp-body input[type=checkbox], .lp-body input[type=radio] {{
  margin-right: 4px; vertical-align: middle;
}}
.lp-bg {{
  margin-top: 4px; padding-top: 4px; border-top: 1px dashed #eee;
  font-size: 11.5px; color: #555;
}}
.lp-bg label {{ display: inline-block; margin-right: 6px; }}
.arrow {{ float: right; color: #666; font-size: 11px; }}
.lp-kde-slider {{
  margin-top: 10px; padding-top: 8px; border-top: 1px solid #eee;
}}
.lp-kde-row {{
  display: flex; justify-content: space-between;
  font-size: 11px; color: #555; margin-bottom: 4px;
}}
.lp-kde-ticks {{
  display: flex; justify-content: space-between;
  font-size: 9px; color: #999; margin-top: 2px;
}}
#layer-panel input[type="range"] {{
  -webkit-appearance: none;
  width: 100%; height: 4px;
  background: linear-gradient(to right, #FFF3B0, #B71C1C);
  border-radius: 2px; outline: none; cursor: pointer;
}}
#layer-panel input[type="range"]::-webkit-slider-thumb {{
  -webkit-appearance: none;
  width: 14px; height: 14px;
  background: #C0392B; border-radius: 50%;
  cursor: pointer; border: 2px solid white;
  box-shadow: 0 1px 3px rgba(0,0,0,0.3);
}}
#layer-panel input[type="range"]::-moz-range-thumb {{
  width: 14px; height: 14px;
  background: #C0392B; border-radius: 50%;
  cursor: pointer; border: 2px solid white;
}}
</style>
<script>
function lpToggle(header) {{
  var body = header.nextElementSibling;
  var arrow = header.querySelector('.arrow');
  if (body.style.display === 'none') {{
    body.style.display = 'block';
    arrow.textContent = '▼';
  }} else {{
    body.style.display = 'none';
    arrow.textContent = '▶';
  }}
}}
function _lpResolve(name) {{
  if (!name || name === 'null') return null;
  if (window[name]) return window[name];
  try {{ return eval(name); }} catch (e) {{ return null; }}
}}
function lpLayer(checkbox, layerName) {{
  var layer = _lpResolve(layerName);
  if (!layer) return;
  if (checkbox.checked) {{
    {map_var}.addLayer(layer);
  }} else {{
    {map_var}.removeLayer(layer);
  }}
}}
function updateKdeRadius(value) {{
  var radius = parseInt(value);
  var blur = Math.round(radius * 0.75);
  var lbl = document.getElementById('kde-radius-val');
  if (lbl) lbl.textContent = radius;

  function apply(layer) {{
    if (!layer || typeof layer.setOptions !== 'function') return false;
    try {{
      layer.setOptions({{radius: radius, blur: blur}});
      if (typeof layer.redraw === 'function') layer.redraw();
      return true;
    }} catch(e) {{
      console.error('[KDE] setOptions 실패:', e);
      return false;
    }}
  }}

  var applied = 0;
  // 1차: 직접 노출된 HeatLayer
  if (apply(window.heatLayer_students)) applied++;
  if (apply(window.heatLayer_redev)) applied++;

  // 2차 폴백: FeatureGroup 자식 layer 순회
  if (applied === 0) {{
    [window.fg_kde_students_ref, window.fg_kde_redev_ref].forEach(function(fg) {{
      if (!fg || typeof fg.eachLayer !== 'function') return;
      fg.eachLayer(function(layer) {{
        if (apply(layer)) applied++;
      }});
    }});
  }}

  console.log('[KDE] radius:', radius, 'applied:', applied);
}}
// base tile lazy resolve (window 노출 이후 호출 보장)
function _lpBgGet() {{
  return {{
    osm: _lpResolve('{osm_v}'),
    gray: _lpResolve('{gray_v}'),
    black: _lpResolve('{black_v}')
  }};
}}
function lpBg(key) {{
  var tiles = _lpBgGet();
  Object.keys(tiles).forEach(function(k) {{
    var t = tiles[k];
    if (t) {{ try {{ {map_var}.removeLayer(t); }} catch(e) {{}} }}
  }});
  if (key !== 'off' && tiles[key]) {{
    {map_var}.addLayer(tiles[key]);
  }}
}}
function _lpInit() {{
  // 기본 OFF로 지정된 FG들 제거
  var offLayers = {off_layers!r};
  offLayers.forEach(function(name) {{
    var l = _lpResolve(name);
    if (l) {{ try {{ {map_var}.removeLayer(l); }} catch(e) {{}} }}
  }});
  // base tile: OSM만 활성
  lpBg('osm');
}}
// folium의 IIFE 끝나는 시점 보장: load + setTimeout 둘 다
if (document.readyState === 'complete') {{
  setTimeout(_lpInit, 50);
}} else {{
  window.addEventListener('load', function() {{ setTimeout(_lpInit, 50); }});
}}
</script>
"""
    m.get_root().html.add_child(folium.Element(panel_html))
    return m


def _verify_counts(projects, redev_fgs, marker_counts, priority_fgs):
    """건수 일치 검증 + 우선순위 레이어 검증."""
    from src.config import DATA_PROCESSED
    csv_path = DATA_PROCESSED / "redev_projects_geocoded.csv"
    df_csv = pd.read_csv(csv_path)
    n_csv_total = len(df_csv)
    n_csv_geo_ok = df_csv["lat"].notna().sum()
    n_csv_active = ((df_csv["lat"].notna()) & (df_csv["상태"] == "진행")).sum()
    n_projects = len(projects)
    n_markers_total = sum(marker_counts.values())

    print("\n   === 건수 일치 검증 ===")
    print(f"   CSV 총 행수:                  {n_csv_total}")
    print(f"   CSV 지오코딩 성공:            {n_csv_geo_ok}")
    print(f"   CSV 진행 + 좌표 있음:         {n_csv_active}")
    print(f"   load_redev_projects() 반환:    {n_projects}")
    print(f"   임박도별 마커 합:              {n_markers_total}")
    for g in IMMINENCE_GROUPS:
        print(f"     · {g['label']:<14}  {marker_counts[g['key']]:>3}건")
    if n_csv_active == n_projects == n_markers_total:
        print(f"   ✅ 진행 사업 / 마커 / 코드 반환 모두 {n_projects}건으로 일치")
    else:
        print(f"   ⚠️  불일치! {n_csv_active} vs {n_projects} vs {n_markers_total}")

    n_priority = len(priority_fgs)
    print(f"   우선순위 FeatureGroup 등록:    {n_priority}개")
    if n_priority == 0:
        print(f"   ⚠️  우선순위 레이어 미생성! (CSV 경로 확인)")


if __name__ == "__main__":
    df = pd.read_csv(DATA_PROCESSED / "schools_with_impact.csv")
    out = build_map(df, include_tram=False)
    print(f"✅ 통합 맵 저장 → {out}")
