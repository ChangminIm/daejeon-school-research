"""학교 팝업 단일 빌더 — 모든 학교 관련 마커가 동일 팝업 사용.

build_school_popup(school_row, ctx) → HTML 문자열
"""
import pandas as pd
from src.config import ESTIMATED_STUDENT_RATES

SCHOOL_HEADER_COLOR = "#34495E"

# Phase B-2 사후 검증 상수 (slope_hypothesis.py 산출 결과, src/extract_slope.py 보고)
SLOPE_AVG_ALL = 4.4    # 대전 학교 243교 slope_300m_mean 평균 (실측 4.58° → 표기 4.4°)
SLOPE_AVG_BUS14 = 8.0  # 현행 통학차량 14교 평균 (실측 8.05°)

# 임박도별 점 색상 (1km 영향사업 한 줄)
_IMMINENCE_COLOR = {
    "1_공사중":   "#C0392B",
    "2_관리처분": "#E74C3C",
    "3_사업시행": "#E67E22",
    "4_조합설립": "#F1C40F",
    "5_초기":     "#F1C40F",
    "6_입안":     "#95A5A6",
    "9_미정":     "#95A5A6",
}

POPUP_CSS = """
<style>
.sp-popup { min-width: 320px; max-width: 380px; font-family: 'Malgun Gothic', sans-serif;
            font-size: 13px; line-height: 1.5;
            max-height: 500px; overflow-y: auto; }
.sp-popup::-webkit-scrollbar { width: 6px; }
.sp-popup::-webkit-scrollbar-track { background: transparent; }
.sp-popup::-webkit-scrollbar-thumb { background: #c8ccd4; border-radius: 3px; }
.sp-popup::-webkit-scrollbar-thumb:hover { background: #a0a4ac; }
.sp-popup hr { border: none; border-top: 1px solid #eee; margin: 8px 0; }
.sp-popup .sp-title { font-size: 16px; font-weight: 700; color: #34495E; }
.sp-popup .sp-rank-badge {
    background: #FFE082; color: #C0392B; padding: 2px 8px; border-radius: 4px;
    font-weight: 900; font-size: 15px; margin-right: 6px;
}
.sp-popup .sp-sub { font-size: 11px; color: #777; margin-top: 2px; }
.sp-popup .sp-section { font-weight: 700; font-size: 13.5px; margin-bottom: 3px; }
.sp-popup table { font-size: 13px; line-height: 1.6; }
.sp-popup td.k { color: #666; padding-right: 10px; }
.sp-popup .sp-star { color: #C0392B; font-weight: 900; }
.sp-popup .sp-warn { color: #C62828; }
.sp-popup .sp-bus-yes { color: #1ABC9C; font-weight: 700; }
.sp-popup .sp-bus-no  { color: #999; }
.sp-popup .sp-proj-row { margin: 3px 0; font-size: 12px; padding-left: 4px; }
/* 영향사업 details 블록 */
.sp-popup .sp-impact-details > summary {
    cursor: pointer; list-style: none;
    user-select: none;
}
.sp-popup .sp-impact-details > summary::-webkit-details-marker { display: none; }
.sp-popup .sp-impact-details > summary::marker { content: ''; }
.sp-popup .sp-impact-details[open] > summary { margin-bottom: 4px; }
.sp-popup .sp-impact-body { padding-left: 2px; }
</style>
"""


def _rank_strong(rank, total=243, threshold=30):
    """순위 표기. 상위 N위 이내면 ★ 강조."""
    if pd.isna(rank):
        return '<span style="color:#999;">미산정</span>'
    rank = int(rank)
    if rank <= threshold:
        return f'<span class="sp-star">★ {rank}위</span> / {total}교'
    return f'{rank}위 / {total}교'


def _build_impact_block(영향사업목록_str, school_level, project_lookup):
    """1km 영향사업 details 블록 — ≤5건은 펼침, ≥6건은 접힘."""
    if pd.isna(영향사업목록_str) or not str(영향사업목록_str).strip():
        return (
            '<div class="sp-section">🏗️ 1km 영향권 내 사업</div>'
            '<div style="color:#888;font-size:12px;">영향권 내 사업 없음</div>'
        )

    parts = [p.strip() for p in str(영향사업목록_str).split(";") if p.strip()]
    count = len(parts)

    items = []
    total_seda = 0
    for part in parts:
        if "(" in part and part.endswith(")"):
            name, imm = part.rsplit("(", 1)
            name, imm = name.strip(), imm.rstrip(")").strip()
        else:
            name, imm = part, ""
        color = _IMMINENCE_COLOR.get(imm, "#888")
        info = project_lookup.get(name, {})
        seda = info.get("세대수")
        seda_str = f"{int(seda):,}세대" if pd.notna(seda) else "세대미정"
        if pd.notna(seda):
            total_seda += seda
        구분 = info.get("구분", "")
        추진 = info.get("추진현황", "")
        items.append(
            f'<div class="sp-proj-row">'
            f'<span style="color:{color};font-size:13px;">●</span> '
            f'<b>{name}</b> '
            f'<span style="color:#888;font-size:11px;">({구분}·{추진})</span><br>'
            f'<span style="color:#999;font-size:11px;margin-left:14px;">{seda_str}</span>'
            f'</div>'
        )

    rate = ESTIMATED_STUDENT_RATES.get(school_level, 0)
    inflow = int(round(total_seda * rate)) if total_seda else 0

    open_attr = "open" if count <= 5 else ""
    arrow = "" if count <= 5 else ' <span style="color:#888;font-weight:500;font-size:12px;">— 펼치기 ▶</span>'
    summary_html = (
        f'🏗️ 1km 영향권 내 사업 ({count}건, '
        f'예상 학생 유입 <span style="color:#C0392B;">{inflow:,}명</span>){arrow}'
    )

    return (
        f'<details class="sp-impact-details" {open_attr}>'
        f'<summary class="sp-section">{summary_html}</summary>'
        f'<div class="sp-impact-body">{"".join(items)}</div>'
        f'</details>'
    )


def _build_bus_row(bus_info):
    """현행 통학차량 운영 상세."""
    if not bus_info:
        return '<div class="sp-bus-no">미운영</div>'
    students = int(bus_info.get("이용학생수", 0) or 0)
    cars = int(bus_info.get("차량대수", 0) or 0)
    cost_total = int(bus_info.get("총비용_천원", 0) or 0)
    per_cap = bus_info.get("학생1인당_총비용_천원")
    car_detail = bus_info.get("차량상세", "")
    bigo = bus_info.get("비고", "")
    warn = ""
    if students == 0:
        warn = ' <span class="sp-warn">⚠️ 운영 미가동</span>'
    elif pd.notna(per_cap) and per_cap >= 10000:
        warn = f' <span class="sp-warn">⚠️ 1인당 {per_cap:,.0f}천원</span>'
    per_cap_str = f"{per_cap:,.0f}천원" if pd.notna(per_cap) else "-"
    bigo_html = ""
    if pd.notna(bigo) and str(bigo).strip() and str(bigo) != "nan":
        bigo_html = f'<div style="color:#777;font-size:11.5px;margin-top:3px;">비고: {bigo}</div>'
    return (
        f'<div class="sp-bus-yes">🚌 운영 중</div>{warn}'
        f'<div style="font-size:12.5px;margin-top:3px;">'
        f'이용학생 <b>{students}명</b> · 차량 <b>{cars}대</b>'
        f'<span style="color:#888;"> ({car_detail})</span><br>'
        f'총비용 <b>{cost_total:,}천원/년</b><br>'
        f'학생1인당 <b>{per_cap_str}/년</b>'
        f'</div>{bigo_html}'
    )


def build_school_popup(school_row, ctx):
    """학교 팝업 HTML 생성.

    Args:
        school_row: 학교 데이터 한 행 (Series 또는 dict).
            필수: 학교명, 학교급, 구, 학생수합계(또는 학생수), lat, lon
            선택: 주소, 동, 설립, 영향사업목록, 최근접_거리m, 최근접_사업
        ctx: dict
            - priority_lookup: {학교명: {적격성순위, 도시개발영향순위, 미래시나리오순위,
                                       영향사업목록, 전체순위, 미운영순위, ...}}
            - bus_lookup: {정식학교명: {이용학생수, 차량대수, 총비용_천원, ...}}
            - project_lookup: {사업명: {세대수, 구분, 추진현황, ...}}
            - total_schools: 전체 학교 수 (default 243)
    Returns:
        HTML 문자열
    """
    s = school_row if isinstance(school_row, dict) else school_row.to_dict()
    name = s["학교명"]
    pri = ctx.get("priority_lookup", {}).get(name, {})
    bus = ctx.get("bus_lookup", {}).get(name, {})
    proj_lookup = ctx.get("project_lookup", {})
    total = ctx.get("total_schools", 243)

    students = int(s.get("학생수합계") or s.get("학생수") or 0)
    level = s.get("학교급", "초")

    # [헤더]
    elig_rank = pri.get("적격성순위") or pri.get("전체순위")
    rank_badge = ""
    if elig_rank is not None and not pd.isna(elig_rank) and int(elig_rank) <= 30:
        rank_badge = f'<span class="sp-rank-badge">#{int(elig_rank)}</span>'
    header = (
        f'<div class="sp-title">{rank_badge}{name}</div>'
        f'<div class="sp-sub">{s.get("구","")} {s.get("동","") or ""} '
        f'· {s.get("주소","")}</div>'
    )

    # [기본정보]
    설립 = s.get("설립", "-")
    basic = (
        f'<div>🏫 <b>{level}등학교</b> '
        f'<span style="color:#777;">({설립})</span> · '
        f'학생수 <b style="color:#34495E;">{students:,}명</b></div>'
    )

    # [경사도] Phase B-2 사후 검증 정보 (점수 미반영, 보조 표시)
    slope = ctx.get("slope_lookup", {}).get(name)
    if slope is not None and not pd.isna(slope):
        slope_color = "#C0392B" if slope >= SLOPE_AVG_BUS14 else (
            "#E67E22" if slope >= SLOPE_AVG_ALL else "#34495E"
        )
        slope_html = (
            f'<div style="margin-top:3px;">🏔️ 경사도: '
            f'<b style="color:{slope_color};">{slope:.1f}°</b> '
            f'<span style="color:#888;font-size:11.5px;">'
            f'(대전 학교 평균 {SLOPE_AVG_ALL}°, 현행 14교 평균 {SLOPE_AVG_BUS14}°)'
            f'</span></div>'
        )
        basic = basic + slope_html

    # [우선순위]
    elig_r = pri.get("적격성순위") or pri.get("전체순위")
    dev_r = pri.get("도시개발영향순위")
    fut_r = pri.get("미래시나리오순위")
    priority = (
        f'<div class="sp-section">📊 우선순위</div>'
        f'<table>'
        f'<tr><td class="k">통학지원 적격성</td><td>{_rank_strong(elig_r, total)}</td></tr>'
        f'<tr><td class="k">도시개발 영향</td><td>{_rank_strong(dev_r, total)}</td></tr>'
        f'<tr><td class="k">미래 시나리오</td><td>{_rank_strong(fut_r, total)}</td></tr>'
        f'</table>'
    )

    # [영향사업] — details 블록 (≤5건 펼침/≥6건 접힘)
    impact = _build_impact_block(pri.get("영향사업목록", ""), level, proj_lookup)

    # [현행 통학차량] + 노선 통계 (운영 12교 기준)
    bus_html = _build_bus_row(bus)
    route_info = ctx.get("route_lookup", {}).get(name)
    if route_info:
        bus_html += (
            f'<div style="font-size:11.5px;margin-top:4px;color:#1A237E;">'
            f'🛣️ 노선 평균 경사: <b>{route_info["avg_slope"]:.1f}°</b> '
            f'(노선 {route_info["n"]}건, 총 길이 {route_info["total_length_km"]:.1f} km)'
            f'</div>'
        )
    bus_section = (
        f'<div class="sp-section">🚌 현행 통학차량</div>'
        f'{bus_html}'
    )

    return (
        POPUP_CSS
        + f'<div class="sp-popup">{header}<hr>{basic}<hr>'
        f'{priority}<hr>{impact}<hr>{bus_section}</div>'
    )


def build_popup_context(schools_df):
    """팝업에 필요한 lookup 4개 + total을 반환."""
    from src.config import OUTPUT_TABLES, DATA_EXTERNAL, DATA_PROCESSED

    # priority_lookup: 통학버스운용_미래시나리오.csv (통합 점수 결과)
    pri_lookup = {}
    pri_path = OUTPUT_TABLES / "통학버스운용_미래시나리오.csv"
    if pri_path.exists():
        pri_df = pd.read_csv(pri_path)
        pri_lookup = {row["학교명"]: row.to_dict() for _, row in pri_df.iterrows()}

    # 미운영순위·전체순위 통합 (tonghak_eligibility 결과에서)
    elig_path = OUTPUT_TABLES / "통학지원적격성_학교별.csv"
    if elig_path.exists():
        elig_df = pd.read_csv(elig_path)
        for _, r in elig_df.iterrows():
            d = pri_lookup.setdefault(r["학교명"], {})
            d["전체순위"] = r.get("전체순위")
            d["미운영순위"] = r.get("미운영순위")

    # bus_lookup
    bus_lookup = {}
    bus_csv = DATA_EXTERNAL / "bus" / "대전_현행통학차량_14개교.csv"
    if bus_csv.exists():
        bus_df = pd.read_csv(bus_csv, encoding="utf-8-sig")
        from src.integrated_priority import match_bus_to_schools
        matched, _ = match_bus_to_schools(bus_df, schools_df)
        for _, r in matched.iterrows():
            bus_lookup[r["정식학교명"]] = r.to_dict()

    # project_lookup
    project_lookup = {}
    try:
        from src.coords_data import load_redev_projects
        for p in load_redev_projects(only_active=True):
            project_lookup[p["사업명"]] = p
    except Exception:
        pass

    # slope_lookup (Phase B-2 사후 검증, 점수 미반영)
    slope_lookup = {}
    slope_csv = DATA_PROCESSED / "schools_with_slope.csv"
    if slope_csv.exists():
        sdf = pd.read_csv(slope_csv, encoding="utf-8-sig")
        if "학교명" in sdf.columns and "slope_300m_mean" in sdf.columns:
            slope_lookup = dict(zip(sdf["학교명"], sdf["slope_300m_mean"]))

    # route_lookup: 운영 12교의 노선 통계 (정식 학교명 → {avg_slope, n, total_length_km})
    route_lookup = {}
    route_csv = DATA_PROCESSED / "schools_route_slope_summary.csv"
    if route_csv.exists() and bus_csv.exists():
        rdf = pd.read_csv(route_csv, encoding="utf-8-sig")
        # 약식 학교명 → bus_matched의 정식 학교명 매핑
        try:
            short_to_full = {}
            for _, r in matched.iterrows():
                # bus CSV의 "학교명"(약식)을 정식 학교명에 매핑
                short_to_full[r["학교명"]] = r["정식학교명"]
        except Exception:
            short_to_full = {}
        for _, r in rdf.iterrows():
            short = r["school_short"]
            full = short_to_full.get(short)
            # 기성초 본/분교 분리는 short_to_full에서 약식 1개에 정식 1개만 매핑됨
            # → bus_lookup 전체를 다시 보고 약식 매칭되는 모든 정식 학교명 처리
            full_names = [r2["정식학교명"] for _, r2 in matched.iterrows()
                          if r2["학교명"] == short]
            payload = {
                "avg_slope": float(r["route_avg_slope"]),
                "n": int(r["route_n"]),
                "total_length_km": float(r["route_total_length_m"]) / 1000.0,
            }
            for fn in full_names:
                route_lookup[fn] = payload

    return {
        "priority_lookup": pri_lookup,
        "bus_lookup": bus_lookup,
        "project_lookup": project_lookup,
        "slope_lookup": slope_lookup,
        "route_lookup": route_lookup,
        "total_schools": len(schools_df),
    }
