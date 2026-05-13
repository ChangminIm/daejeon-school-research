"""트램 14공구 공사영향권 분석 + 정거장 접근성

공구별 LineString → buffer → 학교와 공간 join.
정거장 점 → buffer → 학교 도보권 추정.

좌표계: EPSG:4326 저장 / EPSG:5179 분석 (미터)
"""
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point
from src.config import (
    DATA_PROCESSED, OUTPUT_TABLES,
    CRS_WGS84, CRS_KOREA,
)
from src.tram_data import SECTIONS, LANDMARK_COORDS, STATION_COORDS, get_section_route

# 트램 공사 영향권 버퍼 (미터)
TRAM_CONSTRUCTION_BUFFER = 500   # 공사 소음·진동·통학 우회 영향권
TRAM_STATION_BUFFER = 500        # 정거장 도보 5분 권


def sections_to_gdf(road_mode="snap"):
    """14공구 → GeoDataFrame (LineString, WGS84).

    Args:
        road_mode: "snap" (기본) - 각 정거장·랜드마크를 가장 가까운 주간선도로 노드로 스냅 후
                                   직선 연결. 노선이 도로 위에 있으면서 backtrack 없음.
                   "off"      - 원본 좌표 직선 연결 (도로 스냅 없음)
                   "arterial" - 주간선도로 shortest path (정거장 사이 우회 위험)
                   "all"      - 전체 도로망 shortest path (권장 안 함)
    """
    if road_mode == "arterial":
        from src.road_routing import get_arterial_graph, snap_path_to_road
        G = get_arterial_graph()
        do_path = True
    elif road_mode == "all":
        from src.road_routing import get_road_graph, snap_path_to_road
        G = get_road_graph()
        do_path = True
    elif road_mode == "snap":
        from src.road_routing import get_arterial_graph, snap_points_to_road
        G = get_arterial_graph()
        do_path = False
    else:
        G = None
        do_path = False

    rows = []
    for i, s in enumerate(SECTIONS):
        route = get_section_route(i)  # [(lat, lon), ...] — 시점+정거장+waypoint+종점
        if G is not None:
            try:
                if do_path:
                    from src.road_routing import snap_path_to_road
                    route = snap_path_to_road(route, G=G)
                else:
                    from src.road_routing import snap_points_to_road
                    route = snap_points_to_road(route, G=G, max_distance_m=300)
            except Exception as e:
                print(f"⚠️  {s['공구']}공구 도로 스냅 실패 → 원본 폴백: {e}")
        # shapely는 (x=lon, y=lat)
        line = LineString([(lon, lat) for lat, lon in route])
        rows.append({
            "공구": s["공구"],
            "구": ", ".join(s["구"]),
            "시점": s["시점"], "종점": s["종점"],
            "길이km": s["길이km"],
            "정거장수": len(s["정거장번호"]),
            "착공": s["착공"], "준공": s["준공"],
            "기간개월": s["기간개월"],
            "특이공정": s["특이공정"],
            "geometry": line,
        })
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS_WGS84)


def stations_to_gdf(snap_to_road=True):
    """정거장 → GeoDataFrame (Point, WGS84).

    snap_to_road=True면 sections_to_gdf와 동일하게 주간선도로 위로 스냅 (마커-노선 일치).
    """
    no_to_section = {no: s["공구"] for s in SECTIONS for no in s["정거장번호"]}
    coords = dict(STATION_COORDS)
    if snap_to_road:
        from src.road_routing import snap_points_to_road, get_arterial_graph
        G = get_arterial_graph()
        nos = list(coords.keys())
        snapped = snap_points_to_road([coords[n] for n in nos], G=G, max_distance_m=300)
        coords = dict(zip(nos, snapped))

    rows = []
    for st_no, (lat, lon) in coords.items():
        rows.append({
            "정거장번호": st_no,
            "공구": no_to_section[st_no],
            "lat": lat, "lon": lon,
            "geometry": Point(lon, lat),
        })
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS_WGS84)


def compute_construction_impact(schools_gdf, sections_gdf, buffer_m=TRAM_CONSTRUCTION_BUFFER):
    """학교 × 공구 공사영향권 매칭

    각 학교가 어떤 공구의 buffer_m 영향권 안에 들어가는지 계산.
    여러 공구에 동시에 걸칠 수 있으므로 1:N 결과.
    """
    schools_5179 = schools_gdf.to_crs(CRS_KOREA).copy()
    sections_5179 = sections_gdf.to_crs(CRS_KOREA).copy()
    sections_5179["buffer"] = sections_5179.geometry.buffer(buffer_m)

    rows = []
    for _, sch in schools_5179.iterrows():
        for _, sec in sections_5179.iterrows():
            d = sch.geometry.distance(sec.geometry)  # 점-선 거리
            if d <= buffer_m:
                rows.append({
                    "학교명": sch["학교명"],
                    "학교급": sch["학교급"],
                    "구": sch["구"],
                    "학생수합계": sch["학생수합계"],
                    "공구": sec["공구"],
                    "공구_구간": f"{sec['시점']}~{sec['종점']}",
                    "거리m": round(d, 1),
                    "공사착공": sec["착공"],
                    "공사준공": sec["준공"],
                    "공사기간개월": sec["기간개월"],
                    "특이공정": sec["특이공정"],
                })
    return pd.DataFrame(rows)


def compute_station_access(schools_gdf, stations_gdf, buffer_m=TRAM_STATION_BUFFER):
    """학교별 최근접 정거장과 거리"""
    schools_5179 = schools_gdf.to_crs(CRS_KOREA).copy()
    stations_5179 = stations_gdf.to_crs(CRS_KOREA).copy()

    rows = []
    for _, sch in schools_5179.iterrows():
        best_d, best_st, best_sec = float("inf"), None, None
        for _, st in stations_5179.iterrows():
            d = sch.geometry.distance(st.geometry)
            if d < best_d:
                best_d, best_st, best_sec = d, st["정거장번호"], st["공구"]
        rows.append({
            "학교명": sch["학교명"],
            "학교급": sch["학교급"],
            "구": sch["구"],
            "최근접_정거장": int(best_st) if best_st is not None else None,
            "최근접_공구": int(best_sec) if best_sec is not None else None,
            "거리m": round(best_d, 1),
            "도보권_500m이내": best_d <= 500,
            "도보권_1km이내": best_d <= 1000,
        })
    return pd.DataFrame(rows)


def summarize_by_section(impact_df, schools_df):
    """공구별 영향 학교수·학생수 요약"""
    if len(impact_df) == 0:
        return pd.DataFrame()
    agg = impact_df.groupby(["공구", "공구_구간", "공사착공", "공사준공", "공사기간개월"]).agg(
        영향_학교수=("학교명", "nunique"),
        영향_학생수=("학생수합계", "sum"),
        평균_거리m=("거리m", "mean"),
    ).reset_index()
    agg["평균_거리m"] = agg["평균_거리m"].round(1)
    return agg.sort_values("공구")


if __name__ == "__main__":
    schools_df = pd.read_csv(DATA_PROCESSED / "schools_geocoded.csv")
    schools_df = schools_df.dropna(subset=["lat", "lon"])
    schools_gdf = gpd.GeoDataFrame(
        schools_df,
        geometry=[Point(xy) for xy in zip(schools_df["lon"], schools_df["lat"])],
        crs=CRS_WGS84,
    )

    sections_gdf = sections_to_gdf()
    stations_gdf = stations_to_gdf()

    # 공사 영향권 분석
    impact = compute_construction_impact(schools_gdf, sections_gdf)
    section_summary = summarize_by_section(impact, schools_df)

    # 정거장 접근성
    access = compute_station_access(schools_gdf, stations_gdf)

    # 저장
    impact.to_csv(OUTPUT_TABLES / "트램_공사영향권_상세.csv", index=False, encoding="utf-8-sig")
    section_summary.to_csv(OUTPUT_TABLES / "트램_공사영향권_요약.csv", index=False, encoding="utf-8-sig")
    access.to_csv(OUTPUT_TABLES / "트램_정거장_접근성.csv", index=False, encoding="utf-8-sig")

    print(f"공구 수: {len(sections_gdf)}, 정거장 수: {len(stations_gdf)}")
    print(f"공사 영향 학교 매칭 건수: {len(impact)} (학교는 여러 공구에 중복 가능)")
    print(f"공사 영향 받는 고유 학교 수: {impact['학교명'].nunique() if len(impact) else 0}")
    print(f"\n=== 공구별 요약 ===")
    print(section_summary.to_string(index=False) if len(section_summary) else "(영향 학교 없음)")
    print(f"\n=== 정거장 도보 500m 내 학교 ===")
    walkable = access[access["도보권_500m이내"]]
    print(f"  {len(walkable)}교")
