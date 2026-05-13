"""영향권 분석 & 학생발생률 추정 (재개발 120건 VWorld 지오코딩판)

좌표계: EPSG:4326 저장 / EPSG:5179 분석 (미터)
사업: 상태='진행'만 영향권 계산 대상.
세대수 NaN: 학생수 추정 제외, 영향권 분석은 포함.
"""
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from src.config import (
    DATA_PROCESSED, OUTPUT_TABLES,
    CRS_WGS84, CRS_KOREA, BUFFER_DISTANCES, ESTIMATED_STUDENT_RATES,
)
from src.coords_data import load_redev_projects


def schools_to_gdf(schools_df):
    df = schools_df.dropna(subset=["lat", "lon"]).copy()
    geometry = [Point(xy) for xy in zip(df["lon"], df["lat"])]
    return gpd.GeoDataFrame(df, geometry=geometry, crs=CRS_WGS84)


def projects_to_gdf(projects=None):
    """재개발사업 → GeoDataFrame. projects=None이면 load_redev_projects()."""
    if projects is None:
        projects = load_redev_projects(only_active=True)
    rows = []
    for p in projects:
        lat, lon = p["좌표"]
        rows.append({
            **{k: v for k, v in p.items() if k != "좌표"},
            "lat": lat, "lon": lon,
            "geometry": Point(lon, lat),
        })
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS_WGS84)


def compute_impact_zones(schools_gdf, projects_gdf, buffer_m=1000):
    """각 학교가 가장 가까운 사업·거리·영향권 클래스 부여."""
    schools_5179 = schools_gdf.to_crs(CRS_KOREA)
    projects_5179 = projects_gdf.to_crs(CRS_KOREA)

    nearest_proj, distances = [], []
    proj_geoms = list(zip(projects_5179["사업명"].tolist(), projects_5179.geometry.tolist()))
    for sch in schools_5179.geometry:
        min_d, min_p = float("inf"), None
        for name, g in proj_geoms:
            d = sch.distance(g)
            if d < min_d:
                min_d, min_p = d, name
        nearest_proj.append(min_p)
        distances.append(min_d)

    schools_5179["최근접_사업"] = nearest_proj
    schools_5179["최근접_거리m"] = distances

    def classify(d):
        if d <= 1000: return "1km 이내"
        if d <= 1500: return "1~1.5km"
        return "1.5km 초과"
    schools_5179["영향권"] = schools_5179["최근접_거리m"].apply(classify)
    return schools_5179.to_crs(CRS_WGS84)


def _safe_est_students(seda, rate):
    """세대수 NaN이면 None, 아니면 round(int)."""
    if pd.isna(seda):
        return None
    return int(round(seda * rate))


def estimate_student_generation(projects_gdf, rates=None):
    if rates is None:
        rates = ESTIMATED_STUDENT_RATES
    df = projects_gdf.copy()
    for level, rate in rates.items():
        df[f"예상_{level}등생"] = df["세대수"].apply(lambda s: _safe_est_students(s, rate))
    return df


def summarize_impact(schools_gdf, projects_gdf):
    """사업별 영향권 요약."""
    rows = []
    for _, p in projects_gdf.iterrows():
        seda = p.get("세대수", None)
        est_elem = _safe_est_students(seda, ESTIMATED_STUDENT_RATES["초"])
        est_mid = _safe_est_students(seda, ESTIMATED_STUDENT_RATES["중"])

        in_1km = schools_gdf[
            (schools_gdf["최근접_사업"] == p["사업명"]) &
            (schools_gdf["최근접_거리m"] <= 1000)
        ]
        in_15km = schools_gdf[
            (schools_gdf["최근접_사업"] == p["사업명"]) &
            (schools_gdf["최근접_거리m"] <= 1500)
        ]
        rows.append({
            "사업명": p["사업명"],
            "구분": p.get("구분", ""),
            "구": p["구"], "동": p.get("동", ""),
            "추진현황": p.get("추진현황", ""),
            "통학영향_임박도": p.get("통학영향_임박도", ""),
            "세대수": seda,
            "예상_초등생": est_elem,
            "예상_중등생": est_mid,
            "1km_학교수": len(in_1km),
            "1km_학생수": int(in_1km["학생수합계"].sum()) if len(in_1km) else 0,
            "1.5km_학교수": len(in_15km),
            "1.5km_학생수": int(in_15km["학생수합계"].sum()) if len(in_15km) else 0,
        })
    return pd.DataFrame(rows)


def summarize_by_imminence(impact_summary_df):
    """통학영향_임박도별 집계 (사업수, 세대수, 예상학생, 1km영향)."""
    df = impact_summary_df.copy()
    df["임박도_정렬키"] = df["통학영향_임박도"].astype(str)
    agg = df.groupby("통학영향_임박도", dropna=False).agg(**{
        "사업수": ("사업명", "count"),
        "총_세대수": ("세대수", "sum"),
        "예상_초등생합": ("예상_초등생", "sum"),
        "예상_중등생합": ("예상_중등생", "sum"),
        "영향_1km_학교수합": ("1km_학교수", "sum"),
        "영향_1km_학생수합": ("1km_학생수", "sum"),
        "영향_1.5km_학교수합": ("1.5km_학교수", "sum"),
        "영향_1.5km_학생수합": ("1.5km_학생수", "sum"),
    }).reset_index()
    return agg.sort_values("통학영향_임박도")


if __name__ == "__main__":
    schools_df = pd.read_csv(DATA_PROCESSED / "schools_geocoded.csv")
    schools_gdf = schools_to_gdf(schools_df)
    projects_gdf = projects_to_gdf()

    print(f"[analysis] 학교 {len(schools_gdf)}교 / 진행 사업 {len(projects_gdf)}건")

    schools_gdf = compute_impact_zones(schools_gdf, projects_gdf)
    summary = summarize_impact(schools_gdf, projects_gdf)
    imminence = summarize_by_imminence(summary)

    schools_gdf.drop(columns=["geometry"]).to_csv(
        DATA_PROCESSED / "schools_with_impact.csv",
        index=False, encoding="utf-8-sig",
    )
    summary.to_csv(OUTPUT_TABLES / "재개발영향권_요약.csv",
                   index=False, encoding="utf-8-sig")
    imminence.to_csv(OUTPUT_TABLES / "재개발영향권_요약_임박도별.csv",
                     index=False, encoding="utf-8-sig")

    print(f"\n✅ outputs/tables/재개발영향권_요약.csv ({len(summary)}건)")
    print(f"✅ outputs/tables/재개발영향권_요약_임박도별.csv")
    print(f"\n=== 임박도별 ===")
    print(imminence.to_string(index=False))
