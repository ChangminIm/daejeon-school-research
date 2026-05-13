"""통학지원 우선순위 분석

학교별 5개 지표 0~1 정규화 후 가중합 → 종합점수.

지표:
  1) 임박도가중 (0.30): 1km 영향권 사업의 임박도 가중치 합
  2) 학생유입   (0.25): 1km 영향권 사업의 예상 학생 발생수 (학교급별)
  3) 수용여석   (0.20): 학생수 백분위 (포화도 대리)
  4) 외곽성     (0.15): 자치구 중심에서 학교까지 거리
  5) 트램접근성 (0.10): 정거장 500m 이내 = 보너스(차감)

산출:
  - outputs/tables/통학지원_우선순위_학교별.csv (전체)
  - outputs/tables/통학지원_우선순위_상위30교.csv
  - outputs/figures/우선순위_히트맵.png (구 × 학교급 분포)
"""
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from src.config import (
    DATA_PROCESSED, OUTPUT_TABLES, OUTPUT_FIGURES,
    CRS_WGS84, CRS_KOREA, GU_COLORS,
    ESTIMATED_STUDENT_RATES, PRIORITY_WEIGHTS, IMMINENCE_WEIGHTS,
)
from src.coords_data import load_redev_projects, GU_CENTERS

BUFFER_M = 1000  # 1km 영향권
TRAM_STATION_M = 500


def _normalize(series, invert=False):
    """min-max 정규화 0~1. invert=True면 1-x."""
    s = series.astype(float)
    if s.max() == s.min():
        return pd.Series(np.zeros(len(s)), index=s.index)
    out = (s - s.min()) / (s.max() - s.min())
    return 1.0 - out if invert else out


def _impact_projects_for_school(school_geom_5179, projects_5179):
    """학교 점 기준 1km 이내 사업 리스트(인덱스) 반환."""
    return projects_5179[
        projects_5179.geometry.distance(school_geom_5179) <= BUFFER_M
    ].index.tolist()


def _tram_distance_for_school(school_geom_5179, stations_5179=None):
    """학교 점에서 가장 가까운 트램 정거장 거리(m). 데이터 없으면 None."""
    if stations_5179 is None or len(stations_5179) == 0:
        return None
    return float(stations_5179.geometry.distance(school_geom_5179).min())


def _load_tram_stations_5179():
    """트램 정거장 GeoDataFrame (EPSG:5179). 가용 시 반환."""
    try:
        from src.analysis_tram import stations_to_gdf
        gdf = stations_to_gdf(snap_to_road=True).to_crs(CRS_KOREA)
        return gdf
    except Exception:
        return None


def compute_priority(schools_df, projects=None, weights=None,
                     use_tram=True, verbose=True):
    """학교별 우선순위 점수 계산.

    Args:
        schools_df: schools_with_impact 또는 schools_geocoded
        projects: 사업 리스트(dict). None이면 load_redev_projects()
        weights: 지표 가중치 dict. None이면 config.PRIORITY_WEIGHTS
        use_tram: 트램 정거장 이용 (있으면)

    Returns:
        DataFrame — 모든 학교 + 지표/점수/순위/영향사업목록
    """
    weights = weights or PRIORITY_WEIGHTS
    projects = projects or load_redev_projects(only_active=True)

    schools_df = schools_df.dropna(subset=["lat", "lon"]).copy()

    # 학교 GDF (5179)
    sch_gdf = gpd.GeoDataFrame(
        schools_df,
        geometry=[Point(xy) for xy in zip(schools_df["lon"], schools_df["lat"])],
        crs=CRS_WGS84,
    ).to_crs(CRS_KOREA)

    # 사업 GDF (5179) - 진행 사업만, 좌표 있음
    proj_records = []
    for p in projects:
        proj_records.append({
            "사업명": p["사업명"],
            "구": p["구"],
            "세대수": p.get("세대수"),
            "임박도": p.get("통학영향_임박도", ""),
            "geometry": Point(p["좌표"][1], p["좌표"][0]),
        })
    proj_gdf = gpd.GeoDataFrame(
        proj_records, geometry="geometry", crs=CRS_WGS84
    ).to_crs(CRS_KOREA)

    # 트램 정거장
    stations_5179 = _load_tram_stations_5179() if use_tram else None
    if stations_5179 is not None and verbose:
        print(f"  · 트램 정거장 {len(stations_5179)}개 사용 (보너스 지표)")
    elif verbose:
        print("  · 트램 정거장 없음 → 지표5 = 0")

    # 자치구 중심 좌표 (WGS84) → 5179
    gu_centers_5179 = {}
    for gu, (lat, lon) in GU_CENTERS.items():
        gu_centers_5179[gu] = (
            gpd.GeoSeries([Point(lon, lat)], crs=CRS_WGS84).to_crs(CRS_KOREA).iloc[0]
        )

    # === 학교별 원시 지표 계산 ===
    rows = []
    for idx, s in sch_gdf.iterrows():
        sch_geom = s.geometry
        level = s["학교급"]   # "초" 또는 "중"
        rate = ESTIMATED_STUDENT_RATES.get(level, 0)

        # 1km 영향권 사업
        impact_idx = _impact_projects_for_school(sch_geom, proj_gdf)
        impact_proj = proj_gdf.loc[impact_idx]

        # 지표1: 임박도 가중치 합
        imm_score = sum(
            IMMINENCE_WEIGHTS.get(p["임박도"], 0) for _, p in impact_proj.iterrows()
        )

        # 지표2: 예상 학생 유입 (학교급별 발생률)
        inflow = 0.0
        for _, p in impact_proj.iterrows():
            seda = p["세대수"]
            if pd.notna(seda):
                inflow += seda * rate

        # 지표3: 학생수 (원값 그대로, 정규화는 나중에)
        students = s.get("학생수합계", 0) or 0

        # 지표4: 자치구 중심에서 거리 (m)
        gu_center = gu_centers_5179.get(s["구"])
        if gu_center is not None:
            edge_dist = float(sch_geom.distance(gu_center))
        else:
            edge_dist = 0.0

        # 지표5: 트램 정거장 거리 (m, None이면 -1로 표시)
        tram_dist = _tram_distance_for_school(sch_geom, stations_5179)

        # 영향사업 목록 문자열 (추적용)
        impact_proj_names = "; ".join(
            f"{p['사업명']}({p['임박도']})"
            for _, p in impact_proj.iterrows()
        ) if len(impact_proj) > 0 else ""

        rows.append({
            "학교명": s["학교명"],
            "학교급": level,
            "구": s["구"],
            "동": s.get("동", ""),
            "학생수": int(students),
            "lat": s.get("lat"), "lon": s.get("lon"),
            "_지표1_원": imm_score,
            "_지표2_원": inflow,
            "_지표3_원": students,
            "_지표4_원": edge_dist,
            "_지표5_원": tram_dist if tram_dist is not None else np.nan,
            "영향사업수": len(impact_proj),
            "영향사업목록": impact_proj_names,
        })
    df = pd.DataFrame(rows)

    # === 정규화 ===
    df["지표1_임박도"] = _normalize(df["_지표1_원"])
    df["지표2_학생유입"] = _normalize(df["_지표2_원"])
    df["지표3_수용여석"] = _normalize(df["_지표3_원"])
    df["지표4_외곽성"] = _normalize(df["_지표4_원"])

    # 지표5: 정거장 거리 짧을수록 보너스(차감) — 0~1 후 1-x로 invert
    if df["_지표5_원"].notna().any():
        bonus = pd.Series(np.zeros(len(df)), index=df.index)
        within = df["_지표5_원"] <= TRAM_STATION_M
        if within.any():
            # 500m 이내만 보너스, 500m 초과는 0
            d = df.loc[within, "_지표5_원"]
            bonus.loc[within] = 1.0 - (d / TRAM_STATION_M)
        df["지표5_트램접근성"] = bonus
    else:
        df["지표5_트램접근성"] = 0.0

    # === 종합점수 (지표5는 보너스이므로 차감) ===
    df["종합점수"] = (
        df["지표1_임박도"] * weights["지표1_임박도가중"]
        + df["지표2_학생유입"] * weights["지표2_학생유입"]
        + df["지표3_수용여석"] * weights["지표3_수용여석"]
        + df["지표4_외곽성"] * weights["지표4_외곽성"]
        - df["지표5_트램접근성"] * weights["지표5_트램접근성"]
    ).round(4)

    df["순위"] = df["종합점수"].rank(ascending=False, method="min").astype(int)
    df = df.sort_values("순위").reset_index(drop=True)

    # 점수 기여도 설명 컬럼
    df["점수설명"] = df.apply(_explain_score, axis=1, weights=weights)
    return df


def _explain_score(row, weights):
    """학교 한 행의 지표 기여도 한 줄 요약."""
    parts = []
    for ind_key, w_key in [
        ("지표1_임박도", "지표1_임박도가중"),
        ("지표2_학생유입", "지표2_학생유입"),
        ("지표3_수용여석", "지표3_수용여석"),
        ("지표4_외곽성", "지표4_외곽성"),
    ]:
        contrib = row[ind_key] * weights[w_key]
        parts.append(f"{ind_key.split('_')[1]}={contrib:.3f}")
    tram_contrib = row["지표5_트램접근성"] * weights["지표5_트램접근성"]
    if tram_contrib > 0:
        parts.append(f"트램보너스=-{tram_contrib:.3f}")
    return " | ".join(parts)


def save_results(df, top_n=30):
    """전체 + 상위 N교 + 히트맵 저장."""
    full_out = OUTPUT_TABLES / "통학지원_우선순위_학교별.csv"
    df.to_csv(full_out, index=False, encoding="utf-8-sig")

    top = df.head(top_n).copy()
    top_out = OUTPUT_TABLES / f"통학지원_우선순위_상위{top_n}교.csv"
    top.to_csv(top_out, index=False, encoding="utf-8-sig")

    # 히트맵: 구 × 학교급별 상위 학교수 분포
    pivot = (
        df.head(50)
        .groupby(["구", "학교급"])
        .size()
        .unstack(fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Top 50 priority schools — by gu × level", fontsize=11)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            ax.text(j, i, int(v), ha="center", va="center",
                    color="white" if v > pivot.values.max()/2 else "black")
    plt.colorbar(im, ax=ax, label="count")
    plt.tight_layout()
    fig_out = OUTPUT_FIGURES / "우선순위_히트맵.png"
    plt.savefig(fig_out, dpi=200, bbox_inches="tight")
    plt.close()

    return full_out, top_out, fig_out


def run():
    schools_df = pd.read_csv(DATA_PROCESSED / "schools_with_impact.csv")
    print(f"[priority] 학교 {len(schools_df)}교 분석 시작")
    df = compute_priority(schools_df, verbose=True)
    full, top, fig = save_results(df, top_n=30)

    print(f"\n✅ 전체: {full.name} ({len(df)}교)")
    print(f"✅ 상위 30: {top.name}")
    print(f"✅ 히트맵: {fig.name}")
    print(f"\n=== 상위 10교 ===")
    show_cols = ["순위", "학교명", "학교급", "구", "학생수", "종합점수",
                 "영향사업수", "점수설명"]
    print(df[show_cols].head(10).to_string(index=False))
    return df


if __name__ == "__main__":
    run()
