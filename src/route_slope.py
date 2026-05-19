"""Phase 2 PART B: 노선 경사 프로파일 + 운영비 격차 회귀.

흐름:
  1. _re SHP 로드 + 학교명 정규화 (운영 12교 + 명목 1)
  2. 10m 간격 점 샘플링 → slope raster 값 추출 (5179 → 5186 변환)
  3. 노선별 통계 (mean/max/pct_over_5/pct_over_10/sinuosity)
  4. 학교별 집계 → schools_route_slope_summary.csv
  5. 학교 위치 경사 vs 노선 평균 경사 Wilcoxon
  6. 운영비 회귀 (OLS, 1인당비용 ~ 길이·경사·학생수·차량대수)
  7. 정류장 500m 버퍼 vs 신규 검토 30교 매칭
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import statsmodels.api as sm
from scipy import stats as sp_stats
from pyproj import Transformer
from shapely.geometry import Point

from src.config import (
    ROUTES_SHP, STOPS_SHP, DATA_PROCESSED, DATA_EXTERNAL,
    OUTPUT_TABLES, BUS_OPERATING_12_SHORT,
)

SLOPE_TIF = DATA_EXTERNAL / "dem" / "대전_slope_5m.tif"
DEM_TIF = DATA_EXTERNAL / "dem" / "대전_dem_5m.tif"

OUT_ROUTES_SLOPE = DATA_PROCESSED / "routes_with_slope.csv"
OUT_SCHOOLS_SUM = DATA_PROCESSED / "schools_route_slope_summary.csv"
OUT_RISK_TOP10 = OUTPUT_TABLES / "노선경사_위험구간상위10.csv"
OUT_REG_RESULT = OUTPUT_TABLES / "운영비격차_회귀결과.csv"
OUT_STOPS_NEAR = OUTPUT_TABLES / "신규검토30교_정류장근접.csv"

# SHP 학교명 → 약식 학교명 정규화
# PART A 진단 결과: ? 표기 잔존, 일부 표기 단축
ROUTE_NAME_NORMALIZE = {
    "동명초등학교":            "동명초",
    "기성초등학교·길헌분":     "기성초 및 길헌분교",  # 본교+분교 합성
    "남선초등학교":            "남선초",
    "산서초등학교":            "산서초",
    "산내초등학교":            "산내초",
    "산흥초등학교":            "산흥초",
    "세천초등학교":            "세천초",
    "계산초등학교":            "계산초",
    "진잠초등학교":            "진잠초",
    "구즉초등학교":            "구즉초",
    "대전원신흥초등학교 ?":    "대전원신흥초 복용분교장",  # ? 표기는 분교장 의미
    "신탄진용정초등학교 ?":    "신탄진용정초",            # 신탄진용정+장동 공유 노선
}

SAMPLE_INTERVAL_M = 10


# ===== 데이터 로딩 =====

def load_routes():
    g = gpd.read_file(ROUTES_SHP, encoding="cp949")
    if g.crs.to_epsg() != 5179:
        g = g.to_crs("EPSG:5179")
    g["short"] = g["이름"].map(ROUTE_NAME_NORMALIZE).fillna(g["이름"])
    return g


def load_stops():
    g = gpd.read_file(STOPS_SHP, encoding="cp949")
    if g.crs.to_epsg() != 5179:
        g = g.to_crs("EPSG:5179")
    if "이름" in g.columns:
        g["short"] = g["이름"].map(ROUTE_NAME_NORMALIZE).fillna(g["이름"])
    return g


# ===== 점 샘플링 + 경사도 추출 =====

def _sample_points(line, interval=SAMPLE_INTERVAL_M):
    """라인을 interval 미터 간격으로 점 샘플링."""
    if line.geom_type == "MultiLineString":
        parts = list(line.geoms)
    else:
        parts = [line]
    pts = []
    for part in parts:
        L = part.length
        n = max(int(L / interval), 1)
        for i in range(n + 1):
            pts.append(part.interpolate(i * interval if i < n else L))
    return pts


def _extract_slope_along_routes(routes_5179):
    """각 노선마다 점 샘플링 + slope 값 추출 + 노선별 통계."""
    # 5179 → 5186 변환기 (slope raster는 5186)
    tr_5179_to_5186 = Transformer.from_crs("EPSG:5179", "EPSG:5186", always_xy=True)

    print(f"  노선 {len(routes_5179)}개 × 10m 간격 샘플링 + slope 추출...")

    records = []
    with rasterio.open(SLOPE_TIF) as src_slope, rasterio.open(DEM_TIF) as src_dem:
        nodata_slope = src_slope.nodata
        nodata_dem = src_dem.nodata

        for idx, row in routes_5179.iterrows():
            line = row.geometry
            pts_5179 = _sample_points(line)
            if len(pts_5179) < 2:
                continue
            xs_5179 = [p.x for p in pts_5179]
            ys_5179 = [p.y for p in pts_5179]
            xs_5186, ys_5186 = tr_5179_to_5186.transform(xs_5179, ys_5179)
            coords_5186 = list(zip(xs_5186, ys_5186))

            slope_vals = np.array([v[0] for v in src_slope.sample(coords_5186)])
            dem_vals = np.array([v[0] for v in src_dem.sample(coords_5186)])

            slope_vals = np.where(slope_vals == nodata_slope, np.nan, slope_vals)
            dem_vals = np.where(dem_vals == nodata_dem, np.nan, dem_vals)

            valid_slope = slope_vals[np.isfinite(slope_vals)]
            valid_dem = dem_vals[np.isfinite(dem_vals)]

            if len(valid_slope) < 2:
                continue

            # sinuosity
            if line.geom_type == "MultiLineString":
                gparts = list(line.geoms)
                start = gparts[0].coords[0]
                end = gparts[-1].coords[-1]
                actual = sum(p.length for p in gparts)
            else:
                start = line.coords[0]
                end = line.coords[-1]
                actual = line.length
            dx, dy = end[0] - start[0], end[1] - start[1]
            straight = (dx*dx + dy*dy) ** 0.5
            sinuosity = actual / straight if straight > 0 else np.nan

            # elevation gain: 양의 변화 합
            elevation_gain = 0.0
            if len(valid_dem) > 1:
                diffs = np.diff(valid_dem)
                elevation_gain = float(diffs[diffs > 0].sum())

            records.append({
                "Id": row.get("Id"),
                "school_raw": row["이름"],
                "school_short": row["short"],
                "purpose": row.get("목적"),
                "차수": row.get("차수"),
                "n_samples": len(valid_slope),
                "length_m": float(line.length),
                "sinuosity": float(sinuosity),
                "slope_mean": float(valid_slope.mean()),
                "slope_max": float(valid_slope.max()),
                "slope_pct_over_5deg": float((valid_slope >= 5).mean() * 100),
                "slope_pct_over_10deg": float((valid_slope >= 10).mean() * 100),
                "elevation_gain_m": elevation_gain,
                "dem_min_m": float(valid_dem.min()) if len(valid_dem) else np.nan,
                "dem_max_m": float(valid_dem.max()) if len(valid_dem) else np.nan,
            })
    return pd.DataFrame(records)


# ===== 학교별 집계 =====

def aggregate_schools(routes_slope):
    """학교별 노선 종합 지표."""
    def wmean(df, val_col, weight_col):
        w = df[weight_col].fillna(0).values
        v = df[val_col].fillna(0).values
        if w.sum() == 0:
            return np.nan
        return float((v * w).sum() / w.sum())

    out = []
    for school, sub in routes_slope.groupby("school_short"):
        n_routes = len(sub)
        total_len = sub["length_m"].sum()
        avg_slope_weighted = wmean(sub, "slope_mean", "length_m")
        max_slope = sub["slope_max"].max()
        pct_steep_10 = wmean(sub, "slope_pct_over_10deg", "length_m")
        pct_steep_5 = wmean(sub, "slope_pct_over_5deg", "length_m")
        avg_sinuosity = sub["sinuosity"].mean()

        # 등교/하교 비대칭
        deung = sub[sub["purpose"] == "등교"]["slope_mean"]
        ha = sub[sub["purpose"] == "하교"]["slope_mean"]
        asym = (deung.mean() - ha.mean()) if len(deung) and len(ha) else np.nan

        out.append({
            "school_short": school,
            "route_n": n_routes,
            "route_total_length_m": total_len,
            "route_avg_slope": avg_slope_weighted,
            "route_max_slope": max_slope,
            "route_pct_steep_10deg": pct_steep_10,
            "route_pct_steep_5deg": pct_steep_5,
            "route_avg_sinuosity": avg_sinuosity,
            "asymmetry_deg": asym,
        })
    return pd.DataFrame(out).sort_values("route_avg_slope", ascending=False)


# ===== B-3: 학교 위치 vs 노선 경사 Wilcoxon =====

def compare_position_vs_route(schools_sum):
    print("\n" + "=" * 72)
    print("[B-3] 학교 위치 경사 vs 노선 평균 경사 (Wilcoxon signed-rank)")
    print("=" * 72)

    # schools_with_slope에서 학교명 매칭으로 slope_300m_mean 가져오기
    sch = pd.read_csv(DATA_PROCESSED / "schools_with_slope.csv", encoding="utf-8-sig")

    # 운영 12교 정식 학교명 매칭
    bus_df = pd.read_csv(DATA_EXTERNAL / "bus" / "대전_현행통학차량_14개교.csv",
                          encoding="utf-8-sig")
    from src.integrated_priority import match_bus_to_schools
    matched, _ = match_bus_to_schools(bus_df, sch)
    # matched: 학교명(약식), 정식학교명 컬럼
    matched["slope_300m_mean"] = matched["정식학교명"].map(
        dict(zip(sch["학교명"], sch["slope_300m_mean"]))
    )
    # 약식 단축
    matched["short"] = matched["학교명"]

    # 운영 12교만 (흥도/신탄진용정 제외)
    op_only = matched[matched["short"].isin(BUS_OPERATING_12_SHORT)].copy()

    # 학교명 약식이 schools_sum의 school_short와 매칭
    # 단, "기성초 및 길헌분교"는 schools_sum에선 하나로, matched에선 본교+분교 두 행
    # → 본교 기준으로 통일 (slope_300m_mean 평균 사용 or 본교 값)
    pos_by_short = op_only.groupby("short")["slope_300m_mean"].mean()

    cmp = schools_sum[["school_short", "route_avg_slope"]].merge(
        pos_by_short.rename("position_slope").reset_index().rename(
            columns={"short": "school_short"}),
        on="school_short", how="inner"
    )

    print(f"\n  {'학교':<24} {'위치 경사':>10} {'노선 평균 경사':>16} {'차이':>10}")
    print(f"  {'-'*24} {'-'*10} {'-'*16} {'-'*10}")
    for _, r in cmp.iterrows():
        diff = r["route_avg_slope"] - r["position_slope"]
        print(f"  {r['school_short']:<24} {r['position_slope']:>9.2f}° "
              f"{r['route_avg_slope']:>15.2f}° {diff:>+9.2f}°")

    pos = cmp["position_slope"].values
    route = cmp["route_avg_slope"].values
    n = len(pos)
    if n >= 6:
        stat, p = sp_stats.wilcoxon(route, pos, alternative="two-sided")
        diff_mean = float((route - pos).mean())
        print(f"\n  Wilcoxon signed-rank: stat={stat:.1f}, p={p:.4f}")
        print(f"  평균 차이 = 노선 - 위치 = {diff_mean:+.2f}°")
        if p < 0.05:
            verdict = "유의" if diff_mean > 0 else "유의 (역방향)"
        else:
            verdict = "차이 없음"
        print(f"  → 결론: 노선과 학교 위치 경사 차이 {verdict}")

    return cmp


# ===== B-4: 운영비 격차 회귀 =====

def regression_operating_cost(schools_sum):
    print("\n" + "=" * 72)
    print("[B-4] 운영비 격차 OLS 회귀")
    print("=" * 72)

    bus = pd.read_csv(DATA_EXTERNAL / "bus" / "대전_현행통학차량_14개교.csv",
                       encoding="utf-8-sig")

    # 운영 12교만 + 1인당 비용 NaN 제외
    op = bus[bus["학교명"].isin(BUS_OPERATING_12_SHORT)].copy()
    op = op.dropna(subset=["학생1인당_총비용_천원"])

    # schools_sum 매칭
    df = op.merge(
        schools_sum[["school_short", "route_total_length_m", "route_avg_slope",
                      "route_max_slope", "route_avg_sinuosity"]],
        left_on="학교명", right_on="school_short", how="left"
    )
    df = df.dropna(subset=["route_avg_slope", "학생1인당_총비용_천원"])

    print(f"  표본: {len(df)}교")
    print(df[["학교명", "이용학생수", "차량대수", "route_total_length_m",
              "route_avg_slope", "학생1인당_총비용_천원"]].to_string(index=False))

    if len(df) < 6:
        print("  [경고] 표본 너무 적음. 회귀 신뢰도 낮음.")

    y = df["학생1인당_총비용_천원"].astype(float).values
    X_raw = df[["route_total_length_m", "route_avg_slope",
                "이용학생수", "차량대수"]].astype(float).values
    feature_names = ["노선길이_총m", "노선평균경사", "이용학생수", "차량대수"]

    # 표준화
    X = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0, ddof=0)
    X_const = sm.add_constant(X, has_constant="add")

    try:
        res = sm.OLS(y, X_const).fit()
    except Exception as e:
        print(f"  [에러] OLS 실패: {e}")
        return None

    print(f"\n  R² = {res.rsquared:.3f}  |  Adj R² = {res.rsquared_adj:.3f}  |  "
          f"F-stat = {res.fvalue:.2f}, p = {res.f_pvalue:.4f}")
    print(f"\n  {'변수':<14} {'β':>10} {'SE':>8} {'t':>7} {'p':>10}")
    print(f"  {'-'*14} {'-'*10} {'-'*8} {'-'*7} {'-'*10}")
    print(f"  {'(절편)':<14} {res.params[0]:>10.1f} {res.bse[0]:>8.1f} "
          f"{res.params[0]/res.bse[0]:>7.2f} {res.pvalues[0]:>10.4f}")
    rows = []
    for i, name in enumerate(feature_names):
        j = i + 1
        beta = res.params[j]
        se = res.bse[j]
        t = beta / se
        p = res.pvalues[j]
        sig = "★" if p < 0.05 else "·" if p < 0.10 else " "
        print(f"  {sig} {name:<13} {beta:>10.1f} {se:>8.1f} {t:>7.2f} {p:>10.4f}")
        rows.append({"변수": name, "β_표준화": round(beta, 2),
                      "SE": round(se, 2), "t": round(t, 2),
                      "p": round(p, 4), "유의": sig.strip() or "·"})

    # CSV 저장
    out_df = pd.DataFrame(rows + [{
        "변수": "R²", "β_표준화": round(res.rsquared, 3),
        "SE": "", "t": "", "p": round(res.f_pvalue, 4), "유의": ""
    }])
    out_df.to_csv(OUT_REG_RESULT, index=False, encoding="utf-8-sig")
    print(f"\n  → {OUT_REG_RESULT}")

    # 해석
    print(f"\n  [부호 해석 (운영 12교 기준)]")
    for i, name in enumerate(feature_names):
        beta = res.params[i+1]
        p = res.pvalues[i+1]
        if p < 0.10:
            direction = "증가" if beta > 0 else "감소"
            print(f"    {name} ↑ → 1인당 비용 {direction} (β={beta:+.0f}, p={p:.3f})")

    return res, df


# ===== B-5: 정류장 500m 버퍼 vs 신규 검토 30교 =====

def stops_near_top30():
    print("\n" + "=" * 72)
    print("[B-5] 정류장 500m 버퍼 vs 신규 검토 30교")
    print("=" * 72)

    stops = load_stops()
    print(f"  정류장: {len(stops)}개 (운영 12 + 명목 1 학교의 정류장 합산)")

    top30 = pd.read_csv(OUTPUT_TABLES / "신규검토대상_상위30교.csv")
    sch = pd.read_csv(DATA_PROCESSED / "schools_with_impact.csv")

    valid = sch[sch["학교명"].isin(top30["학교명"])].dropna(subset=["lat", "lon"]).copy()
    top30_gdf = gpd.GeoDataFrame(
        valid,
        geometry=[Point(xy) for xy in zip(valid["lon"], valid["lat"])],
        crs="EPSG:4326",
    ).to_crs("EPSG:5179")

    # 정류장 union → 500m 버퍼
    stops_buf = stops.buffer(500).unary_union

    # 30교 중 정류장 500m 이내
    top30_gdf["near_stop_500m"] = top30_gdf.geometry.within(stops_buf)
    n_near = int(top30_gdf["near_stop_500m"].sum())
    n_far = len(top30_gdf) - n_near
    print(f"\n  정류장 500m 이내 신규 검토 학교: {n_near}/{len(top30_gdf)} ({n_near/len(top30_gdf)*100:.1f}%)")
    print(f"  정류장 500m 밖 (신규 노선 필요): {n_far}/{len(top30_gdf)}")

    # 매칭 30교 정보 표 + 어느 학교의 정류장과 가까운지
    # 가장 가까운 정류장의 학교 이름
    out_rows = []
    for _, r in top30_gdf.iterrows():
        dists = stops.geometry.distance(r.geometry)
        nearest_idx = dists.idxmin()
        nearest_dist = float(dists.iloc[nearest_idx])
        nearest_school = stops.iloc[nearest_idx].get("이름", "")
        rank = top30.loc[top30["학교명"] == r["학교명"], "미운영순위"]
        rank = int(rank.iloc[0]) if len(rank) else None
        out_rows.append({
            "학교명": r["학교명"],
            "구": r.get("구", ""),
            "미운영순위": rank,
            "최근접_정류장_거리m": round(nearest_dist, 0),
            "최근접_정류장_소속학교": nearest_school,
            "500m_이내": "Y" if nearest_dist <= 500 else "",
            "분류": "기존 노선 연장 가능" if nearest_dist <= 500 else "신규 노선 필요",
        })
    out = pd.DataFrame(out_rows).sort_values("미운영순위")
    out.to_csv(OUT_STOPS_NEAR, index=False, encoding="utf-8-sig")
    print(f"\n  → {OUT_STOPS_NEAR}")
    print(f"\n  [신규 30교 중 500m 이내]")
    near_rows = out[out["500m_이내"] == "Y"]
    print(near_rows.to_string(index=False))

    return out, n_near, n_far


def main():
    print("=" * 72)
    print("Phase 2 PART B — 노선 경사 프로파일 + 운영비 회귀")
    print("=" * 72)

    print("\n[1] 노선 SHP 로드 + 학교명 정규화")
    routes = load_routes()
    print(f"  features: {len(routes)} (운영 12 + 명목 1 명단 매칭)")

    print("\n[2] 10m 간격 샘플링 + slope/DEM 값 추출")
    routes_slope = _extract_slope_along_routes(routes)
    routes_slope.to_csv(OUT_ROUTES_SLOPE, index=False, encoding="utf-8-sig")
    print(f"  → {OUT_ROUTES_SLOPE} ({len(routes_slope)} 노선)")

    print("\n[3] 학교별 집계")
    schools_sum = aggregate_schools(routes_slope)
    schools_sum.to_csv(OUT_SCHOOLS_SUM, index=False, encoding="utf-8-sig")
    print(f"  → {OUT_SCHOOLS_SUM}")
    print()
    print(schools_sum.to_string(index=False))

    # 위험구간 상위 10
    print("\n[4] 노선 경사 위험구간 상위 10")
    risk = routes_slope.nlargest(10, "slope_pct_over_10deg")[[
        "school_short", "purpose", "차수", "length_m",
        "slope_mean", "slope_max", "slope_pct_over_10deg"
    ]]
    risk.to_csv(OUT_RISK_TOP10, index=False, encoding="utf-8-sig")
    print(risk.to_string(index=False))

    # B-3
    compare_position_vs_route(schools_sum)

    # B-4
    regression_operating_cost(schools_sum)

    # B-5
    stops_near_top30()

    print("\n" + "=" * 72)
    print("[DONE] Phase 2 PART B 분석 완료")
    print("=" * 72)


if __name__ == "__main__":
    main()
