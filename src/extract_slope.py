"""학교·재개발 사업지 경사도 추출 (Phase A).

입력
  data/processed/schools_geocoded.csv        (EPSG:4326)
  data/processed/redev_projects_geocoded.csv (EPSG:4326)
  data/external/dem/대전_slope_5m.tif        (EPSG:5186, degree)

출력
  data/processed/schools_with_slope.csv
    + slope_point, slope_300m_mean, slope_300m_max
  data/processed/redev_with_slope.csv
    + slope_point, slope_300m_mean, slope_300m_max, slope_500m_mean, slope_500m_max

추가 콘솔 출력
  - 분포 통계, 단계별 평균, 14교 상세 표 + Mann-Whitney p-value
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from scipy import stats as sp_stats

from src.config import DATA_PROCESSED, DATA_EXTERNAL

SLOPE_TIF = DATA_EXTERNAL / "dem" / "대전_slope_5m.tif"

SCHOOLS_CSV = DATA_PROCESSED / "schools_geocoded.csv"
REDEV_CSV = DATA_PROCESSED / "redev_projects_geocoded.csv"

OUT_SCHOOLS = DATA_PROCESSED / "schools_with_slope.csv"
OUT_REDEV = DATA_PROCESSED / "redev_with_slope.csv"

ANALYSIS_CRS = "EPSG:5186"

# 추진 단계 그룹 (가설 검증·출력용)
STAGE_GROUPS = {
    "공사중":       ["1_공사중"],
    "관리처분":     ["2_관리처분"],
    "사업시행":     ["3_사업시행"],
    "조합·추진위": ["4_조합설립", "5_초기"],
    "입안·미정":   ["6_입안", "9_미정"],
}


def _read_csv_bom(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def _to_5186_points(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """lon/lat 컬럼으로 4326 점 GDF 만들고 5186으로 변환."""
    valid = df.dropna(subset=["lat", "lon"]).copy()
    gdf = gpd.GeoDataFrame(
        valid,
        geometry=gpd.points_from_xy(valid["lon"], valid["lat"]),
        crs="EPSG:4326",
    ).to_crs(ANALYSIS_CRS)
    return gdf


def _sample_point(src: rasterio.io.DatasetReader, x: float, y: float, nodata) -> float:
    """단일 점 픽셀값. nodata/범위밖이면 NaN."""
    try:
        val = next(src.sample([(x, y)]))[0]
    except (StopIteration, IndexError):
        return np.nan
    if val is None or not np.isfinite(val) or val == nodata:
        return np.nan
    return float(val)


def _zonal_buffer(src, point_geom, radius_m: float, nodata) -> dict:
    """점 주변 buffer(radius_m, 5186 미터) zonal mean/max."""
    buf = point_geom.buffer(radius_m)
    try:
        out, _ = rio_mask(src, [buf], crop=True, all_touched=True,
                          nodata=nodata, filled=True)
    except (ValueError, Exception):
        return {"mean": np.nan, "max": np.nan}
    arr = out[0]
    valid_mask = np.isfinite(arr) & (arr != nodata)
    if valid_mask.sum() == 0:
        return {"mean": np.nan, "max": np.nan}
    vals = arr[valid_mask]
    return {"mean": float(vals.mean()), "max": float(vals.max())}


def _extract(gdf: gpd.GeoDataFrame, radii: list[int]) -> pd.DataFrame:
    """gdf(EPSG:5186) 각 점에 대해 point 값 + 각 radius zonal."""
    out_rows = []
    with rasterio.open(SLOPE_TIF) as src:
        nodata = src.nodata
        for idx, row in gdf.iterrows():
            p = row.geometry
            rec = {"_idx": idx}
            rec["slope_point"] = _sample_point(src, p.x, p.y, nodata)
            for r in radii:
                z = _zonal_buffer(src, p, r, nodata)
                rec[f"slope_{r}m_mean"] = z["mean"]
                rec[f"slope_{r}m_max"] = z["max"]
            out_rows.append(rec)
    return pd.DataFrame(out_rows).set_index("_idx")


# ===== 학교 =====

def extract_school_slope():
    print("\n" + "=" * 70)
    print("[학교 경사도 추출]")
    print("=" * 70)

    df = _read_csv_bom(SCHOOLS_CSV)
    print(f"  입력: {len(df)}교")

    gdf = _to_5186_points(df)
    print(f"  유효 좌표: {len(gdf)}교")

    slope_df = _extract(gdf, radii=[300])

    # 원 인덱스에 다시 붙임
    out = df.copy()
    for col in ["slope_point", "slope_300m_mean", "slope_300m_max"]:
        out[col] = np.nan
    out.loc[gdf.index, ["slope_point", "slope_300m_mean", "slope_300m_max"]] = (
        slope_df[["slope_point", "slope_300m_mean", "slope_300m_max"]].values
    )

    out.to_csv(OUT_SCHOOLS, index=False, encoding="utf-8-sig")
    print(f"  저장: {OUT_SCHOOLS}")

    # === 콘솔 보고 ===
    valid = out["slope_300m_mean"].notna()
    n_ok = int(valid.sum())
    n_nan = int((~valid).sum())
    print(f"\n  추출 성공: {n_ok}교 / NaN: {n_nan}교")
    if n_nan > 0:
        nan_names = out.loc[~valid, "학교명"].tolist()
        print(f"  NaN 학교: {nan_names}")

    s = out.loc[valid, "slope_300m_mean"]
    print(f"\n  [slope_300m_mean 분포]")
    print(f"    min   = {s.min():.2f}°")
    print(f"    Q1    = {s.quantile(0.25):.2f}°")
    print(f"    median= {s.median():.2f}°")
    print(f"    mean  = {s.mean():.2f}°")
    print(f"    Q3    = {s.quantile(0.75):.2f}°")
    print(f"    max   = {s.max():.2f}°")

    return out


# ===== 14교 매칭 + 상세 표 =====

def _bus14_names(schools_df):
    """현행 14교의 정식 학교명 set."""
    bus_csv = DATA_EXTERNAL / "bus" / "대전_현행통학차량_14개교.csv"
    if not bus_csv.exists():
        print(f"  [경고] 14교 CSV 없음: {bus_csv}")
        return set()
    bus_df = pd.read_csv(bus_csv, encoding="utf-8-sig")
    from src.integrated_priority import match_bus_to_schools
    matched, unmatched = match_bus_to_schools(bus_df, schools_df)
    if unmatched:
        print(f"  [경고] 14교 매칭 실패: {unmatched}")
    return set(matched["정식학교명"].tolist())


def report_bus14_table(schools_with_slope: pd.DataFrame):
    print("\n" + "=" * 70)
    print("[현행 14교 경사도 상세 (Phase B-2 회귀 사전 검토)]")
    print("=" * 70)

    bus14 = _bus14_names(schools_with_slope)
    if not bus14:
        return

    df = schools_with_slope.copy()
    df["is_bus14"] = df["학교명"].isin(bus14)

    bus14_rows = df[df["is_bus14"]].copy()
    others = df[~df["is_bus14"]].copy()

    print(f"\n  14교 매칭: {len(bus14_rows)}건 (CSV 14행 + 기성초 본교/분교 분리로 15건 가능)")
    print(f"\n  {'학교명':<25} {'slope_point':>12} {'slope_300m_mean':>17} {'slope_300m_max':>16}")
    print(f"  {'-'*25} {'-'*12} {'-'*17} {'-'*16}")
    for _, r in bus14_rows.sort_values("학교명").iterrows():
        sp = f"{r['slope_point']:.2f}°" if pd.notna(r["slope_point"]) else "NaN"
        sm = f"{r['slope_300m_mean']:.2f}°" if pd.notna(r["slope_300m_mean"]) else "NaN"
        smx = f"{r['slope_300m_max']:.2f}°" if pd.notna(r["slope_300m_max"]) else "NaN"
        print(f"  {r['학교명']:<25} {sp:>12} {sm:>17} {smx:>16}")
    print(f"  {'-'*25} {'-'*12} {'-'*17} {'-'*16}")
    print(f"  {'평균':<25} "
          f"{bus14_rows['slope_point'].mean():>11.2f}° "
          f"{bus14_rows['slope_300m_mean'].mean():>16.2f}° "
          f"{bus14_rows['slope_300m_max'].mean():>15.2f}°")

    # Mann-Whitney U: 14교 vs 나머지
    a = bus14_rows["slope_300m_mean"].dropna().values
    b = others["slope_300m_mean"].dropna().values
    if len(a) >= 2 and len(b) >= 2:
        u, p = sp_stats.mannwhitneyu(a, b, alternative="two-sided")
        diff = a.mean() - b.mean()
        sign = "가파름" if diff > 0 else "완만"
        print(f"\n  14교 vs 나머지 ({len(a)} vs {len(b)})")
        print(f"    14교 평균   = {a.mean():.2f}°")
        print(f"    나머지 평균 = {b.mean():.2f}°")
        print(f"    차이        = {diff:+.2f}° (14교가 {sign})")
        print(f"    Mann-Whitney U={u:.1f}, p={p:.4f}")


# ===== 재개발 =====

def extract_redev_slope():
    print("\n" + "=" * 70)
    print("[재개발 경사도 추출]")
    print("=" * 70)

    df = _read_csv_bom(REDEV_CSV)
    print(f"  입력: {len(df)}건")
    print(f"  지오코딩 성공: {df['lat'].notna().sum()}건 / NaN: {df['lat'].isna().sum()}건")

    gdf = _to_5186_points(df)
    print(f"  분석 대상: {len(gdf)}건")

    slope_df = _extract(gdf, radii=[300, 500])

    out = df.copy()
    cols = ["slope_point", "slope_300m_mean", "slope_300m_max",
            "slope_500m_mean", "slope_500m_max"]
    for c in cols:
        out[c] = np.nan
    out.loc[gdf.index, cols] = slope_df[cols].values

    out.to_csv(OUT_REDEV, index=False, encoding="utf-8-sig")
    print(f"  저장: {OUT_REDEV}")

    # === 콘솔 보고 ===
    valid = out["slope_500m_mean"].notna()
    n_ok = int(valid.sum())
    n_nan = int((~valid).sum())
    print(f"\n  추출 성공: {n_ok}건 / NaN: {n_nan}건")

    # 추진 단계별 평균 (slope_500m_mean)
    print(f"\n  [추진 단계별 slope_500m_mean 평균]")
    stage_col = "통학영향_임박도"
    for label, codes in STAGE_GROUPS.items():
        sub = out[out[stage_col].isin(codes)]
        sub_valid = sub.dropna(subset=["slope_500m_mean"])
        if len(sub_valid) == 0:
            print(f"    {label:<10}: -    (n=0)")
        else:
            m = sub_valid["slope_500m_mean"].mean()
            print(f"    {label:<10}: {m:5.2f}°  (n={len(sub_valid)}, "
                  f"단계={'+'.join(codes)})")

    # 0_완료 별도 (사용자 명시 5그룹 외 참고)
    done = out[out[stage_col] == "0_완료"].dropna(subset=["slope_500m_mean"])
    if len(done) > 0:
        print(f"    {'(완료)':<10}: {done['slope_500m_mean'].mean():5.2f}°  "
              f"(n={len(done)}, 단계=0_완료, 참고용)")

    return out


def main():
    if not SLOPE_TIF.exists():
        raise FileNotFoundError(f"Slope 래스터 없음: {SLOPE_TIF}")

    schools_with_slope = extract_school_slope()
    report_bus14_table(schools_with_slope)
    extract_redev_slope()

    print("\n" + "=" * 70)
    print("[DONE] Phase A 완료")
    print(f"  - {OUT_SCHOOLS}")
    print(f"  - {OUT_REDEV}")
    print("=" * 70)


if __name__ == "__main__":
    main()
