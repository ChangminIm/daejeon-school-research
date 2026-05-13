"""통학지원 적격성 점수 — 3개 지표 단순화 모델

지표 (가중치 합 = 1.0):
  1) 동급학교밀도 (0.50): 반경 2km 동급 학교 수 — 적을수록 ↑ (invert)
  2) 도심거리     (0.30): 대전시청에서 거리         — 멀수록 ↑
  3) 학교규모     (0.20): 학생수                    — 적을수록 ↑ (invert)

※ 자치구 중심 거리는 제거 (도심거리로 일원화).
※ 시내버스 접근성, 14교 보너스는 점수에 반영 안 함 (검증용 컬럼만 유지).
"""
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from src.config import (
    DATA_PROCESSED, DATA_EXTERNAL, OUTPUT_TABLES,
    CRS_WGS84, CRS_KOREA, DAEJEON_CITYHALL,
    ELIGIBILITY_WEIGHTS,
)

DENSITY_RADIUS_M = 2000


def _minmax(series):
    s = series.astype(float)
    if s.max() == s.min():
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.min()) / (s.max() - s.min())


def _get_bus14_school_names(schools_df):
    """현행 14교의 정식 학교명 set (검증용 컬럼만)"""
    bus_csv = DATA_EXTERNAL / "bus" / "대전_현행통학차량_14개교.csv"
    if not bus_csv.exists():
        return set()
    bus_df = pd.read_csv(bus_csv, encoding="utf-8-sig")
    from src.integrated_priority import match_bus_to_schools
    matched, _ = match_bus_to_schools(bus_df, schools_df)
    return set(matched["정식학교명"].tolist())


def compute_eligibility(schools_df, verbose=True, debug_cases=None):
    """3개 지표 적격성 점수 산출."""
    sch = schools_df.dropna(subset=["lat", "lon"]).copy().reset_index(drop=True)

    gdf = gpd.GeoDataFrame(
        sch,
        geometry=[Point(xy) for xy in zip(sch["lon"], sch["lat"])],
        crs=CRS_WGS84,
    ).to_crs(CRS_KOREA)

    city_hall = (
        gpd.GeoDataFrame(
            geometry=[Point(DAEJEON_CITYHALL[1], DAEJEON_CITYHALL[0])],
            crs=CRS_WGS84,
        )
        .to_crs(CRS_KOREA)
        .geometry.iloc[0]
    )

    # 원값
    gdf["_원_도심거리m"] = gdf.geometry.distance(city_hall).astype(float)
    gdf["_원_학생수"] = gdf["학생수합계"].astype(float)

    # 동급 학교 밀도 (반경 2km 동급)
    densities = []
    for i, row in gdf.iterrows():
        same_level = gdf[gdf["학교급"] == row["학교급"]]
        d = same_level.geometry.distance(row.geometry)
        densities.append(max(int((d <= DENSITY_RADIUS_M).sum() - 1), 0))
    gdf["_원_학교밀도"] = densities

    # 정규화 (정책 방향)
    gdf["적격성_도심거리"]   = _minmax(gdf["_원_도심거리m"])           # 멀수록 ↑
    gdf["적격성_학교규모"]   = 1.0 - _minmax(gdf["_원_학생수"])         # 적을수록 ↑
    gdf["적격성_학교밀도"]   = 1.0 - _minmax(gdf["_원_학교밀도"])       # 적을수록 ↑

    # 가중합 (3개 지표만)
    gdf["적격성점수"] = (
        gdf["적격성_학교밀도"] * ELIGIBILITY_WEIGHTS["동급학교밀도"]
        + gdf["적격성_도심거리"] * ELIGIBILITY_WEIGHTS["도심거리"]
        + gdf["적격성_학교규모"] * ELIGIBILITY_WEIGHTS["학교규모"]
    ).round(4)

    # 전체순위 (243교 중 적격성 순위)
    gdf["전체순위"] = gdf["적격성점수"].rank(ascending=False, method="min").astype(int)
    # 호환용 alias (integrated_priority 등 기존 코드)
    gdf["적격성순위"] = gdf["전체순위"]

    # 14교 표시
    bus14 = _get_bus14_school_names(sch)
    gdf["현행운영여부"] = gdf["학교명"].apply(lambda n: "Y" if n in bus14 else "")

    # 미운영순위 (14교 제외 학교 중 적격성 순위) — 14교는 NaN
    is_unop = gdf["현행운영여부"] != "Y"
    unop_rank = gdf.loc[is_unop, "적격성점수"].rank(
        ascending=False, method="min"
    ).astype(int)
    gdf["미운영순위"] = np.nan
    gdf.loc[is_unop, "미운영순위"] = unop_rank

    result = pd.DataFrame(gdf.drop(columns=["geometry"]))

    if verbose:
        _print_debug(result, debug_cases)
    return result


def _print_debug(df, cases=None):
    if cases is None:
        cases = ["한밭초등학교", "대전진잠초등학교", "대전구즉초등학교",
                 "대전둔산초등학교", "대전산내초등학교"]
    print("\n" + "─"*100)
    print("[적격성 점수 산식 검증 — 5개 케이스 (3지표)]")
    print("─"*100)
    cols = [
        "학교명", "구", "학생수합계",
        "_원_도심거리m", "_원_학교밀도",
        "적격성_도심거리", "적격성_학교규모", "적격성_학교밀도",
        "적격성점수", "적격성순위", "현행운영여부",
    ]
    avail = [c for c in cols if c in df.columns]
    subset = df[df["학교명"].isin(cases)].copy()
    missing = [c for c in cases if c not in subset["학교명"].tolist()]
    if missing:
        for m in missing:
            kw = m.replace("대전", "").replace("초등학교", "").replace("중학교", "")
            cand = df[df["학교명"].str.contains(kw, regex=False)]
            if len(cand) > 0:
                subset = pd.concat([subset, cand.head(1)])
    print(subset[avail].to_string(index=False))
    hb = subset[subset["학교명"].str.contains("한밭초", regex=False)]
    if len(hb) > 0:
        rank = int(hb["적격성순위"].iloc[0])
        verdict = "✅ 정상" if rank >= 200 else f"⚠️  {rank}위 (재검토)"
        print(f"\n  → 한밭초등학교 적격성 {rank}위  {verdict}")


def run():
    schools = pd.read_csv(DATA_PROCESSED / "schools_with_impact.csv")
    df = compute_eligibility(schools, verbose=True)
    full_out = OUTPUT_TABLES / "통학지원적격성_학교별.csv"
    df.to_csv(full_out, index=False, encoding="utf-8-sig")

    # 신규 검토 대상 = 미운영 학교 중 적격성 상위 30교
    new_targets = df[df["미운영순위"].notna()].sort_values("미운영순위").head(30)
    new_out = OUTPUT_TABLES / "신규검토대상_상위30교.csv"
    new_targets.to_csv(new_out, index=False, encoding="utf-8-sig")

    print(f"\n=== 신규 검토 대상 30교 (현행 14교 제외, 적격성 상위) ===")
    show = ["미운영순위", "전체순위", "학교명", "학교급", "구", "학생수합계", "적격성점수"]
    print(new_targets[show].head(30).to_string(index=False))

    bus14 = df[df["현행운영여부"] == "Y"]
    print(f"\n=== 점수 신뢰도: 현행 14교 적격성 전체순위 ===")
    bus_show = ["전체순위", "학교명", "학생수합계", "적격성점수"]
    print(bus14.sort_values("전체순위")[bus_show].to_string(index=False))
    print(f"\n  현행 14교 평균 전체순위: {bus14['전체순위'].mean():.1f}위 / 243교")
    print(f"  매칭: {len(bus14)}건")

    # 신규 30교에 14교 포함 검증
    new_names = set(new_targets["학교명"])
    bus_names = set(bus14["학교명"])
    overlap = new_names & bus_names
    print(f"  신규 30교 중 14교 포함: {len(overlap)}건 (0이어야 정상)")
    return df


if __name__ == "__main__":
    run()
