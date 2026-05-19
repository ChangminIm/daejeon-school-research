"""운영 12교 기준 회귀 재실행 + 운영상태 컬럼이 추가된 학교 데이터 생성.

산출
  data/processed/통학차량_운영현황_정확.csv  (14교 × 운영상태)
  data/processed/schools_with_slope_v2.csv  (243교 × 운영상태)
  콘솔: 회귀 결과 (운영 12교 기준, statsmodels Logit + sklearn)
"""
from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import statsmodels.api as sm

from src.config import (
    DATA_PROCESSED, DATA_EXTERNAL, CRS_WGS84, CRS_KOREA, DAEJEON_CITYHALL,
    BUS_OPERATING_12_SHORT, BUS_PLANNED_1_SHORT, BUS_NOMINAL_1_SHORT,
)

DENSITY_RADIUS_M = 2000  # tonghak_eligibility 동일

OUT_BUS_STATUS = DATA_PROCESSED / "통학차량_운영현황_정확.csv"
OUT_SCHOOLS_V2 = DATA_PROCESSED / "schools_with_slope_v2.csv"


# ===== 운영현황 CSV =====

def build_bus_status_csv():
    """data/external/bus/대전_현행통학차량_14개교.csv + 운영상태 컬럼."""
    bus_csv = DATA_EXTERNAL / "bus" / "대전_현행통학차량_14개교.csv"
    df = pd.read_csv(bus_csv, encoding="utf-8-sig")

    def status_of(name):
        if name in BUS_OPERATING_12_SHORT:
            return "운영"
        if name in BUS_PLANNED_1_SHORT:
            return "예정"
        if name in BUS_NOMINAL_1_SHORT:
            return "명목"
        return "미분류"

    df["운영상태"] = df["학교명"].map(status_of)
    df.to_csv(OUT_BUS_STATUS, index=False, encoding="utf-8-sig")

    print(f"\n[운영현황 CSV] {OUT_BUS_STATUS}")
    print(df[["학교명", "이용학생수", "차량대수", "운영상태"]].to_string(index=False))
    return df


# ===== 운영 12교 정식 학교명 set =====

def get_operating_set(schools_df):
    """운영 12교의 정식 학교명 set (match_bus_to_schools 활용)."""
    bus_csv = DATA_EXTERNAL / "bus" / "대전_현행통학차량_14개교.csv"
    bus_df = pd.read_csv(bus_csv, encoding="utf-8-sig")

    # 운영상태별로 분리
    bus_df["__상태"] = bus_df["학교명"].map(lambda n: (
        "운영" if n in BUS_OPERATING_12_SHORT
        else "예정" if n in BUS_PLANNED_1_SHORT
        else "명목" if n in BUS_NOMINAL_1_SHORT
        else "미분류"
    ))

    from src.integrated_priority import match_bus_to_schools

    def names_of(state):
        sub = bus_df[bus_df["__상태"] == state]
        if len(sub) == 0:
            return set()
        m, _ = match_bus_to_schools(sub, schools_df)
        return set(m["정식학교명"].tolist())

    return {
        "운영": names_of("운영"),
        "예정": names_of("예정"),
        "명목": names_of("명목"),
    }


# ===== schools_with_slope_v2.csv =====

def build_schools_v2():
    """schools_with_slope.csv + 운영상태 컬럼 + 회귀용 파생컬럼."""
    sch = pd.read_csv(DATA_PROCESSED / "schools_with_slope.csv", encoding="utf-8-sig")
    state_sets = get_operating_set(sch)

    def state_of(name):
        for state, names in state_sets.items():
            if name in names:
                return state
        return "미운영"
    sch["운영상태"] = sch["학교명"].apply(state_of)

    # === 회귀 변수 계산 (학생수·도심거리·학교밀도·경사도) ===
    # 도심거리는 시청까지 m (5179)
    valid = sch.dropna(subset=["lat", "lon"]).copy()
    gdf = gpd.GeoDataFrame(
        valid,
        geometry=[Point(xy) for xy in zip(valid["lon"], valid["lat"])],
        crs=CRS_WGS84,
    ).to_crs(CRS_KOREA)

    city_hall = (
        gpd.GeoDataFrame(
            geometry=[Point(DAEJEON_CITYHALL[1], DAEJEON_CITYHALL[0])],
            crs=CRS_WGS84,
        ).to_crs(CRS_KOREA).geometry.iloc[0]
    )
    gdf["도심거리_km"] = (gdf.geometry.distance(city_hall) / 1000.0).astype(float)

    # 동급 학교 밀도 (반경 2km 동급 학교 수 - 1)
    densities = []
    for i, row in gdf.iterrows():
        same_level = gdf[gdf["학교급"] == row["학교급"]]
        d = same_level.geometry.distance(row.geometry)
        densities.append(max(int((d <= DENSITY_RADIUS_M).sum() - 1), 0))
    gdf["학교밀도_2km"] = densities

    # 학생수 (이미 schools_with_slope에 있음)
    out = pd.DataFrame(gdf.drop(columns=["geometry"]))
    # 회귀 종속변수: 운영 12교만 y=1
    out["y_운영12"] = (out["운영상태"] == "운영").astype(int)

    out.to_csv(OUT_SCHOOLS_V2, index=False, encoding="utf-8-sig")
    print(f"\n[schools_with_slope_v2] {OUT_SCHOOLS_V2}")
    print(f"  운영상태 분포:\n{out['운영상태'].value_counts().to_string()}")
    return out


# ===== 회귀 재실행 =====

def run_regression(df):
    print("\n" + "=" * 72)
    print("[회귀] 운영 12교 기준 로지스틱 회귀 (표준화)")
    print("=" * 72)

    # 분석 대상: 운영 12교(y=1) vs 미운영(y=0)
    # 예정 1교(흥도초) + 명목 1교(신탄진용정초)는 분석에서 제외 (애매)
    use = df[df["운영상태"].isin(["운영", "미운영"])].copy()
    use = use.dropna(subset=["slope_300m_mean", "도심거리_km", "학교밀도_2km",
                              "학생수합계"])

    y = use["y_운영12"].astype(int).values
    X_raw = use[["학생수합계", "도심거리_km", "학교밀도_2km", "slope_300m_mean"]].astype(float).values

    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    print(f"  표본: {len(use)}교 (y=1 운영 12교: {n_pos}, y=0 미운영: {n_neg})")

    # 표준화 (평균 0, 표준편차 1)
    X = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0, ddof=0)
    X_const = sm.add_constant(X, has_constant="add")
    feature_names = ["학생수", "도심거리", "학교밀도", "경사도"]

    # statsmodels Logit
    model = sm.Logit(y, X_const)
    try:
        res = model.fit(disp=False, method="lbfgs", maxiter=500)
    except Exception as e:
        print(f"  [에러] Logit fit 실패: {e}")
        return

    params = res.params
    pvals = res.pvalues
    ci = res.conf_int()
    se = res.bse

    print(f"\n  AIC = {res.aic:.2f}  |  BIC = {res.bic:.2f}  |  "
          f"Pseudo R² = {res.prsquared:.3f}")
    print(f"  로그우도: {res.llf:.2f}   |   수렴: {res.mle_retvals.get('converged')}")

    print(f"\n  {'변수':<10} {'β':>8} {'SE':>7} {'z':>7} {'p':>10} {'95% CI':>22}")
    print(f"  {'-'*10} {'-'*8} {'-'*7} {'-'*7} {'-'*10} {'-'*22}")
    print(f"  {'(절편)':<10} {params[0]:>8.3f} {se[0]:>7.3f} "
          f"{params[0]/se[0]:>7.2f} {pvals[0]:>10.4f} "
          f"[{ci[0,0]:>+6.2f},{ci[0,1]:>+6.2f}]")
    for i, name in enumerate(feature_names):
        j = i + 1
        z = params[j] / se[j]
        print(f"  {name:<10} {params[j]:>8.3f} {se[j]:>7.3f} "
              f"{z:>7.2f} {pvals[j]:>10.4f} "
              f"[{ci[j,0]:>+6.2f},{ci[j,1]:>+6.2f}]")

    # 분류 매칭률 (sklearn)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    proba = clf.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)

    n_tp = int(((pred == 1) & (y == 1)).sum())
    n_fn = int(((pred == 0) & (y == 1)).sum())
    print(f"\n  [매칭률] 운영 12교 중 모델이 1로 예측한 수: {n_tp}/{n_pos} "
          f"({n_tp/n_pos*100:.1f}%)")
    # 상위 N 매칭: probability 상위 12교가 운영 12교와 얼마나 일치하는지
    use["__prob"] = proba
    top12 = use.sort_values("__prob", ascending=False).head(12)
    top12_hit = int((top12["y_운영12"] == 1).sum())
    print(f"  [확률 상위 12교가 운영 12교에 포함되는 수]: {top12_hit}/12 "
          f"({top12_hit/12*100:.1f}%)")

    # 변수별 효과 방향 한줄 요약
    print(f"\n  [부호 해석]")
    for i, name in enumerate(feature_names):
        beta = params[i+1]
        p = pvals[i+1]
        sig = "★" if p < 0.05 else "·" if p < 0.10 else " "
        sign = "+" if beta > 0 else "-"
        print(f"    {sig} {name}: β = {beta:+.3f}, p = {p:.3f}  "
              f"({sign} 방향 → 운영 학교일수록 {name}이 {'높음' if beta > 0 else '낮음'})")

    return res


def main():
    print("=" * 72)
    print("운영 12교 기준 데이터 정정 + 회귀 재실행")
    print("=" * 72)

    print("\n[1] 운영현황 CSV 생성 (14교 × 운영상태)")
    build_bus_status_csv()

    print("\n[2] schools_with_slope_v2.csv 생성 (243교 × 운영상태 + 회귀변수)")
    df_v2 = build_schools_v2()

    print("\n[3] 운영 12교 기준 회귀 재실행")
    run_regression(df_v2)


if __name__ == "__main__":
    main()
