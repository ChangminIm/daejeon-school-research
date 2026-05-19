"""받은 통학차량 노선/정류소 SHP 진단.

입력 (외부):
  D:/04_제안서/03_기타/대전광역시/데이터/대전광역시 통학차량 운행 노선(260519)/
    1.대전광역시 통학차량 전체 노선(260519)/
      대전광역시 통학차량_노선도.shp        (Line, EPSG:5179)
      대전광역시 통학차량_탑승장소.shp      (Point)

산출:
  콘솔 보고
  data/processed/routes_summary.csv  (학교별 요약)
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import pandas as pd
import geopandas as gpd

from src.config import DATA_PROCESSED

BASE = Path(
    r"D:\04_제안서\03_기타\대전광역시\데이터"
    r"\대전광역시 통학차량 운행 노선(260519)"
    r"\1.대전광역시 통학차량 전체 노선(260519)"
)
ROUTE_SHP = BASE / "대전광역시 통학차량_노선도.shp"
STOP_SHP = BASE / "대전광역시 통학차량_탑승장소.shp"

OUT_SUMMARY = DATA_PROCESSED / "routes_summary.csv"

# 14교 약식 명단 (사용자 명시)
BUS14_SHORT = [
    "진잠초", "구즉초", "남선초", "기성초", "흥도초", "계산초",
    "장동초", "동명초", "산내초", "산서초", "산흥초", "세천초",
    "신탄진용정초", "대전원신흥초 복용분교장",
]

HR = "─" * 72


# ===== 공통 헬퍼 =====

def _load(shp_path: Path) -> gpd.GeoDataFrame:
    return gpd.read_file(shp_path, encoding="cp949")


def _null_or_zero_rate(s: pd.Series, zero_values=(0, "", "-", "차량번호없음", "없음")):
    n = len(s)
    if n == 0:
        return 0.0, 0
    missing = s.isna().sum()
    empty_or_dummy = s.fillna("").astype(str).isin([str(v) for v in zero_values]).sum()
    # 숫자 0 처리
    if pd.api.types.is_numeric_dtype(s):
        zero_num = (s == 0).fillna(False).sum()
    else:
        zero_num = 0
    total_missing = int(missing + empty_or_dummy + zero_num)
    total_missing = min(total_missing, n)
    return total_missing / n * 100, total_missing


# ===== 노선 SHP 분석 =====

def inspect_routes(g_route: gpd.GeoDataFrame):
    print("\n" + "=" * 72)
    print("[1] 노선 SHP 진단")
    print("=" * 72)
    print(f"  파일      : {ROUTE_SHP.name}")
    print(f"  CRS       : {g_route.crs}  (EPSG:{g_route.crs.to_epsg()})")
    print(f"  features  : {len(g_route)}")
    print(f"  geom type : {g_route.geom_type.value_counts().to_dict()}")

    print(f"\n  컬럼 목록 + dtype:")
    for c in g_route.columns:
        print(f"    {c:<14}  {str(g_route[c].dtype)}")

    # 샘플 5건
    print(f"\n  샘플 5건 (geometry 제외):")
    cols_no_geom = [c for c in g_route.columns if c != "geometry"]
    sample = g_route[cols_no_geom].head(5).to_string(index=True, max_colwidth=22)
    print(sample)

    # 그룹별 카운트
    print(f"\n  [지역 분포]")
    print("    " + g_route["지역"].value_counts(dropna=False).to_string().replace("\n", "\n    "))
    print(f"\n  [학급 분포]")
    print("    " + g_route["학급"].value_counts(dropna=False).to_string().replace("\n", "\n    "))
    print(f"\n  [목적 분포 (등하교)]")
    print("    " + g_route["목적"].value_counts(dropna=False).to_string().replace("\n", "\n    "))
    print(f"\n  [차수 분포]")
    print("    " + g_route["차수"].value_counts(dropna=False).sort_index().to_string().replace("\n", "\n    "))
    print(f"\n  [호차 분포]")
    print("    " + g_route["호차"].value_counts(dropna=False).sort_index().to_string().replace("\n", "\n    "))

    # 학교별 분포
    print(f"\n  [학교(이름)별 feature 수]")
    school_counts = g_route["이름"].value_counts(dropna=False).sort_values(ascending=False)
    for name, cnt in school_counts.items():
        print(f"    {str(name):<32}  {cnt:>3}건")
    print(f"    총 {len(school_counts)}개 학교, 합 {school_counts.sum()}건")

    # NULL/0 비율
    print(f"\n  [컬럼별 NULL/0/더미 비율]")
    candidates = ["길이", "인승", "차량No", "승하차인원", "비고"]
    for c in candidates:
        if c in g_route.columns:
            rate, n_bad = _null_or_zero_rate(g_route[c])
            head_vals = g_route[c].dropna().head(3).tolist()
            print(f"    {c:<10}  {rate:>5.1f}%  (n_bad={n_bad}/{len(g_route)})  sample={head_vals}")


# ===== 정류소 SHP 분석 =====

def inspect_stops(g_stop: gpd.GeoDataFrame):
    print("\n" + "=" * 72)
    print("[2] 정류소 SHP 진단")
    print("=" * 72)
    print(f"  파일      : {STOP_SHP.name}")
    print(f"  CRS       : {g_stop.crs}  (EPSG:{g_stop.crs.to_epsg()})")
    print(f"  features  : {len(g_stop)}")
    print(f"  geom type : {g_stop.geom_type.value_counts().to_dict()}")

    print(f"\n  컬럼 목록 + dtype:")
    for c in g_stop.columns:
        print(f"    {c:<14}  {str(g_stop[c].dtype)}")

    print(f"\n  샘플 5건 (geometry 제외):")
    cols_no_geom = [c for c in g_stop.columns if c != "geometry"]
    print(g_stop[cols_no_geom].head(5).to_string(index=True, max_colwidth=22))

    # 학교명 컬럼 추정 (이름·학교명·학교 등 우선)
    name_col = None
    for cand in ["이름", "학교명", "학교", "학교이름"]:
        if cand in g_stop.columns:
            name_col = cand
            break
    if name_col:
        print(f"\n  [학교({name_col})별 정류소 수]")
        cnts = g_stop[name_col].value_counts(dropna=False).sort_values(ascending=False)
        for name, cnt in cnts.items():
            print(f"    {str(name):<32}  {cnt:>3}개")
        print(f"    총 {len(cnts)}개 학교, 합 {cnts.sum()}개 정류소")
    else:
        print(f"\n  [경고] 학교명 컬럼을 찾지 못함")

    # 정류소명 컬럼 추정
    stop_name_col = None
    for cand in ["정류소명", "정류장명", "탑승장소", "장소명", "정류소"]:
        if cand in g_stop.columns:
            stop_name_col = cand
            break
    if stop_name_col:
        sample_names = g_stop[stop_name_col].dropna().head(5).tolist()
        print(f"\n  정류소명 컬럼 = '{stop_name_col}', 샘플: {sample_names}")
    else:
        print(f"\n  [경고] 정류소명 컬럼 후보 없음")
    return name_col


# ===== 14교 매칭 =====

def check_bus14_match(g_route: gpd.GeoDataFrame):
    print("\n" + "=" * 72)
    print("[3] 14교 매칭 확인")
    print("=" * 72)

    school_names_in_shp = set(g_route["이름"].dropna().unique())
    print(f"  SHP 내 고유 학교: {len(school_names_in_shp)}개")

    matched = {}
    unmatched = []
    for short in BUS14_SHORT:
        # "초" 단독은 너무 짧음 — "흥도초" 같은 약식을 SHP의 정식 이름("흥도초등학교")에 부분 매칭
        # 약식에서 "초"를 "초등학교"로 확장하거나, 단순 contains 사용
        key = short.replace(" ", "")  # 공백 제거 키
        candidates = [
            full for full in school_names_in_shp
            if (short in full) or (key in full.replace(" ", ""))
        ]
        # 분교장 분리: "기성초" 매칭 시 본교 + 분교 둘 다 잡힘 — 분리 처리 안 하고 모두 표시
        if candidates:
            matched[short] = candidates
        else:
            unmatched.append(short)

    print(f"\n  매칭 결과: {len(matched)}/14교")
    for short, fulls in matched.items():
        flag = " (다중매칭)" if len(fulls) > 1 else ""
        print(f"    ✓ {short:<24} → {fulls}{flag}")

    if unmatched:
        print(f"\n  ✗ 매칭 실패 ({len(unmatched)}교):")
        for u in unmatched:
            print(f"    - {u}")
            # 부분 매칭 후보 제안 (앞 2~3 글자)
            stem = u.replace("초", "").replace(" ", "")[:3]
            suggest = [n for n in school_names_in_shp if stem and stem in n][:3]
            if suggest:
                print(f"      후보(stem='{stem}'): {suggest}")
    else:
        print(f"\n  모든 14교가 SHP에 존재")

    # SHP에 있는데 14교 명단에 없는 학교
    matched_fulls = {f for fulls in matched.values() for f in fulls}
    extra = school_names_in_shp - matched_fulls
    if extra:
        print(f"\n  [참고] SHP에 있으나 14교 명단 외 학교: {sorted(extra)}")

    return matched, unmatched


# ===== Geometry 검증 + 학교별 요약 =====

def inspect_geometry(g_route: gpd.GeoDataFrame, g_stop: gpd.GeoDataFrame, stop_name_col=None):
    print("\n" + "=" * 72)
    print("[4] 노선 geometry 검증 (EPSG:5179, 미터 단위)")
    print("=" * 72)

    # 5179이 이미 미터 단위 (CRS_KOREA 동일)
    g = g_route.copy()
    if g.crs.to_epsg() != 5179:
        g = g.to_crs("EPSG:5179")
    g["length_m"] = g.geometry.length

    print(f"  전체 노선 평균 길이: {g['length_m'].mean():,.0f} m "
          f"(중앙값 {g['length_m'].median():,.0f}, "
          f"min {g['length_m'].min():,.0f}, max {g['length_m'].max():,.0f})")

    # 등교 vs 하교
    print(f"\n  [목적별 평균 길이]")
    for purpose, sub in g.groupby("목적"):
        print(f"    {str(purpose):<8}  n={len(sub):>3}  평균 {sub['length_m'].mean():,.0f} m  "
              f"중앙값 {sub['length_m'].median():,.0f} m")

    # raw 길이 컬럼과 비교
    print(f"\n  [raw '길이' 컬럼 vs 계산 length_m 비교]")
    if "길이" in g.columns and pd.api.types.is_numeric_dtype(g["길이"]):
        raw = g["길이"].dropna()
        if (raw < 100).all():
            scale = "km 추정"
            g["raw_m"] = g["길이"] * 1000
        else:
            scale = "m 추정"
            g["raw_m"] = g["길이"]
        diff = (g["raw_m"] - g["length_m"]).dropna()
        print(f"    raw '길이' 컬럼 분포: min={raw.min()}, max={raw.max()}, mean={raw.mean():.2f} ({scale})")
        if len(diff):
            print(f"    raw(m 환산) vs 계산 length_m 차이: mean={diff.mean():,.0f}m, "
                  f"abs mean={diff.abs().mean():,.0f}m")
    else:
        print("    raw '길이' 컬럼 없거나 비수치")

    # 학교별 집계
    agg_route = g.groupby("이름").agg(
        노선수=("Id", "count"),
        총길이_m=("length_m", "sum"),
        평균길이_m=("length_m", "mean"),
        등교노선=("목적", lambda s: (s == "등교").sum()),
        하교노선=("목적", lambda s: (s == "하교").sum()),
        호차수=("호차", "nunique"),
    ).round(0).reset_index()

    # 정류소도 학교별 카운트
    if g_stop is not None and len(g_stop) > 0:
        # 학교명 컬럼 추정
        stop_school_col = None
        for cand in ["이름", "학교명", "학교"]:
            if cand in g_stop.columns:
                stop_school_col = cand
                break
        if stop_school_col:
            stop_cnt = (g_stop.groupby(stop_school_col).size()
                        .rename("정류소수").reset_index()
                        .rename(columns={stop_school_col: "이름"}))
            agg_route = agg_route.merge(stop_cnt, on="이름", how="left")
            agg_route["정류소수"] = agg_route["정류소수"].fillna(0).astype(int)

    # 등하교 길이 차이
    print(f"\n  [학교별 등교 vs 하교 평균 길이 차이 (등교 - 하교)]")
    for school, sub in g.groupby("이름"):
        eg = sub[sub["목적"] == "등교"]["length_m"]
        hg = sub[sub["목적"] == "하교"]["length_m"]
        if len(eg) and len(hg):
            diff_m = eg.mean() - hg.mean()
            print(f"    {str(school):<32}  등교평균 {eg.mean():>5,.0f}m  하교평균 {hg.mean():>5,.0f}m  Δ {diff_m:+,.0f}m")

    return agg_route


# ===== 데이터 품질 보고 =====

def quality_report(g_route, g_stop):
    print("\n" + "=" * 72)
    print("[5] 데이터 품질 보고")
    print("=" * 72)
    print("  사용 가능 (확실):")
    print("    · geometry (LineString, EPSG:5179)")
    print("    · 이름 (학교명, 정식 명칭)")
    print("    · 목적 (등교/하교)")
    print("    · 차수 (회차)")
    print("    · 호차 (차량 일련번호)")
    print("    · 학급 (초등학교/중학교 등)")

    print("\n  검증 필요/부정확:")
    # 길이 컬럼
    if "길이" in g_route.columns:
        raw_med = g_route["길이"].median()
        unit = "km" if raw_med < 100 else "m"
        print(f"    · 길이 컬럼: median={raw_med} ({unit} 추정). geometry.length로 직접 산출 권장")
    if "승하차인원" in g_route.columns:
        rate, _ = _null_or_zero_rate(g_route["승하차인원"])
        med = g_route["승하차인원"].median()
        print(f"    · 승하차인원: median={med}, NULL/0 비율 {rate:.1f}%")
    if "인승" in g_route.columns:
        ins_counts = g_route["인승"].value_counts().to_dict()
        print(f"    · 인승(차량 정원): 분포={ins_counts}")

    print("\n  빠진 정보:")
    if "차량No" in g_route.columns:
        dummy = (g_route["차량No"].astype(str).isin(["차량번호없음", "없음", "-", ""]) | g_route["차량No"].isna()).sum()
        print(f"    · 차량No 더미값('차량번호없음' 등) 비율: {dummy/len(g_route)*100:.1f}%")
    print("    · 운행 시간표 (출발/도착 시각) 없음")
    print("    · 학생 명단/탑승자 정보 없음 (개인정보 사유)")

    if g_stop is not None:
        # 정류소: 정류소명 같은 컬럼 유무 재확인
        cols_stop = [c for c in g_stop.columns if c != "geometry"]
        print(f"\n  정류소 SHP 컬럼: {cols_stop}")


# ===== 메인 =====

def main():
    print(HR)
    print("통학차량 노선/정류소 SHP 진단")
    print(HR)

    if not ROUTE_SHP.exists():
        raise FileNotFoundError(ROUTE_SHP)
    if not STOP_SHP.exists():
        raise FileNotFoundError(STOP_SHP)

    g_route = _load(ROUTE_SHP)
    g_stop = _load(STOP_SHP)

    inspect_routes(g_route)
    stop_school_col = inspect_stops(g_stop)
    check_bus14_match(g_route)
    agg_route = inspect_geometry(g_route, g_stop, stop_school_col)
    quality_report(g_route, g_stop)

    # 요약 CSV
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    agg_route.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")
    print("\n" + "=" * 72)
    print(f"[저장] {OUT_SUMMARY}")
    print(agg_route.to_string(index=False))
    print("=" * 72)


if __name__ == "__main__":
    main()
