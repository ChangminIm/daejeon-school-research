"""신규 _re 통학차량 노선 SHP 진단.

이전 작업 (src/inspect_routes.py, 커밋 6263120)과 비교:
  · feature 수 변화 (185 → ?)
  · "? 표기" 정리 여부 (대전원신흥초 ?, 신탄진용정초 ?)
  · 흥도초 포함 여부 (핵심)
  · sinuosity (실 경로 vs 직선) 표현 일관성

src/config.py의 ROUTES_SHP·STOPS_SHP 사용.
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

from src.config import ROUTES_SHP, STOPS_SHP, BUS_ALL_14_SHORT

HR = "─" * 72

# 이전 185교 기준 비교 데이터 (커밋 6263120 콘솔 보고)
PREV_COUNTS = {
    "세천초등학교": 32, "남선초등학교": 30, "기성초등학교·길헌분": 26,
    "산흥초등학교": 22, "동명초등학교": 15, "계산초등학교": 12,
    "신탄진용정초등학교 ?": 11, "산서초등학교": 10, "진잠초등학교": 9,
    "산내초등학교": 7, "대전원신흥초등학교 ?": 6, "구즉초등학교": 5,
}
PREV_TOTAL = 185


def _load(shp_path):
    return gpd.read_file(shp_path, encoding="cp949")


def basic_info(g):
    print("\n" + "=" * 72)
    print("[A-2] 기본 진단")
    print("=" * 72)
    print(f"  파일      : {ROUTES_SHP}")
    print(f"  CRS       : {g.crs}  (EPSG:{g.crs.to_epsg()})")
    print(f"  features  : {len(g)}  (이전 {PREV_TOTAL}, Δ={len(g)-PREV_TOTAL:+d})")
    print(f"  geom type : {g.geom_type.value_counts().to_dict()}")
    print(f"  컬럼      : {list(g.columns)}")

    print(f"\n  샘플 5건:")
    cols_no_geom = [c for c in g.columns if c != "geometry"]
    print(g[cols_no_geom].head(5).to_string(index=True, max_colwidth=22))


def check_school_names(g):
    print("\n" + "=" * 72)
    print("[A-3] 학교명 표기 변화 + 흥도초 포함 여부")
    print("=" * 72)

    new_counts = g["이름"].value_counts(dropna=False).to_dict()
    new_names = set(new_counts.keys())
    prev_names = set(PREV_COUNTS.keys())

    print(f"\n  [학교명별 feature 수 (이전 → 새 _re)]")
    all_keys = sorted(prev_names | new_names)
    print(f"    {'학교명':<32} {'이전':>5} {'새':>5} {'Δ':>5}")
    print(f"    {'-'*32} {'-'*5} {'-'*5} {'-'*5}")
    for k in all_keys:
        prev = PREV_COUNTS.get(k, 0)
        new = new_counts.get(k, 0)
        delta = new - prev
        flag = ""
        if k not in prev_names:
            flag = " ★신규"
        elif k not in new_names:
            flag = " ✗누락"
        elif "?" in k:
            flag = " (? 표기 잔존)"
        print(f"    {k:<32} {prev:>5} {new:>5} {delta:>+5}{flag}")
    print(f"    {'-'*32} {'-'*5} {'-'*5} {'-'*5}")
    print(f"    {'(합)':<32} {PREV_TOTAL:>5} {len(g):>5} "
          f"{len(g)-PREV_TOTAL:>+5}")

    # 흥도초 포함 여부
    print(f"\n  [흥도초 포함 여부] — 가장 중요")
    hung = [n for n in new_names if "흥도" in n]
    if hung:
        for nm in hung:
            cnt = new_counts.get(nm, 0)
            print(f"    ★ '{nm}': {cnt}건 → 흥도초 노선 데이터 추가됨!")
            print(f"      → 운영 시작 신호 가능성. Phase 분류 재검토 필요할 수도.")
    else:
        print(f"    ✗ 흥도초 없음. 기존 PLANNED_1 분류 유지.")

    # ? 표기 정리
    print(f"\n  [? 표기 정리]")
    prev_q = [k for k in prev_names if "?" in k]
    for old in prev_q:
        # 새 이름에 그 학교의 유사 표기 있나
        stem = old.split()[0]  # "대전원신흥초등학교"
        candidates = [n for n in new_names if stem in n]
        if old in new_names:
            print(f"    [잔존] '{old}': 이전과 동일")
        elif candidates:
            print(f"    [정리?] '{old}' → 후보: {candidates}")
        else:
            print(f"    [실종] '{old}' → 새 _re에 매칭 없음. stem={stem}")

    return new_names


def check_sinuosity(g):
    print("\n" + "=" * 72)
    print("[A-4] 노선 sinuosity (실 경로 vs 직선)")
    print("=" * 72)
    print("  sinuosity = (실제 라인 길이) / (시작-끝 직선 거리)")
    print("  <1.2 거의 직선 ⚠️ /  1.2~1.5 일반 도로 ✓  /  >1.5 산악 도로")

    g_m = g.to_crs("EPSG:5179").copy()

    def calc_sin(line):
        if line.geom_type == "MultiLineString":
            # 첫 part의 끝점 ↔ 마지막 part의 끝점
            parts = list(line.geoms)
            start = parts[0].coords[0]
            end = parts[-1].coords[-1]
            actual = sum(p.length for p in parts)
        else:
            start = line.coords[0]
            end = line.coords[-1]
            actual = line.length
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        straight = (dx*dx + dy*dy) ** 0.5
        return actual / straight if straight > 0 else np.nan

    g_m["sinuosity"] = g_m.geometry.apply(calc_sin)
    g_m["length_m"] = g_m.geometry.length

    # 학교별 평균
    agg = g_m.groupby("이름").agg(
        노선수=("sinuosity", "count"),
        평균_sinuosity=("sinuosity", "mean"),
        평균길이_m=("length_m", "mean"),
    ).round(2).sort_values("평균_sinuosity")

    print(f"\n  [학교별 평균 sinuosity]")
    print(f"    {'학교':<32} {'노선수':>5} {'평균_sin':>10} {'평균길이m':>10}  표현방식")
    print(f"    {'-'*32} {'-'*5} {'-'*10} {'-'*10}  {'-'*15}")
    for school, row in agg.iterrows():
        s = row["평균_sinuosity"]
        if pd.isna(s):
            kind = "NaN"
        elif s < 1.2:
            kind = "직선형 ⚠️"
        elif s < 1.5:
            kind = "도로형 ✓"
        else:
            kind = "산악형 (정밀)"
        print(f"    {str(school):<32} {int(row['노선수']):>5} "
              f"{s:>10.2f} {row['평균길이_m']:>10,.0f}  {kind}")

    return g_m, agg


def summarize_for_phase2(g_m, new_names):
    print("\n" + "=" * 72)
    print("[A-5] PART B 진행 가능 여부 판정")
    print("=" * 72)

    # 운영 12교 매칭
    print(f"\n  [14교 명단 vs 새 _re SHP 학교명]")
    # 14교 매핑
    matched = {}
    unmatched = []
    for short in BUS_ALL_14_SHORT:
        key = short.replace(" ", "").replace("및", "")
        cands = [n for n in new_names if (short in n) or (key in n.replace(" ", ""))]
        if cands:
            matched[short] = cands
        else:
            unmatched.append(short)

    for short, fulls in matched.items():
        flag = " (다중)" if len(fulls) > 1 else ""
        print(f"    ✓ {short:<24} → {fulls}{flag}")
    if unmatched:
        print(f"\n  ✗ 매칭 실패: {unmatched}")

    # sinuosity 일관성 평가
    sin_std = g_m.groupby("이름")["sinuosity"].mean().std()
    print(f"\n  학교별 평균 sinuosity 편차 (표준편차): {sin_std:.2f}")

    print(f"\n  → 결론:")
    print(f"     - feature 수: {len(g_m)} (이전 185)")
    if "흥도초" in unmatched:
        print(f"     - 흥도초: SHP에 없음 → PLANNED_1 분류 유지")
    else:
        print(f"     - 흥도초: SHP에 있음 → 운영 시작 신호, 분류 검토 필요")
    if sin_std > 0.2:
        print(f"     - sinuosity 학교별 편차 큼 → 노선 경사 비교 시 한계 명시 필요")
    print(f"     - PART B (노선 경사 프로파일) 진행 가능")


def main():
    print(HR)
    print("Phase 2 PART A — 신규 _re 노선 SHP 진단")
    print(HR)

    g = _load(ROUTES_SHP)
    basic_info(g)
    new_names = check_school_names(g)
    g_m, agg = check_sinuosity(g)
    summarize_for_phase2(g_m, new_names)

    print("\n" + HR)
    print("[DONE] PART A")
    print(HR)


if __name__ == "__main__":
    main()
