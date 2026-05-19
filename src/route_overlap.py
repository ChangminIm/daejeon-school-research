"""Phase 3: 노선 중복 분석 → 공동활용 후보 권역 도출.

흐름:
  1. 운영 12교 + 명목 1교(13학교) 노선 dissolve (학교별 multiline)
  2. 100m 버퍼 + 쌍별 교차
  3. 중첩률 ≥ 20% 쌍 → 그래프 연결성으로 권역 묶기
  4. 권역별 polygon (시각화용) + 중심점 + 포함 학교
"""
from __future__ import annotations

import sys
import json
from itertools import combinations

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.ops import unary_union
from shapely.geometry import mapping

from src.config import DATA_PROCESSED, OUTPUT_TABLES, DATA_GEOJSON
from src.route_slope import load_routes

OVERLAP_THRESHOLD = 0.20
BUFFER_M = 100

OUT_PAIRS = OUTPUT_TABLES / "노선중복_쌍별.csv"
OUT_REGIONS = OUTPUT_TABLES / "공동활용_후보권역.csv"
OUT_OVERLAP_GEOJSON = DATA_GEOJSON / "노선중첩영역.geojson"
OUT_REGION_GEOJSON = DATA_GEOJSON / "공동활용_권역_polygon.geojson"


def main():
    print("=" * 72)
    print("Phase 3: 노선 중복 분석 (공동활용 후보 권역 도출)")
    print("=" * 72)

    # [1] 학교별 노선 dissolve
    routes_5179 = load_routes()
    school_lines = routes_5179.dissolve(by="short")[["geometry"]]
    print(f"\n  학교 수: {len(school_lines)}")
    print(f"  학교명: {sorted(school_lines.index.tolist())}")

    # 학교별 노선 총 길이
    school_lines["total_length_m"] = school_lines.geometry.length

    # 버퍼
    school_buffers = school_lines.copy()
    school_buffers["geometry"] = school_lines.geometry.buffer(BUFFER_M)

    # [2] 쌍별 교차
    print(f"\n  쌍별 교차 ({BUFFER_M}m 버퍼) ...")
    pairs = []
    overlap_polys = []  # 시각화용 polygon
    for s1, s2 in combinations(school_lines.index, 2):
        b1 = school_buffers.loc[s1, "geometry"]
        b2 = school_buffers.loc[s2, "geometry"]
        inter = b1.intersection(b2)
        if inter.is_empty or inter.area <= 0:
            continue
        l1_total = school_lines.loc[s1, "total_length_m"]
        l2_total = school_lines.loc[s2, "total_length_m"]
        l1 = school_lines.loc[s1, "geometry"]
        l2 = school_lines.loc[s2, "geometry"]
        l1_in = l1.intersection(b2)
        l2_in = l2.intersection(b1)
        rate_a = (l1_in.length / l1_total) if l1_total > 0 else 0
        rate_b = (l2_in.length / l2_total) if l2_total > 0 else 0
        pairs.append({
            "학교_A": s1, "학교_B": s2,
            "중첩면적_m2": round(inter.area, 0),
            "중첩길이_A_m": round(l1_in.length, 0),
            "중첩길이_B_m": round(l2_in.length, 0),
            "중첩률_A": round(rate_a, 3),
            "중첩률_B": round(rate_b, 3),
            "중첩률_max": round(max(rate_a, rate_b), 3),
        })
        if max(rate_a, rate_b) >= OVERLAP_THRESHOLD:
            overlap_polys.append({
                "type": "Feature",
                "properties": {
                    "school_A": s1, "school_B": s2,
                    "rate_A": round(rate_a, 3),
                    "rate_B": round(rate_b, 3),
                    "rate_max": round(max(rate_a, rate_b), 3),
                },
                "geometry": mapping(inter),
            })

    pairs_df = pd.DataFrame(pairs).sort_values("중첩률_max", ascending=False)
    OUT_PAIRS.parent.mkdir(parents=True, exist_ok=True)
    pairs_df.to_csv(OUT_PAIRS, index=False, encoding="utf-8-sig")
    print(f"  → {OUT_PAIRS} ({len(pairs_df)} 쌍)")

    # 상위 출력
    print(f"\n  [상위 중복 학교 쌍]")
    print(pairs_df.head(15).to_string(index=False))

    # [3] 그래프로 권역 묶기
    print(f"\n  공동활용 후보 (중첩률 ≥ {OVERLAP_THRESHOLD*100:.0f}%)")
    candidate = pairs_df[pairs_df["중첩률_max"] >= OVERLAP_THRESHOLD]
    print(f"  {len(candidate)} 쌍이 임계값 충족")

    G = nx.Graph()
    # 모든 13학교를 노드로
    for s in school_lines.index:
        G.add_node(s)
    for _, r in candidate.iterrows():
        G.add_edge(r["학교_A"], r["학교_B"], weight=r["중첩률_max"])

    regions = []
    region_polys = []  # 권역 polygon (시각화용)
    region_idx = 0
    for component in sorted(nx.connected_components(G), key=lambda c: -len(c)):
        if len(component) < 2:
            continue
        region_idx += 1
        schools = sorted(component)
        sub_lines = [school_lines.loc[s, "geometry"] for s in schools]
        merged = unary_union(sub_lines)
        c = merged.centroid
        total_len = sum(school_lines.loc[s, "total_length_m"] for s in schools)

        # 권역 polygon (학교들 노선 union의 100m 버퍼 → 시각화용)
        region_buf = unary_union([school_buffers.loc[s, "geometry"] for s in schools])

        # 평균 중첩률 (이 권역 안 학교들 사이 쌍별 중첩률 평균)
        in_region = candidate[
            candidate["학교_A"].isin(schools) & candidate["학교_B"].isin(schools)
        ]
        avg_rate = in_region["중첩률_max"].mean() if len(in_region) else 0

        regions.append({
            "권역": f"R{region_idx}",
            "포함학교수": len(schools),
            "포함학교": ", ".join(schools),
            "총노선길이_m": round(total_len, 0),
            "권역중첩률평균_max": round(avg_rate, 3),
            "중심_x_5179": round(c.x, 1),
            "중심_y_5179": round(c.y, 1),
            "정책메모": "공동활용 통학차량 도입 검토",
        })
        region_polys.append({
            "type": "Feature",
            "properties": {
                "region": f"R{region_idx}",
                "schools": ", ".join(schools),
                "n_schools": len(schools),
                "avg_rate": round(avg_rate, 3),
            },
            "geometry": mapping(region_buf),
        })

    regions_df = pd.DataFrame(regions)
    regions_df.to_csv(OUT_REGIONS, index=False, encoding="utf-8-sig")
    print(f"\n  → {OUT_REGIONS}")

    print(f"\n  [공동활용 후보 권역 결과]")
    if len(regions_df) == 0:
        print(f"    분석 결과 중첩률 {OVERLAP_THRESHOLD*100:.0f}% 이상 학교 쌍 없음")
        print(f"    → 운영 12교 노선은 공간적으로 분리되어 있어 공동활용 즉시 가능 권역 부재")
        print(f"    (참고: 본 분석은 신탄진용정초 명목 노선 포함 13학교 기준)")
    else:
        print(regions_df.to_string(index=False))
        total_schools_in_regions = sum(r["포함학교수"] for r in regions)
        print(f"\n    총 공동활용 가능 학교: {total_schools_in_regions}교 / {len(school_lines)}학교")
        # 이론적 차량 절감: 권역별 학교수 - 1 (학교당 1대 → 권역당 1~2대 가정)
        saving = sum(r["포함학교수"] - 1 for r in regions)
        print(f"    이론적 차량 절감 가능: {saving}대 "
              f"(학교당 1대 → 권역당 1~2대 가정 기준 상한)")

    # GeoJSON 저장 (인터랙티브 맵용 — 4326으로 변환)
    if overlap_polys:
        gdf_overlap = gpd.GeoDataFrame.from_features(overlap_polys, crs="EPSG:5179")
        gdf_overlap = gdf_overlap.to_crs("EPSG:4326")
        OUT_OVERLAP_GEOJSON.parent.mkdir(parents=True, exist_ok=True)
        gdf_overlap.to_file(OUT_OVERLAP_GEOJSON, driver="GeoJSON", encoding="utf-8")
        print(f"\n  → {OUT_OVERLAP_GEOJSON} ({len(gdf_overlap)} 중첩 polygon)")

    if region_polys:
        gdf_region = gpd.GeoDataFrame.from_features(region_polys, crs="EPSG:5179")
        gdf_region = gdf_region.to_crs("EPSG:4326")
        gdf_region.to_file(OUT_REGION_GEOJSON, driver="GeoJSON", encoding="utf-8")
        print(f"  → {OUT_REGION_GEOJSON} ({len(gdf_region)} 권역 polygon)")

    print("\n" + "=" * 72)
    print("[DONE] Phase 3 노선 중복 분석")
    print("=" * 72)

    return school_lines, pairs_df, regions_df


if __name__ == "__main__":
    main()
