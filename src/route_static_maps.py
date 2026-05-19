"""Phase 2 정적 도면 3장 — 노선 경사 프로파일·학교 위치 vs 노선·운영비 회귀.

07_노선경사프로파일.png
08_노선vs학교위치_경사비교.png
09_운영비vs노선경사_회귀.png
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import contextily as ctx
from pyproj import Transformer
from PIL import Image

from src.config import (
    DATA_PROCESSED, DATA_EXTERNAL, OUTPUT_FIGURES, OUTPUT_TABLES,
    ROUTES_SHP,
)
from src.static_maps import (
    GU_LABEL_POS_4326, COLOR_OP, COLOR_PLAN, COLOR_NOM,
    SLOPE_COLOR_STOPS, _gu_label_pos_3857, _draw_sigungu,
    _set_extent, _add_basemap, _footer, FIG_W, FIG_H, DPI, FOOTER,
)

rcParams["font.family"] = "Malgun Gothic"
rcParams["font.sans-serif"] = ["Malgun Gothic", "Apple SD Gothic Neo",
                                "NanumGothic", "Arial", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

ADMIN_SHP = DATA_EXTERNAL / "admin" / "daejeon_signungu.shp"
SLOPE_PNG = DATA_PROCESSED / "대전_slope_overlay.png"
SLOPE_BOUNDS = DATA_PROCESSED / "대전_slope_overlay_bounds.json"

CRS_3857 = "EPSG:3857"


def _png_extent_3857():
    b = json.loads(SLOPE_BOUNDS.read_text(encoding="utf-8"))
    t = Transformer.from_crs("EPSG:4326", CRS_3857, always_xy=True)
    l, btm = t.transform(b["west"], b["south"])
    r, top = t.transform(b["east"], b["north"])
    return l, r, btm, top


def _setup(title, subtitle=""):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
    ax.set_aspect("equal")
    ax.set_axis_off()
    if subtitle:
        fig.suptitle(title, fontsize=15, fontweight="bold", y=0.96)
        fig.text(0.5, 0.915, subtitle, ha="center", fontsize=10.5, color="#555")
    else:
        ax.set_title(title, fontsize=15, fontweight="bold", pad=10)
    return fig, ax


def _save(fig, name):
    out = OUTPUT_FIGURES / name
    fig.tight_layout(rect=(0, 0.025, 1, 0.96))
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    size_kb = out.stat().st_size / 1024
    print(f"  ✓ {name} ({size_kb:.1f} KB)")
    return out


# ===== 도면 7: 노선 경사 프로파일 =====

def _route_color_for_slope_static(slope):
    """인터랙티브 맵의 _route_color_for_slope와 같은 색상 (5단계 분할)."""
    if pd.isna(slope):
        return "#888"
    if slope < 3:    return "#2D8B43"
    if slope < 6:    return "#91C266"
    if slope < 9:    return "#D4B36A"
    if slope < 12:   return "#B85C2A"
    return "#8B1A1A"


def figure_07_route_slope_profile(sigungu, routes_3857, routes_slope_df, schools_3857=None):
    fig, ax = _setup(
        "통학차량 노선 경사 프로파일",
        "운영 12교 + 명목 1교 노선 (183개) — 평균 경사 5단계 색상, 인터랙티브 맵과 일관"
    )
    _set_extent(ax, sigungu)
    _add_basemap(ax, alpha=0.6)

    # 경사도 음영 (옅게)
    if SLOPE_PNG.exists() and SLOPE_BOUNDS.exists():
        img = np.array(Image.open(SLOPE_PNG))
        l, r, btm, top = _png_extent_3857()
        ax.imshow(img, extent=[l, r, btm, top], origin="upper",
                  interpolation="bilinear", alpha=0.45, zorder=2)

    rs = routes_slope_df.set_index("Id")["slope_mean"].to_dict()
    pct10 = routes_slope_df.set_index("Id")["slope_pct_over_10deg"].to_dict()

    for _, row in routes_3857.iterrows():
        rid = row.get("Id")
        sm = rs.get(rid, np.nan)
        if pd.isna(sm):
            continue
        p10 = pct10.get(rid, 0)
        lw = 1.4 + (p10 / 100) * 3.5
        color = _route_color_for_slope_static(sm)
        if row.geometry.geom_type == "MultiLineString":
            for part in row.geometry.geoms:
                xs, ys = part.coords.xy
                ax.plot(list(xs), list(ys), color=color, linewidth=lw,
                        alpha=0.85, zorder=4)
        else:
            xs, ys = row.geometry.coords.xy
            ax.plot(list(xs), list(ys), color=color, linewidth=lw,
                    alpha=0.85, zorder=4)

    # 운영 학교 🚌 마커 (운영 12교 + 예정·명목)
    if schools_3857 is not None:
        from src.static_maps import _split_schools_by_state, COLOR_OP, COLOR_PLAN, COLOR_NOM
        op, pl, nm, _ = _split_schools_by_state(schools_3857)
        ax.scatter(op.geometry.x, op.geometry.y, s=130, c=COLOR_OP,
                   marker="P", edgecolors="#0E6B5A", linewidths=1.2,
                   alpha=0.98, zorder=8, label="운영 12교")
        ax.scatter(pl.geometry.x, pl.geometry.y, s=130, c=COLOR_PLAN,
                   marker="P", edgecolors="#9C6510", linewidths=1.2,
                   alpha=0.95, zorder=8, label="예정 1교")
        ax.scatter(nm.geometry.x, nm.geometry.y, s=130, c=COLOR_NOM,
                   marker="P", edgecolors="#3A4143", linewidths=1.2,
                   alpha=0.9, zorder=8, label="명목 1교")

    _draw_sigungu(ax, sigungu)

    # 노선 경사 5단계 범례 (우상단, 흰 박스)
    legend_elems = [
        Patch(facecolor="#2D8B43", label="< 3°  (평지)"),
        Patch(facecolor="#91C266", label="3 ~ 6°"),
        Patch(facecolor="#D4B36A", label="6 ~ 9°"),
        Patch(facecolor="#B85C2A", label="9 ~ 12°"),
        Patch(facecolor="#8B1A1A", label="≥ 12°  (가파름)"),
    ]
    leg1 = ax.legend(handles=legend_elems, loc="upper right",
                     title="노선 평균 경사", title_fontsize=9.5,
                     fontsize=9, framealpha=0.95)
    ax.add_artist(leg1)

    # 학교 분류 범례 (좌하단)
    if schools_3857 is not None:
        ax.legend(loc="lower left", fontsize=9.5, framealpha=0.92, title="운영 학교",
                  title_fontsize=10)

    # 굵기 범례 (좌상단)
    legend_text = "노선 굵기 ∝ 10° 이상 구간 비율"
    ax.text(0.02, 0.97, legend_text, transform=ax.transAxes,
            fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#888",
                      alpha=0.92, linewidth=0.5))

    cap = ("운영 12교 노선 평균 4.4°, 학교 위치 평균 8.5°. "
           "도로가 우회 경로 활용 (Wilcoxon p = 0.007)")
    fig.text(0.5, 0.04, cap, ha="center", fontsize=10.5,
             color="#0D47A1", fontweight="bold")

    _footer(fig, "통학차량 노선 경사 프로파일")
    return _save(fig, "07_노선경사프로파일.png")


# ===== 도면 8: 노선 vs 학교 위치 경사 비교 =====

def figure_08_compare_bars(schools_sum, cmp_df):
    """cmp_df: school_short, position_slope, route_avg_slope"""
    # 학교 위치 경사 큰 순으로 정렬
    d = cmp_df.sort_values("position_slope", ascending=False)
    x = np.arange(len(d))
    width = 0.36

    fig, ax = plt.subplots(figsize=(13, 7), dpi=DPI)
    bars_pos = ax.bar(x - width/2, d["position_slope"], width=width,
                       color="#1976D2", alpha=0.85,
                       label="학교 위치 (반경 300m 평균)")
    bars_route = ax.bar(x + width/2, d["route_avg_slope"], width=width,
                         color="#E67E22", alpha=0.85,
                         label="노선 평균 (길이 가중)")

    # 값 라벨
    for b in bars_pos:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.15,
                f"{b.get_height():.1f}", ha="center", fontsize=9, color="#1976D2")
    for b in bars_route:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.15,
                f"{b.get_height():.1f}", ha="center", fontsize=9, color="#E67E22")

    ax.set_xticks(x)
    ax.set_xticklabels(d["school_short"], rotation=22, ha="right", fontsize=10)
    ax.set_ylabel("경사도 (°)", fontsize=11.5)
    ax.set_title("학교 위치 경사 vs 통학버스 노선 평균 경사 (운영 12교 + 명목 1교)",
                  fontsize=14, fontweight="bold", pad=10)
    ax.legend(loc="upper right", fontsize=11, framealpha=0.95)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_ylim(0, max(d["position_slope"].max(), d["route_avg_slope"].max()) * 1.18)

    # 캡션
    diff = (d["route_avg_slope"] - d["position_slope"]).mean()
    cap = (f"평균 차이 = 노선 − 위치 = {diff:+.2f}°  ·  "
           f"Wilcoxon signed-rank p = 0.007 → 노선이 위치보다 통계적으로 더 평탄")
    fig.text(0.5, 0.015, cap, ha="center", fontsize=10.5,
             color="#0D47A1", fontweight="bold")

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = OUTPUT_FIGURES / "08_노선vs학교위치_경사비교.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {out.name} ({out.stat().st_size/1024:.1f} KB)")
    return out


# ===== 도면 9: 운영비 vs 노선 변수 회귀 =====

def figure_09_cost_regression(reg_df, ols_res):
    """reg_df: 학교명, 이용학생수, 차량대수, route_total_length_m,
       route_avg_slope, 학생1인당_총비용_천원"""
    # 4개 산점도 패널 (2x2): 각 변수 vs 1인당 비용 + 단변량 R²
    from scipy.stats import linregress

    y_col = "학생1인당_총비용_천원"
    panels = [
        ("route_total_length_m", "노선 총 길이 (m)", lambda v: v),
        ("route_avg_slope",       "노선 평균 경사도 (°)", lambda v: v),
        ("이용학생수",            "이용 학생 수 (명)", lambda v: v),
        ("차량대수",              "차량 대수", lambda v: v),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), dpi=DPI)
    axes = axes.ravel()

    for ax, (col, xlabel, _f) in zip(axes, panels):
        x = reg_df[col].astype(float).values
        y = reg_df[y_col].astype(float).values
        ax.scatter(x, y, s=120, c="#1976D2", alpha=0.7,
                    edgecolors="white", linewidths=1.0, zorder=4)
        # 학교명 라벨
        for xi, yi, name in zip(x, y, reg_df["학교명"]):
            ax.annotate(name, xy=(xi, yi), xytext=(6, 6),
                         textcoords="offset points", fontsize=8.5,
                         color="#444")
        # 회귀선 + R²
        if len(x) >= 3 and x.std() > 0:
            slope, intercept, r, p, se = linregress(x, y)
            xx = np.linspace(x.min(), x.max(), 100)
            yy = slope * xx + intercept
            ax.plot(xx, yy, color="#C0392B", linewidth=1.6, alpha=0.85,
                    label=f"R² = {r**2:.3f}, p = {p:.3f}")
            ax.legend(loc="upper right", fontsize=10, framealpha=0.92)

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("1인당 비용 (천원/년)", fontsize=11)
        ax.grid(axis="both", linestyle="--", alpha=0.3)
        ax.set_title(xlabel.split(" (")[0], fontsize=11.5, fontweight="bold")

    fig.suptitle("운영비 격차 회귀 — 1인당 비용 vs 노선·학생 변수 (운영 11교)",
                  fontsize=14, fontweight="bold", y=0.99)
    fig.text(0.5, 0.02,
              f"다변량 OLS: R²={ols_res.rsquared:.2f}, F p={ols_res.f_pvalue:.2f} (표본 11교에 변수 4개 → 자유도 부족)",
              ha="center", fontsize=10.5, color="#666", style="italic")

    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    out = OUTPUT_FIGURES / "09_운영비vs노선경사_회귀.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {out.name} ({out.stat().st_size/1024:.1f} KB)")
    return out


def figure_09_route_overlap(sigungu, routes_3857, schools_3857):
    """노선 중복 / 공동활용 후보 권역."""
    import json as _json
    from src.config import DATA_GEOJSON

    overlap_path = DATA_GEOJSON / "노선중첩영역.geojson"
    region_path = DATA_GEOJSON / "공동활용_권역_polygon.geojson"
    regions_csv = OUTPUT_TABLES / "공동활용_후보권역.csv"

    fig, ax = _setup(
        "통학차량 노선 중복 — 공동활용 후보 권역",
        "운영 12교 + 명목 1교 노선 중첩 영역 식별 (100m 버퍼 ∩, 중첩률 ≥ 20% 권역)"
    )
    _set_extent(ax, sigungu)
    _add_basemap(ax, alpha=0.55)

    # 12교 노선 모두 회색 옅게
    for _, row in routes_3857.iterrows():
        if row.geometry.geom_type == "MultiLineString":
            for part in row.geometry.geoms:
                xs, ys = part.coords.xy
                ax.plot(list(xs), list(ys), color="#666", linewidth=1.3,
                        alpha=0.55, zorder=3)
        else:
            xs, ys = row.geometry.coords.xy
            ax.plot(list(xs), list(ys), color="#666", linewidth=1.3,
                    alpha=0.55, zorder=3)

    # 권역 polygon (4326 → 3857) — 점선 박스 대신 fill 반투명
    if region_path.exists():
        gdf_region = gpd.read_file(region_path).to_crs(CRS_3857)
        gdf_region.plot(ax=ax, facecolor="#FFE082", alpha=0.30,
                         edgecolor="#E67E22", linewidth=2.0, linestyle="--",
                         zorder=4)
        # 권역명 라벨
        for _, r in gdf_region.iterrows():
            c = r.geometry.centroid
            schools = r.get("schools", "")
            ax.annotate(
                f"{r['region']}\n{schools}",
                xy=(c.x, c.y), fontsize=10, color="#5D4037",
                ha="center", va="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          ec="#E67E22", alpha=0.92, linewidth=0.8),
                zorder=12,
            )

    # 중첩 영역 (실제 노선 100m 버퍼 ∩)
    if overlap_path.exists():
        gdf_overlap = gpd.read_file(overlap_path).to_crs(CRS_3857)
        gdf_overlap.plot(ax=ax, facecolor="#C0392B", alpha=0.45,
                         edgecolor="#8B0000", linewidth=1.2, zorder=5)

    # 운영 12교 마커
    from src.static_maps import _split_schools_by_state, COLOR_OP, COLOR_PLAN, COLOR_NOM
    op, pl, nm, _ = _split_schools_by_state(schools_3857)
    ax.scatter(op.geometry.x, op.geometry.y, s=130, c=COLOR_OP,
               marker="P", edgecolors="#0E6B5A", linewidths=1.2,
               alpha=0.98, zorder=10, label="운영 12교")
    ax.scatter(pl.geometry.x, pl.geometry.y, s=130, c=COLOR_PLAN,
               marker="P", edgecolors="#9C6510", linewidths=1.2,
               alpha=0.95, zorder=10, label="예정 1교")
    ax.scatter(nm.geometry.x, nm.geometry.y, s=130, c=COLOR_NOM,
               marker="P", edgecolors="#3A4143", linewidths=1.2,
               alpha=0.9, zorder=10, label="명목 1교")

    _draw_sigungu(ax, sigungu)

    # 캡션
    n_regions = 0
    region_summary = ""
    if regions_csv.exists():
        rdf = pd.read_csv(regions_csv, encoding="utf-8-sig")
        n_regions = len(rdf)
        if n_regions > 0:
            parts = [f"{r['권역']} {r['포함학교']} (중첩 {r['권역중첩률평균_max']*100:.0f}%)"
                     for _, r in rdf.iterrows()]
            region_summary = "  ·  ".join(parts)

    if n_regions > 0:
        cap = (f"공동활용 가능 권역 {n_regions}개 식별 — {region_summary}. "
               f"통학차량 운영 효율화의 공간적 근거")
    else:
        cap = "분석 결과 중첩률 20% 이상 학교 쌍 없음 — 노선이 공간적으로 분리되어 공동활용 즉시 가능 권역 부재"
    fig.text(0.5, 0.04, cap, ha="center", fontsize=10,
             color="#C0392B", fontweight="bold")

    # 범례
    from matplotlib.patches import Patch as _Patch
    legend_elems = [
        _Patch(facecolor="#C0392B", edgecolor="#8B0000", alpha=0.5,
               label="노선 100m 버퍼 중첩 영역"),
        _Patch(facecolor="#FFE082", edgecolor="#E67E22", alpha=0.4,
               linestyle="--", label="공동활용 후보 권역"),
        Line2D([], [], color="#666", linewidth=1.5, alpha=0.7,
               label="통학버스 노선 (전체)"),
    ]
    ax.legend(handles=legend_elems, loc="upper right", fontsize=9.5,
              framealpha=0.95)

    _footer(fig, "통학차량 노선 중복 — 공동활용 후보 권역")
    return _save(fig, "09_노선중복_공동활용권역.png")


def main():
    print("=" * 72)
    print("Phase 2 정적 도면 (07·08·09 — 09는 노선 중복 분석으로 교체)")
    print("=" * 72)

    from src.route_slope import (
        load_routes, aggregate_schools, compare_position_vs_route,
        regression_operating_cost, OUT_ROUTES_SLOPE, OUT_SCHOOLS_SUM,
    )

    sigungu = gpd.read_file(ADMIN_SHP).to_crs(CRS_3857)

    print("\n[1] 데이터 로드")
    routes_5179 = load_routes()
    routes_3857 = routes_5179.to_crs(CRS_3857)
    routes_slope_df = pd.read_csv(OUT_ROUTES_SLOPE, encoding="utf-8-sig")
    schools_sum = pd.read_csv(OUT_SCHOOLS_SUM, encoding="utf-8-sig")
    print(f"  routes: {len(routes_3857)}, routes_slope_df: {len(routes_slope_df)}, "
          f"schools_sum: {len(schools_sum)}")

    # 학교 GDF (3857)
    from src.static_maps import _load_schools
    schools_3857 = _load_schools()

    print("\n[2] 도면 07 — 노선 경사 프로파일 (인터랙티브와 색상 일관)")
    figure_07_route_slope_profile(sigungu, routes_3857, routes_slope_df, schools_3857)

    print("\n[3] 도면 08 — 학교 위치 vs 노선 경사 비교")
    cmp_df = compare_position_vs_route(schools_sum)
    figure_08_compare_bars(schools_sum, cmp_df)

    # 09번 운영비 회귀 도면은 제거됨 (docs/운영비_회귀_시도_메모.md로 이동)
    print("\n  [skip] 도면 09 운영비 회귀 — 본 챕터(외부환경분석) 미반영,")
    print("         docs/운영비_회귀_시도_메모.md로 별도 보관")

    print("\n[5] 도면 09 — 노선 중복 / 공동활용 후보 권역 (신규)")
    figure_09_route_overlap(sigungu, routes_3857, schools_3857)

    print("\n" + "=" * 72)
    print("[DONE] 도면 3장 생성")
    print("=" * 72)


if __name__ == "__main__":
    main()
