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

def figure_07_route_slope_profile(sigungu, routes_3857, routes_slope_df):
    fig, ax = _setup(
        "통학버스 노선의 경사 프로파일",
        "운영 12교 + 명목 1교 노선 (183개) — 노선 평균 경사로 색상, 10도 이상 비율 굵기"
    )
    _set_extent(ax, sigungu)
    _add_basemap(ax, alpha=0.6)

    # 경사도 음영 (옅게)
    if SLOPE_PNG.exists() and SLOPE_BOUNDS.exists():
        img = np.array(Image.open(SLOPE_PNG))
        l, r, btm, top = _png_extent_3857()
        ax.imshow(img, extent=[l, r, btm, top], origin="upper",
                  interpolation="bilinear", alpha=0.5, zorder=2)

    # 노선을 평균 경사로 색칠
    cmap = LinearSegmentedColormap.from_list(
        "route_slope", [(0.0, "#2D8B43"), (0.4, "#F1C40F"), (0.7, "#E67E22"), (1.0, "#C0392B")]
    )
    vmin, vmax = 1.0, 9.0
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)

    # routes_3857에 평균 경사 값 부여 (Id 기준 merge)
    rs = routes_slope_df.set_index("Id")["slope_mean"].to_dict()
    pct10 = routes_slope_df.set_index("Id")["slope_pct_over_10deg"].to_dict()

    for _, row in routes_3857.iterrows():
        rid = row.get("Id")
        sm = rs.get(rid, np.nan)
        if pd.isna(sm):
            continue
        p10 = pct10.get(rid, 0)
        # 10도 이상 비율이 높을수록 굵게
        lw = 1.2 + (p10 / 100) * 3.5
        color = cmap(norm(sm))
        if row.geometry.geom_type == "MultiLineString":
            for part in row.geometry.geoms:
                xs, ys = part.coords.xy
                ax.plot(list(xs), list(ys), color=color, linewidth=lw,
                        alpha=0.85, zorder=4)
        else:
            xs, ys = row.geometry.coords.xy
            ax.plot(list(xs), list(ys), color=color, linewidth=lw,
                    alpha=0.85, zorder=4)

    _draw_sigungu(ax, sigungu)

    # 컬러바 (우하단 inset)
    cbar_ax = ax.inset_axes([0.62, 0.07, 0.34, 0.022])
    grad = np.linspace(vmin, vmax, 256).reshape(1, -1)
    cbar_ax.imshow(grad, cmap=cmap, aspect="auto", extent=[vmin, vmax, 0, 1])
    cbar_ax.set_xticks([1, 3, 5, 7, 9])
    cbar_ax.set_xticklabels(["1°", "3°", "5°", "7°", "9°+"], fontsize=8.5)
    cbar_ax.set_yticks([])
    cbar_ax.set_title("노선 평균 경사도", fontsize=9, pad=2)

    # 굵기 범례 + 캡션
    legend_text = (
        "노선 굵기 ∝ 10° 이상 구간 비율\n"
        "(굵을수록 위험 구간 길음)"
    )
    ax.text(0.02, 0.04, legend_text, transform=ax.transAxes,
            fontsize=9, verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#888",
                      alpha=0.92, linewidth=0.5))

    cap = ("노선 평균 4.4° · 학교 위치 평균 8.5° · Wilcoxon p=0.007 → "
           "노선이 학교 위치보다 평탄 (도로가 산기슭 우회)")
    fig.text(0.5, 0.04, cap, ha="center", fontsize=10.5,
             color="#0D47A1", fontweight="bold")

    _footer(fig, "통학버스 노선의 경사 프로파일")
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


def main():
    print("=" * 72)
    print("Phase 2 정적 도면 3장 (07·08·09)")
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

    print("\n[2] 도면 07 — 노선 경사 프로파일")
    figure_07_route_slope_profile(sigungu, routes_3857, routes_slope_df)

    print("\n[3] 도면 08 — 학교 위치 vs 노선 경사 비교")
    cmp_df = compare_position_vs_route(schools_sum)
    figure_08_compare_bars(schools_sum, cmp_df)

    print("\n[4] 도면 09 — 운영비 회귀")
    res, reg_df = regression_operating_cost(schools_sum)
    figure_09_cost_regression(reg_df, res)

    print("\n" + "=" * 72)
    print("[DONE] 도면 3장 생성")
    print("=" * 72)


if __name__ == "__main__":
    main()
