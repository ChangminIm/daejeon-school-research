"""보고서용 정적 도면 (운영 12교 기준 v2).

- 종합지도 / 재개발 단계+KDE 합본 / 적격성 상위 30교 / 운영-검토 비교 /
  경사도 음영 + 운영 학교 / 학생 분포 KDE (행정구역 마스킹)
- A4 가로 (11.7×8.3), 300dpi, contextily 배경
- 제목에 번호 없음. 파일명은 정렬용으로 01_~ 유지.
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
import matplotlib.patheffects as patheffects
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as MplPath
from scipy.stats import gaussian_kde
import contextily as ctx
from pyproj import Transformer
from PIL import Image
from shapely.ops import unary_union
from adjustText import adjust_text

from src.config import (
    DATA_PROCESSED, DATA_EXTERNAL, OUTPUT_FIGURES, OUTPUT_TABLES,
    BUS_OPERATING_12_SHORT, BUS_PLANNED_1_SHORT, BUS_NOMINAL_1_SHORT,
)

rcParams["font.family"] = "Malgun Gothic"
rcParams["font.sans-serif"] = ["Malgun Gothic", "Apple SD Gothic Neo",
                                "NanumGothic", "Arial", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

ADMIN_SHP = DATA_EXTERNAL / "admin" / "daejeon_signungu.shp"
SLOPE_PNG = DATA_PROCESSED / "대전_slope_overlay.png"
SLOPE_BOUNDS = DATA_PROCESSED / "대전_slope_overlay_bounds.json"

CRS_3857 = "EPSG:3857"
CRS_4326 = "EPSG:4326"

FIG_W, FIG_H, DPI = 11.7, 8.3, 300
FOOTER = "데이터 기준일: 2026.3.31 · 국립공주대학교 지리학과 (장동호 · 박종철 · 임창민)"

# 자치구 라벨 위치 — 사용자 명시 (학교 분포 피한 위치, 4326)
GU_LABEL_POS_4326 = {
    "유성구": (127.3450, 36.3850),
    "대덕구": (127.4250, 36.4150),
    "동구":   (127.4650, 36.3050),
    "중구":   (127.4280, 36.3180),
    "서구":   (127.3700, 36.3450),
}

# 운영 분류 색상
COLOR_OP = "#1ABC9C"     # 운영 12 (청록)
COLOR_PLAN = "#F39C12"   # 예정 1 — 흥도초 (노랑)
COLOR_NOM = "#7F8C8D"    # 명목 1 — 신탄진용정초 (회색)
COLOR_TOP30 = "#C0392B"  # 신규 검토 30교 (빨강)

# 임박도 색상
IMM_COLOR = {
    "1_공사중":   "#C0392B",
    "2_관리처분": "#E74C3C",
    "3_사업시행": "#E67E22",
    "4_조합설립": "#F1C40F",
    "5_초기":     "#F1C40F",
    "6_입안":     "#95A5A6",
    "9_미정":     "#95A5A6",
}
IMM_GROUP_LABEL = {
    "공사중":     ["1_공사중"],
    "관리처분":   ["2_관리처분"],
    "사업시행":   ["3_사업시행"],
    "조합·추진위": ["4_조합설립", "5_초기"],
    "입안·미정":  ["6_입안", "9_미정"],
}
IMM_GROUP_COLOR = {
    "공사중":     "#C0392B",
    "관리처분":   "#E74C3C",
    "사업시행":   "#F1C40F",
    "조합·추진위": "#27AE60",
    "입안·미정":  "#7F8C8D",
}

SLOPE_COLOR_STOPS = [
    (0.00, "#2D8B43"), (0.15, "#91C266"), (0.40, "#D4B36A"),
    (0.65, "#B85C2A"), (1.00, "#8B1A1A"),
]


# ===== 데이터 로더 =====

def _load_sigungu():
    return gpd.read_file(ADMIN_SHP).to_crs(CRS_3857)


def _load_schools():
    """schools_with_slope_v2.csv 우선, 없으면 with_slope + impact merge."""
    v2 = DATA_PROCESSED / "schools_with_slope_v2.csv"
    if v2.exists():
        sch = pd.read_csv(v2, encoding="utf-8-sig")
    else:
        sch = pd.read_csv(DATA_PROCESSED / "schools_with_impact.csv")
        slope = pd.read_csv(DATA_PROCESSED / "schools_with_slope.csv")
        smap = dict(zip(slope["학교명"], slope["slope_300m_mean"]))
        sch["slope_300m_mean"] = sch["학교명"].map(smap)
        sch["운영상태"] = "미운영"

    sch = sch.dropna(subset=["lat", "lon"]).copy()
    return gpd.GeoDataFrame(
        sch, geometry=gpd.points_from_xy(sch["lon"], sch["lat"], crs=CRS_4326)
    ).to_crs(CRS_3857)


def _load_redev():
    df = pd.read_csv(DATA_PROCESSED / "redev_projects_geocoded.csv")
    df = df.dropna(subset=["lat", "lon"]).copy()
    return gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["lon"], df["lat"], crs=CRS_4326)
    ).to_crs(CRS_3857)


def _load_top30():
    return pd.read_csv(OUTPUT_TABLES / "신규검토대상_상위30교.csv")


# ===== 공통 그리기 =====

def _gu_label_pos_3857():
    t = Transformer.from_crs(CRS_4326, CRS_3857, always_xy=True)
    return {name: t.transform(lon, lat) for name, (lon, lat) in GU_LABEL_POS_4326.items()}


def _setup(title, subtitle=""):
    """제목만, 부제는 무시 (일괄 정책)."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, fontsize=16, fontweight="bold", pad=12)
    return fig, ax


def _draw_sigungu(ax, sigungu, labels=True):
    sigungu.boundary.plot(ax=ax, color="#222", linewidth=0.7, alpha=0.75,
                           zorder=12)
    if labels:
        pos_3857 = _gu_label_pos_3857()
        for name, (x, y) in pos_3857.items():
            txt = ax.text(
                x, y, name,
                fontsize=11.5, color="#222", fontweight="bold",
                ha="center", va="center", zorder=20,
            )
            txt.set_path_effects([
                patheffects.withStroke(linewidth=3.5, foreground="white")
            ])


# 범례 위치 표준 (지도 안)
LEGEND_LOWER_RIGHT = dict(
    loc="lower right", bbox_to_anchor=(0.98, 0.02),
    framealpha=0.92, facecolor="white", edgecolor="gray",
)
LEGEND_UPPER_RIGHT = dict(
    loc="upper right", bbox_to_anchor=(0.98, 0.98),
    framealpha=0.92, facecolor="white", edgecolor="gray",
)


def _set_extent(ax, sigungu, pad_frac=0.05):
    """대전 영역에 5% 여유만 — 주변(세종/충북) 자동 잘림."""
    minx, miny, maxx, maxy = sigungu.total_bounds
    w, h = maxx - minx, maxy - miny
    ax.set_xlim(minx - w*pad_frac, maxx + w*pad_frac)
    ax.set_ylim(miny - h*pad_frac, maxy + h*pad_frac)


def _add_basemap(ax, alpha=1.0):
    try:
        ctx.add_basemap(ax, crs=CRS_3857,
                         source=ctx.providers.CartoDB.PositronNoLabels,
                         attribution=False, alpha=alpha)
    except Exception as e:
        print(f"   [경고] basemap 실패: {e}")


def _make_clip_patch(sigungu_3857, ax):
    """자치구 union → matplotlib PathPatch (set_clip_path용)."""
    union = unary_union(sigungu_3857.geometry.tolist())

    def poly_segments(poly):
        verts, codes = [], []
        x, y = poly.exterior.coords.xy
        x, y = list(x), list(y)
        for i, (xi, yi) in enumerate(zip(x, y)):
            verts.append((xi, yi))
            codes.append(MplPath.MOVETO if i == 0 else MplPath.LINETO)
        codes[-1] = MplPath.CLOSEPOLY
        for ring in poly.interiors:
            xr, yr = ring.coords.xy
            xr, yr = list(xr), list(yr)
            start = len(verts)
            for i, (xi, yi) in enumerate(zip(xr, yr)):
                verts.append((xi, yi))
                codes.append(MplPath.MOVETO if i == 0 else MplPath.LINETO)
            codes[-1] = MplPath.CLOSEPOLY
        return verts, codes

    all_verts, all_codes = [], []
    if hasattr(union, "geoms"):
        for p in union.geoms:
            v, c = poly_segments(p)
            all_verts.extend(v)
            all_codes.extend(c)
    else:
        v, c = poly_segments(union)
        all_verts.extend(v)
        all_codes.extend(c)

    path = MplPath(all_verts, all_codes)
    return PathPatch(path, transform=ax.transData, facecolor="none", edgecolor="none")


def _footer(fig, page_title):
    """footer 표시 안 함 (일괄 정책)."""
    return


def _save(fig, name):
    out = OUTPUT_FIGURES / name
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    size_kb = out.stat().st_size / 1024
    print(f"  ✓ {name} ({size_kb:.1f} KB)")
    return out


# ===== 운영 분류 분리 helper =====

def _split_schools_by_state(schools_gdf):
    """schools_with_slope_v2 운영상태 컬럼으로 분리."""
    op = schools_gdf[schools_gdf["운영상태"] == "운영"]
    pl = schools_gdf[schools_gdf["운영상태"] == "예정"]
    nm = schools_gdf[schools_gdf["운영상태"] == "명목"]
    others = schools_gdf[~schools_gdf["운영상태"].isin(["운영", "예정", "명목"])]
    return op, pl, nm, others


# ===== 도면 1: 종합지도 =====

def figure_01_overview(sigungu, schools, redev, top30):
    fig, ax = _setup("대전 학교·도시개발 분포 현황")
    _set_extent(ax, sigungu)
    _add_basemap(ax)

    # 재개발 사업 점 — 임박도별 색상 유지
    redev_active = redev[redev["통학영향_임박도"].isin(IMM_COLOR.keys())]
    for stage, sub in redev_active.groupby("통학영향_임박도"):
        color = IMM_COLOR.get(stage, "#888")
        ax.scatter(sub.geometry.x, sub.geometry.y, s=18, c=color,
                   marker="D", alpha=0.6, edgecolors="none", zorder=3)

    # 학교 (회색)
    elem = schools[schools["학교급"] == "초"]
    midd = schools[schools["학교급"] == "중"]
    ax.scatter(elem.geometry.x, elem.geometry.y, s=14, c="#555",
               marker="o", alpha=0.55, edgecolors="white", linewidths=0.4, zorder=4)
    ax.scatter(midd.geometry.x, midd.geometry.y, s=16, c="#555",
               marker="s", alpha=0.55, edgecolors="white", linewidths=0.4, zorder=4)

    # 신규 검토 대상
    top30_names = set(top30["학교명"])
    top30_g = schools[schools["학교명"].isin(top30_names)]
    ax.scatter(top30_g.geometry.x, top30_g.geometry.y, s=120, c=COLOR_TOP30,
               marker="*", edgecolors="white", linewidths=0.8, alpha=0.95, zorder=6)

    # 운영 학교 — 운영/예정/명목 단일 색상 통합 (청록 P)
    op, pl, nm, _ = _split_schools_by_state(schools)
    all_bus = pd.concat([op, pl, nm])
    ax.scatter(all_bus.geometry.x, all_bus.geometry.y, s=140, c=COLOR_OP,
               marker="P", edgecolors="white", linewidths=0.9, alpha=0.95, zorder=7)

    _draw_sigungu(ax, sigungu)

    legend_elems = [
        Line2D([], [], marker="*", color="w", markerfacecolor=COLOR_TOP30,
               markersize=14, label="신규 검토 대상"),
        Line2D([], [], marker="P", color="w", markerfacecolor=COLOR_OP,
               markersize=12, label="현행 운영 학교"),
        Line2D([], [], marker="o", color="w", markerfacecolor="#555",
               markersize=7, label="초등학교 / 중학교"),
        Line2D([], [], marker="D", color="w", markerfacecolor="#C0392B",
               markersize=7, label="재개발 사업"),
    ]
    ax.legend(handles=legend_elems, loc="lower left", fontsize=9.5, framealpha=0.92)

    return _save(fig, "01_종합지도.png")


# ===== 도면 2: 재개발 단계 + KDE 도시개발압력 합본 =====

def figure_02_redev_with_kde(sigungu, redev):
    fig, ax = _setup("도시개발 분포 및 압력")
    _set_extent(ax, sigungu)
    _add_basemap(ax)

    # KDE — 세대수 가중 (강조)
    valid_redev = redev[redev["세대수"].fillna(0) > 0].copy()
    if len(valid_redev) > 3:
        xy = np.array([(p.x, p.y) for p in valid_redev.geometry]).T
        w = valid_redev["세대수"].astype(float).values
        w = w / w.sum()
        kde = gaussian_kde(xy, weights=w, bw_method=0.20)
        minx, miny, maxx, maxy = sigungu.total_bounds
        pad = (maxx - minx) * 0.03
        xx, yy = np.meshgrid(
            np.linspace(minx - pad, maxx + pad, 240),
            np.linspace(miny - pad, maxy + pad, 240),
        )
        zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        cmap_kde = LinearSegmentedColormap.from_list(
            "yor", [(0.0, "#FFF3B0"), (0.4, "#FFA34D"), (1.0, "#C0392B")]
        )
        cs = ax.contourf(xx, yy, zz, levels=14, cmap=cmap_kde, alpha=0.75, zorder=2)
        clip_patch = _make_clip_patch(sigungu, ax)
        ax.add_patch(clip_patch)
        cs.set_clip_path(clip_patch)

        cbar_ax = ax.inset_axes([0.68, 0.94, 0.27, 0.022])
        grad = np.linspace(0, 1, 256).reshape(1, -1)
        cbar_ax.imshow(grad, cmap=cmap_kde, aspect="auto")
        cbar_ax.set_xticks([0, 255])
        cbar_ax.set_xticklabels(["낮음", "높음"], fontsize=8.5)
        cbar_ax.set_yticks([])
        cbar_ax.set_title("도시개발 압력 (세대수 가중)", fontsize=9, pad=2)

    # 도시개발 사업 점 — 일률 단일 진청, 더 작게/흐리게 (KDE 강조)
    redev_active = redev[redev["통학영향_임박도"].isin(IMM_COLOR.keys())]
    ax.scatter(redev_active.geometry.x, redev_active.geometry.y,
               s=20, c="#1A237E", alpha=0.6,
               edgecolors="white", linewidths=0.3, zorder=6,
               label="● 재개발 사업 (진행)")

    _draw_sigungu(ax, sigungu)

    # 점 범례 — 우하단
    ax.legend(**LEGEND_LOWER_RIGHT, fontsize=10)

    return _save(fig, "02_재개발임박도.png")


# ===== 도면 3: 적격성 상위 30교 (adjustText + 정의 박스) =====

def figure_03_top30(sigungu, schools, top30):
    fig, ax = _setup("통학지원 적격성 상위 30교")
    _set_extent(ax, sigungu)
    _add_basemap(ax)

    # 전체 학교 옅게
    ax.scatter(schools.geometry.x, schools.geometry.y, s=10, c="#bbb",
               alpha=0.5, edgecolors="none", zorder=3)

    # 상위 30교 등급별
    top30_with_geom = schools.merge(top30[["학교명", "미운영순위"]], on="학교명", how="inner")
    for _, r in top30_with_geom.iterrows():
        rank = int(r["미운영순위"])
        if rank <= 5:
            c, s = "#C0392B", 240
        elif rank <= 15:
            c, s = "#E67E22", 160
        else:
            c, s = "#F1C40F", 110
        ax.scatter(r.geometry.x, r.geometry.y, s=s, c=c, marker="*",
                   edgecolors="white", linewidths=0.9, alpha=0.95, zorder=5)

    # 상위 5교 학교명 라벨 (adjustText, 박스로 가독성 강조)
    top5 = top30_with_geom[top30_with_geom["미운영순위"] <= 5].sort_values("미운영순위")
    texts = []
    for _, r in top5.iterrows():
        txt = ax.text(
            r.geometry.x, r.geometry.y,
            f"{int(r['미운영순위'])}. {r['학교명']}",
            fontsize=10, color="#222", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="gray",
                      alpha=0.95, linewidth=0.7),
            zorder=15,
        )
        texts.append(txt)
    if texts:
        adjust_text(
            texts, ax=ax,
            arrowprops=dict(arrowstyle="-", color="#666", lw=0.6),
            expand=(1.8, 2.0),
            force_text=(1.0, 1.2),
            force_points=(0.5, 0.5),
        )

    _draw_sigungu(ax, sigungu)

    # 상위 학교 범례 — 우상단 (지도 안)
    legend_elems = [
        Line2D([], [], marker="*", color="w", markerfacecolor="#C0392B",
               markersize=17, label="상위 1~5교"),
        Line2D([], [], marker="*", color="w", markerfacecolor="#E67E22",
               markersize=14, label="상위 6~15교"),
        Line2D([], [], marker="*", color="w", markerfacecolor="#F1C40F",
               markersize=12, label="상위 16~30교"),
        Line2D([], [], marker="o", color="w", markerfacecolor="#bbb",
               markersize=6, label="기타 학교"),
    ]
    ax.legend(handles=legend_elems, **LEGEND_UPPER_RIGHT,
              fontsize=10, handletextpad=1.0, borderpad=0.8)

    # 정의 박스 — 우하단 (범례와 분리)
    def_text = (
        "신규 검토 대상\n"
        "= 적격성 점수 상위 + 현행 미운영"
    )
    ax.text(0.98, 0.02, def_text, transform=ax.transAxes,
            fontsize=10.5, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray",
                      alpha=0.95, linewidth=0.8))

    return _save(fig, "03_적격성상위30교.png")


# ===== 도면 4: 운영-검토 비교 (운영/예정/명목 색상 분리) =====

# figure_04_compare 제거됨 (04번 도면 보고서에서 제외, 사용자 결정)
# 04 번호는 인용·문서 영향 회피를 위해 빈 슬롯으로 둠


# ===== 도면 5: 경사도 음영 + 운영 학교 (범례 분리) =====

def _png_extent_3857():
    b = json.loads(SLOPE_BOUNDS.read_text(encoding="utf-8"))
    t = Transformer.from_crs(CRS_4326, CRS_3857, always_xy=True)
    l, btm = t.transform(b["west"], b["south"])
    r, top = t.transform(b["east"], b["north"])
    return l, r, btm, top


def figure_05_slope_with_operating(sigungu, schools, top30):
    fig, ax = _setup("대전 경사도와 운영 학교 분포")
    _set_extent(ax, sigungu)
    _add_basemap(ax, alpha=0.6)

    # 경사도 음영 PNG
    if SLOPE_PNG.exists() and SLOPE_BOUNDS.exists():
        img = np.array(Image.open(SLOPE_PNG))
        l, r, btm, top = _png_extent_3857()
        ax.imshow(img, extent=[l, r, btm, top], origin="upper",
                  interpolation="bilinear", zorder=2)

    _draw_sigungu(ax, sigungu)

    # 신규 검토 30교
    t30 = schools[schools["학교명"].isin(set(top30["학교명"]))]
    ax.scatter(t30.geometry.x, t30.geometry.y, s=110, c="#F1C40F",
               marker="*", edgecolors="#222", linewidths=0.8,
               alpha=0.95, zorder=5)

    # 운영/예정/명목
    op, pl, nm, _ = _split_schools_by_state(schools)
    ax.scatter(op.geometry.x, op.geometry.y, s=140, c=COLOR_OP,
               marker="P", edgecolors="#0E6B5A", linewidths=1.2,
               alpha=0.98, zorder=7)
    ax.scatter(pl.geometry.x, pl.geometry.y, s=140, c=COLOR_PLAN,
               marker="P", edgecolors="#9C6510", linewidths=1.2,
               alpha=0.95, zorder=7)
    ax.scatter(nm.geometry.x, nm.geometry.y, s=140, c=COLOR_NOM,
               marker="P", edgecolors="#3A4143", linewidths=1.2,
               alpha=0.9, zorder=7)

    # 학교 분류 범례 — 우상단 (지도 안)
    legend_elems = [
        Line2D([], [], marker="P", color="w", markerfacecolor=COLOR_OP,
               markersize=13, label="운영 12교"),
        Line2D([], [], marker="P", color="w", markerfacecolor=COLOR_PLAN,
               markersize=13, label="예정 1교"),
        Line2D([], [], marker="P", color="w", markerfacecolor=COLOR_NOM,
               markersize=13, label="명목 1교"),
        Line2D([], [], marker="*", color="w", markerfacecolor="#F1C40F",
               markersize=14, markeredgecolor="#222", label="신규 검토 대상"),
    ]
    ax.legend(handles=legend_elems, **LEGEND_UPPER_RIGHT, fontsize=10)

    # 경사도 컬러바 — 우하단 (지도 안, 가로 형태)
    cmap = LinearSegmentedColormap.from_list("slope", SLOPE_COLOR_STOPS)
    cbar_ax = ax.inset_axes([0.62, 0.05, 0.34, 0.022])
    grad = np.linspace(0, 30, 256).reshape(1, -1)
    cbar_ax.imshow(grad, cmap=cmap, aspect="auto", extent=[0, 30, 0, 1])
    cbar_ax.set_yticks([])
    cbar_ax.set_xticks([0, 7.5, 15, 22.5, 30])
    cbar_ax.set_xticklabels(["0°", "7.5°", "15°", "22.5°", "30°+"], fontsize=8.5)
    cbar_ax.set_title("경사도", fontsize=9, pad=2)

    return _save(fig, "05_경사도음영_운영학교.png")


# ===== 도면 6: 학생 분포 KDE (행정구역 마스킹) =====

def figure_06_kde_students(sigungu, schools, top30):
    fig, ax = _setup("학생 분포 밀도 및 신규 통학지원 검토 대상")
    _set_extent(ax, sigungu)
    _add_basemap(ax)

    # 학생수 가중 KDE
    xy = np.array([(p.x, p.y) for p in schools.geometry]).T
    weights = schools["학생수합계"].astype(float).values
    weights = weights / weights.sum()
    kde = gaussian_kde(xy, weights=weights, bw_method=0.18)

    minx, miny, maxx, maxy = sigungu.total_bounds
    pad = (maxx - minx) * 0.03
    xx, yy = np.meshgrid(
        np.linspace(minx - pad, maxx + pad, 240),
        np.linspace(miny - pad, maxy + pad, 240),
    )
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    cmap_kde = LinearSegmentedColormap.from_list(
        "yor_stud", [(0.0, "#FFF3B0"), (0.4, "#FFA34D"), (1.0, "#C0392B")]
    )
    cs = ax.contourf(xx, yy, zz, levels=14, cmap=cmap_kde, alpha=0.55, zorder=2)

    # 행정구역 마스킹 (대전 외부 투명)
    clip_patch = _make_clip_patch(sigungu, ax)
    ax.add_patch(clip_patch)
    cs.set_clip_path(clip_patch)

    _draw_sigungu(ax, sigungu)

    # 신규 검토 대상
    t30 = schools[schools["학교명"].isin(set(top30["학교명"]))]
    ax.scatter(t30.geometry.x, t30.geometry.y, s=110, c="#1976D2",
               marker="*", edgecolors="white", linewidths=0.9,
               alpha=0.95, zorder=6, label="신규 검토 대상")

    # KDE 컬러바 (우상단 inset)
    cbar_ax = ax.inset_axes([0.68, 0.92, 0.27, 0.02])
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    cbar_ax.imshow(grad, cmap=cmap_kde, aspect="auto")
    cbar_ax.set_xticks([0, 255])
    cbar_ax.set_xticklabels(["낮음", "높음"], fontsize=8.5)
    cbar_ax.set_yticks([])
    cbar_ax.set_title("학생 분포 밀도 (가중 KDE)", fontsize=9, pad=2)

    ax.legend(**LEGEND_LOWER_RIGHT, fontsize=10)
    return _save(fig, "06_KDE학생분포.png")


# ===== main =====

def main():
    print("=" * 72)
    print("보고서용 정적 도면 (운영 12교 기준 v2)")
    print("=" * 72)

    sigungu = _load_sigungu()
    schools = _load_schools()
    redev = _load_redev()
    top30 = _load_top30()

    op_n = (schools["운영상태"] == "운영").sum()
    pl_n = (schools["운영상태"] == "예정").sum()
    nm_n = (schools["운영상태"] == "명목").sum()
    print(f"\n  자치구: {len(sigungu)}개, 학교: {len(schools)}교 "
          f"(운영 {op_n} + 예정 {pl_n} + 명목 {nm_n}), "
          f"재개발: {len(redev)}건, 상위30: {len(top30)}")

    print()
    figure_01_overview(sigungu, schools, redev, top30)
    figure_02_redev_with_kde(sigungu, redev)
    figure_03_top30(sigungu, schools, top30)
    # 04: 제외 (보고서 비포함, 사용자 결정)
    figure_05_slope_with_operating(sigungu, schools, top30)
    figure_06_kde_students(sigungu, schools, top30)

    # 기존 04_14교vs상위30교.png 와 05_경사도음영_14교.png 삭제 (새 파일명으로 교체됨)
    for old in ["04_14교vs상위30교.png", "05_경사도음영_14교.png"]:
        oldp = OUTPUT_FIGURES / old
        if oldp.exists():
            oldp.unlink()
            print(f"  - 구파일 삭제: {old}")

    print("\n" + "=" * 72)
    print("[DONE] 보고서용 정적 도면 갱신 완료")
    print("=" * 72)


if __name__ == "__main__":
    main()
