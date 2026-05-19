"""보고서용 정적 도면 6장.

매트플롯립 + contextily(CartoDB Positron) + 자치구 경계 + 한글 라벨.
A4 가로(11.7x8.3), 300dpi.

산출: outputs/figures/
  01_종합지도.png
  02_재개발임박도.png
  03_적격성상위30교.png
  04_14교vs상위30교.png
  05_경사도음영_14교.png
  06_KDE학생분포.png
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
from scipy.stats import gaussian_kde
import contextily as ctx
from pyproj import Transformer
from PIL import Image

from src.config import DATA_PROCESSED, DATA_EXTERNAL, OUTPUT_FIGURES, OUTPUT_TABLES

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
IMM_LABEL = {
    "1_공사중": "공사중", "2_관리처분": "관리처분", "3_사업시행": "사업시행",
    "4_조합설립": "조합·추진위", "5_초기": "조합·추진위",
    "6_입안": "입안·미정", "9_미정": "입안·미정",
}

SLOPE_COLOR_STOPS = [
    (0.00, "#2D8B43"), (0.15, "#91C266"), (0.40, "#D4B36A"),
    (0.65, "#B85C2A"), (1.00, "#8B1A1A"),
]


# ===== 데이터 로더 =====

def _load_sigungu():
    return gpd.read_file(ADMIN_SHP).to_crs(CRS_3857)


def _load_schools():
    sch = pd.read_csv(DATA_PROCESSED / "schools_with_impact.csv")
    slope = pd.read_csv(DATA_PROCESSED / "schools_with_slope.csv")
    smap = dict(zip(slope["학교명"], slope["slope_300m_mean"]))
    sch["slope_300m_mean"] = sch["학교명"].map(smap)
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


def _bus14_set(schools_df_csv):
    bus_csv = DATA_EXTERNAL / "bus" / "대전_현행통학차량_14개교.csv"
    if not bus_csv.exists():
        return set()
    bus_df = pd.read_csv(bus_csv, encoding="utf-8-sig")
    from src.integrated_priority import match_bus_to_schools
    matched, _ = match_bus_to_schools(bus_df, schools_df_csv)
    return set(matched["정식학교명"].tolist())


# ===== 공통 그리기 =====

def _setup(title, subtitle=""):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
    ax.set_aspect("equal")
    ax.set_axis_off()
    full_title = title
    if subtitle:
        full_title = f"{title}\n"
        fig.suptitle(title, fontsize=15, fontweight="bold", y=0.96)
        fig.text(0.5, 0.915, subtitle, ha="center", fontsize=10.5, color="#555")
    else:
        ax.set_title(title, fontsize=15, fontweight="bold", pad=10)
    return fig, ax


def _draw_sigungu(ax, sigungu, with_labels=True):
    sigungu.boundary.plot(ax=ax, color="#222", linewidth=0.7, alpha=0.75)
    if with_labels:
        for _, r in sigungu.iterrows():
            c = r.geometry.representative_point()
            ax.annotate(
                r["SIGUNGU_NM"],
                xy=(c.x, c.y), fontsize=11, color="#222",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#888",
                          alpha=0.85, linewidth=0.4),
            )


def _set_extent(ax, sigungu, pad_frac=0.04):
    minx, miny, maxx, maxy = sigungu.total_bounds
    w, h = maxx - minx, maxy - miny
    ax.set_xlim(minx - w*pad_frac, maxx + w*pad_frac)
    ax.set_ylim(miny - h*pad_frac, maxy + h*pad_frac)


def _add_basemap(ax):
    try:
        ctx.add_basemap(ax, crs=CRS_3857,
                         source=ctx.providers.CartoDB.PositronNoLabels,
                         attribution=False)
    except Exception as e:
        print(f"   [경고] basemap 실패: {e}")


def _footer(fig, page_title):
    fig.text(0.01, 0.012, page_title, fontsize=8.5, color="#666", ha="left")
    fig.text(0.99, 0.012, FOOTER, fontsize=8.5, color="#666", ha="right")


def _save(fig, name):
    out = OUTPUT_FIGURES / name
    fig.tight_layout(rect=(0, 0.025, 1, 0.96))
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    size_kb = out.stat().st_size / 1024
    print(f"  ✓ {name} ({size_kb:.1f} KB)")
    return out


# ===== 도면 1: 종합지도 =====

def figure_01_overview(sigungu, schools, redev, top30, bus14):
    fig, ax = _setup(
        "01. 종합 분석 지도",
        "학교 243교 · 도시정비사업 진행 110건 · 적격성 상위 30교 · 현행 통학버스 14교"
    )
    _set_extent(ax, sigungu)
    _add_basemap(ax)
    _draw_sigungu(ax, sigungu)

    # 진행 사업만 (입안·미정 제외하지 않고 모두)
    redev_active = redev[redev["통학영향_임박도"].isin(IMM_COLOR.keys())]
    for stage, sub in redev_active.groupby("통학영향_임박도"):
        color = IMM_COLOR.get(stage, "#888")
        ax.scatter(sub.geometry.x, sub.geometry.y, s=18, c=color,
                   marker="D", alpha=0.6, edgecolors="none", zorder=3)

    # 학교: 초등 원, 중학교 사각형 (모두 회색)
    elem = schools[schools["학교급"] == "초"]
    midd = schools[schools["학교급"] == "중"]
    ax.scatter(elem.geometry.x, elem.geometry.y, s=14, c="#555",
               marker="o", alpha=0.6, edgecolors="white", linewidths=0.4, zorder=4)
    ax.scatter(midd.geometry.x, midd.geometry.y, s=16, c="#555",
               marker="s", alpha=0.6, edgecolors="white", linewidths=0.4, zorder=4)

    # 적격성 상위 30교: ★ 강조 (학교명 기준)
    top30_names = set(top30["학교명"])
    top30_g = schools[schools["학교명"].isin(top30_names)]
    ax.scatter(top30_g.geometry.x, top30_g.geometry.y, s=120, c="#C0392B",
               marker="*", edgecolors="white", linewidths=0.8, alpha=0.95, zorder=6)

    # 14교: 청록 사각형 강조
    bus_g = schools[schools["학교명"].isin(bus14)]
    ax.scatter(bus_g.geometry.x, bus_g.geometry.y, s=130, c="#1ABC9C",
               marker="P", edgecolors="white", linewidths=0.8, alpha=0.95, zorder=6)

    legend_elems = [
        Line2D([], [], marker="*", color="w", markerfacecolor="#C0392B",
               markersize=14, label="★ 적격성 상위 30교"),
        Line2D([], [], marker="P", color="w", markerfacecolor="#1ABC9C",
               markersize=12, label="현행 통학버스 14교"),
        Line2D([], [], marker="o", color="w", markerfacecolor="#555",
               markersize=7, label="초등학교"),
        Line2D([], [], marker="s", color="w", markerfacecolor="#555",
               markersize=7, label="중학교"),
        Line2D([], [], marker="D", color="w", markerfacecolor="#C0392B",
               markersize=7, label="재개발 (임박도별 색상)"),
    ]
    ax.legend(handles=legend_elems, loc="lower left", fontsize=9, framealpha=0.92)

    _footer(fig, "01. 종합 분석 지도")
    return _save(fig, "01_종합지도.png")


# ===== 도면 2: 재개발 임박도 =====

def figure_02_redev_imminence(sigungu, redev):
    fig, ax = _setup(
        "02. 재개발 추진단계별 분포",
        "대전 도시정비사업 120건 — 단계(임박도)별 색상 구분"
    )
    _set_extent(ax, sigungu)
    _add_basemap(ax)
    _draw_sigungu(ax, sigungu)

    # 그룹별 마커
    seda_total = 0
    counts = {}
    for stage in IMM_COLOR.keys():
        sub = redev[redev["통학영향_임박도"] == stage]
        if len(sub) == 0:
            continue
        ax.scatter(sub.geometry.x, sub.geometry.y, s=70,
                   c=IMM_COLOR[stage], alpha=0.75,
                   edgecolors="white", linewidths=0.7, zorder=4,
                   label=f"{IMM_LABEL[stage]} ({stage[0]}단계)")
        counts[stage] = len(sub)
        if "세대수" in sub.columns:
            seda_total += int(sub["세대수"].fillna(0).sum())

    # 단계별 사업수+세대수 텍스트 박스
    rows = []
    for stage in ["1_공사중", "2_관리처분", "3_사업시행", "4_조합설립", "5_초기", "6_입안", "9_미정"]:
        if stage not in counts:
            continue
        sub = redev[redev["통학영향_임박도"] == stage]
        seda = int(sub["세대수"].fillna(0).sum()) if "세대수" in sub.columns else 0
        rows.append(f"{IMM_LABEL[stage]}: {counts[stage]}건 · {seda:,}세대")
    text_box = "재개발 단계별 집계\n" + "\n".join(rows)
    ax.text(0.01, 0.99, text_box, transform=ax.transAxes,
            fontsize=9.5, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#aaa", alpha=0.95))

    # 임박도 범례 우하단
    legend_elems = [
        Patch(color="#C0392B", label="공사중"),
        Patch(color="#E74C3C", label="관리처분"),
        Patch(color="#E67E22", label="사업시행"),
        Patch(color="#F1C40F", label="조합·추진위"),
        Patch(color="#95A5A6", label="입안·미정"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=9, framealpha=0.92,
              title="추진단계", title_fontsize=10)

    _footer(fig, "02. 재개발 추진단계별 분포")
    return _save(fig, "02_재개발임박도.png")


# ===== 도면 3: 적격성 상위 30교 =====

def figure_03_top30(sigungu, schools, top30):
    fig, ax = _setup(
        "03. 통학지원 적격성 상위 30교",
        "현행 14교 제외 · 미운영 학교 중 적격성 점수 상위 30교"
    )
    _set_extent(ax, sigungu)
    _add_basemap(ax)
    _draw_sigungu(ax, sigungu)

    # 전체 학교 옅게
    ax.scatter(schools.geometry.x, schools.geometry.y, s=10, c="#bbb",
               alpha=0.5, edgecolors="none", zorder=3)

    # 상위 30교 등급별
    top30_with_geom = schools.merge(top30[["학교명", "미운영순위"]], on="학교명", how="inner")
    for _, r in top30_with_geom.iterrows():
        rank = int(r["미운영순위"])
        if rank <= 5:
            c, s = "#C0392B", 200
        elif rank <= 15:
            c, s = "#E67E22", 140
        else:
            c, s = "#F1C40F", 100
        ax.scatter(r.geometry.x, r.geometry.y, s=s, c=c, marker="*",
                   edgecolors="white", linewidths=0.9, alpha=0.95, zorder=5)

    # 상위 5교 학교명 라벨
    top5 = top30_with_geom[top30_with_geom["미운영순위"] <= 5].sort_values("미운영순위")
    for _, r in top5.iterrows():
        ax.annotate(
            f"{int(r['미운영순위'])}. {r['학교명']}",
            xy=(r.geometry.x, r.geometry.y),
            xytext=(8, 8), textcoords="offset points",
            fontsize=9.5, color="#222", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#C0392B", alpha=0.92),
        )

    legend_elems = [
        Line2D([], [], marker="*", color="w", markerfacecolor="#C0392B",
               markersize=15, label="상위 1~5교"),
        Line2D([], [], marker="*", color="w", markerfacecolor="#E67E22",
               markersize=12, label="상위 6~15교"),
        Line2D([], [], marker="*", color="w", markerfacecolor="#F1C40F",
               markersize=10, label="상위 16~30교"),
        Line2D([], [], marker="o", color="w", markerfacecolor="#bbb",
               markersize=5, label="기타 학교"),
    ]
    ax.legend(handles=legend_elems, loc="lower left", fontsize=9.5, framealpha=0.92)

    _footer(fig, "03. 통학지원 적격성 상위 30교")
    return _save(fig, "03_적격성상위30교.png")


# ===== 도면 4: 14교 vs 적격성 상위 30교 =====

def figure_04_compare(sigungu, schools, top30, bus14):
    fig, ax = _setup(
        "04. 현행 14교 vs 신규 검토 30교",
        "현행 운영 vs 신규 검토 후보 — 위치 비교"
    )
    _set_extent(ax, sigungu)
    _add_basemap(ax)
    _draw_sigungu(ax, sigungu)

    # 기타 학교 옅게
    others = schools[
        ~schools["학교명"].isin(set(top30["학교명"]) | bus14)
    ]
    ax.scatter(others.geometry.x, others.geometry.y, s=8, c="#ddd",
               alpha=0.5, edgecolors="none", zorder=3)

    # 적격성 상위 30교
    t30 = schools[schools["학교명"].isin(set(top30["학교명"]))]
    ax.scatter(t30.geometry.x, t30.geometry.y, s=160, c="#F1C40F",
               marker="*", edgecolors="#C0392B", linewidths=1.4,
               alpha=0.95, zorder=5, label="★ 신규 검토 30교")

    # 14교
    b14 = schools[schools["학교명"].isin(bus14)]
    ax.scatter(b14.geometry.x, b14.geometry.y, s=160, c="#1ABC9C",
               marker="P", edgecolors="#0E6B5A", linewidths=1.2,
               alpha=0.95, zorder=6, label="현행 14교 (통학버스 운영)")

    # 통계 박스
    n_b14 = len(b14)
    n_t30 = len(t30)
    text_box = (
        f"현행 통학버스: {n_b14}교\n"
        f"신규 검토 후보: {n_t30}교\n"
        f"중복 없음 (현행 14교는 검토에서 제외됨)"
    )
    ax.text(0.01, 0.99, text_box, transform=ax.transAxes,
            fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#aaa", alpha=0.95))

    ax.legend(loc="lower left", fontsize=10, framealpha=0.92)
    _footer(fig, "04. 현행 14교 vs 신규 검토 30교")
    return _save(fig, "04_14교vs상위30교.png")


# ===== 도면 5: 경사도 음영 + 14교 =====

def _png_extent_3857():
    """slope_overlay PNG의 bounds(4326) → 3857 사각형 extent."""
    b = json.loads(SLOPE_BOUNDS.read_text(encoding="utf-8"))
    t = Transformer.from_crs(CRS_4326, CRS_3857, always_xy=True)
    l, btm = t.transform(b["west"], b["south"])
    r, top = t.transform(b["east"], b["north"])
    return l, r, btm, top


def figure_05_slope_bus14(sigungu, schools, top30, bus14):
    fig, ax = _setup(
        "05. 경사도 음영 + 현행 14교 분포",
        "5m DEM 기반 경사도 (녹색=평지 → 빨강=20°+) · 14교 평균 경사 8.0°"
    )
    _set_extent(ax, sigungu)
    # 배경지도 (옅게)
    try:
        ctx.add_basemap(ax, crs=CRS_3857,
                         source=ctx.providers.CartoDB.PositronNoLabels,
                         attribution=False, alpha=0.6)
    except Exception as e:
        print(f"   [경고] basemap 실패: {e}")

    # 경사도 음영 PNG 오버레이
    if SLOPE_PNG.exists() and SLOPE_BOUNDS.exists():
        img = np.array(Image.open(SLOPE_PNG))
        l, r, btm, top = _png_extent_3857()
        ax.imshow(img, extent=[l, r, btm, top], origin="upper",
                  interpolation="bilinear", zorder=2)
    else:
        print("   [경고] slope PNG/bounds 없음")

    _draw_sigungu(ax, sigungu)

    # 적격성 30교
    t30 = schools[schools["학교명"].isin(set(top30["학교명"]))]
    ax.scatter(t30.geometry.x, t30.geometry.y, s=110, c="#F1C40F",
               marker="*", edgecolors="#222", linewidths=0.8,
               alpha=0.95, zorder=5, label="★ 적격성 상위 30교")
    # 14교
    b14 = schools[schools["학교명"].isin(bus14)]
    ax.scatter(b14.geometry.x, b14.geometry.y, s=140, c="#1ABC9C",
               marker="P", edgecolors="#0E6B5A", linewidths=1.3,
               alpha=0.98, zorder=6, label="현행 14교 (통학버스 운영)")

    # 캡션
    cap = "현행 14교는 평균 경사 8.0°로 명확히 산악권 분포 — Mann-Whitney p<0.0001"
    fig.text(0.5, 0.04, cap, ha="center", fontsize=10.5,
             color="#0D47A1", fontweight="bold")

    # 경사도 색상 범례 (가로 색띠) — ax 내부 inset
    cmap = LinearSegmentedColormap.from_list("slope", SLOPE_COLOR_STOPS)
    cbar_ax = ax.inset_axes([0.32, 0.04, 0.36, 0.025])
    grad = np.linspace(0, 30, 256).reshape(1, -1)
    cbar_ax.imshow(grad, cmap=cmap, aspect="auto")
    cbar_ax.set_xticks([0, 64, 128, 192, 255])
    cbar_ax.set_xticklabels(["0°", "7.5°", "15°", "22.5°", "30°+"], fontsize=8.5)
    cbar_ax.set_yticks([])
    cbar_ax.set_title("경사도 (degree)", fontsize=9, pad=2)

    ax.legend(loc="lower left", fontsize=10, framealpha=0.92)
    _footer(fig, "05. 경사도 음영 + 현행 14교 분포")
    return _save(fig, "05_경사도음영_14교.png")


# ===== 도면 6: 학생 분포 KDE =====

def figure_06_kde_students(sigungu, schools, top30):
    fig, ax = _setup(
        "06. 학생 분포 KDE (학생수 가중)",
        "학교 학생수 가중 커널 밀도 — 학생은 도심 밀집, 신규 검토는 외곽 분산"
    )
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

    cs = ax.contourf(xx, yy, zz, levels=14, cmap="YlOrRd", alpha=0.55, zorder=2)

    _draw_sigungu(ax, sigungu)

    # 적격성 30교 오버레이
    t30 = schools[schools["학교명"].isin(set(top30["학교명"]))]
    ax.scatter(t30.geometry.x, t30.geometry.y, s=110, c="#1976D2",
               marker="*", edgecolors="white", linewidths=0.9,
               alpha=0.95, zorder=6, label="★ 적격성 상위 30교")

    # 캡션
    cap = "학생은 도심에 밀집되어 있으나 통학지원 신규 검토 학교는 외곽에 분산"
    fig.text(0.5, 0.04, cap, ha="center", fontsize=10.5,
             color="#C0392B", fontweight="bold")

    ax.legend(loc="lower left", fontsize=10, framealpha=0.92)
    _footer(fig, "06. 학생 분포 KDE (학생수 가중)")
    return _save(fig, "06_KDE학생분포.png")


# ===== main =====

def main():
    print("=" * 70)
    print("보고서용 정적 도면 6장 생성")
    print("=" * 70)

    sigungu = _load_sigungu()
    schools = _load_schools()
    redev = _load_redev()
    top30 = _load_top30()
    schools_raw = pd.read_csv(DATA_PROCESSED / "schools_with_impact.csv")
    bus14 = _bus14_set(schools_raw)

    print(f"\n  자치구: {len(sigungu)}개, 학교: {len(schools)}교, "
          f"재개발: {len(redev)}건, 상위30: {len(top30)}, 14교: {len(bus14)}건")

    print()
    figure_01_overview(sigungu, schools, redev, top30, bus14)
    figure_02_redev_imminence(sigungu, redev)
    figure_03_top30(sigungu, schools, top30)
    figure_04_compare(sigungu, schools, top30, bus14)
    figure_05_slope_bus14(sigungu, schools, top30, bus14)
    figure_06_kde_students(sigungu, schools, top30)

    print("\n" + "=" * 70)
    print("[DONE] 6장 모두 생성 → outputs/figures/")
    print("=" * 70)


if __name__ == "__main__":
    main()
