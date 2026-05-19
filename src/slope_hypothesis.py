"""Phase B-1: 가설 검증.

가설: "신규 아파트 단지는 (대전 평균 / 학교보다) 경사가 급하다"

그룹
  A 신규 아파트  : 재개발 중 (공사중 + 관리처분)의 slope_500m_mean
  B 진행 전체   : 진행 재개발(1·2·3·4·5단계)의 slope_500m_mean — 0_완료, 6_입안, 9_미정 제외
  C 학교 245교   : 학교의 slope_300m_mean
  D 대전 영역   : 대전 slope 래스터 random sample 1000개 (참고용)

검정
  A vs C : Mann-Whitney U (+ 정규성 시 t-test)
  A vs B : Mann-Whitney U
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
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats as sp_stats

from src.config import DATA_PROCESSED, DATA_EXTERNAL, OUTPUT_FIGURES

rcParams["font.family"] = "Malgun Gothic"
rcParams["axes.unicode_minus"] = False

SLOPE_TIF = DATA_EXTERNAL / "dem" / "대전_slope_5m.tif"
SCHOOLS_SLOPE_CSV = DATA_PROCESSED / "schools_with_slope.csv"
REDEV_SLOPE_CSV = DATA_PROCESSED / "redev_with_slope.csv"

OUT_PNG = OUTPUT_FIGURES / "slope_hypothesis.png"

NEW_APT_STAGES = ["1_공사중", "2_관리처분"]
PROGRESS_STAGES = ["1_공사중", "2_관리처분", "3_사업시행", "4_조합설립", "5_초기"]


def _load_groups():
    schools = pd.read_csv(SCHOOLS_SLOPE_CSV, encoding="utf-8-sig")
    redev = pd.read_csv(REDEV_SLOPE_CSV, encoding="utf-8-sig")

    A = redev.loc[
        redev["통학영향_임박도"].isin(NEW_APT_STAGES), "slope_500m_mean"
    ].dropna().values

    B = redev.loc[
        redev["통학영향_임박도"].isin(PROGRESS_STAGES), "slope_500m_mean"
    ].dropna().values

    C = schools["slope_300m_mean"].dropna().values

    # D: 대전 slope random sample
    with rasterio.open(SLOPE_TIF) as src:
        nodata = src.nodata
        arr = src.read(1)
    valid_mask = np.isfinite(arr) & (arr != nodata)
    valid_vals = arr[valid_mask]
    rng = np.random.default_rng(42)
    n_sample = min(1000, len(valid_vals))
    D = rng.choice(valid_vals, size=n_sample, replace=False)

    return {"A": A, "B": B, "C": C, "D": D}


def _desc_row(name, label, arr):
    return {
        "그룹": name,
        "설명": label,
        "n": len(arr),
        "mean": float(np.mean(arr)) if len(arr) else np.nan,
        "median": float(np.median(arr)) if len(arr) else np.nan,
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else np.nan,
    }


def _print_desc(groups):
    rows = [
        _desc_row("A", "신규 아파트 (공사중+관리처분)", groups["A"]),
        _desc_row("B", "진행 전체 (1·2·3·4·5단계)",   groups["B"]),
        _desc_row("C", "학교 245교 (slope_300m_mean)", groups["C"]),
        _desc_row("D", "대전 영역 random 1000",        groups["D"]),
    ]
    print("\n[그룹별 기술통계]")
    print(f"  {'그룹':<4} {'설명':<32} {'n':>5} {'mean':>8} {'median':>8} {'std':>8}")
    print(f"  {'-'*4} {'-'*32} {'-'*5} {'-'*8} {'-'*8} {'-'*8}")
    for r in rows:
        print(f"  {r['그룹']:<4} {r['설명']:<32} "
              f"{r['n']:>5d} "
              f"{r['mean']:>7.2f}° "
              f"{r['median']:>7.2f}° "
              f"{r['std']:>7.2f}°")


def _shapiro(arr, name):
    if len(arr) < 3:
        return None
    n = min(len(arr), 5000)
    sample = arr if len(arr) <= n else np.random.default_rng(0).choice(arr, n, replace=False)
    try:
        stat, p = sp_stats.shapiro(sample)
        return {"name": name, "stat": stat, "p": p, "normal": p > 0.05}
    except Exception:
        return None


def _test_two_groups(a, b, label_a, label_b):
    print(f"\n  --- {label_a} vs {label_b} ---")
    print(f"  n_{label_a}={len(a)}, n_{label_b}={len(b)}")

    # Mann-Whitney U (양측)
    u, p_u = sp_stats.mannwhitneyu(a, b, alternative="two-sided")
    print(f"  Mann-Whitney U = {u:.1f}, p = {p_u:.4f}")

    # 정규성
    sh_a = _shapiro(a, label_a)
    sh_b = _shapiro(b, label_b)
    both_normal = sh_a and sh_b and sh_a["normal"] and sh_b["normal"]
    if sh_a and sh_b:
        print(f"  Shapiro-Wilk: {label_a} p={sh_a['p']:.4f} (정규성 {'OK' if sh_a['normal'] else 'X'}), "
              f"{label_b} p={sh_b['p']:.4f} ({'OK' if sh_b['normal'] else 'X'})")

    # t-test (참고용)
    t, p_t = sp_stats.ttest_ind(a, b, equal_var=False)
    print(f"  Welch t-test: t={t:.3f}, p={p_t:.4f}{' (정규성 충족, 신뢰)' if both_normal else ' (참고용)'}")

    # 결론
    diff = a.mean() - b.mean()
    sig = p_u < 0.05
    if sig:
        sign = "가파름" if diff > 0 else "완만함"
        print(f"  → 결론: {label_a}이 {label_b}보다 {abs(diff):.2f}° {sign} (p={p_u:.4f}, 유의)")
    else:
        print(f"  → 결론: 통계적 차이 없음 (Δ={diff:+.2f}°, p={p_u:.4f})")

    return {"u": u, "p_mwu": p_u, "t": t, "p_t": p_t, "diff": diff}


def _plot_boxplot(groups, test_results):
    data = [groups["A"], groups["B"], groups["C"], groups["D"]]
    labels = [
        f"A 신규 아파트\n(공사중+관리처분)\nn={len(groups['A'])}",
        f"B 진행 전체\n(1·2·3·4·5단계)\nn={len(groups['B'])}",
        f"C 학교 245교\n(300m mean)\nn={len(groups['C'])}",
        f"D 대전 영역\n(random 1000)\nn={len(groups['D'])}",
    ]
    # 색상: 공사중·관리처분=빨강+주황 톤 → A는 빨강, B는 주황, C 학교=파랑, D 대전=회색
    colors = ["#D32F2F", "#F57C00", "#1976D2", "#9E9E9E"]

    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black",
                       markersize=7),
        medianprops=dict(color="black", linewidth=1.5),
        flierprops=dict(marker=".", markersize=3, alpha=0.4),
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

    ax.set_ylabel("경사도 (°)", fontsize=12)
    ax.set_title("현행 통학차량 운영 학교의 경사도 분포 — 점수 산식의 사후 검증",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(bottom=0)

    # 각 박스 위에 mean 텍스트
    for i, arr in enumerate(data, start=1):
        if len(arr) > 0:
            ax.text(i, arr.max() * 0.95 + 1, f"μ={np.mean(arr):.1f}°",
                    ha="center", fontsize=9, color="#333")

    # 캡션: 가설 검정 + 14교 사후 검증 결론 (Phase B-2)
    p_ac = test_results["A_vs_C"]["p_mwu"]
    p_ab = test_results["A_vs_B"]["p_mwu"]
    cap_top = (
        f"Mann-Whitney U  |  A vs C: p={p_ac:.4f}"
        f"{' *' if p_ac < 0.05 else ''}"
        f"   |   A vs B: p={p_ab:.4f}"
        f"{' *' if p_ab < 0.05 else ''}"
    )
    cap_bottom = (
        "사후 검증: 현행 14교 평균 8.0°, 나머지 학교 평균 4.4°, "
        "Mann-Whitney p<0.0001 → 14교는 통계적으로 더 가파른 곳에 위치 "
        "(점수 산식 외 보조 신호)"
    )
    fig.text(0.5, 0.04, cap_top, ha="center", fontsize=10, style="italic", color="#555")
    fig.text(0.5, 0.01, cap_bottom, ha="center", fontsize=9.5, color="#C0392B")

    fig.tight_layout(rect=(0, 0.07, 1, 1))
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[boxplot] 저장: {OUT_PNG}")


def main():
    print("=" * 70)
    print("Phase B-1: 신규 아파트 경사도 가설 검증")
    print("=" * 70)

    groups = _load_groups()
    _print_desc(groups)

    print("\n[가설 검정]")
    results = {}
    results["A_vs_C"] = _test_two_groups(groups["A"], groups["C"],
                                         "A (신규 아파트)", "C (학교 245교)")
    results["A_vs_B"] = _test_two_groups(groups["A"], groups["B"],
                                         "A (신규 아파트)", "B (진행 전체)")

    _plot_boxplot(groups, results)

    print("\n" + "=" * 70)
    print("[DONE] Phase B-1")
    print("=" * 70)


if __name__ == "__main__":
    main()
