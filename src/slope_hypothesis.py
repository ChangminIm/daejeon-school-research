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


def _bus14_set(schools_df):
    """현행 통학버스 14교의 정식 학교명 set."""
    bus_csv = DATA_EXTERNAL / "bus" / "대전_현행통학차량_14개교.csv"
    if not bus_csv.exists():
        return set()
    bus_df = pd.read_csv(bus_csv, encoding="utf-8-sig")
    from src.integrated_priority import match_bus_to_schools
    matched, _ = match_bus_to_schools(bus_df, schools_df)
    return set(matched["정식학교명"].tolist())


def _load_groups():
    schools = pd.read_csv(SCHOOLS_SLOPE_CSV, encoding="utf-8-sig")
    redev = pd.read_csv(REDEV_SLOPE_CSV, encoding="utf-8-sig")

    A = redev.loc[
        redev["통학영향_임박도"].isin(NEW_APT_STAGES), "slope_500m_mean"
    ].dropna().values

    B = redev.loc[
        redev["통학영향_임박도"].isin(PROGRESS_STAGES), "slope_500m_mean"
    ].dropna().values

    # C: 학교 전체 (가설 검정 A vs C용 — 기존 호환 유지)
    C = schools["slope_300m_mean"].dropna().values

    # C': 14교 제외 학교, E: 14교 (사후 검증)
    bus14 = _bus14_set(schools)
    is_bus14 = schools["학교명"].isin(bus14)
    C_prime = schools.loc[~is_bus14, "slope_300m_mean"].dropna().values
    E = schools.loc[is_bus14, "slope_300m_mean"].dropna().values

    # D: 대전 slope random sample
    with rasterio.open(SLOPE_TIF) as src:
        nodata = src.nodata
        arr = src.read(1)
    valid_mask = np.isfinite(arr) & (arr != nodata)
    valid_vals = arr[valid_mask]
    rng = np.random.default_rng(42)
    n_sample = min(1000, len(valid_vals))
    D = rng.choice(valid_vals, size=n_sample, replace=False)

    return {"A": A, "B": B, "C": C, "C_prime": C_prime, "E": E, "D": D}


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
        _desc_row("A",  "신규 아파트 (공사중+관리처분)",      groups["A"]),
        _desc_row("B",  "진행 전체 (1·2·3·4·5단계)",         groups["B"]),
        _desc_row("C'", "학교 (14교 제외, n=228)",            groups["C_prime"]),
        _desc_row("E",  "현행 14교 ★ (통학버스 운영)",       groups["E"]),
        _desc_row("D",  "대전 영역 random 1000",              groups["D"]),
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
    data = [groups["A"], groups["B"], groups["C_prime"], groups["E"], groups["D"]]
    labels = [
        f"A 신규 아파트\n(공사중+관리처분)\nn={len(groups['A'])}",
        f"B 진행 전체\n(1·2·3·4·5단계)\nn={len(groups['B'])}",
        f"C' 학교\n(14교 제외)\nn={len(groups['C_prime'])}",
        f"★ E 현행 14교 ★\n(통학버스 운영)\nn={len(groups['E'])}",
        f"D 대전 영역\n(random 1000)\nn={len(groups['D'])}",
    ]
    # 색상: A 빨강 / B 주황 / C' 파랑(중간톤) / E 진네이비(강조) / D 회색
    colors = ["#D32F2F", "#F57C00", "#1976D2", "#0D47A1", "#9E9E9E"]

    fig, ax = plt.subplots(figsize=(14, 7))
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
    for i, (patch, c) in enumerate(zip(bp["boxes"], colors)):
        patch.set_facecolor(c)
        # E(14교, idx=3)는 강조: alpha 높이고 테두리 굵게
        if i == 3:
            patch.set_alpha(0.85)
            patch.set_edgecolor("#0D47A1")
            patch.set_linewidth(2.0)
        else:
            patch.set_alpha(0.7)

    ax.set_ylabel("경사도 (°)", fontsize=12)
    ax.set_title("통학차량 운영 학교의 경사도 분포",
                 fontsize=16, fontweight="bold", pad=12)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(bottom=0)

    # 각 박스 위에 mean 텍스트
    for i, arr in enumerate(data, start=1):
        if len(arr) > 0:
            ax.text(i, arr.max() * 0.95 + 1, f"μ={np.mean(arr):.1f}°",
                    ha="center", fontsize=9, color="#333")

    # 캡션·통계 수치 삭제 (일괄 정책) — 박스 위 μ 표시는 유지

    fig.tight_layout()
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
                                         "A (신규 아파트)", "C (학교 243교)")
    results["A_vs_B"] = _test_two_groups(groups["A"], groups["B"],
                                         "A (신규 아파트)", "B (진행 전체)")
    results["E_vs_Cprime"] = _test_two_groups(groups["E"], groups["C_prime"],
                                              "E (현행 14교)", "C' (학교 14교 제외)")

    _plot_boxplot(groups, results)

    print("\n" + "=" * 70)
    print("[DONE] Phase B-1")
    print("=" * 70)


if __name__ == "__main__":
    main()
