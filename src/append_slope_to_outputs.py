"""적격성·우선순위 출력 CSV에 slope_300m_mean 컬럼 후처리 추가.

Phase B-2 결정에 따른 후처리 (점수 산식 미변경).
schools_with_slope.csv → 학교명 기준 join → 기존 CSV 덮어쓰기.

대상 (모두 outputs/tables/):
- 통학지원적격성_학교별.csv
- 통학지원적격성_상위30교.csv
- 통학지원_우선순위_학교별.csv
- 통학지원_우선순위_상위30교.csv
"""
from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import pandas as pd

from src.config import DATA_PROCESSED, OUTPUT_TABLES

SLOPE_CSV = DATA_PROCESSED / "schools_with_slope.csv"
SLOPE_COL = "slope_300m_mean"

TARGETS = [
    "통학지원적격성_학교별.csv",
    "통학지원적격성_상위30교.csv",
    "통학지원_우선순위_학교별.csv",
    "통학지원_우선순위_상위30교.csv",
]


def main():
    slope_df = pd.read_csv(SLOPE_CSV, encoding="utf-8-sig")
    if "학교명" not in slope_df.columns or SLOPE_COL not in slope_df.columns:
        raise KeyError(f"slope CSV에 필요한 컬럼 없음: 학교명/{SLOPE_COL}")
    slope_map = dict(zip(slope_df["학교명"], slope_df[SLOPE_COL]))
    print(f"[slope] 학교 {len(slope_map)}교, slope_300m_mean 평균 "
          f"{slope_df[SLOPE_COL].mean():.2f}°")

    for fname in TARGETS:
        path = OUTPUT_TABLES / fname
        if not path.exists():
            print(f"  [skip] {fname} (파일 없음)")
            continue
        df = pd.read_csv(path)
        if "학교명" not in df.columns:
            print(f"  [skip] {fname} (학교명 컬럼 없음)")
            continue

        prev_exists = SLOPE_COL in df.columns
        df[SLOPE_COL] = df["학교명"].map(slope_map)
        n_match = df[SLOPE_COL].notna().sum()
        df.to_csv(path, index=False, encoding="utf-8-sig")
        tag = "갱신" if prev_exists else "추가"
        print(f"  [{tag}] {fname}: {len(df)}행, slope 매칭 {n_match}건")

    print("\n[DONE] CSV slope_300m_mean 컬럼 후처리 완료")


if __name__ == "__main__":
    main()
