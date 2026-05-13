"""통학버스 운용 종합 우선순위 분석

두 점수로 분리:
  - 도시개발영향 점수: PRIORITY_WEIGHTS 지표1·2 (임박도+학생유입) 결합
  - 적격성 점수:       지표3·4 (수용여석+외곽성) 결합 (트램=0 보류)

종합점수 = 도시개발영향 × 0.5 + 적격성 × 0.5 (INTEGRATED_WEIGHTS)

산출:
  - outputs/tables/도시개발영향_상위30교.csv
  - outputs/tables/통학지원적격성_상위30교.csv
  - outputs/tables/통학버스운용_종합우선순위.csv (전체 + 종합 상위 30교)
"""
import re
import pandas as pd
from src.config import (
    DATA_PROCESSED, DATA_EXTERNAL, OUTPUT_TABLES,
    INTEGRATED_WEIGHTS, DEVIMPACT_SUB_WEIGHTS, ELIGIBILITY_SUB_WEIGHTS,
)
from src.priority_analysis import compute_priority


BUS_CSV = DATA_EXTERNAL / "bus" / "대전_현행통학차량_14개교.csv"


def _load_bus14():
    """현행 통학차량 14개교 CSV 로드"""
    if not BUS_CSV.exists():
        return None
    return pd.read_csv(BUS_CSV, encoding="utf-8-sig")


def match_bus_to_schools(bus_df, schools_df):
    """버스 CSV의 약식 학교명을 schools의 정식 학교명으로 정밀 매칭.

    규칙:
      - 학교급 = '초' (초등학교)만 매칭, 중학교 절대 매칭 X
      - "기성초 및 길헌분교" → 기성초등학교 + 기성초등학교길헌분교장 (둘 다, 2건)
      - "대전원신흥초 복용분교장" → 분교장만 (본교 제외, 1건)
      - "진잠초" → 대전진잠초등학교만 (진잠중학교 매칭 안 됨, 1건)
      - 기타: 정식 학교명에 키워드 포함하는 초등학교 1건
      → 결과 정확히 15건 (14행 CSV + 기성초 본교/분교 분리로 +1)
    """
    if bus_df is None:
        return pd.DataFrame(), []

    # 초등학교만 추출
    elem_only = schools_df[schools_df.get("학교급", "초") == "초"].copy() \
                if "학교급" in schools_df.columns else schools_df.copy()
    elem_names = elem_only["학교명"].tolist()

    def find_all_matches(short):
        """short(약식명) → 매칭된 정식 학교명 리스트 반환 (1개 또는 2개)"""
        results = []
        if "및" in short:
            # "기성초 및 길헌분교" → 본교 + 분교 둘 다
            parts = [p.strip() for p in re.split(r"\s*및\s*", short)]
            # parts: ["기성초", "길헌분교"]
            base = parts[0].replace("대전", "").strip()  # "기성초"
            sub = parts[1].strip()                        # "길헌분교"
            # 본교 (분교 단어가 없는 매칭)
            for n in elem_names:
                if base in n and "분교" not in n:
                    results.append(n)
                    break
            # 분교 (분교 단어 포함 + base 포함)
            for n in elem_names:
                if base in n and ("분교" in n or sub.replace("분교","").strip() in n):
                    if n not in results:
                        results.append(n)
                        break
            return results
        elif "분교장" in short or "분교" in short:
            # "대전원신흥초 복용분교장" → 분교장만, 본교 제외
            key = short.replace(" ", "")
            key_main = re.sub(r"(복용)?분교장?$", "", key).replace("대전", "")
            for n in elem_names:
                if "분교" in n and key_main in n:
                    results.append(n)
                    return results
            return results
        else:
            # 일반: 약식 → 정식 학교명 (분교 제외)
            key = short.replace("대전", "").strip()
            for n in elem_names:
                if key in n and "분교" not in n:
                    results.append(n)
                    return results
            return results

    matched_rows = []
    unmatched = []
    for _, row in bus_df.iterrows():
        short = str(row["학교명"])
        matches = find_all_matches(short)
        if not matches:
            unmatched.append(short)
            continue
        for full in matches:
            sch_row = elem_only[elem_only["학교명"] == full].iloc[0]
            matched_rows.append({
                **row.to_dict(),
                "정식학교명": full,
                "lat": sch_row.get("lat"),
                "lon": sch_row.get("lon"),
                "구": sch_row.get("구"),
            })
    return pd.DataFrame(matched_rows), unmatched


def compute_integrated(schools_df, verbose=True):
    """학교별 도시개발영향(priority_analysis) + 적격성(tonghak_eligibility) + 미래시나리오.

    Returns:
        DataFrame (정렬: 미래시나리오순위)
    """
    from src.tonghak_eligibility import compute_eligibility

    if verbose:
        print("  · 도시개발영향 지표 (priority_analysis) 계산 중...")
    pdf = compute_priority(schools_df, verbose=False)

    # === 도시개발영향 점수 ===
    pdf["도시개발영향점수"] = (
        pdf["지표1_임박도"] * DEVIMPACT_SUB_WEIGHTS["지표1_임박도가중"]
        + pdf["지표2_학생유입"] * DEVIMPACT_SUB_WEIGHTS["지표2_학생유입"]
    ).round(4)

    # === 적격성 점수 — 새 모듈(tonghak_eligibility)에서 5개 지표 산출 ===
    if verbose:
        print("  · 적격성 5개 지표 (tonghak_eligibility) 계산 중...")
    elig_df = compute_eligibility(schools_df, verbose=False)
    elig_lookup = elig_df.set_index("학교명")[["적격성점수"]].to_dict("index")

    pdf["적격성점수"] = pdf["학교명"].apply(
        lambda n: elig_lookup.get(n, {}).get("적격성점수", 0.0)
    )

    # === 미래 시나리오 점수 ===
    pdf["미래시나리오점수"] = (
        pdf["도시개발영향점수"] * INTEGRATED_WEIGHTS["도시개발영향"]
        + pdf["적격성점수"] * INTEGRATED_WEIGHTS["적격성"]
    ).round(4)

    pdf["도시개발영향순위"] = pdf["도시개발영향점수"].rank(ascending=False, method="min").astype(int)
    pdf["적격성순위"] = pdf["적격성점수"].rank(ascending=False, method="min").astype(int)
    pdf["미래시나리오순위"] = pdf["미래시나리오점수"].rank(ascending=False, method="min").astype(int)

    return pdf.sort_values("미래시나리오순위").reset_index(drop=True)


def run(verbose=True):
    schools_df = pd.read_csv(DATA_PROCESSED / "schools_with_impact.csv")
    print(f"[integrated_priority] 학교 {len(schools_df)}교")

    integ_df = compute_integrated(schools_df, verbose=verbose)

    # 현행 14교 매칭
    bus_df = _load_bus14()
    if bus_df is not None:
        bus_matched, unmatched = match_bus_to_schools(bus_df, integ_df)
        bus_school_names = set(bus_matched["정식학교명"].tolist())
        print(f"  · 현행 14교 매칭: {len(bus_matched)}/14 성공")
        if unmatched:
            print(f"  · 매칭 실패: {unmatched}")
    else:
        bus_school_names = set()
        bus_matched = pd.DataFrame()
        unmatched = []
        print("  · 버스 CSV 없음")

    integ_df["현행운영여부"] = integ_df["학교명"].apply(
        lambda n: "Y" if n in bus_school_names else ""
    )

    # === 저장 ===
    cols_main = [
        "미래시나리오순위", "학교명", "학교급", "구", "동", "학생수",
        "도시개발영향점수", "도시개발영향순위",
        "적격성점수", "적격성순위",
        "미래시나리오점수",
        "현행운영여부",
        "영향사업수", "영향사업목록",
        "lat", "lon",
    ]
    main_out = OUTPUT_TABLES / "통학버스운용_미래시나리오.csv"
    integ_df[cols_main].to_csv(main_out, index=False, encoding="utf-8-sig")

    top_integ = integ_df.head(30)
    top_integ_out = OUTPUT_TABLES / "통학버스운용_미래시나리오_상위30교.csv"
    top_integ[cols_main].to_csv(top_integ_out, index=False, encoding="utf-8-sig")

    # 3) 도시개발영향 상위 30 (별도 정렬)
    dev_top = integ_df.sort_values("도시개발영향순위").head(30)
    dev_out = OUTPUT_TABLES / "도시개발영향_상위30교.csv"
    dev_top[cols_main].to_csv(dev_out, index=False, encoding="utf-8-sig")

    # 4) 적격성 상위 30
    elig_top = integ_df.sort_values("적격성순위").head(30)
    elig_out = OUTPUT_TABLES / "통학지원적격성_상위30교.csv"
    elig_top[cols_main].to_csv(elig_out, index=False, encoding="utf-8-sig")

    print(f"\n✅ {main_out.name} (전체 {len(integ_df)}교)")
    print(f"✅ {top_integ_out.name}")
    print(f"✅ {dev_out.name}")
    print(f"✅ {elig_out.name}")

    # 콘솔 출력: 미래시나리오 상위 20교
    print("\n=== 미래 시나리오 적격성 상위 20교 ===")
    show = ["미래시나리오순위", "학교명", "학교급", "구",
            "도시개발영향점수", "적격성점수", "미래시나리오점수", "현행운영여부"]
    print(integ_df[show].head(20).to_string(index=False))

    overlap = integ_df.head(30)["현행운영여부"].eq("Y").sum()
    print(f"\n📊 미래시나리오 상위 30교 중 현행 14교 포함: {overlap}교 / 14교")
    if overlap == 0:
        print("   ⚠️  0교 — 점수 설계 재검토 필요")
    elif overlap >= 12:
        print("   ⚠️  거의 다 들어감 — 점수가 단순 운영여부 추정에 가까움")
    else:
        print("   ✅ 적정 overlap (분석 유효성 OK)")

    return integ_df, bus_matched, unmatched


if __name__ == "__main__":
    run()
