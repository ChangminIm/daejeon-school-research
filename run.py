"""메인 파이프라인 실행

순서:
1. parse_schools         → schools.csv
2. geocode (학교)        → schools_geocoded.csv (SHP > official > demo)
3. geocode_redev (재개발) → redev_projects_geocoded.csv (이미 있으면 스킵)
4. analysis              → schools_with_impact.csv + 요약표 2종
5. viz_map               → outputs/maps/*.html
"""
from src import (
    parse_schools, geocode, geocode_redev, analysis,
    priority_analysis, tonghak_eligibility, integrated_priority, viz_map,
)
from src.config import DATA_PROCESSED, OUTPUT_TABLES, OUTPUT_MAPS, DATA_EXTERNAL
import pandas as pd


def main():
    print("="*60)
    print("대전광역시교육청 통학지원 - 외부환경분석 파이프라인")
    print("="*60)

    # 1. 학교 데이터 파싱
    print("\n[1/5] 학교 데이터 파싱...")
    schools = parse_schools.parse()
    schools.to_csv(DATA_PROCESSED / "schools.csv", index=False, encoding="utf-8-sig")
    print(f"   ✅ {len(schools)}개 학교")

    # 2. 학교 좌표 부여
    print("\n[2/5] 학교 좌표 부여...")
    shp_dir = DATA_EXTERNAL / "schools"
    official_path = DATA_EXTERNAL / "전국초중등학교위치표준데이터.csv"
    if (shp_dir / "초등학교_위치정보_WGS.shp").exists():
        print("   📍 사용자 SHP 사용")
        geocoded = geocode.geocode_from_shp(schools, shp_dir)
    elif official_path.exists():
        print("   📍 공공데이터포털 좌표 사용")
        geocoded = geocode.geocode_official(schools, official_path)
    else:
        print("   📍 데모 모드")
        geocoded = geocode.geocode_demo(schools)
    geocoded.to_csv(DATA_PROCESSED / "schools_geocoded.csv",
                    index=False, encoding="utf-8-sig")
    print(f"   ✅ 좌표 부여: {geocoded['lat'].notna().sum()}/{len(geocoded)}")

    # 3. 재개발 지오코딩 (캐시 우선)
    print("\n[3/5] 재개발 지오코딩 (VWorld)...")
    redev_csv = DATA_PROCESSED / "redev_projects_geocoded.csv"
    if redev_csv.exists():
        df_redev = pd.read_csv(redev_csv)
        print(f"   ⏭️  스킵: 기존 결과 사용 ({len(df_redev)}건)")
    else:
        df_redev = geocode_redev.run(force=False)
    n_ok = (df_redev["지오코딩방식"] != "실패").sum() if "지오코딩방식" in df_redev else len(df_redev)
    print(f"   ✅ 지오코딩 성공: {n_ok}/{len(df_redev)} ({n_ok/len(df_redev)*100:.1f}%)")

    # 4. 영향권 분석
    print("\n[4/5] 재개발 영향권 분석...")
    schools_gdf = analysis.schools_to_gdf(geocoded)
    projects_gdf = analysis.projects_to_gdf()  # 진행 사업만
    schools_gdf = analysis.compute_impact_zones(schools_gdf, projects_gdf)
    summary = analysis.summarize_impact(schools_gdf, projects_gdf)
    imminence = analysis.summarize_by_imminence(summary)

    summary.to_csv(OUTPUT_TABLES / "재개발영향권_요약.csv",
                   index=False, encoding="utf-8-sig")
    imminence.to_csv(OUTPUT_TABLES / "재개발영향권_요약_임박도별.csv",
                     index=False, encoding="utf-8-sig")
    schools_with_impact = schools_gdf.drop(columns=["geometry"])
    schools_with_impact.to_csv(DATA_PROCESSED / "schools_with_impact.csv",
                               index=False, encoding="utf-8-sig")
    print(f"   ✅ 진행 사업 {len(projects_gdf)}건 분석")
    print("\n   === 임박도별 영향권 요약 ===")
    print(imminence.to_string(index=False))

    # 5. 5개 지표 우선순위 분석 (기존, 내부 유틸로 사용)
    print("\n[5/7] 5개 지표 우선순위 분석 (기반)...")
    priority_df = priority_analysis.compute_priority(schools_with_impact, verbose=False)
    priority_analysis.save_results(priority_df, top_n=30)
    print(f"   ✅ {len(priority_df)}교 5개 지표 산출")

    # 5.5 통학지원 적격성 (3지표) + 신규 검토 대상 30교 CSV
    print("\n[5.5/7] 통학지원 적격성 (3지표) + 신규 검토 대상 30교 추출...")
    tonghak_eligibility.run()

    # 6. 통합 우선순위 (도시개발영향 / 적격성 / 미래시나리오)
    print("\n[6/7] 통합 우선순위 (도시개발영향 + 적격성 → 미래시나리오)...")
    integ_df, bus_matched, unmatched = integrated_priority.run(verbose=False)

    # 7. Folium 맵
    print("\n[7/7] Folium 인터랙티브 맵 생성...")
    map_path = viz_map.build_map(schools_with_impact, include_tram=False)
    print(f"   ✅ 맵 저장: {map_path}")

    # 점수 신뢰도 검증
    _print_reliability(integ_df)

    print("\n" + "="*60)
    print(f"🎉 완료: {map_path}")
    print("="*60)


def _print_reliability(integ_df):
    """run.py 끝에 점수 신뢰도 자동 검증 출력."""
    import pandas as pd

    print("\n" + "─"*60)
    print("[점수 신뢰도 검증]")
    print("─"*60)

    elig_top30 = integ_df.sort_values("적격성순위").head(30)

    # 1. 적격성 상위 30교 중 현행 14교 포함
    overlap = elig_top30["현행운영여부"].eq("Y").sum()
    if 9 <= overlap <= 12:
        verdict = "✅ 양호 (9~12교 권장 범위)"
    elif overlap < 5:
        verdict = "⚠️  5교 미만 — 점수 설계 재검토"
    elif overlap >= 14:
        verdict = "⚠️  사실상 14교 전부 — 점수가 운영여부 추정에 가까움"
    else:
        verdict = "  중간 (검토 권장)"
    print(f"1. 적격성 상위 30교 ∩ 현행 14교: {overlap}교 / 14교  {verdict}")

    # 2. 적격성 상위 30교의 도심 비율 (서구/중구/동구 도심 = 도심으로 간주)
    # 외곽: 대덕구·유성구·서구 도안/관저/가수원, 동구 외곽 — 간단히 학생수 평균으로 대리
    # 학생수 < 전체 중앙값이면 외곽으로 추정
    median_students = integ_df["학생수"].median()
    elig_top30_urban_count = (elig_top30["학생수"] > median_students).sum()
    urban_pct = elig_top30_urban_count / 30 * 100
    verdict2 = "⚠️  도심 비율 30% 초과 — 외곽 가중치 늘릴 필요" if urban_pct > 30 \
               else "✅ 적정 (외곽 학교 위주)"
    print(f"2. 적격성 상위 30교 도심 학교 비율: {urban_pct:.0f}%  {verdict2}")
    print(f"   (도심 판정 = 학생수 > 전체 중앙값 {median_students:.0f}명)")

    # 3. 적격성 상위 10교
    print(f"\n3. 적격성 상위 10교 (육안 검증):")
    show3 = ["적격성순위", "학교명", "학교급", "구", "학생수", "적격성점수", "현행운영여부"]
    print(integ_df.sort_values("적격성순위").head(10)[show3].to_string(index=False))

    # 4. 도시개발 영향 상위 10교
    print(f"\n4. 도시개발 영향 상위 10교:")
    show4 = ["도시개발영향순위", "학교명", "학교급", "구", "학생수", "도시개발영향점수", "현행운영여부"]
    print(integ_df.sort_values("도시개발영향순위").head(10)[show4].to_string(index=False))

    # 5. 미래 시나리오 적격성 상위 10교
    print(f"\n5. 미래 시나리오 적격성 상위 10교 (적격성 0.70 + 도시개발 0.30):")
    show5 = ["미래시나리오순위", "학교명", "학교급", "구", "학생수",
             "적격성점수", "도시개발영향점수", "미래시나리오점수", "현행운영여부"]
    print(integ_df.sort_values("미래시나리오순위").head(10)[show5].to_string(index=False))


if __name__ == "__main__":
    main()
