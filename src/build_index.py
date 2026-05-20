"""GitHub Pages 랜딩 페이지(index.html) 빌더.

outputs/tables/*.csv 와 outputs/figures/*.png 를 글로빙해 동적 카드 생성.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import pandas as pd

from src.config import ROOT, OUTPUT_TABLES, OUTPUT_FIGURES

OUT_INDEX = ROOT / "index.html"

REPO_RAW = "https://raw.githubusercontent.com/ChangminIm/daejeon-school-research/main"
REPO_BLOB = "https://github.com/ChangminIm/daejeon-school-research/blob/main"

# CSV 카테고리 매핑 — 순서대로 표시됨 (분석 흐름 → 최종 결과)
# 파일명에 포함된 키워드(위에서 아래로 매칭, 첫 매칭 우선)
TABLE_CATEGORIES = [
    ("외부환경 자료", [
        "도시개발", "KDE", "재개발", "재건축", "트램",
    ]),
    ("통학버스 운영 현황", [
        "운영현황", "운영_정확", "통학버스운용", "시나리오",
    ]),
    ("노선 중복 분석", [
        "노선중복", "공동활용", "노선경사",
    ]),
    ("실행 가능성", [
        "정류장근접", "정류장_근접",
    ]),
    ("회귀·검증", [
        "회귀결과", "회귀",
    ]),
    ("점수·우선순위 (최종)", [
        "적격성_학교별", "적격성_상위30교", "적격성",
        "14교_적격성_분석결과", "우선순위", "신규검토",
    ]),
]

CATEGORY_EMOJI = ["1⃣", "2⃣", "3⃣", "4⃣", "5⃣", "6⃣"]
ETC_CATEGORY = "기타"


def categorize_csv(filename: str) -> str:
    """파일명 → 카테고리. 첫 매칭 우선."""
    for category, keywords in TABLE_CATEGORIES:
        for kw in keywords:
            if kw in filename:
                return category
    return ETC_CATEGORY


# CSV별 부가 설명 — 매핑된 파일만 표시, 없으면 행수만
CSV_DESC = {
    "통학지원_우선순위_학교별.csv":
        "243교 통학지원 우선순위 종합 점수 (적격성 0.70 + 도시개발영향 0.30)",
    "통학지원_우선순위_상위30교.csv":
        "통합 우선순위 상위 30교 추출",
    "통학지원적격성_학교별.csv":
        "동급학교밀도 0.50 + 도심거리 0.30 + 학교규모 0.20 가중합",
    "통학지원적격성_상위30교.csv":
        "적격성 점수 상위 30교",
    "통학지원적격성_상위30교_v2.csv":
        "적격성 산정 v2 (가중치 조정안, 검토용)",
    "신규검토대상_상위30교.csv":
        "현행 14교 제외, 미운영 학교 중 적격성 상위 30교",
    "도시개발영향_상위30교.csv":
        "재개발·재건축 1km 영향권 가중 점수 상위 30교",
    "재개발영향권_요약.csv":
        "정비사업별 1km/1.5km 영향권 내 학교 집계",
    "재개발영향권_요약_임박도별.csv":
        "사업 추진단계(임박도)별 영향 학교 집계",
    "트램_공사영향권_상세.csv":
        "트램 14개 공구 공사 영향 학교 상세",
    "트램_공사영향권_요약.csv":
        "공구별 영향 학교 수 요약",
    "트램_정거장_접근성.csv":
        "45개 정거장-학교 도보 접근성",
    "통학버스운용_미래시나리오.csv":
        "단기·중기·장기 시나리오별 통학버스 수요",
    "통학버스운용_미래시나리오_상위30교.csv":
        "미래 시나리오 운용 우선 학교 30교",
}

# 정적 도면 부가 설명 (운영 12교 기준 v2)
FIG_DESC = {
    "01_종합지도.png":
        "학교 243교 · 재개발 110건 · 적격성 상위 30교 · 운영 12교 + 예정·명목 각 1",
    "02_재개발임박도.png":
        "재개발 사업 단계별 점 + KDE 도시개발 압력 (세대수 가중) — 대전 영역 마스킹",
    "03_적격성상위30교.png":
        "신규 검토 대상 30교 — 등급별(1~5/6~15/16~30) + 상위 5교 자동배치 라벨",
    "05_경사도음영_운영학교.png":
        "5m DEM 경사도 음영 + 운영 12교 분포 (평균 8°) — 산악권 집중 (p<0.0001)",
    "06_KDE학생분포.png":
        "학생수 가중 KDE — 행정구역 마스킹, 도심 밀집 vs 신규 검토 외곽 분산",
    "07_노선경사프로파일.png":
        "통학차량 183개 노선 + 평균 경사 5단계 색상(인터랙티브와 일관) + 운영 학교 마커",
    "08_노선vs학교위치_경사비교.png":
        "학교 위치(파랑) vs 노선 평균(주황) 막대 — Wilcoxon p=0.007, 노선이 평탄",
    "09_노선중복_공동활용권역.png":
        "노선 100m 버퍼 중첩 영역 + 공동활용 후보 권역 (남선·진잠, 산내·산흥)",
    "09-1_공동활용권역_남선진잠.png":
        "공동활용 후보 권역 확대 — 남선초·진잠초 노선·정류장·중첩 영역 (중첩률 31%)",
    "09-2_공동활용권역_산내산흥.png":
        "공동활용 후보 권역 확대 — 산내초·산흥초 노선·정류장·중첩 영역 (중첩률 48%)",
    "10_경사도분포_그룹별.png":
        "학교 위치 경사도 그룹별 분포 — 신규 아파트·재개발·일반학교·운영 14교·대전 전역",
}

# 정적 도면 제외 목록 (개발용 미리보기 등)
FIG_EXCLUDE = {
    "dem_clip_preview.png",  # 개발용, 보고서 아님
    "우선순위_히트맵.png",   # 이미 폐기됨 (혹시 잔존 시)
}


def _count_rows(csv_path: Path) -> int:
    try:
        df = pd.read_csv(csv_path)
        return len(df)
    except Exception:
        return -1


def _scan_csvs():
    items = []
    for p in sorted(OUTPUT_TABLES.glob("*.csv")):
        items.append({
            "file": p.name,
            "rows": _count_rows(p),
            "size_kb": p.stat().st_size / 1024,
            "desc": CSV_DESC.get(p.name, ""),
        })
    return items


def _figure_sort_key(name: str):
    """번호 있는 파일(01_..., 09-1_...) 먼저, 번호 없는 파일은 뒤로."""
    if len(name) >= 2 and name[:2].isdigit():
        return (0, name)
    return (1, name)


def _figure_title(name: str) -> str:
    """파일명에서 .png 제거하고 첫 번호 뒤 '_'를 '. '로 변환.
    예: '01_종합지도.png'                       → '01. 종합지도'
        '09-1_공동활용권역_남선진잠.png'        → '09-1. 공동활용권역 남선진잠'
        '10_경사도분포_그룹별.png'              → '10. 경사도분포 그룹별'
    """
    stem = name.removesuffix(".png")
    # 09-1, 09-2 같은 하위번호 처리
    if len(stem) >= 5 and stem[:2].isdigit() and stem[2] == "-" and stem[3].isdigit() and stem[4] == "_":
        return stem[:4] + ". " + stem[5:].replace("_", " ")
    if len(stem) >= 3 and stem[:2].isdigit() and stem[2] == "_":
        return stem[:2] + ". " + stem[3:].replace("_", " ")
    return stem


def _scan_figures():
    items = []
    for p in sorted(OUTPUT_FIGURES.glob("*.png"), key=lambda x: _figure_sort_key(x.name)):
        if p.name in FIG_EXCLUDE:
            continue
        items.append({
            "file": p.name,
            "title": _figure_title(p.name),
            "size_kb": p.stat().st_size / 1024,
            "desc": FIG_DESC.get(p.name, ""),
        })
    return items


def _esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;"))


def _csv_card_html(item: dict) -> str:
    file = item["file"]
    rows = item["rows"]
    desc = item["desc"]
    title = file.replace(".csv", "")
    rows_str = f"{rows:,}행" if rows >= 0 else "행수 미상"
    desc_html = f'<p class="desc">{_esc(desc)}</p>' if desc else ""
    return (
        '      <div class="card">\n'
        '        <div class="icon">📊</div>\n'
        f'        <h4>{_esc(title)}</h4>\n'
        f'        <div class="meta-line">{rows_str} · {item["size_kb"]:.1f} KB</div>\n'
        f'        {desc_html}\n'
        f'        <a class="download" href="outputs/tables/{_esc(file)}" download>📥 CSV 다운로드</a>\n'
        '      </div>\n'
    )


def _figure_card_html(item: dict) -> str:
    file = item["file"]
    title = item["title"]
    desc = item["desc"]
    desc_html = f'<p class="desc">{_esc(desc)}</p>' if desc else ""
    return (
        '      <div class="card figure-card">\n'
        f'        <a class="thumb" href="outputs/figures/{_esc(file)}" target="_blank" rel="noopener">\n'
        f'          <img src="outputs/figures/{_esc(file)}" alt="{_esc(title)}" loading="lazy">\n'
        '        </a>\n'
        f'        <h4>{_esc(title)}</h4>\n'
        f'        <div class="meta-line">{item["size_kb"]:.1f} KB · 클릭 시 원본</div>\n'
        f'        {desc_html}\n'
        f'        <a class="download" href="outputs/figures/{_esc(file)}" download>📥 PNG 다운로드</a>\n'
        '      </div>\n'
    )


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>대전광역시교육청 통학 지원 운영 방안 연구</title>
<meta name="description" content="GIS 기반 통학환경 종합 분석 플랫폼 — 대전 초·중학교 243교, 도시정비사업 120건, 트램 14개 공구 분석.">
<meta property="og:title" content="대전광역시교육청 통학 지원 운영 방안 연구">
<meta property="og:description" content="GIS 기반 통학환경 종합 분석 플랫폼 — 대전 초·중학교 243교, 도시정비사업 120건 분석.">
<meta property="og:type" content="website">
<!-- 이 파일은 src/build_index.py 로 생성됩니다. 직접 수정하지 마세요. -->
<link rel="preconnect" href="https://cdn.jsdelivr.net">
<link rel="stylesheet" as="style" crossorigin
      href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.min.css">
<style>
  :root {
    --bg: #f6f7fb;
    --surface: #ffffff;
    --surface-hover: #fafbff;
    --border: #e6e8ef;
    --primary: #1d4ed8;
    --primary-light: #eaf0ff;
    --text: #1f2937;
    --muted: #6b7280;
    --accent: #0ea5e9;
    --shadow: 0 1px 2px rgba(15, 23, 42, 0.04), 0 4px 16px rgba(15, 23, 42, 0.06);
    --shadow-hover: 0 4px 8px rgba(15, 23, 42, 0.06), 0 12px 30px rgba(15, 23, 42, 0.10);
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0; padding: 0;
    font-family: "Pretendard Variable", Pretendard, -apple-system, BlinkMacSystemFont,
                 "Apple SD Gothic Neo", "Noto Sans KR", "Segoe UI", Roboto, system-ui, sans-serif;
    color: var(--text);
    background: var(--bg);
    line-height: 1.55;
    -webkit-font-smoothing: antialiased;
  }
  a { color: var(--primary); text-decoration: none; }
  a:hover { text-decoration: underline; }
  .container { max-width: 1100px; margin: 0 auto; padding: 0 24px; }

  header.hero {
    background: linear-gradient(140deg, #1e3a8a 0%, #1d4ed8 60%, #0ea5e9 100%);
    color: white;
    padding: 56px 0 64px;
  }
  header.hero h1 {
    margin: 0 0 12px;
    font-size: clamp(1.6rem, 4.2vw, 2.4rem);
    font-weight: 800;
    letter-spacing: -0.02em;
  }
  header.hero p.subtitle { margin: 0; font-size: 1.02rem; opacity: 0.92; }
  header.hero p.meta {
    margin: 14px 0 0;
    font-size: 0.78rem; opacity: 0.85; line-height: 1.6;
  }
  header.hero .tag {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 999px;
    padding: 4px 12px;
    font-size: 0.78rem;
    margin-bottom: 14px;
    font-weight: 600;
    letter-spacing: 0.02em;
  }

  section { padding: 44px 0; }
  section h2 {
    font-size: 1.35rem; font-weight: 700;
    margin: 0 0 18px;
    display: flex; align-items: center; gap: 8px;
    letter-spacing: -0.01em;
  }
  section h2 .num { color: var(--primary); font-feature-settings: "tnum"; }
  section h2 .count { font-size: 0.85rem; color: var(--muted); font-weight: 500; }

  .stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px;
    margin-bottom: 36px;
  }
  .stat {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 18px;
    box-shadow: var(--shadow);
  }
  .stat .label { font-size: 0.82rem; color: var(--muted); margin-bottom: 4px; }
  .stat .value { font-size: 1.5rem; font-weight: 800; color: var(--primary); letter-spacing: -0.02em; }
  .stat .unit { font-size: 0.85rem; font-weight: 500; color: var(--text); margin-left: 2px; }

  .main-map-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    overflow: hidden;
    box-shadow: var(--shadow);
    transition: box-shadow 0.2s, transform 0.2s;
    margin-bottom: 36px;
  }
  .main-map-card:hover { box-shadow: var(--shadow-hover); transform: translateY(-2px); }
  .main-map-card a { display: block; padding: 28px; color: inherit; text-decoration: none; }
  .main-map-card .icon { font-size: 2rem; margin-bottom: 8px; }
  .main-map-card h3 { margin: 0 0 6px; font-size: 1.2rem; font-weight: 700; color: var(--text); }
  .main-map-card p { margin: 0; color: var(--muted); font-size: 0.95rem; }
  .main-map-card .cta {
    display: inline-block;
    margin-top: 14px; padding: 8px 16px;
    background: var(--primary); color: white;
    border-radius: 8px; font-size: 0.88rem; font-weight: 600;
  }

  .card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 14px;
  }
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px;
    box-shadow: var(--shadow);
    display: flex; flex-direction: column; gap: 6px;
    transition: box-shadow 0.18s, transform 0.18s, border-color 0.18s;
  }
  .card:hover { box-shadow: var(--shadow-hover); transform: translateY(-2px); border-color: #d4dcef; }
  .card .icon { font-size: 1.4rem; }
  .card h4 {
    margin: 0; font-size: 0.98rem; font-weight: 600; color: var(--text);
    word-break: keep-all;
  }
  .card .meta-line { font-size: 0.78rem; color: var(--muted); }
  .card .desc { color: var(--muted); font-size: 0.83rem; flex: 1; margin: 4px 0 0; }
  .card .download {
    display: inline-flex; align-items: center; gap: 4px;
    margin-top: 8px; padding: 6px 12px;
    background: var(--primary-light); color: var(--primary);
    border-radius: 6px; font-size: 0.8rem; font-weight: 600;
    align-self: flex-start;
  }

  /* 정적 도면 카드 — 썸네일 영역 */
  .figure-card { padding: 0; }
  .figure-card .thumb {
    display: block; aspect-ratio: 16 / 9;
    background: #f3f4f8; overflow: hidden;
    border-bottom: 1px solid var(--border);
  }
  .figure-card .thumb img {
    width: 100%; height: 100%; object-fit: contain;
    transition: transform 0.18s;
  }
  .figure-card:hover .thumb img { transform: scale(1.02); }
  .figure-card h4,
  .figure-card .meta-line,
  .figure-card .desc { padding-left: 18px; padding-right: 18px; }
  .figure-card h4 { padding-top: 14px; }
  .figure-card .download { margin-left: 18px; margin-bottom: 18px; }

  .table-category { margin-top: 14px; }
  .table-category:first-child { margin-top: 0; }
  .table-category h3 {
    margin: 0 0 10px;
    font-size: 1.05rem; font-weight: 700;
    color: var(--text);
    display: flex; align-items: center; gap: 8px;
    padding: 6px 0 6px 12px;
    border-left: 4px solid var(--primary);
  }
  .table-category h3 .num { color: var(--primary); font-feature-settings: "tnum"; }
  .table-category h3 .count { font-size: 0.78rem; color: var(--muted); font-weight: 500; }

  footer {
    background: #0f172a;
    color: #cbd5e1;
    padding: 32px 0;
    margin-top: 40px;
    font-size: 0.88rem;
  }
  footer h5 { margin: 0 0 6px; color: white; font-size: 0.98rem; }
  footer p { margin: 4px 0; }
  footer a { color: #93c5fd; }

  @media (max-width: 600px) {
    header.hero { padding: 36px 0 44px; }
    section { padding: 28px 0; }
    .main-map-card a { padding: 20px; }
  }
</style>
</head>
<body>

<header class="hero">
  <div class="container">
    <span class="tag">대전광역시교육청 연구용역</span>
    <h1>대전광역시교육청 통학 지원 운영 방안 연구</h1>
    <p class="subtitle">GIS 기반 통학환경 종합 분석 플랫폼</p>
    <p class="meta">
      📅 데이터 기준일: 2026. 3. 31.<br>
      🔬 현재 진행: 외부환경 분석 (도시개발 + 경사도) — 다른 분석 영역 추가 예정<br>
      🚌 통학차량 14교 = 운영 12교 + 예정 1교(흥도초, 임시배치) + 명목 1교(신탄진용정초, 이용 0명)<br>
      👥 연구진: 국립공주대학교 지리학과 장동호 · 박종철 · 임창민
    </p>
  </div>
</header>

<section>
  <div class="container">

    <div class="stat-grid">
      <div class="stat"><div class="label">분석 대상</div><div class="value">243<span class="unit">교</span></div></div>
      <div class="stat"><div class="label">도시정비사업</div><div class="value">120<span class="unit">건</span></div></div>
      <div class="stat"><div class="label">트램 공구</div><div class="value">14<span class="unit">개</span></div></div>
      <div class="stat"><div class="label">트램 정거장</div><div class="value">45<span class="unit">개</span></div></div>
      <div class="stat"><div class="label">우선순위 도출</div><div class="value">30<span class="unit">교</span></div></div>
    </div>

    <h2><span class="num">🗺️</span> 메인 인터랙티브 맵</h2>
    <div class="main-map-card">
      <a href="outputs/maps/대전_외부환경분석_도시개발.html">
        <div class="icon">🗺️</div>
        <h3>대전 외부환경분석 종합 맵</h3>
        <p>학교 243교 · 정비사업 120건 영향권(1km/1.5km) · 경사도 음영 · 우선순위 점수 · 적격성 분석 — 단일 인터랙티브 맵에서 모두 확인</p>
        <span class="cta">맵 열기 →</span>
      </a>
    </div>

    <h2 style="margin-top: 40px;"><span class="num">🖼️</span> 정적 도면 <span class="count">(총 {n_figures}개)</span></h2>
    <div class="card-grid">
{figure_cards}    </div>

    <h2 style="margin-top: 40px;"><span class="num">📊</span> 분석표 다운로드 <span class="count">(총 {n_csvs}개)</span></h2>
{csv_sections}
  </div>
</section>

<footer>
  <div class="container">
    <h5>대전광역시교육청 통학지원 운영방안 연구</h5>
    <p>국립공주대학교 정책연구용역 · 외부환경분석 (도시개발 + 경사도) 파트</p>
    <p>© 2026 임창민 (Changmin Im) · 국립공주대학교 지리학과</p>
    <p style="margin-top: 12px;">
      <a href="https://github.com/ChangminIm/daejeon-school-research">GitHub Repository</a>
      &nbsp;·&nbsp; 코드 라이선스: MIT
    </p>
  </div>
</footer>

</body>
</html>
"""


def _build_csv_sections_html(csvs):
    """CSV 목록을 카테고리별 그룹핑 → 섹션 HTML 생성.
    카테고리 순서는 TABLE_CATEGORIES 정의 순서. '기타'는 마지막에.
    """
    grouped = defaultdict(list)
    for c in csvs:
        grouped[categorize_csv(c["file"])].append(c)

    parts = []
    for idx, (category, _) in enumerate(TABLE_CATEGORIES):
        items = grouped.get(category, [])
        if not items:
            continue
        emoji = CATEGORY_EMOJI[idx] if idx < len(CATEGORY_EMOJI) else ""
        cards = "".join(_csv_card_html(it) for it in items)
        parts.append(
            f'    <div class="table-category">\n'
            f'      <h3><span class="num">{emoji}</span> {_esc(category)} '
            f'<span class="count">({len(items)}개)</span></h3>\n'
            f'      <div class="card-grid">\n{cards}      </div>\n'
            f'    </div>\n'
        )

    # 기타 카테고리 (있으면 마지막에)
    etc_items = grouped.get(ETC_CATEGORY, [])
    if etc_items:
        cards = "".join(_csv_card_html(it) for it in etc_items)
        parts.append(
            f'    <div class="table-category">\n'
            f'      <h3>📁 {_esc(ETC_CATEGORY)} '
            f'<span class="count">({len(etc_items)}개)</span></h3>\n'
            f'      <div class="card-grid">\n{cards}      </div>\n'
            f'    </div>\n'
        )

    return "".join(parts), grouped


def main():
    print("=" * 70)
    print("index.html 빌드 (outputs/tables, outputs/figures 글로빙)")
    print("=" * 70)

    csvs = _scan_csvs()
    figs = _scan_figures()

    print(f"\n[CSV] {len(csvs)}개")
    for c in csvs:
        flag = " (desc)" if c["desc"] else ""
        print(f"  · {c['file']} — {c['rows']:>5}행, {c['size_kb']:>6.1f} KB{flag}")

    print(f"\n[PNG] {len(figs)}개")
    for f in figs:
        flag = " (desc)" if f["desc"] else ""
        print(f"  · {f['file']} — {f['size_kb']:>6.1f} KB{flag}")

    csv_sections, grouped = _build_csv_sections_html(csvs)

    # 카테고리 분류 결과 요약
    print(f"\n[카테고리 분류]")
    for idx, (category, _) in enumerate(TABLE_CATEGORIES):
        n = len(grouped.get(category, []))
        emoji = CATEGORY_EMOJI[idx] if idx < len(CATEGORY_EMOJI) else " "
        print(f"  {emoji} {category}: {n}개")
        for it in grouped.get(category, []):
            print(f"      - {it['file']}")
    etc_n = len(grouped.get(ETC_CATEGORY, []))
    if etc_n:
        print(f"  📁 기타: {etc_n}개 ⚠️ 매핑 필요")
        for it in grouped.get(ETC_CATEGORY, []):
            print(f"      - {it['file']}")
    else:
        print(f"  ✓ '기타' 카테고리 0개 — 모든 CSV 분류됨")

    fig_html = "".join(_figure_card_html(f) for f in figs)

    # CSS 안의 {} 와 충돌하지 않도록 .replace() 사용
    html = (
        HTML_TEMPLATE
        .replace("{csv_sections}", csv_sections)
        .replace("{figure_cards}", fig_html)
        .replace("{n_csvs}", str(len(csvs)))
        .replace("{n_figures}", str(len(figs)))
    )

    OUT_INDEX.write_text(html, encoding="utf-8")
    print(f"\n[DONE] {OUT_INDEX} ({OUT_INDEX.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
