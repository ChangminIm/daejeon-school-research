# 대전광역시교육청 통학지원 운영방안 연구 - 외부환경분석

> **국립공주대학교 정책연구용역** (외부환경분석 / 도시개발 파트)
> 도시개발(재개발·재건축·트램·신규 인프라)이 학교 통학환경에 미치는 영향을 GIS 공간분석으로 정량화한 연구입니다.

🔗 **라이브 데모**: https://changminim.github.io/daejeon-school-research/

---

## 핵심 분석

| 항목 | 규모 |
|---|---|
| 대전 초·중학교 | **245교** (2026.3.31 기준) |
| 도시정비사업 활성 건수 | **120건** (재개발·재건축·도시환경정비 등) |
| 통학지원 적격성 분석 | 학교별 점수화 (거리·안전·도시개발 영향 종합) |
| 신규 통학지원 검토 후보 | **상위 30교** 도출 |
| 도시철도 2호선(트램) | 14개 공구 / 45개 정거장 |

### 분석 차원
1. **개발사업지-학교 위치 공간 중첩** (1km / 1.5km 버퍼)
2. **학생발생률 추정** (경기도교육청 2024 방법론 기반)
3. **트램 공사 영향권** (단기: 2026-28 공사기 / 중기: 2028-30 개통+입주 / 장기: 2030+)
4. **통학버스 운영 시나리오** (현행 14교 → 미래 확장 검토)
5. **시나리오별 통학지원 우선순위 도출**

---

## 빠른 시작

### 1. 환경 세팅

```bash
git clone https://github.com/ChangminIm/daejeon-school-research.git
cd daejeon-school-research

# 가상환경 권장 (Python 3.11+)
python -m venv .venv
.venv\Scripts\activate         # Windows
# source .venv/bin/activate    # macOS/Linux

pip install -r requirements.txt
```

### 2. API 키 설정

```bash
# .env.example을 복사하여 .env로 만들고 본인 VWorld 인증키 입력
copy .env.example .env         # Windows
# cp .env.example .env         # macOS/Linux
```

`.env` 안에 VWorld(국토교통부 공간정보 오픈플랫폼) 인증키를 넣으세요:
```
VWORLD_API_KEY=여기에_본인_VWorld_인증키_입력
```
신청: https://www.vworld.kr/dev/v4dv_2ddataguide2_s001.do

### 3. 데이터 준비

`data/raw/`에 교육청 송부 자료(.xlsx) 배치, `data/external/`에 다음을 다운로드:

| 데이터 | 출처 |
|---|---|
| 전국 초중등학교 위치 표준데이터 | [공공데이터포털 #15021148](https://www.data.go.kr/data/15021148/standard.do) |
| 대전 행정동/시군구 경계 SHP | [SGIS 통계지리정보](https://sgis.kostat.go.kr) |
| 대전 도시정비사업 활성 목록 | [대전 정비사업관리시스템](https://hreas.daejeon.go.kr) |
| 대전 통학차량 운용 현황 | 대전광역시교육청 행정과 |
| 도시철도 2호선(트램) 노선·정거장 | 대전광역시 도시철도건설국 |

자세한 데이터 출처는 [docs/data_sources.md](docs/data_sources.md) 참조.

### 4. 실행

```bash
python run.py
```

→ `outputs/maps/` 에 인터랙티브 HTML, `outputs/tables/` 에 분석표, `outputs/figures/` 에 정적 PNG 생성.

---

## 폴더 구조

```
daejeon-school-research/
├── README.md                  # 이 파일
├── LICENSE                    # MIT
├── CLAUDE.md                  # Claude Code 프로젝트 컨텍스트
├── requirements.txt           # 파이썬 의존성
├── .env.example               # 환경변수 템플릿 (본인 키로 .env 작성)
├── run.py                     # 메인 파이프라인 실행 스크립트
├── index.html                 # GitHub Pages 랜딩 페이지
│
├── src/                       # 분석 모듈
│   ├── config.py              # 경로·색상·CRS 중앙관리
│   ├── parse_schools.py       # 교육청 송부자료 파싱
│   ├── geocode.py             # 학교 좌표 부여 (VWorld + 표준데이터)
│   ├── geocode_redev.py       # 정비사업 좌표 부여
│   ├── tram_data.py           # 트램 노선/정거장 데이터
│   ├── analysis.py            # 개발사업 영향권 분석
│   ├── analysis_tram.py       # 트램 공사영향권 분석
│   ├── priority_analysis.py   # 통학지원 우선순위 산정
│   ├── tonghak_eligibility.py # 통학지원 적격성 산정
│   ├── integrated_priority.py # 종합 우선순위 통합
│   ├── road_routing.py        # OSMnx 보행 네트워크 분석
│   ├── popup_builder.py       # Folium 팝업 HTML 빌더
│   ├── viz_map.py             # Folium 인터랙티브 맵 생성
│   └── coords_data.py         # (legacy) 동 단위 데모 좌표
│
├── data/
│   ├── raw/                   # 교육청 송부 원본 (gitignore)
│   ├── external/              # 외부 공개 데이터
│   │   ├── admin/             # 행정구역 SHP
│   │   ├── schools/           # 학교위치 표준데이터
│   │   ├── develop/           # 도시정비사업 120건
│   │   ├── bus/               # 통학차량 운용현황
│   │   └── tram/              # 트램 노선 (대용량 원본 gitignore)
│   ├── processed/             # 전처리 결과 CSV
│   └── geojson/               # GeoJSON 변환본
│
├── outputs/                   # ★ GitHub Pages로 호스팅
│   ├── maps/                  # 인터랙티브 Folium HTML
│   ├── tables/                # 분석 결과 CSV
│   └── figures/               # 정적 PNG (보고서용)
│
└── docs/
    └── data_sources.md        # 데이터 출처 상세
```

---

## 주요 산출물

### 인터랙티브 맵
- `outputs/maps/대전_외부환경분석_도시개발.html` — 메인 종합 맵 (학교 245교 + 정비사업 120건 + 영향권 + 트램)

### 분석표 (`outputs/tables/`)
| 파일 | 내용 |
|---|---|
| `통학지원_우선순위_학교별.csv` | 245교 우선순위 점수 |
| `통학지원_우선순위_상위30교.csv` | 상위 30교 추출 |
| `통학지원적격성_학교별.csv` | 적격성 점수 (거리·안전·환경) |
| `통학지원적격성_상위30교.csv` | 적격성 상위 30교 |
| `신규검토대상_상위30교.csv` | 신규 통학지원 후보 30교 |
| `도시개발영향_상위30교.csv` | 도시개발 영향 큰 학교 |
| `재개발영향권_요약.csv` | 정비사업별 영향권 학교 수 |
| `재개발영향권_요약_임박도별.csv` | 사업 진행단계별 집계 |
| `트램_공사영향권_상세.csv` | 트램 14개 공구 영향 학교 |
| `트램_정거장_접근성.csv` | 45개 정거장 도보 접근성 |
| `통학버스운용_미래시나리오.csv` | 시나리오별 통학버스 운용 |

### 정적 도면
- `outputs/figures/우선순위_히트맵.png` — 우선순위 히트맵 (300dpi)

---

## 좌표계

| 용도 | EPSG |
|---|---|
| 저장 (WGS84 위경도) | **4326** |
| 거리·면적 분석 (한국 직각좌표) | **5179** |

모든 `GeoDataFrame.crs`를 명시 후 작업. `src/config.py` 에서 일괄 관리.

---

## 데이터 출처

| 카테고리 | 출처 | 라이선스 |
|---|---|---|
| 학교 위치·학생수 | 대전광역시교육청 송부자료 (2026.3.31) | 연구용 |
| 학교 위치 좌표 보정 | [공공데이터포털 학교위치 표준데이터 #15021148](https://www.data.go.kr/data/15021148/standard.do) | 공공저작물 |
| 지오코딩 | [VWorld 지오코더 API](https://www.vworld.kr) | 인증키 필요 |
| 행정구역 경계 | [SGIS 통계지리정보](https://sgis.kostat.go.kr) | 공공저작물 |
| 도시정비사업 활성 목록 | [대전광역시 정비사업관리시스템](https://hreas.daejeon.go.kr) | 공공저작물 |
| 통학차량 운용현황 | 대전광역시교육청 (2026) | 연구용 |
| 도시철도 2호선 | 대전광역시 도시철도건설국 | 공개자료 |
| 보행 네트워크 | [OpenStreetMap](https://www.openstreetmap.org) via osmnx | ODbL |

---

## 참고 문헌

- 경기도교육청 (2024). 3기 신도시 지역 적정 학생배치를 위한 학생발생률 연구
- 김지호 외 (2025). I2SFCA 기반 통학구역 접근성 분석
- 이화룡·동재욱 (2011). 도시개발지역 내 학교 적정배치

---

## 저자

**임창민 (Changmin Im)**
국립공주대학교 지리학과
연구진: 장동호·박종철·임창민 (국립공주대학교 지리학과)

## 라이선스

[MIT License](LICENSE) — 코드는 자유롭게 사용 가능합니다.
데이터(특히 `data/raw/`)는 별도 라이선스의 적용을 받습니다.
