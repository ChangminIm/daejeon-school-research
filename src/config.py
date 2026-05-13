"""중앙 설정: 경로, 좌표계, 색상, 분석 파라미터"""
import os
from pathlib import Path
from dotenv import load_dotenv

# ===== 경로 =====
ROOT = Path(__file__).parent.parent

# .env 로드 (있을 때만)
load_dotenv(ROOT / ".env")
DATA_RAW = ROOT / "data" / "raw"
DATA_EXTERNAL = ROOT / "data" / "external"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_GEOJSON = ROOT / "data" / "geojson"
OUTPUT_MAPS = ROOT / "outputs" / "maps"
OUTPUT_FIGURES = ROOT / "outputs" / "figures"
OUTPUT_TABLES = ROOT / "outputs" / "tables"

# ===== 좌표계 =====
CRS_WGS84 = "EPSG:4326"     # 저장 표준 (위경도)
CRS_KOREA = "EPSG:5179"     # 분석용 (한국 직각좌표, 미터 단위)

# ===== 대전 중심 좌표 (지도 초기 위치) =====
DAEJEON_CENTER = (36.350, 127.385)
DAEJEON_ZOOM = 12
DAEJEON_CITYHALL = (36.3504, 127.3845)   # 대전시청 (서구 둔산동) — 도심거리 기준점

# 통학지원 적격성 가중치 (3개 지표, 합 = 1.0)
ELIGIBILITY_WEIGHTS = {
    "동급학교밀도": 0.50,   # 반경 2km 내 동급 학교 수가 적을수록 ↑
    "도심거리":     0.30,   # 대전시청에서 멀수록 ↑
    "학교규모":     0.20,   # 학생수 적을수록 ↑
}

# ===== 색상 (구별) =====
GU_COLORS = {
    "대덕구": "#E74C3C",
    "동구":   "#F39C12",
    "중구":   "#27AE60",
    "서구":   "#3498DB",
    "유성구": "#9B59B6",
}

# ===== 분석 파라미터 =====
BUFFER_DISTANCES = [1000, 1500]  # 미터, 영향권 버퍼

# 학생발생률 (경기도교육청 2024 단순화)
# TODO: 평형/유형별 계수로 정교화 필요
ESTIMATED_STUDENT_RATES = {
    "초": 0.18,
    "중": 0.08,
    "고": 0.07,
}

# 통학거리 기준 (제안서 + 이화룡·동재욱 2011)
TONGHAK_THRESHOLDS = {
    "적정": 1000,        # 1km 이내
    "허용": 1500,        # 1.5km 이내
    "지원필요": 1500,    # 1.5km 초과
}

# ===== 통학지원 우선순위 가중치 =====
# 트램 데이터 보류 상태 → 지표5 가중치 0으로 설정.
# 나머지 4개 지표의 원 비율(0.30:0.25:0.20:0.15)을 유지하면서 합=1.0이 되도록 재정규화.
# (각 가중치 / 0.90)
PRIORITY_WEIGHTS = {
    "지표1_임박도가중":    0.3333,  # 1km 영향권 사업의 임박도 가중치 합 (원 0.30 / 0.90)
    "지표2_학생유입":      0.2778,  # 1km 영향권 사업의 예상 학생 발생수 (원 0.25 / 0.90)
    "지표3_수용여석":      0.2222,  # 학생수 백분위 (원 0.20 / 0.90)
    "지표4_외곽성":        0.1667,  # 자치구 중심에서 학교까지 거리 (원 0.15 / 0.90)
    "지표5_트램접근성":    0.0000,  # 트램 데이터 보류 → 0
}
# 검산: 0.3333 + 0.2778 + 0.2222 + 0.1667 + 0.0 = 1.0000

# 통합 우선순위 = 도시개발영향 점수 × w1 + 적격성 점수 × w2 (합 = 1.0)
# 본 연구는 통학버스 운용 정책 입력값 산출이 목표 → 적격성을 메인으로 격상
INTEGRATED_WEIGHTS = {
    "도시개발영향": 0.30,
    "적격성":       0.70,
}

# 도시개발영향 점수 내부 가중치 (PRIORITY_WEIGHTS 지표1·2의 비율 유지)
DEVIMPACT_SUB_WEIGHTS = {
    "지표1_임박도가중":  0.30 / 0.55,   # ≈ 0.545
    "지표2_학생유입":    0.25 / 0.55,   # ≈ 0.455
}

# 적격성 점수 내부 가중치 (PRIORITY_WEIGHTS 지표3·4의 비율 유지, 트램=0)
ELIGIBILITY_SUB_WEIGHTS = {
    "지표3_수용여석":  0.20 / 0.35,    # ≈ 0.571
    "지표4_외곽성":    0.15 / 0.35,    # ≈ 0.429
}

# 임박도별 가중치 (지표1 산출용)
IMMINENCE_WEIGHTS = {
    "1_공사중":    1.00,
    "2_관리처분":  0.80,
    "3_사업시행":  0.50,
    "4_조합설립":  0.20,
    "5_초기":      0.20,
    "6_입안":      0.05,
    "9_미정":      0.00,
}

# ===== 시나리오 =====
SCENARIOS = {
    "단기": {"기간": "2026-2028", "설명": "트램 공사 기간"},
    "중기": {"기간": "2028-2030", "설명": "트램 개통+주요 재개발 입주"},
    "장기": {"기간": "2030+",     "설명": "통학수요 안정화"},
}

# ===== API 키 =====
VWORLD_API_KEY = os.environ.get("VWORLD_API_KEY")
VWORLD_GEOCODE_URL = "https://api.vworld.kr/req/address"

# 폴더 자동 생성
for p in [DATA_PROCESSED, DATA_GEOJSON, OUTPUT_MAPS, OUTPUT_FIGURES, OUTPUT_TABLES]:
    p.mkdir(parents=True, exist_ok=True)
