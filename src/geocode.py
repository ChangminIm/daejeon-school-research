"""학교 좌표 부여

세 가지 모드:
1. shp:      사용자 제공 SHP (data/external/schools/, 권장)
2. official: 공공데이터포털 학교위치표준데이터 CSV
3. demo:     동 좌표 사전 + jitter (외부 데이터 없이)
"""
import pandas as pd
import numpy as np
import re
import geopandas as gpd
from src.config import DATA_PROCESSED, DATA_EXTERNAL
from src.coords_data import DONG_COORDS, GU_CENTERS


def extract_dong(addr):
    if not isinstance(addr, str):
        return None
    m = re.search(r'\(\s*([가-힣]+동)', addr)
    if m:
        return m.group(1)
    m = re.search(r'(?:대덕구|동구|중구|서구|유성구)\s+([가-힣]+동)', addr)
    if m:
        return m.group(1)
    return None


def geocode_demo(schools_df, seed=42):
    """데모용: 동 좌표 + jitter로 근사 좌표 생성"""
    np.random.seed(seed)
    df = schools_df.copy()
    df["동"] = df["주소"].apply(extract_dong)

    def base_coord(row):
        dong = row["동"]
        if pd.notna(dong) and dong in DONG_COORDS:
            return (*DONG_COORDS[dong], "동")
        if row["구"] in GU_CENTERS:
            return (*GU_CENTERS[row["구"]], "구")
        return (None, None, "없음")

    coords = df.apply(base_coord, axis=1, result_type="expand")
    df["lat_base"], df["lon_base"], df["좌표출처"] = coords[0], coords[1], coords[2]
    df["lat"], df["lon"] = df["lat_base"].copy(), df["lon_base"].copy()

    # 같은 동 내 jitter
    for (gu, dong), idx in df.groupby(["구", "동"]).groups.items():
        n = len(idx)
        if n > 1:
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            r = 0.003  # ~300m
            df.loc[idx, "lat"] = df.loc[idx, "lat_base"].values + r*np.sin(angles) + np.random.normal(0, 0.0008, n)
            df.loc[idx, "lon"] = df.loc[idx, "lon_base"].values + r*np.cos(angles) + np.random.normal(0, 0.0008, n)

    return df


def geocode_official(schools_df, official_csv_path=None):
    """공공데이터포털 학교위치표준데이터로 정확 매칭

    학교위치표준데이터 컬럼 예시 (실제 다운로드 후 확인 필요):
        학교명, 학교급구분, 시도교육청명, 소재지지번주소, 위도, 경도, ...

    Args:
        schools_df: parse_schools로 만든 DF
        official_csv_path: 공공데이터포털 CSV 경로
    """
    if official_csv_path is None:
        official_csv_path = DATA_EXTERNAL / "전국초중등학교위치표준데이터.csv"

    if not official_csv_path.exists():
        raise FileNotFoundError(
            f"학교위치표준데이터가 없습니다: {official_csv_path}\n"
            "공공데이터포털에서 다운로드: "
            "https://www.data.go.kr/data/15021148/standard.do"
        )

    # 인코딩 자동 시도
    for enc in ["utf-8-sig", "cp949", "euc-kr"]:
        try:
            official = pd.read_csv(official_csv_path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue

    # 대전 학교만 필터
    gyo_col = next((c for c in official.columns if "교육청" in c), None)
    if gyo_col:
        official = official[official[gyo_col].str.contains("대전", na=False)]

    # 학교명 컬럼 자동 탐지
    name_col = next((c for c in official.columns if c == "학교명"), None)
    lat_col = next((c for c in official.columns if c in ["위도", "latitude", "lat"]), None)
    lon_col = next((c for c in official.columns if c in ["경도", "longitude", "lon", "lng"]), None)

    if not all([name_col, lat_col, lon_col]):
        raise ValueError(f"필수 컬럼을 찾을 수 없음. 확인: {official.columns.tolist()}")

    # 학교명 정규화 후 join
    def normalize(name):
        return re.sub(r'\s+', '', str(name)).replace("학교", "").strip()

    schools_df = schools_df.copy()
    schools_df["_key"] = schools_df["학교명"].apply(normalize)
    official["_key"] = official[name_col].apply(normalize)

    merged = schools_df.merge(
        official[["_key", lat_col, lon_col]].rename(columns={lat_col: "lat", lon_col: "lon"}),
        on="_key", how="left"
    ).drop(columns=["_key"])

    unmatched = merged[merged["lat"].isna()]
    if len(unmatched) > 0:
        print(f"⚠️  매칭 실패 {len(unmatched)}교 — 수동 확인 필요")
        print(unmatched[["학교급", "학교명", "구"]].to_string())

    return merged


def geocode_from_shp(schools_df, shp_dir=None):
    """사용자 제공 SHP (초·중)으로 학교명 매칭.

    data/external/schools/
        초등학교_위치정보_WGS.shp  (.dbf cp949)
        중학교_위치정보_WGS.shp
    좌표계: WGS84 (X=경도, Y=위도) — .prj 없어도 좌표값이 위경도임
    """
    if shp_dir is None:
        shp_dir = DATA_EXTERNAL / "schools"

    paths = {
        "초": shp_dir / "초등학교_위치정보_WGS.shp",
        "중": shp_dir / "중학교_위치정보_WGS.shp",
    }
    missing = [p for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"SHP 누락: {missing}")

    frames = []
    for level, p in paths.items():
        gdf = gpd.read_file(p, encoding="cp949")
        # .prj 없어 CRS=None이지만 좌표값은 WGS84
        gdf = gdf.rename(columns={"X": "lon_shp", "Y": "lat_shp"})
        gdf["_level"] = level
        frames.append(gdf[["학교명", "lon_shp", "lat_shp", "_level",
                           "표준신주소", "행정동명"]])
    shp_all = pd.concat(frames, ignore_index=True)

    # 학교명 정규화 (공백 제거)
    def normalize(name):
        return re.sub(r"\s+", "", str(name))

    shp_all["_key"] = shp_all["학교명"].apply(normalize)
    schools_df = schools_df.copy()
    schools_df["_key"] = schools_df["학교명"].apply(normalize)

    merged = schools_df.merge(
        shp_all[["_key", "lon_shp", "lat_shp", "행정동명"]],
        on="_key", how="left"
    ).drop(columns=["_key"])

    merged = merged.rename(columns={
        "lon_shp": "lon", "lat_shp": "lat", "행정동명": "동",
    })

    unmatched = merged[merged["lat"].isna()]
    if len(unmatched) > 0:
        print(f"⚠️  매칭 실패 {len(unmatched)}교:")
        print(unmatched[["학교급", "구", "학교명"]].to_string(index=False))

    return merged


if __name__ == "__main__":
    schools = pd.read_csv(DATA_PROCESSED / "schools.csv")

    # 우선순위: schools/ SHP > 공공데이터포털 CSV > demo
    shp_dir = DATA_EXTERNAL / "schools"
    official_path = DATA_EXTERNAL / "전국초중등학교위치표준데이터.csv"

    if (shp_dir / "초등학교_위치정보_WGS.shp").exists():
        print("📍 shp 모드: 사용자 제공 SHP 사용")
        result = geocode_from_shp(schools, shp_dir)
    elif official_path.exists():
        print("📍 official 모드: 공공데이터포털 학교위치표준데이터 사용")
        result = geocode_official(schools, official_path)
    else:
        print("📍 demo 모드: 동 좌표 + jitter")
        result = geocode_demo(schools)

    out = DATA_PROCESSED / "schools_geocoded.csv"
    result.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"✅ 좌표 부여 완료 → {out}")
    print(f"   성공: {result['lat'].notna().sum()}/{len(result)}")
