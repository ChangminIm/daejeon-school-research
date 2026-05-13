"""대전 도시정비사업 120건 VWorld 지오코딩

전체주소 컬럼 → parcel(지번) 우선, road(도로명) 폴백.
결과: data/processed/redev_projects_geocoded.csv
실패: data/processed/redev_unmatched.csv

사용:
    python -m src.geocode_redev          # 기존 결과 있으면 스킵
    python -m src.geocode_redev --force  # 강제 재실행
"""
import sys
import time
import argparse
import requests
import pandas as pd
from tqdm import tqdm
from src.config import (
    DATA_EXTERNAL, DATA_PROCESSED,
    VWORLD_API_KEY, VWORLD_GEOCODE_URL,
)

INPUT_CSV = DATA_EXTERNAL / "develop" / "대전_도시정비사업_활성_120건.csv"
OUTPUT_CSV = DATA_PROCESSED / "redev_projects_geocoded.csv"
UNMATCHED_CSV = DATA_PROCESSED / "redev_unmatched.csv"

REQUEST_SLEEP_SEC = 0.05    # 호출 간 대기
MAX_RETRIES = 3              # 네트워크 에러 재시도


def vworld_geocode(address, addr_type="parcel"):
    """VWorld Geocoder API 한 건 호출.

    Args:
        address: 주소 문자열
        addr_type: "parcel"(지번) 또는 "road"(도로명)

    Returns:
        (lat, lon, status) — 실패 시 (None, None, error_msg)
    """
    params = {
        "service": "address",
        "request": "getCoord",
        "version": "2.0",
        "crs": "epsg:4326",
        "address": address,
        "type": addr_type,
        "format": "json",
        "key": VWORLD_API_KEY,
    }
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(VWORLD_GEOCODE_URL, params=params, timeout=10)
            r.raise_for_status()
            data = r.json().get("response", {})
            status = data.get("status")
            if status == "OK":
                pt = data.get("result", {}).get("point", {})
                lon = float(pt["x"]); lat = float(pt["y"])
                return lat, lon, "OK"
            else:
                # NOT_FOUND, ERROR 등은 재시도 무의미
                return None, None, status or "NO_STATUS"
        except (requests.RequestException, ValueError, KeyError) as e:
            last_err = str(e)
            time.sleep(0.5 * (2 ** attempt))   # 지수 백오프
    return None, None, f"NET_FAIL:{last_err}"


def geocode_one(address):
    """parcel 먼저 시도, 실패 시 road 폴백.

    Returns:
        dict {lat, lon, 지오코딩방식, 오류}
    """
    # 1) 지번
    lat, lon, status = vworld_geocode(address, addr_type="parcel")
    if lat is not None:
        return {"lat": lat, "lon": lon, "지오코딩방식": "parcel", "오류": ""}

    time.sleep(REQUEST_SLEEP_SEC)

    # 2) 도로명 폴백
    lat, lon, status_road = vworld_geocode(address, addr_type="road")
    if lat is not None:
        return {"lat": lat, "lon": lon, "지오코딩방식": "road", "오류": ""}

    return {"lat": None, "lon": None, "지오코딩방식": "실패",
            "오류": f"parcel={status}, road={status_road}"}


def run(force=False):
    if not VWORLD_API_KEY:
        print("❌ VWORLD_API_KEY 미설정. .env 파일 확인")
        sys.exit(1)

    if OUTPUT_CSV.exists() and not force:
        df = pd.read_csv(OUTPUT_CSV)
        print(f"⏭️  스킵: 기존 결과 사용 ({OUTPUT_CSV.name}, {len(df)}건)")
        print(f"   재실행하려면 --force")
        return df

    if not INPUT_CSV.exists():
        print(f"❌ 입력 파일 없음: {INPUT_CSV}")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    print(f"📥 입력: {len(df)}건 / {INPUT_CSV.name}")
    print(f"🌐 VWorld 지오코딩 시작...")

    results = []
    for addr in tqdm(df["전체주소"].astype(str).tolist(), desc="geocoding"):
        results.append(geocode_one(addr))
        time.sleep(REQUEST_SLEEP_SEC)

    geo_df = pd.DataFrame(results)
    out = pd.concat([df.reset_index(drop=True), geo_df], axis=1)

    # 통계
    n_total = len(out)
    n_ok = (out["지오코딩방식"] != "실패").sum()
    n_parcel = (out["지오코딩방식"] == "parcel").sum()
    n_road = (out["지오코딩방식"] == "road").sum()
    n_fail = (out["지오코딩방식"] == "실패").sum()

    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    unmatched = out[out["지오코딩방식"] == "실패"]
    unmatched.to_csv(UNMATCHED_CSV, index=False, encoding="utf-8-sig")

    print()
    print(f"✅ 결과 저장: {OUTPUT_CSV}")
    print(f"   전체: {n_total} / 성공: {n_ok} ({n_ok/n_total*100:.1f}%)")
    print(f"   - parcel(지번) 성공: {n_parcel}")
    print(f"   - road(도로명) 성공: {n_road}")
    print(f"   - 실패: {n_fail}")
    if n_fail > 0:
        print(f"\n⚠️  실패 {n_fail}건 → {UNMATCHED_CSV}")
        print(unmatched[["연번", "구", "구역명", "전체주소", "오류"]].to_string(index=False))

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="기존 결과 무시하고 재실행")
    args = parser.parse_args()
    run(force=args.force)
