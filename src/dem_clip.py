"""대전 경계로 원본 Slope/DEM 래스터(EPSG:5186) 자르기.

원본:
  D:/07_데이터/2019_Slope_5m/Slope_epsg5186.tif  (~17GB)
  D:/07_데이터/2019_Slope_5m/DEM_epsg5186.tif    (~16GB)

마스크: D:/04_제안서/03_기타/대전광역시/shp/daejeon_signungu.shp (EPSG:5179)
출력:   data/external/dem/대전_slope_5m.tif, 대전_dem_5m.tif
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.enums import Resampling

from src.config import DATA_EXTERNAL, OUTPUT_FIGURES

# ===== 입력/출력 경로 =====
SRC_SLOPE = Path(r"D:/07_데이터/2019_Slope_5m/Slope_epsg5186.tif")
SRC_DEM   = Path(r"D:/07_데이터/2019_Slope_5m/DEM_epsg5186.tif")
MASK_SHP  = Path(r"D:/04_제안서/03_기타/대전광역시/shp/daejeon_signungu.shp")

OUT_DIR = DATA_EXTERNAL / "dem"
OUT_SLOPE = OUT_DIR / "대전_slope_5m.tif"
OUT_DEM   = OUT_DIR / "대전_dem_5m.tif"

# 원본은 EPSG:5186 (KGD2002 중부원점 TM)
RASTER_CRS = "EPSG:5186"
NODATA = -9999.0


def load_mask_geoms(shp_path: Path, target_crs: str):
    """SHP를 읽어 target_crs로 변환 후 geometry 리스트 반환."""
    gdf = gpd.read_file(shp_path)
    print(f"  [mask] SHP 좌표계: {gdf.crs}")
    print(f"  [mask] feature 수: {len(gdf)}")

    if gdf.crs is None:
        raise ValueError("SHP에 좌표계 정보 없음")

    if gdf.crs.to_string() != target_crs:
        print(f"  [mask] {gdf.crs.to_string()} → {target_crs} 변환")
        gdf = gdf.to_crs(target_crs)

    return [geom.__geo_interface__ for geom in gdf.geometry if geom is not None]


def clip_raster_to_daejeon(
    raster_path: Path,
    output_path: Path,
    mask_geoms: list,
    label: str = "",
) -> dict:
    """rasterio.mask로 라스터 클립.

    - all_touched=True: 경계 픽셀 포함
    - crop=True: bounding box 자르기 (메모리 절약)
    - nodata=-9999: 대전 밖
    - LZW 압축
    """
    print(f"\n=== {label} 자르기 시작 ===")
    print(f"  입력: {raster_path}")
    print(f"  출력: {output_path}")

    in_size = raster_path.stat().st_size
    print(f"  입력 파일 크기: {in_size / 1024**3:.2f} GB")

    with rasterio.open(raster_path) as src:
        print(f"  [입력] CRS: {src.crs}")
        print(f"  [입력] shape: {src.width} x {src.height}")
        print(f"  [입력] dtype: {src.dtypes[0]}, nodata: {src.nodata}")
        print(f"  [입력] 픽셀 크기: {src.res}")

        src_epsg = src.crs.to_epsg() if src.crs else None
        expected_epsg = int(RASTER_CRS.split(":")[1])
        if src_epsg != expected_epsg:
            print(f"  WARN: raster EPSG({src_epsg}) != expected({expected_epsg})")

        out_image, out_transform = rio_mask(
            src,
            mask_geoms,
            all_touched=True,
            crop=True,
            nodata=NODATA,
            filled=True,
        )

        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "nodata": NODATA,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "predictor": 3 if np.issubdtype(out_image.dtype, np.floating) else 2,
        })

        with rasterio.open(output_path, "w", **out_meta) as dst:
            dst.write(out_image)

    out_size = output_path.stat().st_size

    with rasterio.open(output_path) as dst:
        arr = dst.read(1, masked=True)
        valid = arr.compressed()
        stats = {
            "min": float(valid.min()) if valid.size else float("nan"),
            "max": float(valid.max()) if valid.size else float("nan"),
            "mean": float(valid.mean()) if valid.size else float("nan"),
            "valid_pixels": int(valid.size),
            "total_pixels": int(arr.size),
        }
        print(f"  [출력] CRS: {dst.crs}")
        print(f"  [출력] shape: {dst.width} x {dst.height} (= {dst.width * dst.height:,} px)")
        print(f"  [출력] 유효 픽셀: {stats['valid_pixels']:,} / {stats['total_pixels']:,} "
              f"({stats['valid_pixels']/stats['total_pixels']*100:.1f}%)")
        print(f"  [출력] nodata: {dst.nodata}")
        print(f"  [출력] 압축: {dst.profile.get('compress')}")
        print(f"  [통계] min={stats['min']:.3f}, max={stats['max']:.3f}, mean={stats['mean']:.3f}")
        print(f"  [파일] {in_size / 1024**3:.2f} GB → {out_size / 1024**2:.2f} MB "
              f"(압축률 {in_size / out_size:.0f}배)")

    return {
        "in_size_gb": in_size / 1024**3,
        "out_size_mb": out_size / 1024**2,
        "width": out_meta["width"],
        "height": out_meta["height"],
        **stats,
    }


def make_preview(slope_path: Path, dem_path: Path, out_png: Path, max_dim: int = 800):
    """자른 두 래스터를 다운샘플링해서 2패널 PNG로 저장 (모양 확인용)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams["font.family"] = "Malgun Gothic"
    rcParams["axes.unicode_minus"] = False

    def _read_downsampled(path: Path):
        with rasterio.open(path) as src:
            scale = max(src.width, src.height) / max_dim
            scale = max(scale, 1.0)
            out_w = int(src.width / scale)
            out_h = int(src.height / scale)
            arr = src.read(
                1,
                out_shape=(out_h, out_w),
                resampling=Resampling.average,
                masked=True,
            )
            bounds = src.bounds
            return arr, bounds

    dem_arr, dem_bounds = _read_downsampled(dem_path)
    slope_arr, slope_bounds = _read_downsampled(slope_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    im0 = axes[0].imshow(
        dem_arr,
        cmap="terrain",
        extent=(dem_bounds.left, dem_bounds.right, dem_bounds.bottom, dem_bounds.top),
    )
    axes[0].set_title("DEM (표고, m)")
    axes[0].set_aspect("equal")
    plt.colorbar(im0, ax=axes[0], fraction=0.04, label="m")

    im1 = axes[1].imshow(
        slope_arr,
        cmap="YlOrRd",
        extent=(slope_bounds.left, slope_bounds.right, slope_bounds.bottom, slope_bounds.top),
    )
    axes[1].set_title("Slope (경사, °)")
    axes[1].set_aspect("equal")
    plt.colorbar(im1, ax=axes[1], fraction=0.04, label="degree")

    fig.suptitle("대전 클립 결과 미리보기 (EPSG:5186)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[미리보기] 저장: {out_png}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("대전 Slope/DEM 클립 (EPSG:5186)")
    print("=" * 70)

    if not SRC_SLOPE.exists():
        raise FileNotFoundError(SRC_SLOPE)
    if not SRC_DEM.exists():
        raise FileNotFoundError(SRC_DEM)
    if not MASK_SHP.exists():
        raise FileNotFoundError(MASK_SHP)

    print("\n[1] 마스크 SHP 로드 + 좌표계 정합")
    mask_geoms = load_mask_geoms(MASK_SHP, RASTER_CRS)
    print(f"  → 마스크 geometry {len(mask_geoms)}개 준비")

    print("\n[2] Slope 자르기")
    slope_stats = clip_raster_to_daejeon(SRC_SLOPE, OUT_SLOPE, mask_geoms, "Slope")

    print("\n[3] DEM 자르기")
    dem_stats = clip_raster_to_daejeon(SRC_DEM, OUT_DEM, mask_geoms, "DEM")

    print("\n[4] 단위/범위 점검")
    print(f"  Slope mean = {slope_stats['mean']:.2f}")
    if 0 <= slope_stats["mean"] <= 30:
        print("    → 도(degree) 단위로 추정 (대전 평균 5~15도 기대)")
    elif 30 < slope_stats["mean"] <= 200:
        print("    → 백분율(percent) 단위 가능성")
    else:
        print("    → 단위 확인 필요")

    print(f"  DEM range = {dem_stats['min']:.1f} ~ {dem_stats['max']:.1f} m "
          f"(평지~산지: 약 30~600m 기대)")

    print("\n[5] 미리보기 PNG 생성")
    preview_png = OUTPUT_FIGURES / "dem_clip_preview.png"
    make_preview(OUT_SLOPE, OUT_DEM, preview_png)

    print("\n" + "=" * 70)
    print("[DONE]")
    print(f"  Slope: {OUT_SLOPE}  ({slope_stats['out_size_mb']:.1f} MB)")
    print(f"  DEM  : {OUT_DEM}  ({dem_stats['out_size_mb']:.1f} MB)")
    print(f"  Preview: {preview_png}")
    print("=" * 70)


if __name__ == "__main__":
    main()
