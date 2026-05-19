"""대전 slope 래스터 → Folium ImageOverlay용 RGBA PNG + bounds JSON.

흐름:
  1) 5m 5186 → 50m 5186 (factor=10, average resampling)
  2) 5186 → 4326 reproject (bilinear)
  3) 색상 매핑 (녹색→갈색→빨강) + nodata 투명
  4) RGBA PNG 저장 + bounds(4326) JSON 저장
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import rasterio
from rasterio.enums import Resampling as RioResampling
from rasterio.warp import calculate_default_transform, reproject
from rasterio.transform import array_bounds
import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import LinearSegmentedColormap, Normalize
from PIL import Image

from src.config import DATA_EXTERNAL, DATA_PROCESSED

SRC_TIF = DATA_EXTERNAL / "dem" / "대전_slope_5m.tif"
OUT_PNG = DATA_PROCESSED / "대전_slope_overlay.png"
OUT_BOUNDS = DATA_PROCESSED / "대전_slope_overlay_bounds.json"

DOWNSAMPLE_FACTOR = 10  # 5m → 50m
ALPHA_DATA = 0.55
VMAX = 30.0  # 30°+에서 채도 포화

COLOR_STOPS = [
    (0.00, "#2D8B43"),  # 평지 (0°)
    (0.15, "#91C266"),  # 약 (~5°)
    (0.40, "#D4B36A"),  # 중 (~12°)
    (0.65, "#B85C2A"),  # 강 (~20°)
    (1.00, "#8B1A1A"),  # 매우 가파름 (30°+)
]


def main():
    if not SRC_TIF.exists():
        raise FileNotFoundError(SRC_TIF)

    print("=" * 70)
    print("대전 경사도 ImageOverlay PNG 생성")
    print("=" * 70)

    # [1] 다운샘플 (5186 좌표계 유지)
    print(f"\n[1] 다운샘플 (factor={DOWNSAMPLE_FACTOR}, 5m → 50m)")
    with rasterio.open(SRC_TIF) as src:
        src_nodata = src.nodata
        src_crs = src.crs
        out_w = src.width // DOWNSAMPLE_FACTOR
        out_h = src.height // DOWNSAMPLE_FACTOR
        downsampled = src.read(
            1,
            out_shape=(out_h, out_w),
            resampling=RioResampling.average,
            masked=False,
        )
        # 다운샘플 후 transform 갱신
        ds_transform = src.transform * src.transform.scale(
            src.width / out_w, src.height / out_h
        )
    print(f"    {src.width}x{src.height} → {out_w}x{out_h}")
    print(f"    src CRS: {src_crs.to_string()}")

    # [2] EPSG:4326 reproject
    print(f"\n[2] EPSG:5186 → EPSG:4326 reproject (bilinear)")
    dst_crs = "EPSG:4326"
    ds_left, ds_bottom, ds_right, ds_top = array_bounds(out_h, out_w, ds_transform)
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, dst_crs, out_w, out_h,
        left=ds_left, bottom=ds_bottom, right=ds_right, top=ds_top,
    )
    dst_arr = np.full((dst_height, dst_width), src_nodata, dtype=np.float32)
    reproject(
        source=downsampled,
        destination=dst_arr,
        src_transform=ds_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=RioResampling.bilinear,
        src_nodata=src_nodata,
        dst_nodata=src_nodata,
    )
    print(f"    → {dst_width}x{dst_height} @ EPSG:4326")

    # bounds (4326)
    bounds_left = float(dst_transform[2])
    bounds_top = float(dst_transform[5])
    bounds_right = float(bounds_left + dst_transform[0] * dst_width)
    bounds_bottom = float(bounds_top + dst_transform[4] * dst_height)

    # [3] 색상 매핑
    print(f"\n[3] 색상 매핑 (녹색→갈색→빨강, vmin=0, vmax={VMAX})")
    cmap = LinearSegmentedColormap.from_list("terrain_slope", COLOR_STOPS)
    norm = Normalize(vmin=0, vmax=VMAX, clip=True)

    valid = np.isfinite(dst_arr) & (dst_arr != src_nodata)
    arr_for_cmap = np.where(valid, dst_arr, 0)  # nodata는 0으로 채움 (어차피 alpha=0)
    rgba = cmap(norm(arr_for_cmap))  # (H, W, 4) float 0~1
    rgba[..., 3] = np.where(valid, ALPHA_DATA, 0.0)

    # [4] PNG 저장
    print(f"\n[4] RGBA PNG 저장")
    rgba_u8 = (rgba * 255).astype(np.uint8)
    Image.fromarray(rgba_u8, mode="RGBA").save(OUT_PNG, optimize=True)

    # [5] bounds JSON
    bounds = {
        "south": bounds_bottom,
        "west": bounds_left,
        "north": bounds_top,
        "east": bounds_right,
    }
    OUT_BOUNDS.write_text(json.dumps(bounds, indent=2), encoding="utf-8")

    # 보고
    png_size = OUT_PNG.stat().st_size
    valid_count = int(valid.sum())
    print("\n" + "=" * 70)
    print("[DONE]")
    print(f"  PNG     : {OUT_PNG}")
    print(f"  크기    : {png_size/1024:.1f} KB ({png_size/1024**2:.2f} MB)")
    print(f"  shape   : {dst_width} x {dst_height}")
    print(f"  유효 px : {valid_count:,} ({valid_count/(dst_height*dst_width)*100:.1f}%)")
    print(f"  bounds  : S={bounds_bottom:.4f}, W={bounds_left:.4f}, "
          f"N={bounds_top:.4f}, E={bounds_right:.4f}")
    print(f"  JSON    : {OUT_BOUNDS}")
    print("=" * 70)


if __name__ == "__main__":
    main()
