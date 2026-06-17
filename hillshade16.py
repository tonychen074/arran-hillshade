"""
16-direction hillshade renderer for DTM tiles.
Handles nodata, computes multi-directional hillshade, applies percentile normalization.
Output: uint8 GeoTIFF (same CRS/transform as input).
"""

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path
import sys


def compute_slope_aspect(dem: np.ndarray, cellsize: float):
    """Horn's formula for slope (radians) and geographic aspect (radians, CW from North)."""
    p = np.pad(dem, 1, mode="edge")

    a = p[:-2, :-2]; b = p[:-2, 1:-1]; c = p[:-2, 2:]
    d = p[1:-1, :-2];                   f = p[1:-1, 2:]
    g = p[2:,  :-2]; h = p[2:,  1:-1]; i = p[2:,  2:]

    dz_dx = ((c + 2*f + i) - (a + 2*d + g)) / (8.0 * cellsize)   # eastward gradient
    dz_dy = ((a + 2*b + c) - (g + 2*h + i)) / (8.0 * cellsize)   # northward gradient

    slope = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))

    # Geographic aspect: clockwise from North
    aspect = np.pi / 2.0 - np.arctan2(dz_dy, dz_dx)
    aspect = np.where(aspect < 0, aspect + 2 * np.pi, aspect)

    return slope.astype(np.float32), aspect.astype(np.float32)


def hillshade_single(slope: np.ndarray, aspect: np.ndarray,
                     azimuth_deg: float, altitude_deg: float = 45.0) -> np.ndarray:
    """Hillshade value in [0, 1] for one sun direction."""
    az = np.radians(azimuth_deg)
    zenith = np.radians(90.0 - altitude_deg)
    hs = (np.cos(zenith) * np.cos(slope) +
          np.sin(zenith) * np.sin(slope) * np.cos(az - aspect))
    return np.clip(hs, 0.0, 1.0).astype(np.float32)


def multidirectional_hillshade(dem: np.ndarray, cellsize: float,
                                n: int = 16, altitude: float = 45.0) -> np.ndarray:
    """Average hillshade from n equally-spaced azimuths."""
    slope, aspect = compute_slope_aspect(dem, cellsize)
    azimuths = np.linspace(0, 360, n, endpoint=False)
    acc = np.zeros_like(dem, dtype=np.float32)
    for az in azimuths:
        acc += hillshade_single(slope, aspect, az, altitude)
    return acc / n


def normalize_percentile(arr: np.ndarray, valid_mask: np.ndarray,
                          p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    """Stretch valid pixels to [0, 1] using percentile clip; nodata pixels set to 0."""
    valid_vals = arr[valid_mask]
    if valid_vals.size == 0:
        return np.zeros_like(arr)
    vmin = np.percentile(valid_vals, p_low)
    vmax = np.percentile(valid_vals, p_high)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    out = np.zeros_like(arr)
    out[valid_mask] = np.clip((arr[valid_mask] - vmin) / (vmax - vmin), 0.0, 1.0)
    return out


def process_tile(src_path: Path, dst_path: Path, n_directions: int = 16,
                 altitude: float = 45.0) -> None:
    with rasterio.open(src_path) as src:
        dem = src.read(1).astype(np.float32)
        nodata = src.nodata
        cellsize = abs(src.transform[0])
        profile = src.profile.copy()

    # Build valid mask: exclude nodata and very large negative sentinel values
    if nodata is not None:
        valid = dem != nodata
    else:
        valid = np.ones(dem.shape, dtype=bool)
    valid &= dem > -9000   # catch -9999 or -3.4e38 sentinels not tagged in profile

    # Replace nodata with local mean so gradient at edges doesn't blow up
    if not valid.all():
        mean_val = dem[valid].mean() if valid.any() else 0.0
        dem[~valid] = mean_val

    hs = multidirectional_hillshade(dem, cellsize, n=n_directions, altitude=altitude)

    hs_norm = normalize_percentile(hs, valid)
    hs_u8 = (hs_norm * 255).astype(np.uint8)
    hs_u8[~valid] = 0  # nodata → black

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    profile.update(dtype=rasterio.uint8, count=1, nodata=0, compress="deflate")
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(hs_u8[np.newaxis, :, :])


def process_directory(src_dir: Path, dst_dir: Path, pattern: str = "*.tif",
                       n_directions: int = 16, altitude: float = 45.0) -> int:
    tiles = sorted(src_dir.glob(pattern))
    if not tiles:
        print(f"  [!] 未找到 TIF 文件: {src_dir}")
        return 0
    for tile in tiles:
        out = dst_dir / (tile.stem + "_hillshade16.tif")
        if out.exists():
            print(f"  跳过 (已存在): {out.name}")
            continue
        print(f"  处理: {tile.name} -> {out.name}")
        process_tile(tile, out, n_directions=n_directions, altitude=altitude)
    return len(tiles)


if __name__ == "__main__":
    JOBS = [
        {
            "name":    "Arran DTM (50cm)",
            "src_dir": Path(r"C:\Users\29775\Arran\DTM\images"),
            "dst_dir": Path(r"C:\Users\29775\Arran\Hillshade"),
        },
        {
            "name":    "Kintyre DTM (1m)",
            "src_dir": Path(r"C:\Users\29775\Scotland_DTM\Kintyre"),
            "dst_dir": Path(r"C:\Users\29775\Scotland_DTM\Kintyre_Hillshade"),
        },
    ]

    N_DIRECTIONS = 16
    ALTITUDE = 45.0

    for job in JOBS:
        print(f"\n=== {job['name']} ===")
        count = process_directory(
            job["src_dir"], job["dst_dir"],
            n_directions=N_DIRECTIONS, altitude=ALTITUDE
        )
        print(f"  完成 {count} 个瓦片")

    print("\n全部完成。")
