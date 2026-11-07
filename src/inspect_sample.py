from pathlib import Path

import rasterio

project_dir = Path(__file__).resolve().parents[1]
raw_dir = project_dir / "data" / "raw"

band_files = sorted(
    path for path in raw_dir.rglob("*")
    if path.is_file() and path.name.upper() in {
        "BAND2.TIF", "BAND3.TIF", "BAND4.TIF"
    }
)

if not band_files:
    raise FileNotFoundError(f"No band TIFFs found inside {raw_dir}")

for path in band_files:
    with rasterio.open(path) as dataset:
        print(f"\nFile: {path.relative_to(project_dir)}")
        print(f"Dimensions: {dataset.width} x {dataset.height}")
        print(f"Band count: {dataset.count}")
        print(f"Pixel types: {dataset.dtypes}")
        print(f"Coordinate system: {dataset.crs}")
        print(f"Pixel size: {dataset.res}")
        print(f"NoData value: {dataset.nodata}")
        print(f"Bounds: {dataset.bounds}")
        print(f"Mask flags: {dataset.mask_flag_enums}")