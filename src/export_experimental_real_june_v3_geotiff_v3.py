"""Export a GEOREFERENCED, EXPERIMENTAL V3 inference result for selected June 2025 tiles.

These are NOT validated cloud-free satellite images. The June-only cloud mask is
an uncalibrated DN proxy; V3 was trained on synthetic clouds, and can create
flat, inaccurate surfaces. April is NOT used in inference. No image alignment
corrections are applied; output uses the original June geospatial grid.

From the project root (no interactive windows, no pandas):
  .\\.venv\\Scripts\\python.exe -u .\\src\\export_experimental_real_june_v3_geotiff_v3.py --self-test
  .\\.venv\\Scripts\\python.exe -u .\\src\\export_experimental_real_june_v3_geotiff_v3.py

Requires existing project files: train_unet_residual_v3.py,
preview_real_june_unet_v3.py, verify_april_june_landmark_registration.py,
preview_real_temporal_transfer_unregistered.py,
audit_april_june_temporal_reference.py, review_april_june_cloud_and_alignment.py.

Writes 256x256 GeoTIFF tiles: original June, experimental V3 composite,
June-only proxy mask and June validity mask, plus a PNG, CSV and provenance JSON.
Never changes source imagery or model checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rasterio
import torch
from PIL import Image, ImageDraw
from rasterio.windows import Window
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import transform as transform_coords

from audit_april_june_temporal_reference import ROOT, locate_bands, preview_rgb
from preview_real_temporal_transfer_unregistered import qc_records, mask_rgb
from verify_april_june_landmark_registration import load_tile
from preview_real_june_unet_v3 import june_only_cloud_proxy, predict
from train_unet_residual_v3 import BEST, ResidualCloudUNet, load_checkpoint

EXPORTER_VERSION = 'V3-GEOREF-SPATIAL-2026-11-10'

DEFAULT_OUT = ROOT / 'outputs' / 'real_june_v3_experimental_geotiff'
QC_CSV = ROOT / 'outputs' / 'april_june_targeted_qc' / 'sampled_cloud_reference_qc.csv'
WARNING = (
    'EXPERIMENTAL, NOT A VALIDATED CLOUD-FREE PRODUCT. The cloud mask is an '
    'uncalibrated June-only digital-number heuristic. The neural model was '
    'trained on synthetic clouds. Actual June ground truth is unavailable. '
    'Clouds, haze and shadows can remain; inferred pixels can be inaccurate. '
    'No April reference pixels or unverified geographic shifts are used.'
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--check-version', action='store_true', help='Print exact exporter version and exit')
    p.add_argument('--tiles', default='T041,T058', help='Comma-separated QC tile IDs')
    p.add_argument('--qc-csv', type=Path, default=QC_CSV)
    p.add_argument('--april-dir', type=Path, default=ROOT / 'data/raw/mumbai_scenes/2025-04-16')
    p.add_argument('--june-dir', type=Path, default=ROOT / 'data/raw/mumbai_scenes/2025-06-03')
    p.add_argument('--checkpoint', type=Path, default=BEST)
    p.add_argument('--out', type=Path, default=DEFAULT_OUT)
    p.add_argument('--tile-size', type=int, default=256)
    p.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    return p.parse_args()


def resolve_device(name):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable, use --device cpu')
    return torch.device(name)



def max_geographic_displacement_m(source_crs, written_crs, affine, width: int, height: int) -> float:
    """Compare actual geographic footprints, not WKT/EPSG label serialization.

    Both CRSs must transform five independent pixel centres to nearly identical
    WGS84 coordinates. Geographic displacement is measured in metres.
    Any failure or invalid coordinate FAILS CLOSED rather than being accepted.
    """
    if source_crs is None or written_crs is None:
        return float('inf')
    pts = [(0.5, 0.5), (width - 0.5, 0.5), (0.5, height - 0.5),
           (width - 0.5, height - 0.5), (width / 2, height / 2)]
    xs, ys = zip(*(affine * pos for pos in pts))
    geodetic = CRS.from_epsg(4326)
    try:
        lon_src, lat_src = transform_coords(source_crs, geodetic, list(xs), list(ys))
        lon_out, lat_out = transform_coords(written_crs, geodetic, list(xs), list(ys))
    except Exception:
        return float('inf')
    lon_src = np.asarray(lon_src, dtype=np.float64)
    lat_src = np.asarray(lat_src, dtype=np.float64)
    lon_out = np.asarray(lon_out, dtype=np.float64)
    lat_out = np.asarray(lat_out, dtype=np.float64)
    if not np.all(np.isfinite([*lon_src, *lat_src, *lon_out, *lat_out])):
        return float('inf')
    # Haversine distance for corresponding pixel centres. This also handles
    # longitude wrap correctly and does not require an external pyproj package.
    rlat1, rlat2 = np.deg2rad(lat_src), np.deg2rad(lat_out)
    delta_lat = rlat2 - rlat1
    delta_lon = np.deg2rad(((lon_out - lon_src + 180) % 360) - 180)
    hav = np.sin(delta_lat / 2)**2 + np.cos(rlat1)*np.cos(rlat2)*np.sin(delta_lon / 2)**2
    return float(np.max(2 * 6371008.8 * np.arcsin(np.sqrt(np.clip(hav, 0, 1)))))


def write_tif(path: Path, array: np.ndarray, transform, crs, valid: np.ndarray, tags: dict,
              descriptions: tuple[str, ...]):
    """Write georeferenced data, preserve validity through a GDAL mask."""
    if array.ndim == 2:
        bands = array[None, ...]
    elif array.ndim == 3:
        bands = np.moveaxis(array, -1, 0)
    else:
        raise ValueError('Expected 2D mask or HWC satellite raster')
    if bands.shape[1:] != valid.shape or bands.dtype not in (np.dtype('uint8'), np.dtype('uint16')):
        raise ValueError('Raster shape or dtype invalid')
    h, w = valid.shape
    profile = dict(driver='GTiff', width=w, height=h, count=bands.shape[0], dtype=bands.dtype,
                   crs=crs, transform=transform, tiled=True, blockxsize=256, blockysize=256,
                   compress='DEFLATE', predictor=2, nodata=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Embed the mask inside each TIFF, rather than requiring a separate .msk sidecar.
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(path, 'w', **profile) as ds:
            ds.write(bands)
            ds.write_mask(valid.astype(np.uint8) * 255)
            ds.update_tags(**{k: str(v) for k, v in tags.items()})
            for index, desc in enumerate(descriptions, start=1):
                ds.set_band_description(index, desc)
    # Verify the same GEOGRAPHIC FOOTPRINT, including the source CRS defined
    # with noncanonical WKT. A real source may report to_epsg()=None while the
    # GeoTIFF writer canonicalizes equivalent WGS84/UTM 43N to EPSG:32643.
    # Five pixel centres must agree within 5 centimetres, much smaller than
    # the native 5-metre pixel. Different geolocation MUST fail.
    with rasterio.open(path) as ds:
        expected_crs = CRS.from_user_input(crs) if crs is not None else None
        actual_crs = ds.crs
        crs_labels_equal = (expected_crs is not None and actual_crs == expected_crs)
        epsg_equal = (expected_crs is not None and actual_crs is not None
                      and expected_crs.to_epsg() is not None
                      and expected_crs.to_epsg() == actual_crs.to_epsg())
        max_location_error_m = max_geographic_displacement_m(
            expected_crs, actual_crs, transform, w, h)
        location_tolerance_m = 0.05
        crs_ok = (crs_labels_equal or epsg_equal or
                  max_location_error_m <= location_tolerance_m)

        expected_affine = np.asarray(tuple(transform)[:6], dtype=np.float64)
        actual_affine = np.asarray(tuple(ds.transform)[:6], dtype=np.float64)
        grid_tolerance = 1e-5
        grid_error = float(np.max(np.abs(actual_affine - expected_affine)))
        grid_ok = bool(np.allclose(actual_affine, expected_affine,
                                   rtol=0, atol=grid_tolerance))
        shape_ok = (ds.count == bands.shape[0] and ds.width == w and ds.height == h)
        if not (crs_ok and grid_ok and shape_ok):
            raise AssertionError(
                f'GeoTIFF GEOGRAPHIC LOCATION mismatch: {path}\n'
                f'  Source CRS EPSG={expected_crs.to_epsg() if expected_crs else None}; '
                f'written EPSG={actual_crs.to_epsg() if actual_crs else None}\n'
                f'  CRS labels equal={crs_labels_equal}; EPSG equal={epsg_equal}\n'
                f'  Max WGS84 footprint drift={max_location_error_m:.9f} m '
                f'(tolerance={location_tolerance_m} m)\n'
                f'  Source transform={transform}\n'
                f'  Export transform={ds.transform}\n'
                f'  Max affine difference={grid_error:.12g} (tolerance {grid_tolerance})\n'
                f'  Bands/width/height source={bands.shape[0]}/{w}/{h}, '
                f'export={ds.count}/{ds.width}/{ds.height}'
            )
        # For audits: source and exported geolocations coincide to 5 cm.
        # An EPSG lookup mismatch by itself is NOT proof of misregistration.
        if not np.array_equal(ds.read(), bands):
            raise AssertionError(f'GeoTIFF data mismatch: {path}')
        if not np.array_equal(ds.dataset_mask() > 0, valid):
            raise AssertionError(f'GeoTIFF validity mask mismatch: {path}')


def sheet(tiles: list[dict], filename: Path):
    labels = ['June original (fixed stretch)', 'June-only proxy (unvalidated)',
              'V3 experimental output (same stretch)', 'Mean absolute DN change']
    width, gap, panel, start_y, step = 4 * 260 + 5 * 12, 12, 260, 100, 305
    img = Image.new('RGB', (width, start_y + step * len(tiles) + 35), 'white')
    d = ImageDraw.Draw(img)
    d.text((12, 8), 'REAL JUNE V3 | EXPERIMENTAL GEOTIFF TILE EXPORT', fill='black')
    d.text((12, 28), 'NOT validated cloud removal. June-only proxy mask. No April pixels or unverified shifts.', fill='black')
    d.text((12, 46), 'Output includes georeferenced experimental tiles; all original June files remain unchanged.', fill='black')
    for col, txt in enumerate(labels):
        d.text((gap + col * (panel + gap), start_y - 21), txt[:38], fill='black')
    for row, item in enumerate(tiles):
        j, valid, mask, out = item['j'], item['valid'], item['mask'], item['result']
        change = np.abs(out.astype(np.float32) - j.astype(np.float32)).mean(axis=2)
        t = np.clip(change / 250.0, 0, 1)
        heat = np.zeros((*change.shape, 3), np.uint8)
        heat[..., 0] = (255 * t).astype(np.uint8)
        heat[..., 1] = (145 * t).astype(np.uint8)
        heat[..., 2] = (65 * t).astype(np.uint8)
        heat[~mask & valid] = [15, 15, 15]
        heat[~valid] = [255, 255, 255]
        panels = [preview_rgb(j, valid), mask_rgb(mask, valid),
                  preview_rgb(out, valid), Image.fromarray(heat, 'RGB')]
        yy = start_y + row * step
        for col, im in enumerate(panels):
            im = im.resize((panel, panel), Image.Resampling.NEAREST if col == 1 else Image.Resampling.BILINEAR)
            img.paste(im, (gap + col * (panel + gap), yy))
        d.text((gap, yy + panel + 9), f"{item['id']}: mask={item['fraction']:.1%}; output=experimental, NOT truth", fill='black')
    filename.parent.mkdir(parents=True, exist_ok=True)
    img.save(filename)


def export(opts):
    if opts.tile_size != 256:
        raise ValueError('This export supports the 256x256 V3 inference tiles only')
    ids = [t.strip().upper() for t in opts.tiles.split(',') if t.strip()]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Specify unique tile IDs, e.g. --tiles T041,T058')
    records = qc_records(opts.qc_csv)
    for tid in ids:
        if tid not in records:
            raise ValueError(f'{tid} is absent from {opts.qc_csv}')
    if not opts.checkpoint.is_file():
        raise FileNotFoundError(f'V3 checkpoint missing: {opts.checkpoint}')
    device = resolve_device(opts.device)
    model, ck = load_checkpoint(opts.checkpoint, device, residual_scale=.25)
    model.eval()
    checkpoint_hash = hashlib.sha256(opts.checkpoint.read_bytes()).hexdigest()
    outdir = opts.out
    outdir.mkdir(parents=True, exist_ok=True)
    print(f'Exporter version: {EXPORTER_VERSION}', flush=True)
    print('REAL JUNE EXPERIMENTAL V3 GEOTIFF EXPORT — NOT VALIDATED CLOUD REMOVAL', flush=True)
    print(f'Device: {device} | checkpoint epoch: {ck.get("epoch", "unknown")} | tiles: {ids}', flush=True)
    print('No April transfer, no shifts, no training. June-only proxy mask.', flush=True)

    june_paths = locate_bands(opts.june_dir)
    june_nir = june_paths['BAND4.tif']
    rows, images = [], []
    for tid in ids:
        r = records[tid]
        a, av, j, jv, pixel_metres = load_tile(opts, r)
        del a, av
        with rasterio.open(june_nir) as src:
            window = Window(int(r['june_col']), int(r['june_row']), opts.tile_size, opts.tile_size)
            if not (window.col_off >= 0 and window.row_off >= 0 and
                    window.col_off + window.width <= src.width and
                    window.row_off + window.height <= src.height):
                raise ValueError(f'{tid}: tile is outside the June raster')
            transform = src.window_transform(window)
            crs = src.crs
            if not np.array_equal(src.read(1, window=window), j[..., 0].astype(np.uint16)):
                raise AssertionError('Loaded June tile does not match original June NIR GeoTIFF')
        mask = june_only_cloud_proxy(j, jv)
        fraction = float(mask.sum() / max(1, int(jv.sum())))
        if not 0.02 <= fraction <= .90 or int((jv & ~mask).sum()) < 100:
            raise ValueError(f'{tid}: unsafe proxy coverage {fraction:.1%}; no GeoTIFF exported')
        model_output = predict(model, j, jv, mask, device)
        original = j.astype(np.uint16)
        recon = np.rint(np.clip(model_output, 0, 1023)).astype(np.uint16)
        recon[~mask] = original[~mask]
        if not np.array_equal(recon[~mask], original[~mask]):
            raise AssertionError('June input outside cloud proxy was unexpectedly changed')
        if np.any(recon[jv] > 1023):
            raise AssertionError('Reconstruction exceeds 10-bit DN range')
        delta = np.abs(recon.astype(np.float32) - original.astype(np.float32))
        change_dn = float(delta[mask].mean()) if mask.any() else 0.
        meta = dict(TILE_ID=tid, IMAGE_DATE='2025-06-03', STATUS='EXPERIMENTAL_UNVALIDATED',
                    CLOUD_MASK='JUNE_ONLY_UNCALIBRATED_DN_PROXY',
                    APRIL_PIXELS_USED='NO', BANDS='NIR_B4,RED_B3,GREEN_B2',
                    CHECKPOINT_SHA256=checkpoint_hash,
                    ANY_REGISTRATION_SHIFT_APPLIED='NO', REAL_CLOUD_ACCURACY_KNOWN='NO')
        folder = outdir / tid
        names = {
            'original': folder / f'{tid}_june_original_B4_B3_B2.tif',
            'experimental': folder / f'{tid}_V3_EXPERIMENTAL_NOT_VALIDATED.tif',
            'proxy': folder / f'{tid}_UNVALIDATED_cloud_proxy_mask.tif',
            'validity': folder / f'{tid}_june_valid_mask.tif',
        }
        write_tif(names['original'], original, transform, crs, jv, meta | dict(TYPE='JUNE_ORIGINAL'),
                  ('NIR / B4', 'Red / B3', 'Green / B2'))
        write_tif(names['experimental'], recon, transform, crs, jv,
                  meta | dict(TYPE='V3_EXPERIMENTAL_INFERENCE'), ('NIR / B4', 'Red / B3', 'Green / B2'))
        write_tif(names['proxy'], mask.astype(np.uint8), transform, crs, jv,
                  meta | dict(TYPE='UNVALIDATED_JUNE_ONLY_PROXY'), ('Potential cloud (1)',))
        write_tif(names['validity'], jv.astype(np.uint8), transform, crs,
                  np.ones_like(jv, dtype=bool), meta | dict(TYPE='JUNE_VALIDITY_MASK'), ('Valid data (1)',))
        row = dict(tile_id=tid, src_row=int(r['june_row']), src_col=int(r['june_col']),
                   tile_width=opts.tile_size, tile_height=opts.tile_size,
                   pixel_spacing_m=round(pixel_metres, 5), crs=str(crs),
                   selected_proxy_fraction=round(fraction, 5),
                   masked_mean_absolute_change_dn=round(change_dn, 2),
                   outside_proxy_identical=True, cloud_mask_validated=False,
                   real_cloud_reconstruction_accuracy_available=False,
                   experimental_geotiff=str(names['experimental'].relative_to(outdir)))
        rows.append(row)
        images.append(dict(id=tid, j=original, valid=jv, mask=mask, result=recon, fraction=fraction))
        print(f'  {tid}: proxy {fraction:.1%} | changed {change_dn:.2f} DN | 4 GeoTIFFs written and verified', flush=True)

    preview_path = outdir / 'experimental_geotiff_preview.png'
    sheet(images, preview_path)
    csv_file = outdir / 'experimental_tile_summary.csv'
    with csv_file.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    doc = outdir / 'READ_ME_FIRST_UNVALIDATED.txt'
    doc.write_text(WARNING + '\n\nThese are 256x256 georeferenced June-grid analysis tiles.\n'
                   'BAND ORDER = NIR(B4), Red(B3), Green(B2), raw 10-bit digital numbers in uint16.\n'
                   'The exported 3-band inference GeoTIFF is an experimental output; do NOT label it cloud-free.\n'
                   'Use the separate proxy and validity GeoTIFF files when inspecting model changes.\n'
                   'Training/validation/test synthetic benchmark files are not included or modified.\n',
                   encoding='utf-8')
    prov = dict(title='Experimental georeferenced V3 inference on real June satellite patches',
                notice=WARNING, checkpoint=str(opts.checkpoint), checkpoint_sha256=checkpoint_hash,
                checkpoint_epoch=ck.get('epoch'), source_june_band4=str(june_nir),
                source_april_used=False, mask_uses_april=False, no_shifts_applied=True,
                original_june_untouched=True, synthetic_training=True, real_cloud_truth_available=False,
                real_cloud_performance_validated=False, real_cloud_mask_validated=False,
                export_pixel_dtype='uint16', band_order=['NIR_B4','Red_B3','Green_B2'],
                normalization_divisor=1023.0, proxy_thresholds='NIR,Red,Green >=220 DN and NDVI < 0.28',
                tiles=rows)
    (outdir / 'experimental_provenance.json').write_text(json.dumps(prov, indent=2), encoding='utf-8')
    print('\nEXPERIMENTAL GEOTIFF EXPORT COMPLETE — NOT VALIDATED CLOUD-FREE', flush=True)
    print('Preview:', preview_path, flush=True)
    print('CSV:', csv_file, flush=True)
    print('Provenance:', outdir / 'experimental_provenance.json', flush=True)
    print('Read first:', doc, flush=True)
    return rows


def self_test():
    rng = np.random.default_rng(81)
    n = 256
    yy, xx = np.indices((n,n))
    base = 140 + 15 * np.sin(xx / 15.) + 9 * np.cos(yy / 13.) + rng.normal(0, 2, (n,n))
    a = np.stack([base+40, base-12, base+3], axis=-1).clip(1,1023).astype(np.uint16)
    j = a.copy()
    j[45:180,50:165] = np.array([440,415,425], dtype=np.uint16)
    valid = np.ones((n,n), bool)
    valid[:5, :5] = False
    a[~valid] = 0; j[~valid] = 0
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        transform = from_origin(500000, 2120000, 5, 5)
        for sub, arr in [('apr',a),('jun',j)]:
            p = root / sub
            p.mkdir()
            for band, idx in [(4,0),(3,1),(2,2)]:
                with rasterio.open(p / f'BAND{band}.tif','w',driver='GTiff',height=n,width=n,
                                   count=1,dtype='uint16',crs='EPSG:32643',transform=transform,
                                   nodata=0) as ds:
                    ds.write(arr[...,idx],1)
        qc = root / 'qc.csv'
        qc.write_text('tile_id,june_col,june_row\nT001,0,0\n',encoding='utf-8')
        model = ResidualCloudUNet(residual_scale=.25)
        ckpt = root / 'ck.pt'
        torch.save(dict(model_state_dict=model.state_dict(),experiment='residual_unet_v3',
                        normalization_divisor=1023.0,residual_scale=.25,epoch=0),ckpt)
        opts = SimpleNamespace(tiles='T001',qc_csv=qc,april_dir=root/'apr',june_dir=root/'jun',
                               checkpoint=ckpt,out=root/'out',tile_size=256,device='cpu')
        report = export(opts)
        assert len(report)==1 and report[0]['outside_proxy_identical']
        # More realistic test: fractional UTM origins and an explicit WKT CRS.
        # This catches false mismatches from GeoTIFF coordinate serialization.
        fine_grid = from_origin(482371.123456, 2134567.654321, 5.0, 5.0)
        write_tif(root/'fractional_grid_crs_test.tif', a, fine_grid,
                  CRS.from_epsg(32643).to_wkt(), valid,
                  {'TYPE':'REALISTIC_GEOREF_QC'}, ('NIR / B4','Red / B3','Green / B2'))
        v3 = opts.out/'T001'/'T001_V3_EXPERIMENTAL_NOT_VALIDATED.tif'
        assert v3.is_file()
        with rasterio.open(v3) as ds:
            assert ds.crs.to_epsg()==32643 and ds.transform==transform
            assert ds.count==3 and ds.dtypes[0]=='uint16'
        assert (opts.out/'experimental_geotiff_preview.png').is_file()
        assert (opts.out/'experimental_tile_summary.csv').is_file()
        assert (opts.out/'experimental_provenance.json').is_file()
        assert (opts.out/'READ_ME_FIRST_UNVALIDATED.txt').is_file()
        # Source CRS metadata can be noncanonical even when the real
        # geographic transform is equivalent to EPSG:32643.
        old_wkt = CRS.from_epsg(32643).to_wkt(version='WKT1_GDAL')
        import re
        old_wkt_without_authority = re.sub(r',AUTHORITY\[\"EPSG\",\"[0-9]+\"\]', '', old_wkt)
        alias_crs = CRS.from_wkt(old_wkt_without_authority)
        assert max_geographic_displacement_m(alias_crs, CRS.from_epsg(32643), fine_grid, n, n) < 0.05
        write_tif(root/'noncanonical_wkt_test.tif', a, fine_grid, alias_crs, valid,
                  {'TYPE':'WKT_EQUIVALENCE_CHECK'}, ('NIR / B4','Red / B3','Green / B2'))
        # Negative test: an incorrect UTM zone is NOT equivalent, and the
        # five-point geographic comparison must reject it.
        wrong_zone = CRS.from_epsg(32644)
        assert max_geographic_displacement_m(alias_crs, wrong_zone, fine_grid, n, n) > 100000
        # Regression for the EXACT legacy WKT format printed by the user's
        # Resourcesat-2A June scene: authority name appears in the WKT, but
        # Rasterio may still return to_epsg()=None for its legacy datum.
        june_legacy_wkt = (
            'PROJCS["WGS 84 / UTM zone 43N",GEOGCS["WGS 84",'
            'DATUM["WGS_1984",SPHEROID["WGS 84",6378137,'
            '298.25722293287,AUTHORITY["EPSG","7030"]],'
            'AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],'
            'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],'
            'AUTHORITY["EPSG","4326"]],PROJECTION["Transverse_Mercator"],'
            'PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",75],'
            'PARAMETER["scale_factor",0.9996],'
            'PARAMETER["false_easting",500000],PARAMETER["false_northing",0],'
            'UNIT["metre",1,AUTHORITY["EPSG","9001"]],'
            'AXIS["Easting",EAST],AXIS["Northing",NORTH],'
            'AUTHORITY["EPSG","32643"]]'
        )
        legacy_crs = CRS.from_wkt(june_legacy_wkt)
        legacy_pos = from_origin(280983.94, 2089937.5, 5, 5)
        assert legacy_crs.to_epsg() is None, 'Expected this WKT to reproduce EPSG=None mismatch'
        assert max_geographic_displacement_m(
            legacy_crs, CRS.from_epsg(32643), legacy_pos, n, n) < 0.05
        write_tif(root/'actual_june_wkt_regression.tif', a, legacy_pos,
                  legacy_crs, valid, {'TYPE':'LEGACY_JUNE_CRS_TEST'},
                  ('NIR / B4','Red / B3','Green / B2'))
    print('EXPERIMENTAL GEOREFERENCED V3 SELF-TEST PASSED | equivalent WKT accepted; wrong UTM zone rejected', flush=True)


if __name__ == '__main__':
    options = parse_args()
    if options.check_version:
        print(f'{EXPORTER_VERSION} | file={Path(__file__).resolve()}')
    elif options.self_test:
        self_test()
    else:
        export(options)
