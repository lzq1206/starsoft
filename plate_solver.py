"""Bundled ASTAP plate solving and local Gaia bright-star matching."""
from __future__ import annotations

import hashlib
import csv
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Callable, Sequence
from urllib.request import Request, urlopen

import numpy as np
import sep
import tifffile
from astropy.io import fits
from astropy.wcs import WCS
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


Progress = Callable[[int, str], None]
_SEIZA_DATA_LOCK = threading.Lock()
_SEIZA_DATA: tuple[object, object] | None = None
_SEIZA_DEEP_DATA: tuple[object, object] | None = None


@dataclass(frozen=True)
class SolverTile:
    x0: int
    y0: int
    x1: int
    y1: int
    wcs: WCS
    pixel_scale_arcsec: float
    wide_field: bool = False
    matched_quads: int = 0
    total_quads: int = 0
    scale_was_inaccurate: bool = False


def _tile_pixel_to_world(tile: SolverTile, pixels: np.ndarray) -> np.ndarray:
    """Convert top-left image pixels to ASTAP's FITS-style bottom-left WCS pixels."""
    coordinates = np.asarray(pixels, dtype=np.float64).copy()
    coordinates[..., 1] = (tile.y1 - tile.y0 - 1) - coordinates[..., 1]
    return np.asarray(tile.wcs.all_pix2world(coordinates, 0), dtype=np.float64)


def _tile_world_to_pixel(tile: SolverTile, world: np.ndarray) -> np.ndarray:
    """Convert ASTAP's FITS-style WCS pixels back to top-left image pixels."""
    coordinates = np.asarray(
        tile.wcs.all_world2pix(world, 0, quiet=True), dtype=np.float64
    )
    coordinates[..., 1] = (tile.y1 - tile.y0 - 1) - coordinates[..., 1]
    return coordinates


def _world_unit_vectors(world: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(world, dtype=np.float64)
    ra = np.radians(coordinates[..., 0])
    dec = np.radians(coordinates[..., 1])
    cos_dec = np.cos(dec)
    return np.stack((cos_dec * np.cos(ra), cos_dec * np.sin(ra), np.sin(dec)), axis=-1)


def _camera_ray(
    x: float,
    y: float,
    image_width: int,
    image_height: int,
    sensor_width_mm: float,
    sensor_height_mm: float,
    focal_mm: float,
) -> np.ndarray:
    # Rectilinear camera projection: pixel displacement maps to sensor-plane
    # displacement, then the lens focal length supplies the ray's z component.
    ray = np.asarray([
        (x - image_width * 0.5) * sensor_width_mm / max(image_width, 1),
        (image_height * 0.5 - y) * sensor_height_mm / max(image_height, 1),
        focal_mm,
    ], dtype=np.float64)
    return ray / max(float(np.linalg.norm(ray)), 1e-12)


def _camera_orientation_from_tile(
    tile: SolverTile,
    image_width: int,
    image_height: int,
    sensor_width_mm: float,
    sensor_height_mm: float,
    focal_mm: float,
) -> np.ndarray | None:
    """Recover camera-to-sky orientation from one local WCS and camera geometry."""
    center_x = (tile.x0 + tile.x1) * 0.5
    center_y = (tile.y0 + tile.y1) * 0.5
    step = 12.0
    global_pixels = np.asarray([
        [center_x, center_y],
        [center_x - step, center_y], [center_x + step, center_y],
        [center_x, center_y - step], [center_x, center_y + step],
    ], dtype=np.float64)
    try:
        local_pixels = global_pixels - np.asarray([tile.x0, tile.y0])
        world = _tile_pixel_to_world(tile, local_pixels)
        sky = _world_unit_vectors(world)
        if not np.isfinite(sky).all():
            return None

        camera = np.asarray([
            _camera_ray(x, y, image_width, image_height,
                        sensor_width_mm, sensor_height_mm, focal_mm)
            for x, y in global_pixels
        ])
        camera_center = camera[0]
        camera_dx = camera[2] - camera[1]
        camera_dy = camera[4] - camera[3]
        sky_center = sky[0]
        sky_dx = sky[2] - sky[1]
        sky_dy = sky[4] - sky[3]

        def orthogonal_frame(
            center: np.ndarray, dx: np.ndarray, dy: np.ndarray
        ) -> np.ndarray | None:
            dx = dx - center * float(np.dot(center, dx))
            dx_norm = float(np.linalg.norm(dx))
            if dx_norm < 1e-12:
                return None
            dx /= dx_norm
            dy = dy - center * float(np.dot(center, dy)) - dx * float(np.dot(dx, dy))
            dy_norm = float(np.linalg.norm(dy))
            if dy_norm < 1e-12:
                return None
            dy /= dy_norm
            return np.column_stack((center, dx, dy))

        camera_frame = orthogonal_frame(camera_center, camera_dx, camera_dy)
        sky_frame = orthogonal_frame(sky_center, sky_dx, sky_dy)
        if camera_frame is None or sky_frame is None:
            return None
        orientation = sky_frame @ camera_frame.T
        if not np.isfinite(orientation).all():
            return None
        return orientation
    except Exception:
        return None


@dataclass(frozen=True)
class CatalogMatch:
    source_id: int | None
    ra_deg: float
    dec_deg: float
    g_mag: float
    bp_mag: float | None
    rp_mag: float | None
    separation_arcsec: float
    catalog_row_index: int | None = None
    name: str | None = None
    position_recovered: bool = False


@dataclass(frozen=True)
class RecoveredCatalogStar:
    g_mag: float
    x: float
    y: float
    flux: float
    peak: float
    fwhm: float
    a: float
    b: float
    theta: float
    signal_to_noise: float
    separation_arcsec: float
    catalog_row_index: int = -1
    ra_deg: float = 0.0
    dec_deg: float = 0.0
    name: str = ""


@dataclass(frozen=True)
class CatalogPosition:
    """A Gaia-derived W08 position, with image evidence kept explicit."""

    x: float
    y: float
    g_mag: float
    state: str
    ra_deg: float
    dec_deg: float
    name: str
    projection_source: str = ""


@dataclass(frozen=True)
class CatalogCoverageArea:
    x0: float
    y0: float
    x1: float
    y1: float
    source: str


@lru_cache(maxsize=1)
def _load_named_star_rows() -> tuple[tuple[float, float, str, str], ...]:
    roots = [Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))]
    roots.append(Path(__file__).resolve().parent)
    catalog_path = next(
        (root / "data" / "hyg_named_stars.csv" for root in roots
         if (root / "data" / "hyg_named_stars.csv").is_file()),
        None,
    )
    if catalog_path is None:
        return ()
    rows: list[tuple[float, float, str, str]] = []
    try:
        with catalog_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    ra = float(row["ra_deg_j2000"])
                    dec = float(row["dec_deg_j2000"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not (math.isfinite(ra) and math.isfinite(dec) and 0.0 <= ra < 360.0 and -90.0 <= dec <= 90.0):
                    continue
                proper = str(row.get("proper_name") or "").strip()
                designation = str(row.get("designation") or "").strip()
                label = proper or designation
                if label:
                    rows.append((ra, dec, label, proper))
    except OSError:
        return ()
    return tuple(rows)


@lru_cache(maxsize=1)
def _named_star_tree() -> tuple[cKDTree | None, np.ndarray]:
    rows = _load_named_star_rows()
    if not rows:
        return None, np.empty((0, 3), dtype=np.float64)
    vectors = _world_unit_vectors(np.asarray([[row[0], row[1]] for row in rows], dtype=np.float64))
    return cKDTree(vectors), vectors


def _named_star_label(ra_deg: float, dec_deg: float) -> str:
    rows = _load_named_star_rows()
    tree, vectors = _named_star_tree()
    if tree is not None and len(rows):
        target = _world_unit_vectors(np.asarray([[ra_deg, dec_deg]], dtype=np.float64))[0]
        distance, index = tree.query(target, k=1)
        # HYG J2000 positions and Gaia DR3-era W08 coordinates can differ by
        # proper motion. Keep the match local enough to avoid naming a neighbour.
        if float(distance) <= 2.0 * math.sin(math.radians(60.0 / 3600.0) * 0.5):
            return rows[int(index)][2]
    ra_hours = (float(ra_deg) % 360.0) / 15.0
    hour = int(ra_hours)
    minute_value = (ra_hours - hour) * 60.0
    minute = int(minute_value)
    second = (minute_value - minute) * 60.0
    sign = "+" if dec_deg >= 0 else "-"
    abs_dec = abs(float(dec_deg))
    dec_degree = int(abs_dec)
    dec_minute_value = (abs_dec - dec_degree) * 60.0
    dec_minute = int(dec_minute_value)
    dec_second = (dec_minute_value - dec_minute) * 60.0
    return f"RA {hour:02d}:{minute:02d}:{second:04.1f} Dec {sign}{dec_degree:02d}:{dec_minute:02d}:{dec_second:04.1f}"


@dataclass(frozen=True)
class CameraProjection:
    orientation: np.ndarray
    focal_mm: float
    center_x: float
    center_y: float
    residual_p90_px: float


@dataclass(frozen=True)
class SeizaLocalSolveResult:
    camera_projection: CameraProjection | None
    catalog_match_count: int
    verified_tiles: tuple[SolverTile, ...]


@dataclass(frozen=True)
class _SolveSource:
    x: float
    y: float
    flux: float


def _fit_wide_camera_projection(
    stars: Sequence[object],
    matches: dict[int, CatalogMatch],
    tiles: Sequence[SolverTile],
    detector_shape: tuple[int, int],
    detector_scale_xy: tuple[float, float],
    info: object,
) -> CameraProjection | None:
    """Fit a rectilinear camera pose from a trusted local WCS and Gaia matches."""
    if len(matches) < 20 or not tiles:
        return None
    detector_height, detector_width = detector_shape
    scale_x, scale_y = detector_scale_xy
    image_width = max(1, int(round(detector_width * scale_x)))
    image_height = max(1, int(round(detector_height * scale_y)))
    sensor_width_mm, sensor_height_mm, focal_mm = _sensor_dimensions(info, detector_shape)
    image_center = np.asarray([image_width * 0.5, image_height * 0.5])
    seed_tile = min(
        tiles,
        key=lambda tile: math.hypot(
            (tile.x0 + tile.x1) * 0.5 - image_center[0],
            (tile.y0 + tile.y1) * 0.5 - image_center[1],
        ),
    )
    orientation0 = _camera_orientation_from_tile(
        seed_tile, image_width, image_height,
        sensor_width_mm, sensor_height_mm, focal_mm,
    )
    if orientation0 is None:
        return None

    pixel_scale_arcsec = max(float(seed_tile.pixel_scale_arcsec), 1e-6)
    # Use only stars inside the seed's low-distortion field, and discard
    # astrometric associations too far from the WCS's own source prediction.
    # A robust loss below handles the remaining accidental nearest neighbours.
    maximum_separation = min(240.0, max(90.0, 3.5 * pixel_scale_arcsec))
    source_pixels: list[tuple[float, float]] = []
    sky_vectors: list[np.ndarray] = []
    for det_id, match in matches.items():
        if not (0 <= det_id < len(stars)) or match.separation_arcsec > maximum_separation:
            continue
        star = stars[det_id]
        x = float(getattr(star, "x")) * scale_x
        y = float(getattr(star, "y")) * scale_y
        if not (seed_tile.x0 <= x < seed_tile.x1 and seed_tile.y0 <= y < seed_tile.y1):
            continue
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        source_pixels.append((x, y))
        sky_vectors.append(_world_unit_vectors([[match.ra_deg, match.dec_deg]])[0])
    if len(source_pixels) < 16:
        return None
    pixels = np.asarray(source_pixels, dtype=np.float64)
    targets = np.asarray(sky_vectors, dtype=np.float64)

    def residual(parameters: np.ndarray) -> np.ndarray:
        effective_focal, center_x, center_y = parameters[:3]
        orientation = Rotation.from_rotvec(parameters[3:]).as_matrix() @ orientation0
        rays = np.column_stack((
            (pixels[:, 0] - center_x) * sensor_width_mm / image_width,
            (center_y - pixels[:, 1]) * sensor_height_mm / image_height,
            np.full(len(pixels), effective_focal),
        ))
        rays /= np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-12)
        projected = rays @ orientation.T
        return ((projected - targets) * (206264.806247 / pixel_scale_arcsec)).ravel()

    initial = np.concatenate((
        [focal_mm, image_center[0], image_center[1]],
        np.zeros(3, dtype=np.float64),
    ))
    bounds = (
        np.asarray([
            focal_mm * 0.65,
            image_center[0] - image_width * 0.03,
            image_center[1] - image_height * 0.03,
            -0.15, -0.15, -0.15,
        ]),
        np.asarray([
            focal_mm * 1.35,
            image_center[0] + image_width * 0.03,
            image_center[1] + image_height * 0.03,
            0.15, 0.15, 0.15,
        ]),
    )
    try:
        fit = least_squares(
            residual, initial, bounds=bounds, loss="soft_l1", f_scale=1.5,
            max_nfev=300,
        )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return None
    errors = np.linalg.norm(residual(fit.x).reshape((-1, 3)), axis=1)
    if not fit.success or not np.isfinite(errors).all():
        return None
    p50, p90 = np.percentile(errors, [50.0, 90.0])
    inlier_count = int(np.count_nonzero(errors <= 6.0))
    if p50 > 3.5 or p90 > 8.0 or inlier_count < 16:
        return None
    orientation = Rotation.from_rotvec(fit.x[3:]).as_matrix() @ orientation0
    return CameraProjection(
        orientation=orientation,
        focal_mm=float(fit.x[0]),
        center_x=float(fit.x[1]),
        center_y=float(fit.x[2]),
        residual_p90_px=float(p90),
    )


# Used only when camera EXIF omits FocalLengthIn35mmFilm. Values are active
# image dimensions; the list intentionally covers camera bodies used in the
# supplied samples and common full-frame and APS-C DSLRs.
_SENSOR_SIZES_MM: dict[str, tuple[float, float]] = {
    "NIKON D4S": (36.0, 23.9),
    "NIKON D4": (36.0, 23.9),
    "NIKON D5": (35.9, 23.9),
    "NIKON D6": (35.9, 23.9),
    "NIKON Z 7": (35.9, 23.9),
    "NIKON Z 6": (35.9, 23.9),
    "NIKON Z 8": (35.9, 23.9),
    "NIKON Z 9": (35.9, 23.9),
    "NIKON D850": (35.9, 23.9),
    "NIKON D810": (35.9, 23.9),
    "CANON EOS 5D MARK IV": (36.0, 24.0),
    "CANON EOS R5": (36.0, 24.0),
    "CANON EOS R6": (36.0, 24.0),
    "SONY ILCE-7RM3": (35.9, 24.0),
    "SONY ILCE-7RM4": (35.7, 23.8),
    "SONY ILCE-7M3": (35.6, 23.8),
    "SONY ILCE-7M4": (35.6, 23.8),
    "SONY ILCE-7RM5": (35.7, 23.8),
    "NIKON D7500": (23.5, 15.7),
    "NIKON D500": (23.5, 15.7),
    "NIKON D7200": (23.5, 15.6),
    "NIKON Z 50": (23.5, 15.7),
    "CANON EOS 90D": (22.3, 14.8),
    "CANON EOS 80D": (22.5, 15.0),
    "SONY ILCE-6400": (23.5, 15.6),
    "SONY ILCE-6700": (23.3, 15.5),
}

_SENSOR_FORMAT_SIZES_MM: dict[str, tuple[float, float]] = {
    "full_frame": (36.0, 24.0),
    "aps_c": (23.5, 15.6),
    "medium_4433": (43.8, 32.9),
    "four_thirds": (17.3, 13.0),
    "one_inch": (13.2, 8.8),
}


def _executable_and_catalogs() -> tuple[Path, Path]:
    override = os.environ.get("STARSOFT_SOLVER_DIR")
    roots: list[Path] = []
    if override:
        roots.append(Path(override))
    executable_dir = Path(sys.executable).resolve().parent
    roots.extend((
        executable_dir / "solver",
        executable_dir.parent / "Resources" / "solver",
        Path(__file__).resolve().parent / "solver",
    ))
    executable_name = "astap_cli.exe" if os.name == "nt" else "astap_cli"
    for root in roots:
        executable = root / executable_name
        catalogs = root / "catalogs"
        if executable.is_file() and catalogs.is_dir() and any(catalogs.iterdir()):
            return executable, catalogs
    raise RuntimeError("本机板解算组件未随程序完整安装。请重新下载完整版本压缩包并解压后运行。")


def _sensor_dimensions(info: object, image_shape: tuple[int, int]) -> tuple[float, float, float]:
    """Return sensor width/height in mm and equivalent focal length in mm."""
    height, width = image_shape
    camera = " ".join(str(getattr(info, "camera", "")).upper().split())
    focal_35mm = getattr(info, "focal_length_35mm", None)
    focal = float(focal_35mm) if focal_35mm and float(focal_35mm) > 0 else None
    selected_format = getattr(info, "sensor_format", None)
    sensor = _SENSOR_FORMAT_SIZES_MM.get(str(selected_format)) if selected_format else None
    if sensor is not None:
        # A manually selected sensor format pairs with actual lens focal length,
        # even when the camera also writes a 35 mm equivalent EXIF value.
        focal = None
    else:
        lens = " ".join(str(getattr(info, "lens", "")).upper().split())
        aps_c_lens = bool(re.search(r"\b(?:DX|DC|EF-S|EFS|DT|DI II|APS-C)\b", lens))
        sensor = next(
            (dimensions for model, dimensions in _SENSOR_SIZES_MM.items() if model in camera),
            None,
        )
        if aps_c_lens:
            sensor = _SENSOR_FORMAT_SIZES_MM["aps_c"]
    if sensor is None and focal is not None:
        # EXIF equivalent focal length is defined against the 36 x 24 mm format.
        sensor = (36.0, 24.0)
    if sensor is None:
        actual_focal = getattr(info, "focal_length", None)
        if actual_focal and float(actual_focal) > 0:
            raise RuntimeError(
                "照片记录了镜头焦距，但没有记录 35 mm 等效焦距，且程序无法识别相机画幅。"
                "为避免用错误视场匹配星表，请使用带相机型号 EXIF 的文件。"
            )
        raise RuntimeError("无法从照片读取相机画幅与焦距，不能可靠解算星表坐标。")
    sensor_width, sensor_height = sensor
    if focal is None:
        actual_focal = getattr(info, "focal_length", None)
        if not actual_focal or float(actual_focal) <= 0:
            raise RuntimeError("照片没有可用的镜头焦距信息，不能可靠解算星表坐标。")
        focal = float(actual_focal)
    if height > width:
        return sensor_height, sensor_width, focal
    return sensor_width, sensor_height, focal


# J2000 coordinates for Mintaka, Alnilam, and Alnitak. This local asterism
# anchor is a last-resort hypothesis only; the fit is accepted only after an
# independent, spatially distributed W08 catalogue match.
_ORION_BELT_J2000_DEG = np.asarray([
    [83.0016667, -0.299094],   # Mintaka (delta Orionis)
    [84.0533875, -1.201919],   # Alnilam (epsilon Orionis)
    [85.1897290, -1.942639],   # Alnitak (zeta Orionis)
], dtype=np.float64)


def _project_camera_catalog(
    vectors: np.ndarray,
    projection: CameraProjection,
    image_width: int,
    image_height: int,
    sensor_width_mm: float,
    sensor_height_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project sky directions through the fitted rectilinear camera model."""
    camera_vectors = np.asarray(vectors, dtype=np.float64) @ projection.orientation
    z = camera_vectors[:, 2]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        x = projection.center_x + (
            projection.focal_mm * camera_vectors[:, 0]
            / np.maximum(z, 1e-9) * image_width / sensor_width_mm
        )
        y = projection.center_y - (
            projection.focal_mm * camera_vectors[:, 1]
            / np.maximum(z, 1e-9) * image_height / sensor_height_mm
        )
    return np.column_stack((x, y)), z


def _camera_catalog_pairs(
    catalog_magnitudes: np.ndarray,
    catalog_vectors: np.ndarray,
    projection: CameraProjection,
    source_pixels: np.ndarray,
    source_ids: np.ndarray,
    source_tree: cKDTree,
    detector_shape: tuple[int, int],
    detector_scale_xy: tuple[float, float],
    sky_mask: np.ndarray | None,
    maximum_distance_px: float,
    info: object,
) -> list[tuple[float, int, int]]:
    """Return one-to-one projected W08/source pairs within a pixel radius."""
    detector_height, detector_width = detector_shape
    scale_x, scale_y = detector_scale_xy
    image_width = max(1, int(round(detector_width * scale_x)))
    image_height = max(1, int(round(detector_height * scale_y)))
    sensor_width_mm, sensor_height_mm, _focal_mm = _sensor_dimensions(info, detector_shape)
    catalog_rows = np.flatnonzero(catalog_magnitudes <= 8.0)
    if not len(catalog_rows) or not len(source_ids):
        return []
    projected, z = _project_camera_catalog(
        catalog_vectors[catalog_rows], projection,
        image_width, image_height, sensor_width_mm, sensor_height_mm,
    )
    valid = (
        np.isfinite(projected).all(axis=1)
        & np.isfinite(z)
        & (z > 0.05)
        & (projected[:, 0] >= 0.0) & (projected[:, 0] < image_width)
        & (projected[:, 1] >= 0.0) & (projected[:, 1] < image_height)
    )
    if sky_mask is not None and sky_mask.shape == detector_shape:
        valid_rows = np.flatnonzero(valid)
        detector_x = np.clip(
            np.floor((projected[valid_rows, 0] + 0.5) / scale_x).astype(np.intp),
            0, detector_width - 1,
        )
        detector_y = np.clip(
            np.floor((projected[valid_rows, 1] + 0.5) / scale_y).astype(np.intp),
            0, detector_height - 1,
        )
        valid[valid_rows] &= ~np.asarray(sky_mask, dtype=bool)[detector_y, detector_x]
    rows = np.flatnonzero(valid)
    if not len(rows):
        return []
    distances, source_rows = source_tree.query(
        projected[rows], k=1, distance_upper_bound=maximum_distance_px
    )
    candidates = [
        (float(distance), int(catalog_rows[row]), int(source_ids[source_row]))
        for row, distance, source_row in zip(rows, distances, source_rows)
        if math.isfinite(float(distance)) and int(source_row) < len(source_ids)
    ]
    used_catalog: set[int] = set()
    used_sources: set[int] = set()
    accepted: list[tuple[float, int, int]] = []
    for distance, catalog_row, source_id in sorted(candidates):
        if catalog_row in used_catalog or source_id in used_sources:
            continue
        used_catalog.add(catalog_row)
        used_sources.add(source_id)
        accepted.append((distance, catalog_row, source_id))
    return accepted


def _positions_from_camera_projection(
    stars: Sequence[object],
    detector: np.ndarray,
    sky_mask: np.ndarray | None,
    detector_scale_xy: tuple[float, float],
    sensor_dimensions_mm: tuple[float, float],
    projection: CameraProjection,
) -> tuple[list[tuple[int, float, float]], float]:
    """Map SEP detections through a catalogue-verified camera pose."""
    detector_height, detector_width = detector.shape
    scale_x, scale_y = detector_scale_xy
    image_width = max(1, int(round(detector_width * scale_x)))
    image_height = max(1, int(round(detector_height * scale_y)))
    sensor_width_mm, sensor_height_mm = sensor_dimensions_mm
    positions: list[tuple[int, float, float]] = []
    for index, star in enumerate(stars):
        x, y = float(getattr(star, "x")), float(getattr(star, "y"))
        detector_x, detector_y = int(round(x)), int(round(y))
        if not (0 <= detector_x < detector_width and 0 <= detector_y < detector_height):
            continue
        if sky_mask is not None and sky_mask.shape == detector.shape and sky_mask[detector_y, detector_x]:
            continue
        full_x = (x + 0.5) * scale_x - 0.5
        full_y = (y + 0.5) * scale_y - 0.5
        ray = _camera_ray(
            full_x, full_y, image_width, image_height,
            sensor_width_mm, sensor_height_mm, projection.focal_mm,
        )
        world = ray @ projection.orientation.T
        ra = math.degrees(math.atan2(world[1], world[0])) % 360.0
        dec = math.degrees(math.asin(float(np.clip(world[2], -1.0, 1.0))))
        if math.isfinite(ra) and math.isfinite(dec):
            positions.append((index, ra, dec))
    pixel_scale = math.degrees(math.atan2(
        sensor_height_mm / image_height, projection.focal_mm
    )) * 3600.0
    return positions, pixel_scale


def _try_orion_belt_camera_fit(
    stars: Sequence[object],
    sky_mask: np.ndarray | None,
    info: object,
    catalog_magnitudes: np.ndarray,
    catalog_vectors: np.ndarray,
    detector_shape: tuple[int, int],
    detector_scale_xy: tuple[float, float],
    progress: Progress | None,
) -> tuple[CameraProjection, int] | None:
    """Try a catalogue-checked Orion belt seed after blind solvers fail."""
    if len(stars) < 3:
        return None
    detector_height, detector_width = detector_shape
    scale_x, scale_y = detector_scale_xy
    image_width = max(1, int(round(detector_width * scale_x)))
    image_height = max(1, int(round(detector_height * scale_y)))
    sensor_width_mm, sensor_height_mm, focal_mm = _sensor_dimensions(info, detector_shape)
    camera_pixels = np.full((len(stars), 2), np.nan, dtype=np.float64)
    valid_ids: list[int] = []
    for index, star in enumerate(stars):
        x, y = float(getattr(star, "x")), float(getattr(star, "y"))
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        detector_x, detector_y = int(round(x)), int(round(y))
        if not (0 <= detector_x < detector_width and 0 <= detector_y < detector_height):
            continue
        if sky_mask is not None and sky_mask.shape == detector_shape and sky_mask[detector_y, detector_x]:
            continue
        camera_pixels[index] = ((x + 0.5) * scale_x - 0.5, (y + 0.5) * scale_y - 0.5)
        valid_ids.append(index)
    if len(valid_ids) < 40:
        return None
    source_ids = np.asarray(valid_ids, dtype=np.intp)
    source_pixels = camera_pixels[source_ids]
    source_tree = cKDTree(source_pixels)

    # Limit the geometric search to the brightest 500 SEP detections. These
    # three named belt stars are bright, and their Gaia magnitudes provide an
    # additional ordering check before any hypothesis reaches the W08 fit.
    candidate_ids = source_ids[source_ids < min(500, len(stars))]
    if len(candidate_ids) < 3:
        return None
    rays = np.asarray([
        _camera_ray(
            camera_pixels[index, 0], camera_pixels[index, 1],
            image_width, image_height,
            sensor_width_mm, sensor_height_mm, focal_mm,
        )
        for index in candidate_ids
    ])
    anchor_tree = cKDTree(catalog_vectors)
    anchor_vectors = _world_unit_vectors(_ORION_BELT_J2000_DEG)
    anchor_distances, anchor_rows = anchor_tree.query(anchor_vectors, k=1)
    anchor_separations = np.degrees(2.0 * np.arcsin(np.clip(anchor_distances, 0.0, 1.0) * 0.5))
    if np.max(anchor_separations) * 3600.0 > 5.0:
        return None
    anchor_vectors = catalog_vectors[np.asarray(anchor_rows, dtype=np.intp)]
    anchor_magnitudes = catalog_magnitudes[np.asarray(anchor_rows, dtype=np.intp)]
    separation_ab, separation_bc, separation_ac = [
        math.degrees(math.acos(float(np.clip(np.dot(anchor_vectors[a], anchor_vectors[b]), -1.0, 1.0))))
        for a, b in ((0, 1), (1, 2), (0, 2))
    ]
    center_pixel_scale = math.degrees(math.atan2(
        sensor_height_mm / image_height, focal_mm
    )) * 3600.0
    angular_tolerance = float(np.clip(center_pixel_scale * 4.0 / 3600.0, 0.08, 0.18))
    pair_angles = np.degrees(np.arccos(np.clip(rays @ rays.T, -1.0, 1.0)))
    long_pairs = np.argwhere(np.triu(np.abs(pair_angles - separation_ac) <= angular_tolerance, 1))
    hypotheses: set[tuple[int, int, int]] = set()
    candidate_rank = {int(source_id): rank for rank, source_id in enumerate(candidate_ids)}
    for first, last in long_pairs:
        first, last = int(first), int(last)
        for mintaka_index, alnitak_index in ((first, last), (last, first)):
            alnilam_indices = np.flatnonzero(
                (np.abs(pair_angles[mintaka_index] - separation_ab) <= angular_tolerance)
                & (np.abs(pair_angles[alnitak_index] - separation_bc) <= angular_tolerance)
            )
            for alnilam_index in alnilam_indices:
                if alnilam_index in (mintaka_index, alnitak_index):
                    continue
                ids = (
                    int(candidate_ids[mintaka_index]),
                    int(candidate_ids[alnilam_index]),
                    int(candidate_ids[alnitak_index]),
                )
                # The catalog says Alnilam is the brightest of these three,
                # and Mintaka is the faintest. Allow rank noise from color and
                # foreground contrast, but reject faint random triangles.
                rank_a, rank_b, rank_c = (candidate_rank[index] for index in ids)
                if max(rank_a, rank_b, rank_c) > 200:
                    continue
                if rank_b > rank_a + 32 or rank_c > rank_a + 48 or rank_b > rank_c + 24:
                    continue
                hypotheses.add(ids)

    if not hypotheses:
        return None
    if progress:
        progress(53, f"检测到 {len(hypotheses)} 个猎户腰带几何候选，正在用 W08 星表交叉验证…")

    viable: list[tuple[float, tuple[int, int, int], np.ndarray]] = []
    for ids in hypotheses:
        observed = camera_pixels[np.asarray(ids, dtype=np.intp)]
        candidate_rays = np.asarray([
            _camera_ray(x, y, image_width, image_height,
                        sensor_width_mm, sensor_height_mm, focal_mm)
            for x, y in observed
        ])
        try:
            orientation = Rotation.align_vectors(anchor_vectors, candidate_rays)[0].as_matrix()
        except (ValueError, np.linalg.LinAlgError):
            continue
        candidate_projection = CameraProjection(
            orientation=orientation,
            focal_mm=focal_mm,
            center_x=image_width * 0.5,
            center_y=image_height * 0.5,
            residual_p90_px=math.inf,
        )
        predicted, z = _project_camera_catalog(
            anchor_vectors, candidate_projection, image_width, image_height,
            sensor_width_mm, sensor_height_mm,
        )
        anchor_errors = np.linalg.norm(predicted - observed, axis=1)
        if np.any(z <= 0.0) or not np.isfinite(anchor_errors).all():
            continue
        if float(np.max(anchor_errors)) > 2.5:
            continue
        viable.append((float(np.percentile(anchor_errors, 90)), ids, orientation))
    viable.sort(key=lambda item: (item[0], item[1]))

    for _anchor_error, ids, orientation0 in viable[:12]:
        initial_projection = CameraProjection(
            orientation=orientation0,
            focal_mm=focal_mm,
            center_x=image_width * 0.5,
            center_y=image_height * 0.5,
            residual_p90_px=math.inf,
        )
        initial_pairs = _camera_catalog_pairs(
            catalog_magnitudes, catalog_vectors, initial_projection,
            source_pixels, source_ids, source_tree,
            detector_shape, detector_scale_xy, sky_mask, 10.0, info,
        )
        if len([pair for pair in initial_pairs if pair[0] <= 8.0]) < 32:
            continue

        parameters = np.asarray([
            focal_mm, image_width * 0.5, image_height * 0.5, 0.0, 0.0, 0.0,
        ], dtype=np.float64)
        projection = initial_projection
        for maximum_fit_distance in (8.0, 6.0):
            pairs = _camera_catalog_pairs(
                catalog_magnitudes, catalog_vectors, projection,
                source_pixels, source_ids, source_tree,
                detector_shape, detector_scale_xy, sky_mask,
                maximum_fit_distance, info,
            )
            pairs = [pair for pair in pairs if pair[0] <= maximum_fit_distance]
            if len(pairs) < 24:
                break
            catalog_rows = np.asarray([pair[1] for pair in pairs], dtype=np.intp)
            source_rows = np.asarray([pair[2] for pair in pairs], dtype=np.intp)
            targets = camera_pixels[source_rows]
            vectors = catalog_vectors[catalog_rows]

            def residual(values: np.ndarray) -> np.ndarray:
                orientation = Rotation.from_rotvec(values[3:]).as_matrix() @ orientation0
                trial = CameraProjection(
                    orientation=orientation,
                    focal_mm=float(values[0]),
                    center_x=float(values[1]),
                    center_y=float(values[2]),
                    residual_p90_px=math.inf,
                )
                projected, _z = _project_camera_catalog(
                    vectors, trial, image_width, image_height,
                    sensor_width_mm, sensor_height_mm,
                )
                return (projected - targets).ravel()

            try:
                fit = least_squares(
                    residual,
                    parameters,
                    bounds=(
                        [focal_mm * 0.8, image_width * 0.47, image_height * 0.47, -0.2, -0.2, -0.2],
                        [focal_mm * 1.2, image_width * 0.53, image_height * 0.53, 0.2, 0.2, 0.2],
                    ),
                    loss="soft_l1",
                    f_scale=1.5,
                    max_nfev=300,
                )
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                break
            if not fit.success or not np.isfinite(fit.x).all():
                break
            parameters = fit.x
            orientation = Rotation.from_rotvec(parameters[3:]).as_matrix() @ orientation0
            fit_errors = np.linalg.norm(residual(parameters).reshape((-1, 2)), axis=1)
            inliers = fit_errors <= 6.0
            if not np.any(inliers):
                break
            residual_p90 = float(np.percentile(fit_errors[inliers], 90.0))
            projection = CameraProjection(
                orientation=orientation,
                focal_mm=float(parameters[0]),
                center_x=float(parameters[1]),
                center_y=float(parameters[2]),
                residual_p90_px=residual_p90,
            )

        final_pairs = _camera_catalog_pairs(
            catalog_magnitudes, catalog_vectors, projection,
            source_pixels, source_ids, source_tree,
            detector_shape, detector_scale_xy, sky_mask, 8.0, info,
        )
        close_pairs = [pair for pair in final_pairs if pair[0] <= 4.0]
        inliers = [pair for pair in final_pairs if pair[0] <= 6.0]
        if len(close_pairs) < 24 or len(inliers) < 40:
            continue
        residuals = np.asarray([pair[0] for pair in inliers], dtype=np.float64)
        if float(np.percentile(residuals, 90.0)) > 6.0:
            continue
        matched_xy = camera_pixels[np.asarray([pair[2] for pair in close_pairs], dtype=np.intp)]
        if float(np.ptp(matched_xy[:, 0])) < image_width * 0.35:
            continue
        if float(np.ptp(matched_xy[:, 1])) < image_height * 0.20:
            continue
        grid_x = np.clip((matched_xy[:, 0] / image_width * 4).astype(np.intp), 0, 3)
        grid_y = np.clip((matched_xy[:, 1] / image_height * 4).astype(np.intp), 0, 3)
        if len(set(zip(grid_x.tolist(), grid_y.tolist()))) < 6:
            continue
        projection = replace(
            projection,
            residual_p90_px=float(np.percentile(residuals, 90.0)),
        )
        return projection, len(close_pairs)
    return None


def _seiza_cache_directory() -> Path:
    if os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Caches"
    else:
        root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return root / "StarSoftFocus" / "seiza"


def _seiza_resources_in(
    directory: Path,
    *,
    catalog_name: str | None = None,
) -> tuple[Path, Path] | None:
    if not directory.is_dir():
        return None
    try:
        index = next(directory.rglob("blind-gaia16.idx"), None)
        if index is None:
            return None
        catalog_candidates = [
            path for path in directory.rglob("stars-*.bin")
            if path.is_file() and path.stat().st_size >= 64 * 1024 * 1024
        ]
        if catalog_name is not None:
            catalog = next(
                (path for path in catalog_candidates if path.name == catalog_name), None
            )
            return (catalog, index) if catalog is not None else None
        if not catalog_candidates:
            return None
        catalog = next(
            (path for preferred in ("stars-deep-gaia17.bin", "stars-gaia.bin")
             for path in catalog_candidates if path.name == preferred),
            max(catalog_candidates, key=lambda path: path.stat().st_size),
        )
        return catalog, index
    except OSError:
        return None


def _download_seiza_catalog_files(
    dataset_names: Sequence[str],
    cache_dir: Path,
    progress: Progress | None,
) -> dict[str, Path]:
    """Download verified Seiza v4 catalog objects with parallel HTTP ranges.

    Seiza's downloader is the normal catalog interface. This range-based
    fallback keeps first-run setup resumable at the HTTP layer when a large
    compressed response is interrupted by a proxy or CDN connection reset.
    The catalog manifest supplies the official artifact key, length, and hash.
    """
    manifest_dir = cache_dir / "manifests"
    manifest_path = manifest_dir / "catalog-bundle-v4.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        manifest_url = "https://downloads.seiza.fyi/data/v4/catalog-bundle-v4.json"
        request = Request(manifest_url, headers={"User-Agent": "StarSoftFocus/catalog-cache"})
        with urlopen(request, timeout=30) as response:
            manifest_bytes = response.read()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        manifest_dir.mkdir(parents=True, exist_ok=True)
        temporary_manifest = manifest_path.with_suffix(".json.partial")
        temporary_manifest.write_bytes(manifest_bytes)
        temporary_manifest.replace(manifest_path)

    entries = {
        str(item.get("name")): item
        for item in manifest.get("files", [])
        if isinstance(item, dict) and item.get("name")
    }
    results: dict[str, Path] = {}
    cache_partial = cache_dir / "partial"
    cache_partial.mkdir(parents=True, exist_ok=True)
    for dataset_name in dataset_names:
        item = entries.get(dataset_name)
        if item is None:
            raise RuntimeError(f"Seiza v4 manifest does not list {dataset_name}.")
        size = int(item["bytes"])
        digest = str(item["sha256"]).lower()
        artifact_key = str(item["key"])
        destination = cache_dir / "objects" / digest / dataset_name
        if destination.is_file() and destination.stat().st_size == size:
            results[dataset_name] = destination
            continue

        url = f"https://downloads.seiza.fyi/data/v4/{artifact_key}"
        chunk_size = 16 * 1024 * 1024
        byte_ranges = [
            (start, min(size - 1, start + chunk_size - 1))
            for start in range(0, size, chunk_size)
        ]
        with tempfile.TemporaryDirectory(
            prefix="starsoft-seiza-download-", dir=cache_partial
        ) as temporary_dir:
            staged_file = Path(temporary_dir) / dataset_name
            with staged_file.open("w+b") as output:
                output.truncate(size)

                def fetch_range(byte_range: tuple[int, int]) -> tuple[int, bytes]:
                    start, end = byte_range
                    expected_range = f"bytes {start}-{end}/{size}"
                    for attempt in range(4):
                        request = Request(
                            url,
                            headers={
                                "User-Agent": "StarSoftFocus/catalog-cache",
                                "Range": f"bytes={start}-{end}",
                            },
                        )
                        try:
                            with urlopen(request, timeout=90) as response:
                                if response.status != 206:
                                    raise RuntimeError(
                                        f"HTTP {response.status}; expected a byte-range response"
                                    )
                                content_range = response.headers.get("Content-Range", "")
                                if content_range != expected_range:
                                    raise RuntimeError(
                                        f"unexpected Content-Range {content_range!r}"
                                    )
                                payload = response.read()
                            if len(payload) != end - start + 1:
                                raise RuntimeError(
                                    f"short catalog range ({len(payload)} bytes)"
                                )
                            return start, payload
                        except Exception:
                            if attempt == 3:
                                raise
                            time.sleep(1.0 + attempt)
                    raise RuntimeError("catalog range retry limit reached")

                worker_count = min(8, max(1, os.cpu_count() or 1), len(byte_ranges))
                with ThreadPoolExecutor(
                    max_workers=worker_count, thread_name_prefix="seiza-catalog"
                ) as executor:
                    futures = [executor.submit(fetch_range, byte_range) for byte_range in byte_ranges]
                    completed_bytes = 0
                    for future in as_completed(futures):
                        start, payload = future.result()
                        output.seek(start)
                        output.write(payload)
                        completed_bytes += len(payload)
                        if progress:
                            percent = int(completed_bytes * 100 / max(size, 1))
                            progress(
                                53,
                                f"正在并行下载 Seiza {dataset_name}：{percent}%（"
                                f"{completed_bytes:,}/{size:,} 字节）…",
                            )

            computed_digest = hashlib.sha256()
            with staged_file.open("rb") as source:
                while chunk := source.read(8 * 1024 * 1024):
                    computed_digest.update(chunk)
            if computed_digest.hexdigest().lower() != digest:
                raise RuntimeError(f"SHA-256 verification failed for {dataset_name}.")
            destination.parent.mkdir(parents=True, exist_ok=True)
            staged_file.replace(destination)
            results[dataset_name] = destination
    return results


def _load_seiza_solver_data(
    progress: Progress | None,
    *,
    allow_download: bool,
    prefer_deep: bool = False,
) -> tuple[object, object] | None:
    """Load local Seiza Gaia data, downloading the verified catalog only on demand."""
    global _SEIZA_DATA, _SEIZA_DEEP_DATA
    with _SEIZA_DATA_LOCK:
        if prefer_deep and _SEIZA_DEEP_DATA is not None:
            return _SEIZA_DEEP_DATA
        if not prefer_deep and _SEIZA_DATA is not None:
            return _SEIZA_DATA
        try:
            import seiza
        except ImportError:
            return None

        search_roots: list[Path] = []
        for variable in ("STARSOFT_SEIZA_DATA", "SEIZA_STAR_DATA"):
            value = os.environ.get(variable)
            if value:
                search_roots.append(Path(value).expanduser())
        executable_dir = Path(sys.executable).resolve().parent
        search_roots.extend((executable_dir / "seiza-data", Path(__file__).resolve().parent / "seiza-data"))
        cache_dir = _seiza_cache_directory()
        search_roots.append(cache_dir)
        requested_catalog = "stars-deep-gaia17.bin" if prefer_deep else None
        resources = next((
            found for root in search_roots
            if (found := _seiza_resources_in(root, catalog_name=requested_catalog)) is not None
        ), None)
        if resources is None:
            if not allow_download:
                return None
            if progress:
                progress(
                    53,
                    "正在下载并校验 Seiza 本机 Gaia 盲解索引与"
                    + ("深度 G≤17 星表…" if prefer_deep else "轻量 G≤15 星表…"),
                )
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                catalog_name = requested_catalog or "stars-gaia.bin"
                paths = _download_seiza_catalog_files(
                    [catalog_name, "blind-gaia16.idx"],
                    cache_dir,
                    progress,
                )
                resources = (
                    Path(paths[catalog_name]),
                    Path(paths["blind-gaia16.idx"]),
                )
            except Exception as error:
                if progress:
                    progress(54, f"Seiza Gaia 数据下载或校验失败：{type(error).__name__}")
                return None
        try:
            catalog = seiza.StarCatalog.open(str(resources[0]))
            index = seiza.BlindIndex.open(str(resources[1]))
        except Exception:
            return None
        loaded = (catalog, index)
        if prefer_deep:
            _SEIZA_DEEP_DATA = loaded
        else:
            _SEIZA_DATA = loaded
        return loaded


def _seiza_wcs_on_solver_grid(
    seiza_wcs: object,
    detector_scale_xy: tuple[float, float],
    detector_shape: tuple[int, int],
) -> WCS:
    """Convert Seiza's top-left detector WCS to ASTAP's bottom-left solver grid."""
    header = fits.Header()
    for key, value in seiza_wcs.fits_header_cards().items():
        header[str(key)] = value
    detector_wcs = WCS(header, relax=True)
    scale_x, scale_y = detector_scale_xy
    matrix = np.asarray(detector_wcs.pixel_scale_matrix, dtype=np.float64)
    solver_wcs = WCS(naxis=2)
    solver_wcs.wcs.ctype = detector_wcs.wcs.ctype
    solver_wcs.wcs.cunit = detector_wcs.wcs.cunit
    solver_wcs.wcs.crval = detector_wcs.wcs.crval
    detector_height = max(int(detector_shape[0]), 1)
    solver_wcs.wcs.crpix = np.asarray([
        (detector_wcs.wcs.crpix[0] - 1.0) * scale_x + 1.0,
        (detector_height - detector_wcs.wcs.crpix[1] + 1.0) * scale_y,
    ])
    solver_wcs.wcs.cd = matrix @ np.diag([1.0 / scale_x, -1.0 / scale_y])
    solver_wcs.wcs.equinox = detector_wcs.wcs.equinox
    if detector_wcs.wcs.radesys:
        solver_wcs.wcs.radesys = detector_wcs.wcs.radesys
    solver_wcs.wcs.set()
    return solver_wcs


def _seiza_catalog_confirmation(
    solution: object,
    stars: Sequence[object],
    detector: np.ndarray,
    sky_mask: np.ndarray | None,
    detector_shape: tuple[int, int],
    catalog_magnitudes: np.ndarray,
    catalog_vectors: np.ndarray,
) -> tuple[int, float, float, float] | None:
    """Require unique, spatially distributed W08 matches before accepting a blind solve."""
    if int(getattr(solution, "matched_stars", 0)) < 10:
        return None
    valid_indices: list[int] = []
    source_pixels: list[tuple[float, float]] = []
    height, width = detector_shape
    for index, star in enumerate(stars[:2000]):
        x = float(getattr(star, "x"))
        y = float(getattr(star, "y"))
        if not (math.isfinite(x) and math.isfinite(y) and 0 <= x < width and 0 <= y < height):
            continue
        if sky_mask is not None and sky_mask.shape == detector.shape and sky_mask[int(y), int(x)]:
            continue
        valid_indices.append(index)
        source_pixels.append((x, y))
    if len(valid_indices) < 80:
        return None

    try:
        world = np.asarray([
            solution.wcs.pixel_to_world(x, y) for x, y in source_pixels
        ], dtype=np.float64)
        finite = (
            np.isfinite(world).all(axis=1)
            & (world[:, 1] >= -90.0) & (world[:, 1] <= 90.0)
        )
        if np.count_nonzero(finite) < 40:
            return None
        world = world[finite]
        source_pixels_array = np.asarray(source_pixels, dtype=np.float64)[finite]
        source_vectors = _world_unit_vectors(world)
    except Exception:
        return None

    tree = cKDTree(catalog_vectors)
    distances, catalog_rows = tree.query(source_vectors, k=1)
    separations = np.degrees(2.0 * np.arcsin(np.clip(distances * 0.5, 0.0, 1.0))) * 3600.0
    scale = max(float(solution.scale_arcsec_px), 1e-6)
    match_radius_arcsec = float(np.clip(scale * 1.25, 45.0, 180.0))
    pairs = [
        (float(separation), index, int(row))
        for index, (separation, row) in enumerate(zip(separations, catalog_rows))
        if int(row) < len(catalog_magnitudes)
        and float(catalog_magnitudes[int(row)]) <= 8.0
        and float(separation) <= match_radius_arcsec
    ]
    used_sources: set[int] = set()
    used_catalog: set[int] = set()
    accepted: list[tuple[float, int]] = []
    for separation, source_index, catalog_index in sorted(pairs):
        if source_index in used_sources or catalog_index in used_catalog:
            continue
        used_sources.add(source_index)
        used_catalog.add(catalog_index)
        accepted.append((separation, source_index))
    if len(accepted) < 12:
        return None

    residuals = np.asarray([item[0] for item in accepted], dtype=np.float64)
    p50, p90 = np.percentile(residuals, [50.0, 90.0])
    matched_xy = source_pixels_array[np.asarray([item[1] for item in accepted], dtype=np.intp)]
    if p50 > scale or p90 > scale * 1.25:
        return None
    if float(np.ptp(matched_xy[:, 0])) < detector.shape[1] * 0.20:
        return None
    if float(np.ptp(matched_xy[:, 1])) < detector.shape[0] * 0.10:
        return None
    return len(accepted), float(p50), float(p90), scale


def _wide_focal_length_bounds(info: object, fallback_focal_mm: float) -> tuple[float, float]:
    lens = " ".join(str(getattr(info, "lens", "")).split())
    match = re.search(r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*mm\b", lens, re.IGNORECASE)
    if match:
        first, second = float(match.group(1)), float(match.group(2))
        if 1.0 <= min(first, second) <= max(first, second) <= 1000.0:
            return min(first, second), max(first, second)
    return fallback_focal_mm, fallback_focal_mm


def _try_seiza_fallback_tile(
    stars: Sequence[object],
    detector: np.ndarray,
    sky_mask: np.ndarray | None,
    info: object,
    catalogs: Path,
    detector_scale_xy: tuple[float, float],
    solver_shape: tuple[int, int],
    sensor_width_mm: float,
    focal_mm: float,
    progress: Progress | None,
    *,
    seiza_data: tuple[object, object] | None = None,
    allow_download: bool,
) -> tuple[SolverTile | None, str | None]:
    try:
        if seiza_data is None:
            seiza_data = _load_seiza_solver_data(progress, allow_download=allow_download)
        if seiza_data is None:
            return None, "未取得本机 Seiza Gaia 盲解数据。"
        import seiza

        catalog_path = next(iter(sorted(catalogs.glob("w08_*.001"))), None)
        if catalog_path is None:
            return None, "程序包缺少 W08 亮星索引，无法独立验证盲解结果。"
        catalog_magnitudes, catalog_vectors = _load_w08_catalog(str(catalog_path.resolve()))
        detector_height, detector_width = detector.shape
        solver_height, solver_width = solver_shape
        source_stars: list[object] = []
        for star in stars[:2000]:
            x, y, flux = (float(getattr(star, key)) for key in ("x", "y", "flux"))
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(flux) and flux > 0):
                continue
            if not (0 <= x < detector_width and 0 <= y < detector_height):
                continue
            if sky_mask is not None and sky_mask.shape == detector.shape and sky_mask[int(y), int(x)]:
                continue
            source_stars.append(star)
        if len(source_stars) < 80:
            return None, f"Seiza 盲解可用天空点源只有 {len(source_stars)} 颗。"

        minimum_focal, maximum_focal = _wide_focal_length_bounds(info, focal_mm)
        detector_scales = [
            206264.806247 * sensor_width_mm
            / (max(float(focal), 1e-6) * max(detector_width, 1))
            for focal in (minimum_focal, maximum_focal)
        ]
        minimum_scale = max(0.1, min(detector_scales) * 0.65)
        maximum_scale = max(detector_scales) * 1.45
        if progress:
            progress(53, "正在用本机 Seiza/Gaia 盲解，并以 W08 亮星作独立复核…")
        solve_counts = list(dict.fromkeys((
            min(1200, len(source_stars)),
            min(700, len(source_stars)),
            min(2000, len(source_stars)),
        )))
        last_error: Exception | None = None
        for source_count in solve_counts:
            if source_count < 80:
                continue
            try:
                solution = seiza.solve_blind(
                    [
                        (float(star.x), float(star.y), float(star.flux))
                        for star in source_stars[:source_count]
                    ],
                    seiza_data[0],
                    seiza_data[1],
                    detector_width,
                    detector_height,
                    min_scale_arcsec_px=minimum_scale,
                    max_scale_arcsec_px=maximum_scale,
                    sip_order=2,
                )
                confirmation = _seiza_catalog_confirmation(
                    solution,
                    source_stars[:source_count],
                    detector,
                    sky_mask,
                    detector.shape,
                    catalog_magnitudes,
                    catalog_vectors,
                )
                if confirmation is None:
                    continue
                match_count, p50, p90, detector_pixel_scale = confirmation
                solver_wcs = _seiza_wcs_on_solver_grid(
                    solution.wcs, detector_scale_xy, detector.shape
                )
                solver_matrix = np.asarray(solver_wcs.pixel_scale_matrix, dtype=np.float64)
                solver_pixel_scale = math.sqrt(abs(float(np.linalg.det(solver_matrix)))) * 3600.0
                tile = SolverTile(
                    0, 0, solver_width, solver_height,
                    solver_wcs, solver_pixel_scale, True,
                    match_count, match_count, False,
                )
                message = (
                    f"Seiza 全幅盲解 {solution.matched_stars} 星，尺度 "
                    f"{detector_pixel_scale:.1f}″/检测像素；W08 一对一确认 "
                    f"{match_count} 星，P50/P90 {p50:.0f}/{p90:.0f}″。"
                )
                return tile, message
            except Exception as error:
                last_error = error
        if last_error is not None:
            return None, f"Seiza 盲解失败：{type(last_error).__name__}: {last_error}"
        return None, "Seiza 盲解候选未通过 W08 一对一匹配、残差或空间分布复核。"
    except Exception as error:
        return None, f"Seiza Gaia 备用解算不可用：{type(error).__name__}: {error}"


def _try_seiza_local_camera_fit(
    stars: Sequence[object],
    detector: np.ndarray,
    sky_mask: np.ndarray | None,
    info: object,
    catalogs: Path,
    detector_scale_xy: tuple[float, float],
    solver_shape: tuple[int, int],
    sensor_width_mm: float,
    sensor_height_mm: float,
    focal_mm: float,
    progress: Progress | None,
    *,
    seiza_data: tuple[object, object],
) -> SeizaLocalSolveResult | None:
    """Blind-solve and retain independently W08-verified wide-field patches.

    Wide lenses can prevent whole-frame asterism solvers from finding a stable
    pattern. This fallback blind-solves overlapping local patches with Seiza,
    cross-checks each solution against the packaged W08/Gaia catalogue, and
    retains each local WCS that has enough accurate, spatially distributed
    matches. A full-frame camera projection is an optional separate result.
    """
    try:
        import seiza

        catalog_path = next(iter(sorted(catalogs.glob("w08_*.001"))), None)
        if catalog_path is None:
            return None
        catalog_magnitudes, catalog_vectors = _load_w08_catalog(str(catalog_path.resolve()))
        detector_height, detector_width = detector.shape
        solver_height, solver_width = solver_shape
        scale_x, scale_y = detector_scale_xy
        minimum_focal, maximum_focal = _wide_focal_length_bounds(info, focal_mm)
        detector_scales = [
            206264.806247 * sensor_width_mm
            / (max(float(focal), 1e-6) * max(detector_width, 1))
            for focal in (minimum_focal, maximum_focal)
        ]
        minimum_scale = max(0.1, min(detector_scales) * 0.65)
        maximum_scale = max(detector_scales) * 1.45

        # Keep the strongest full-frame SEP sources ready for independent
        # validation after a local Seiza patch has supplied a celestial pose.
        source_pixels: list[tuple[float, float]] = []
        source_ids: list[int] = []
        for index, star in enumerate(stars):
            x, y = float(getattr(star, "x")), float(getattr(star, "y"))
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            if not (0.0 <= x < detector_width and 0.0 <= y < detector_height):
                continue
            detector_x, detector_y = int(round(x)), int(round(y))
            detector_x = min(max(detector_x, 0), detector_width - 1)
            detector_y = min(max(detector_y, 0), detector_height - 1)
            if sky_mask is not None and sky_mask.shape == detector.shape and sky_mask[detector_y, detector_x]:
                continue
            source_ids.append(index)
            source_pixels.append(((x + 0.5) * scale_x - 0.5,
                                  (y + 0.5) * scale_y - 0.5))
        if len(source_ids) < 80:
            return None
        source_ids_array = np.asarray(source_ids, dtype=np.intp)
        source_pixels_array = np.asarray(source_pixels, dtype=np.float64)
        source_tree = cKDTree(source_pixels_array)

        # Generate half-overlapping patches at 50, 40, then 32 degree vertical fields.
        # Prioritize center/sky-rich patches, but retain all image quadrants so
        # a forest or mountain mask cannot leave the solver stuck at the center.
        candidates: list[tuple[float, float, tuple[int, int, int, int], float]] = []
        seen_boxes: set[tuple[int, int, int, int]] = set()
        for target_fov in (50.0, 40.0, 32.0):
            half_angle = math.radians(target_fov) * 0.5
            height_fraction = min(
                1.0,
                math.tan(half_angle)
                / max(math.tan(math.radians(_axis_fov_degrees(
                    0, solver_height, solver_height, sensor_height_mm, focal_mm
                )) * 0.5), 1e-8),
            )
            # Preserve the detector's aspect ratio. Equal horizontal and
            # vertical angular spans would make nearly square crops, discarding
            # many useful stars on common 3:2 camera sensors.
            width_fraction = height_fraction
            box_width = min(detector_width, max(900, int(round(detector_width * width_fraction))))
            box_height = min(detector_height, max(700, int(round(detector_height * height_fraction))))

            def axis_starts(length: int, box_size: int) -> list[int]:
                if length <= box_size:
                    return [0]
                last = length - box_size
                stride = max(1, int(box_size * 0.5))
                return sorted(set([0, last, *range(0, last + 1, stride)]))

            for y0 in axis_starts(detector_height, box_height):
                for x0 in axis_starts(detector_width, box_width):
                    box = (x0, y0, x0 + box_width, y0 + box_height)
                    if box in seen_boxes:
                        continue
                    seen_boxes.add(box)
                    x1, y1 = box[2], box[3]
                    inside = (
                        (source_pixels_array[:, 0] >= x0 * scale_x)
                        & (source_pixels_array[:, 0] < x1 * scale_x)
                        & (source_pixels_array[:, 1] >= y0 * scale_y)
                        & (source_pixels_array[:, 1] < y1 * scale_y)
                    )
                    detected_count = int(np.count_nonzero(inside))
                    if detected_count < 24:
                        continue
                    if sky_mask is not None and sky_mask.shape == detector.shape:
                        mask_crop = np.asarray(sky_mask[y0:y1, x0:x1], dtype=bool)
                        sky_fraction = float(np.mean(~mask_crop))
                    else:
                        sky_fraction = 1.0
                    if sky_fraction < 0.12:
                        continue
                    center_distance = math.hypot(
                        (x0 + x1) * 0.5 - detector_width * 0.5,
                        (y0 + y1) * 0.5 - detector_height * 0.5,
                    ) / max(math.hypot(detector_width, detector_height), 1.0)
                    score = detected_count * math.sqrt(max(sky_fraction, 0.01))
                    candidates.append((score, center_distance, box, target_fov))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (-item[0], item[1], item[3]))
        # Spread a bounded set of attempts across the frame. Solving only the
        # highest-density crop tends to leave wide-field corners without a
        # local WCS even when the central crop yields a valid camera pose.
        spread_candidates: list[tuple[float, float, tuple[int, int, int, int], float]] = []
        remaining = list(candidates)
        image_diagonal = max(math.hypot(detector_width, detector_height), 1.0)
        while remaining and len(spread_candidates) < 24:
            if not spread_candidates:
                selected = remaining[0]
            else:
                selected = max(
                    remaining,
                    key=lambda item: item[0] * (
                        0.35 + 0.65 * min(
                            math.hypot(
                                (item[2][0] + item[2][2]) * 0.5 - (prior[2][0] + prior[2][2]) * 0.5,
                                (item[2][1] + item[2][3]) * 0.5 - (prior[2][1] + prior[2][3]) * 0.5,
                            ) / image_diagonal
                            for prior in spread_candidates
                        )
                    ),
                )
            spread_candidates.append(selected)
            remaining.remove(selected)
        candidates = spread_candidates
        if progress:
            progress(54, f"全幅盲解未收敛，正在用 Seiza 分区识别广角星点（{len(candidates)} 个候选区域）…")

        tree = cKDTree(catalog_vectors)
        attempts = 0
        last_error: Exception | None = None
        verified_tiles: list[SolverTile] = []
        best_projection: CameraProjection | None = None
        best_projection_match_count = 0
        sky_cells: set[tuple[int, int]] = set()
        for grid_y in range(3):
            for grid_x in range(4):
                x0 = int(round(detector_width * grid_x / 4))
                x1 = int(round(detector_width * (grid_x + 1) / 4))
                y0 = int(round(detector_height * grid_y / 3))
                y1 = int(round(detector_height * (grid_y + 1) / 3))
                if sky_mask is None or sky_mask.shape != detector.shape:
                    sky_cells.add((grid_x, grid_y))
                elif float(np.mean(~np.asarray(sky_mask[y0:y1, x0:x1], dtype=bool))) >= 0.12:
                    sky_cells.add((grid_x, grid_y))
        for _score, _center_distance, (x0, y0, x1, y1), _target_fov in candidates:
            attempts += 1
            if attempts > 24 or len(verified_tiles) >= 8:
                break
            crop = np.asarray(detector[y0:y1, x0:x1], dtype=np.float32).copy()
            crop_mask: np.ndarray | None = None
            if sky_mask is not None and sky_mask.shape == detector.shape:
                crop_mask = np.asarray(sky_mask[y0:y1, x0:x1], dtype=bool)
                valid = ~crop_mask & np.isfinite(crop)
                if not np.any(valid):
                    continue
                crop[~valid] = float(np.median(crop[valid]))
            crop = np.nan_to_num(crop, nan=0.0, posinf=1.0, neginf=0.0)
            try:
                # Seiza's ranked 500-star set was more robust than flooding its
                # blind matcher with thousands of faint/noisy detections in a
                # wide, unevenly illuminated field.
                local_sources = seiza.detect(crop, max_stars=500)
            except Exception:
                continue
            solve_sources: list[_SolveSource] = []
            for source in local_sources:
                try:
                    if hasattr(source, "x") and hasattr(source, "y") and hasattr(source, "flux"):
                        sx = float(source.x)
                        sy = float(source.y)
                        flux = float(source.flux)
                    else:
                        sx, sy, flux = map(float, source[:3])
                except (IndexError, TypeError, ValueError, AttributeError):
                    continue
                if math.isfinite(sx) and math.isfinite(sy) and math.isfinite(flux) and flux > 0:
                    solve_sources.append(_SolveSource(sx, sy, flux))
            if len(solve_sources) < 40:
                continue
            solve_sources.sort(key=lambda source: source.flux, reverse=True)

            for source_count in dict.fromkeys((
                min(700, len(solve_sources)),
                min(1200, len(solve_sources)),
                len(solve_sources),
            )):
                if source_count < 40:
                    continue
                chosen_sources = solve_sources[:source_count]
                try:
                    solution = seiza.solve_blind(
                        [(source.x, source.y, source.flux) for source in chosen_sources],
                        seiza_data[0],
                        seiza_data[1],
                        crop.shape[1],
                        crop.shape[0],
                        min_scale_arcsec_px=minimum_scale,
                        max_scale_arcsec_px=maximum_scale,
                        sip_order=2,
                    )
                except Exception as error:
                    last_error = error
                    continue

                try:
                    # Match each local WCS prediction to a unique Gaia/W08
                    # bright star and retain source indices in the full SEP list.
                    crop_sep_indices: list[int] = []
                    crop_detector_xy: list[tuple[float, float]] = []
                    for index, star in enumerate(stars):
                        sx, sy = float(getattr(star, "x")), float(getattr(star, "y"))
                        if not (x0 <= sx < x1 and y0 <= sy < y1):
                            continue
                        ix, iy = int(round(sx)), int(round(sy))
                        ix, iy = min(max(ix, 0), detector_width - 1), min(max(iy, 0), detector_height - 1)
                        if sky_mask is not None and sky_mask.shape == detector.shape and sky_mask[iy, ix]:
                            continue
                        crop_sep_indices.append(index)
                        crop_detector_xy.append((sx - x0, sy - y0))
                    if len(crop_sep_indices) < 20:
                        continue
                    local_world = np.asarray([
                        solution.wcs.pixel_to_world(sx, sy) for sx, sy in crop_detector_xy
                    ], dtype=np.float64)
                    valid_world = (
                        np.isfinite(local_world).all(axis=1)
                        & (local_world[:, 1] >= -90.0)
                        & (local_world[:, 1] <= 90.0)
                    )
                    if np.count_nonzero(valid_world) < 20:
                        continue
                    local_vectors = _world_unit_vectors(local_world[valid_world])
                    local_sep_ids = np.asarray(crop_sep_indices, dtype=np.intp)[valid_world]
                    distances, catalog_rows = tree.query(local_vectors, k=1)
                    separations = np.degrees(
                        2.0 * np.arcsin(np.clip(distances * 0.5, 0.0, 1.0))
                    ) * 3600.0
                    match_radius = float(np.clip(solution.scale_arcsec_px * 1.25, 45.0, 180.0))
                    raw_pairs = [
                        (float(separation), int(source_id), int(row))
                        for separation, source_id, row in zip(separations, local_sep_ids, catalog_rows)
                        if int(row) < len(catalog_magnitudes)
                        and catalog_magnitudes[int(row)] <= 8.0
                        and float(separation) <= match_radius
                    ]
                    used_source_ids: set[int] = set()
                    used_catalog_rows: set[int] = set()
                    accepted: dict[int, CatalogMatch] = {}
                    for separation, source_id, row in sorted(raw_pairs):
                        if source_id in used_source_ids or row in used_catalog_rows:
                            continue
                        used_source_ids.add(source_id)
                        used_catalog_rows.add(row)
                        ra = math.degrees(math.atan2(catalog_vectors[row, 1], catalog_vectors[row, 0])) % 360.0
                        dec = math.degrees(math.asin(float(np.clip(catalog_vectors[row, 2], -1.0, 1.0))))
                        accepted[source_id] = CatalogMatch(
                            source_id=None,
                            ra_deg=ra,
                            dec_deg=dec,
                            g_mag=float(catalog_magnitudes[row]),
                            bp_mag=None,
                            rp_mag=None,
                            separation_arcsec=separation,
                        )
                    if len(accepted) < 20:
                        continue

                    local_residuals = np.asarray(
                        [match.separation_arcsec for match in accepted.values()],
                        dtype=np.float64,
                    )
                    local_match_p50 = float(np.percentile(local_residuals, 50.0))
                    local_match_p90 = float(np.percentile(local_residuals, 90.0))
                    matched_detector_xy = np.asarray([
                        [float(getattr(stars[source_id], "x")), float(getattr(stars[source_id], "y"))]
                        for source_id in accepted
                    ], dtype=np.float64)
                    spread_x = float(np.ptp(matched_detector_xy[:, 0])) / max(x1 - x0, 1)
                    spread_y = float(np.ptp(matched_detector_xy[:, 1])) / max(y1 - y0, 1)
                    local_grid_x = np.clip(
                        ((matched_detector_xy[:, 0] - x0) / max(x1 - x0, 1) * 3).astype(np.intp),
                        0, 2,
                    )
                    local_grid_y = np.clip(
                        ((matched_detector_xy[:, 1] - y0) / max(y1 - y0, 1) * 3).astype(np.intp),
                        0, 2,
                    )
                    occupied_local_cells = len(set(zip(local_grid_x.tolist(), local_grid_y.tolist())))
                    local_tile_is_verified = (
                        local_match_p50 <= max(20.0, solution.scale_arcsec_px * 0.6)
                        and local_match_p90 <= max(45.0, solution.scale_arcsec_px * 1.5)
                        and spread_x >= 0.10
                        and spread_y >= 0.10
                        and occupied_local_cells >= 3
                    )

                    tile_wcs = _seiza_wcs_on_solver_grid(
                        solution.wcs, detector_scale_xy, (y1 - y0, x1 - x0)
                    )
                    local_solver_width = max(1, int(round((x1 - x0) * scale_x)))
                    local_solver_height = max(1, int(round((y1 - y0) * scale_y)))
                    tile_x0 = int(round(x0 * scale_x))
                    tile_y0 = int(round(y0 * scale_y))
                    matrix = np.asarray(tile_wcs.pixel_scale_matrix, dtype=np.float64)
                    actual_scale = math.sqrt(abs(float(np.linalg.det(matrix)))) * 3600.0
                    tile = SolverTile(
                        tile_x0, tile_y0,
                        min(solver_width, tile_x0 + local_solver_width),
                        min(solver_height, tile_y0 + local_solver_height),
                        tile_wcs,
                        actual_scale,
                        True,
                        len(accepted),
                        len(accepted),
                        False,
                    )
                    tile_area = max((tile.x1 - tile.x0) * (tile.y1 - tile.y0), 1)
                    overlaps_existing = False
                    for previous in verified_tiles:
                        ix = max(0, min(tile.x1, previous.x1) - max(tile.x0, previous.x0))
                        iy = max(0, min(tile.y1, previous.y1) - max(tile.y0, previous.y0))
                        intersection = ix * iy
                        previous_area = max(
                            (previous.x1 - previous.x0) * (previous.y1 - previous.y0), 1
                        )
                        if intersection / min(tile_area, previous_area) >= 0.78:
                            overlaps_existing = True
                            break
                    if local_tile_is_verified and not overlaps_existing:
                        verified_tiles.append(tile)

                    projection = _fit_wide_camera_projection(
                        stars,
                        accepted,
                        [tile],
                        detector.shape,
                        detector_scale_xy,
                        info,
                    )
                    if projection is not None:
                        pairs = _camera_catalog_pairs(
                            catalog_magnitudes,
                            catalog_vectors,
                            projection,
                            source_pixels_array,
                            source_ids_array,
                            source_tree,
                            detector.shape,
                            detector_scale_xy,
                            sky_mask,
                            8.0,
                            info,
                        )
                        close_pairs = [pair for pair in pairs if pair[0] <= 4.0]
                        inliers = [pair for pair in pairs if pair[0] <= 6.0]
                        if len(close_pairs) >= 24 and len(inliers) >= 40:
                            residuals = np.asarray([pair[0] for pair in inliers], dtype=np.float64)
                            matched_xy = source_pixels_array[
                                np.asarray([pair[2] for pair in close_pairs], dtype=np.intp)
                            ]
                            grid_x = np.clip((matched_xy[:, 0] / solver_width * 4).astype(np.intp), 0, 3)
                            grid_y = np.clip((matched_xy[:, 1] / solver_height * 4).astype(np.intp), 0, 3)
                            spatially_distributed = (
                                float(np.percentile(residuals, 90.0)) <= 6.0
                                and float(np.ptp(matched_xy[:, 0])) >= solver_width * 0.35
                                and float(np.ptp(matched_xy[:, 1])) >= solver_height * 0.20
                                and len(set(zip(grid_x.tolist(), grid_y.tolist()))) >= 6
                            )
                            if spatially_distributed:
                                projection = replace(
                                    projection,
                                    residual_p90_px=float(np.percentile(residuals, 90.0)),
                                )
                                if (
                                    best_projection is None
                                    or len(close_pairs) > best_projection_match_count
                                ):
                                    best_projection = projection
                                    best_projection_match_count = len(close_pairs)
                                if progress:
                                    progress(
                                        55,
                                        f"Seiza 局部解提出的全幅模型通过分布式 W08/Gaia 校验："
                                        f"{len(close_pairs)} 个匹配，P90 {projection.residual_p90_px:.2f} solver px；"
                                        f"已保留 {len(verified_tiles)} 个局部复核图块。",
                                    )

                    # Each crop has already passed a one-to-one W08 match,
                    # residual, and spatial-spread check. Keep it even when a
                    # rectilinear full-frame extrapolation is not trustworthy.
                    break
                except Exception as error:
                    last_error = error
                    continue

            if verified_tiles and sky_cells:
                covered_cells: set[tuple[int, int]] = set()
                for tile in verified_tiles:
                    left = tile.x0 / max(scale_x, 1e-8)
                    top = tile.y0 / max(scale_y, 1e-8)
                    right = tile.x1 / max(scale_x, 1e-8)
                    bottom = tile.y1 / max(scale_y, 1e-8)
                    for grid_x, grid_y in sky_cells:
                        cell_left = detector_width * grid_x / 4
                        cell_right = detector_width * (grid_x + 1) / 4
                        cell_top = detector_height * grid_y / 3
                        cell_bottom = detector_height * (grid_y + 1) / 3
                        overlap_x = max(0.0, min(right, cell_right) - max(left, cell_left))
                        overlap_y = max(0.0, min(bottom, cell_bottom) - max(top, cell_top))
                        if overlap_x * overlap_y >= (cell_right - cell_left) * (cell_bottom - cell_top) * 0.55:
                            covered_cells.add((grid_x, grid_y))
                if len(covered_cells) >= math.ceil(len(sky_cells) * 0.85):
                    break

        if verified_tiles or best_projection is not None:
            if progress:
                if best_projection is not None:
                    progress(
                        55,
                        f"Seiza 已取得 {len(verified_tiles)} 个局部 W08 复核图块；"
                        f"全幅姿态通过 {best_projection_match_count} 个分布式匹配。",
                    )
                else:
                    progress(
                        55,
                        f"Seiza 仅确认了 {len(verified_tiles)} 个局部 WCS 图块；"
                        "其余区域仍标为未解算，不作全幅推算。",
                    )
            return SeizaLocalSolveResult(
                best_projection,
                best_projection_match_count,
                tuple(verified_tiles),
            )
        if progress and last_error is not None:
            progress(55, f"Seiza 局部星表解未通过 W08 局部复核：{type(last_error).__name__}")
        return None
    except Exception as error:
        if progress:
            progress(55, f"Seiza 广角局部盲解不可用：{type(error).__name__}")
        return None


def _positions_from_solved_tiles(
    stars: Sequence[object],
    successful_tiles: Sequence[SolverTile],
    detector_scale_xy: tuple[float, float],
) -> list[tuple[int, float, float]]:
    scale_x, scale_y = detector_scale_xy
    source_positions: list[tuple[int, float, float]] = []
    for index, star in enumerate(stars):
        x = float(getattr(star, "x")) * scale_x
        y = float(getattr(star, "y")) * scale_y
        valid_tiles = [
            tile for tile in successful_tiles
            if tile.x0 <= x < tile.x1 and tile.y0 <= y < tile.y1
        ]
        if not valid_tiles:
            continue
        tile = min(
            valid_tiles,
            key=lambda item: ((x - (item.x0 + item.x1) / 2.0) / max(item.x1 - item.x0, 1)) ** 2
                             + ((y - (item.y0 + item.y1) / 2.0) / max(item.y1 - item.y0, 1)) ** 2,
        )
        try:
            sky = _tile_pixel_to_world(tile, np.asarray([[x - tile.x0, y - tile.y0]]))[0]
            ra, dec = float(sky[0]) % 360.0, float(sky[1])
        except Exception:
            continue
        if math.isfinite(ra) and math.isfinite(dec) and -90.0 <= dec <= 90.0:
            source_positions.append((index, ra, dec))
    return source_positions


def _axis_fov_degrees(
    start: int,
    end: int,
    image_size: int,
    sensor_size_mm: float,
    focal_mm: float,
) -> float:
    center = image_size / 2.0
    mm_per_pixel = sensor_size_mm / image_size
    left = math.atan(((start - center) * mm_per_pixel) / focal_mm)
    right = math.atan(((end - center) * mm_per_pixel) / focal_mm)
    return math.degrees(abs(right - left))


def _local_box_fov_degrees(
    box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    sensor_width_mm: float,
    sensor_height_mm: float,
    focal_mm: float,
) -> tuple[float, float]:
    """Estimate the actual angular spans through an off-axis crop's centre."""
    x0, y0, x1, y1 = box
    center_x, center_y = (x0 + x1) * 0.5, (y0 + y1) * 0.5

    def ray(x: float, y: float) -> np.ndarray:
        plane_x = (x - image_width * 0.5) * sensor_width_mm / max(image_width, 1)
        plane_y = (y - image_height * 0.5) * sensor_height_mm / max(image_height, 1)
        return np.asarray([plane_x, plane_y, focal_mm], dtype=np.float64)

    def angle(first: np.ndarray, second: np.ndarray) -> float:
        cosine = float(np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second)))
        return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))

    horizontal = angle(ray(x0, center_y), ray(x1, center_y))
    vertical = angle(ray(center_x, y0), ray(center_x, y1))
    return horizontal, vertical


def _extract_astap_wcs(
    path: Path, expected_scale_arcsec: float, *, allow_inaccurate_scale: bool = False
) -> WCS | None:
    ini_path = path.with_suffix(".ini")
    wcs_path = path.with_suffix(".wcs")
    if not ini_path.exists() or not wcs_path.exists():
        return None
    ini_text = ini_path.read_text(encoding="utf-8", errors="replace")
    if "PLTSOLVD=T" not in ini_text:
        return None
    scale_was_inaccurate = "scale was inaccurate" in ini_text.lower()
    if scale_was_inaccurate and not allow_inaccurate_scale:
        return None
    try:
        header = fits.Header.fromfile(wcs_path, endcard=False, padding=False)
        wcs = WCS(header, relax=True)
        if not wcs.has_celestial or wcs.pixel_n_dim < 2:
            return None
        matrix = np.asarray(wcs.pixel_scale_matrix, dtype=np.float64)
        actual_scale = math.sqrt(abs(float(np.linalg.det(matrix)))) * 3600.0
    except Exception:
        return None
    if not math.isfinite(actual_scale) or actual_scale <= 0:
        return None
    ratio = actual_scale / max(expected_scale_arcsec, 1e-6)
    minimum_ratio, maximum_ratio = (0.25, 4.0) if allow_inaccurate_scale else (0.55, 1.8)
    if ratio < minimum_ratio or ratio > maximum_ratio:
        return None
    return wcs


def _astap_quad_counts(log_path: Path, stdout: str = "") -> tuple[int, int]:
    """Read ASTAP's independent quad-match count for a candidate solution."""
    text = stdout
    if log_path.is_file():
        text += "\n" + log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(
        r"\b(\d+)\s+of\s+(\d+)\s+quads?\s+selected\s+matching\b",
        text,
        flags=re.IGNORECASE,
    )
    if not matches:
        return 0, 0
    return tuple(map(int, matches[-1]))


def _wide_tile_is_strong(tile: SolverTile) -> bool:
    """A lone wide-field WCS needs more than ASTAP's minimum three quads."""
    return (
        not tile.scale_was_inaccurate
        and tile.matched_quads >= 6
        and tile.total_quads > 0
        and tile.matched_quads / tile.total_quads >= 0.5
    )


def _wide_tiles_agree(first: SolverTile, second: SolverTile) -> bool:
    """Check two overlapping tile solutions at several shared image points."""
    x0, y0 = max(first.x0, second.x0), max(first.y0, second.y0)
    x1, y1 = min(first.x1, second.x1), min(first.y1, second.y1)
    if x1 <= x0 or y1 <= y0:
        return False
    overlap_area = (x1 - x0) * (y1 - y0)
    smaller_tile_area = min(
        (first.x1 - first.x0) * (first.y1 - first.y0),
        (second.x1 - second.x0) * (second.y1 - second.y0),
    )
    if overlap_area / max(smaller_tile_area, 1) > 0.85:
        return False
    if (x1 - x0) < 0.15 * min(first.x1 - first.x0, second.x1 - second.x0):
        return False
    if (y1 - y0) < 0.15 * min(first.y1 - first.y0, second.y1 - second.y0):
        return False

    sample_x = np.linspace(x0 + 0.2 * (x1 - x0), x1 - 0.2 * (x1 - x0), 3)
    sample_y = np.linspace(y0 + 0.2 * (y1 - y0), y1 - 0.2 * (y1 - y0), 3)
    global_pixels = np.asarray([(x, y) for y in sample_y for x in sample_x], dtype=np.float64)
    try:
        first_world = _tile_pixel_to_world(
            first, global_pixels - np.asarray([first.x0, first.y0])
        )
        second_world = _tile_pixel_to_world(
            second, global_pixels - np.asarray([second.x0, second.y0])
        )
    except Exception:
        return False
    valid = (
        np.isfinite(first_world).all(axis=1)
        & np.isfinite(second_world).all(axis=1)
        & (np.abs(first_world[:, 1]) <= 90.0)
        & (np.abs(second_world[:, 1]) <= 90.0)
    )
    if np.count_nonzero(valid) < 5:
        return False

    def unit_vectors(world: np.ndarray) -> np.ndarray:
        ra = np.radians(world[:, 0])
        dec = np.radians(world[:, 1])
        cos_dec = np.cos(dec)
        return np.column_stack((cos_dec * np.cos(ra), cos_dec * np.sin(ra), np.sin(dec)))

    first_vectors = unit_vectors(first_world[valid])
    second_vectors = unit_vectors(second_world[valid])
    cos_separation = np.clip(np.sum(first_vectors * second_vectors, axis=1), -1.0, 1.0)
    separation_arcsec = np.degrees(np.arccos(cos_separation)) * 3600.0
    tolerance_arcsec = max(
        120.0,
        10.0 * max(first.pixel_scale_arcsec, second.pixel_scale_arcsec),
    )
    return float(np.percentile(separation_arcsec, 90)) <= tolerance_arcsec


def _trusted_wide_tiles(candidates: Sequence[SolverTile]) -> list[SolverTile]:
    """Keep strong single-tile solves and weak solves confirmed in overlaps."""
    trusted = {index for index, tile in enumerate(candidates) if _wide_tile_is_strong(tile)}
    for first_index, first in enumerate(candidates):
        for second_index in range(first_index + 1, len(candidates)):
            second = candidates[second_index]
            if _wide_tiles_agree(first, second):
                trusted.add(first_index)
                trusted.add(second_index)
    return [tile for index, tile in enumerate(candidates) if index in trusted]


def _catalog_confirmed_wide_tiles(
    candidates: Sequence[SolverTile],
    stars: Sequence[object],
    detector: np.ndarray,
    sky_mask: np.ndarray | None,
    detector_scale_xy: tuple[float, float],
    catalogs: Path,
) -> list[tuple[SolverTile, int, float]]:
    """Keep weak wide-field ASTAP WCS candidates only when Gaia confirms them."""
    catalog_path = next(iter(sorted(catalogs.glob("w08_*.001"))), None)
    if catalog_path is None or not candidates or not stars:
        return []
    try:
        _magnitudes, catalog_vectors = _load_w08_catalog(str(catalog_path.resolve()))
    except Exception:
        return []
    catalog_tree = cKDTree(catalog_vectors)
    scale_x, scale_y = detector_scale_xy
    detector_height, detector_width = detector.shape
    source_pixels = np.asarray([
        (float(getattr(star, "x")) * scale_x, float(getattr(star, "y")) * scale_y)
        for star in stars
    ], dtype=np.float64)
    valid_sources = np.isfinite(source_pixels).all(axis=1)
    valid_sources &= (
        (source_pixels[:, 0] >= 0) & (source_pixels[:, 0] < detector_width * scale_x)
        & (source_pixels[:, 1] >= 0) & (source_pixels[:, 1] < detector_height * scale_y)
    )
    if sky_mask is not None and sky_mask.shape == detector.shape:
        detector_x = np.clip(
            np.floor(np.nan_to_num(source_pixels[:, 0]) / scale_x).astype(np.intp),
            0, detector_width - 1,
        )
        detector_y = np.clip(
            np.floor(np.nan_to_num(source_pixels[:, 1]) / scale_y).astype(np.intp),
            0, detector_height - 1,
        )
        valid_sources &= ~np.asarray(sky_mask, dtype=bool)[detector_y, detector_x]
    confirmed: list[tuple[SolverTile, int, float]] = []
    for tile in candidates:
        inside = (
            valid_sources
            & (source_pixels[:, 0] >= tile.x0) & (source_pixels[:, 0] < tile.x1)
            & (source_pixels[:, 1] >= tile.y0) & (source_pixels[:, 1] < tile.y1)
        )
        source_indices = np.flatnonzero(inside)
        if len(source_indices) < 12:
            continue
        local_pixels = source_pixels[source_indices] - np.asarray([tile.x0, tile.y0])
        try:
            world = _tile_pixel_to_world(tile, local_pixels)
        except Exception:
            continue
        finite = (
            np.isfinite(world).all(axis=1)
            & (world[:, 1] >= -90.0) & (world[:, 1] <= 90.0)
        )
        source_indices = source_indices[finite]
        local_pixels = local_pixels[finite]
        if len(source_indices) < 12:
            continue
        source_vectors = _world_unit_vectors(world[finite])
        matrix = np.asarray(tile.wcs.pixel_scale_matrix, dtype=np.float64)
        actual_scale = math.sqrt(abs(float(np.linalg.det(matrix)))) * 3600.0
        if not math.isfinite(actual_scale) or actual_scale <= 0:
            continue
        max_separation_arcsec = float(np.clip(actual_scale * 1.8, 45.0, 180.0))
        chord_limit = 2.0 * math.sin(math.radians(max_separation_arcsec / 3600.0) * 0.5)
        distances, rows = catalog_tree.query(
            source_vectors, k=1, distance_upper_bound=chord_limit
        )
        pairs = []
        for local_index, (distance, row) in enumerate(zip(distances, rows)):
            row = int(row)
            if row >= len(catalog_vectors) or not math.isfinite(float(distance)):
                continue
            separation = math.degrees(2.0 * math.asin(min(float(distance), 2.0) * 0.5)) * 3600.0
            pairs.append((separation, local_index, row))
        used_sources: set[int] = set()
        used_catalog: set[int] = set()
        accepted = []
        for separation, local_index, row in sorted(pairs):
            if local_index in used_sources or row in used_catalog:
                continue
            used_sources.add(local_index)
            used_catalog.add(row)
            accepted.append((separation, local_index))
        if len(accepted) < 8:
            continue
        residuals = np.asarray([pair[0] for pair in accepted], dtype=np.float64)
        matched_local = np.asarray([pair[1] for pair in accepted], dtype=np.intp)
        p50, p90 = np.percentile(residuals, [50.0, 90.0])
        if p50 > actual_scale * 0.6 or p90 > actual_scale * 1.5:
            continue
        matched_xy = local_pixels[matched_local]
        width, height = max(tile.x1 - tile.x0, 1), max(tile.y1 - tile.y0, 1)
        spread_x = float(np.ptp(matched_xy[:, 0]) / width)
        spread_y = float(np.ptp(matched_xy[:, 1]) / height)
        if spread_x < 0.10 or spread_y < 0.10:
            continue
        if tile.scale_was_inaccurate:
            tile = replace(tile, pixel_scale_arcsec=actual_scale)
        confirmed.append((tile, len(accepted), float(p90)))
    return confirmed


def _solver_image(detector: np.ndarray, mask: np.ndarray | None, crop: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = crop
    tile = np.asarray(detector[y0:y1, x0:x1], dtype=np.float32).copy()
    if mask is not None:
        tile_mask = np.asarray(mask[y0:y1, x0:x1], dtype=bool)
        valid = ~tile_mask & np.isfinite(tile)
        if np.any(valid):
            tile[~valid] = float(np.median(tile[valid]))
    tile = np.nan_to_num(tile, nan=0.0, posinf=1.0, neginf=0.0)
    low, high = np.percentile(tile, (0.2, 99.8))
    if high <= low:
        raise RuntimeError("图像中没有足够的星点对比度用于板解算。")
    # This is a solver-only 16-bit TIFF proxy. The source working image is not
    # rescaled or quantized by this operation.
    np.subtract(tile, low, out=tile)
    np.divide(tile, high - low, out=tile)
    np.clip(tile, 0.0, 1.0, out=tile)
    return np.rint(tile * 65535.0).astype(np.uint16)


def _solver_crop_from_map(
    source: np.ndarray,
    box: tuple[int, int, int, int],
    sky_mask: np.ndarray | None,
    detector_scale_xy: tuple[float, float],
) -> np.ndarray:
    """Locally stretch a 16-bit solve crop without changing source pixels."""
    x0, y0, x1, y1 = box
    tile = np.asarray(source[y0:y1, x0:x1], dtype=np.float32).copy()
    valid = np.isfinite(tile)
    crop_mask: np.ndarray | None = None
    if sky_mask is not None and sky_mask.ndim == 2:
        scale_x, scale_y = detector_scale_xy
        x_indices = np.rint((np.arange(x0, x1, dtype=np.float64) + 0.5) / scale_x - 0.5)
        y_indices = np.rint((np.arange(y0, y1, dtype=np.float64) + 0.5) / scale_y - 0.5)
        x_indices = np.clip(x_indices.astype(np.intp), 0, sky_mask.shape[1] - 1)
        y_indices = np.clip(y_indices.astype(np.intp), 0, sky_mask.shape[0] - 1)
        crop_mask = np.asarray(sky_mask, dtype=bool)[np.ix_(y_indices, x_indices)]
        if crop_mask.shape == tile.shape:
            valid &= ~crop_mask
        else:
            crop_mask = None
    if not np.any(valid):
        raise RuntimeError("本地图块的有效天空像素不足以生成板解算图像。")
    tile[~valid] = float(np.median(tile[valid]))
    low, high = np.percentile(tile[valid], (0.2, 99.8))
    if not math.isfinite(float(low)) or not math.isfinite(float(high)) or high <= low:
        raise RuntimeError("本地图块的星点对比度不足以生成板解算图像。")
    np.subtract(tile, low, out=tile)
    np.divide(tile, high - low, out=tile)
    np.clip(tile, 0.0, 1.0, out=tile)
    return np.rint(tile * 65535.0).astype(np.uint16)


def _positions_to_sky(
    stars: Sequence[object],
    detector: np.ndarray,
    sky_mask: np.ndarray | None,
    info: object,
    solver_image_path: Path,
    detector_scale_xy: tuple[float, float],
    primary_image_path: Path | None,
    progress: Progress | None,
) -> tuple[
    list[tuple[int, float, float]], float, float,
    list[SolverTile], CameraProjection | None,
]:
    executable, catalogs = _executable_and_catalogs()
    detector_height, detector_width = detector.shape
    detector_scale_x, detector_scale_y = detector_scale_xy
    solver_height = max(1, int(round(detector_height * detector_scale_y)))
    solver_width = max(1, int(round(detector_width * detector_scale_x)))
    if not solver_image_path.is_file():
        raise RuntimeError("板解算用的临时 16 位图像不存在。")
    sensor_width_mm, sensor_height_mm, focal_mm = _sensor_dimensions(info, detector.shape)
    full_fov_height = _axis_fov_degrees(0, solver_height, solver_height, sensor_height_mm, focal_mm)
    full_fov_width = _axis_fov_degrees(0, solver_width, solver_width, sensor_width_mm, focal_mm)
    if not 0.15 <= min(full_fov_height, full_fov_width) <= 180:
        raise RuntimeError("相机镜头视场估算超出本地板解算支持范围。")
    preliminary_camera_result: SeizaLocalSolveResult | None = None
    preliminary_local_tiles: list[SolverTile] = []

    # If the Seiza data is already cached, try its fast blind solve before
    # ASTAP's multi-tile sweep. A verified full-frame solution avoids spending
    # up to the ASTAP tile-search budget on a field Seiza can solve directly.
    local_seiza_data = _load_seiza_solver_data(None, allow_download=False)
    seiza_attempted_locally = local_seiza_data is not None
    seiza_local_message: str | None = None
    if local_seiza_data is not None:
        seiza_tile, seiza_local_message = _try_seiza_fallback_tile(
            stars,
            detector,
            sky_mask,
            info,
            catalogs,
            detector_scale_xy,
            (solver_height, solver_width),
            sensor_width_mm,
            focal_mm,
            progress,
            seiza_data=local_seiza_data,
            allow_download=False,
        )
        if seiza_tile is not None:
            if progress:
                progress(54, f"本机 Seiza/Gaia 盲解已通过 W08 复核：{seiza_local_message}")
            positions = _positions_from_solved_tiles(
                stars, [seiza_tile], detector_scale_xy
            )
            return positions, seiza_tile.pixel_scale_arcsec, full_fov_height, [seiza_tile], None

    # The supplied 14 mm twilight frame is especially difficult for blind
    # quad solvers. Before launching the full tile sweep, cheaply test the
    # recognizable Orion belt against SEP positions and then require a
    # distributed W08 Gaia match over the complete field.
    w08_path = next(iter(sorted(catalogs.glob("w08_*.001"))), None)
    if w08_path is not None:
        try:
            catalog_magnitudes, catalog_vectors = _load_w08_catalog(str(w08_path.resolve()))
            orion_result = _try_orion_belt_camera_fit(
                stars,
                sky_mask,
                info,
                catalog_magnitudes,
                catalog_vectors,
                detector.shape,
                detector_scale_xy,
                progress,
            )
            if orion_result is not None:
                camera_projection, anchor_catalog_matches = orion_result
                positions, pixel_scale = _positions_from_camera_projection(
                    stars,
                    detector,
                    sky_mask,
                    detector_scale_xy,
                    (sensor_width_mm, sensor_height_mm),
                    camera_projection,
                )
                if progress:
                    progress(
                        54,
                        f"猎户腰带锚点通过 W08 Gaia 全幅复核：{anchor_catalog_matches} 个分布式匹配，"
                        f"P90 {camera_projection.residual_p90_px:.1f} solver px。",
                    )
                return positions, pixel_scale, full_fov_height, [], camera_projection
        except Exception as error:
            if progress:
                progress(53, f"猎户腰带星表锚点备用校验失败，继续盲解：{type(error).__name__}")

    # Wide-angle distortion can defeat a whole-frame asterism solve even when
    # the sky contains enough Gaia stars. Try local Seiza blind solves before
    # ASTAP's larger crop sweep, then require an independently verified camera
    # pose across the full image before mapping any SEP source to the sky.
    if local_seiza_data is not None:
        local_camera_result = _try_seiza_local_camera_fit(
            stars,
            detector,
            sky_mask,
            info,
            catalogs,
            detector_scale_xy,
            (solver_height, solver_width),
            sensor_width_mm,
            sensor_height_mm,
            focal_mm,
            progress,
            seiza_data=local_seiza_data,
        )
        if local_camera_result is not None:
            preliminary_camera_result = local_camera_result
            preliminary_local_tiles.extend(local_camera_result.verified_tiles)
            if progress and local_camera_result.camera_projection is not None:
                camera_projection = local_camera_result.camera_projection
                progress(
                    55,
                    f"广角相机候选模型通过分布式 Gaia/W08 校验：{local_camera_result.catalog_match_count} 个匹配，"
                    f"P90 {camera_projection.residual_p90_px:.2f} px；继续检查重叠图块和边缘区域。",
                )
            elif progress:
                progress(
                    55,
                    f"Seiza 只取得 {len(local_camera_result.verified_tiles)} 个局部 WCS；"
                    "继续搜索其他图块，尚未确认的区域不会按全幅识别处理。",
                )

    valid_height = solver_height
    if sky_mask is not None and sky_mask.shape == detector.shape:
        valid_rows = np.mean(~np.asarray(sky_mask, dtype=bool), axis=1)
        sky_rows = np.flatnonzero(valid_rows >= 0.03)
        if len(sky_rows):
            valid_height = min(solver_height, int(math.ceil((int(sky_rows[-1]) + 1) * detector_scale_y)))
    # ASTAP's -fov parameter is the image-height field, so a wide landscape
    # sensor may have a horizontal field above 80 degrees and still be a
    # supported single-frame solve.
    # Keep the blind full-frame attempt within ASTAP W08's documented 80-degree
    # height limit; wider images are handled by overlapping local fields.
    can_solve_full = full_fov_height <= 80.0
    boxes: list[tuple[int, int, int, int]] = []
    if can_solve_full:
        boxes.append((0, 0, solver_width, solver_height))

    # ASTAP W08 supports fields up to 80 degrees. Use overlapping full-detail
    # crops for very wide views, and as a fallback if the full image is too
    # distorted or contains too much foreground for one reliable solve.
    # Keep wide-angle blind solves near 50 degrees to retain enough stars for
    # ASTAP's W08 quad matcher without producing excessive overlapping crops.
    # Convert angular width to a sensor fraction with tan(FOV/2); a linear FOV
    # ratio understates the actual angle on rectilinear 14 mm frames.
    target_tile_fov = math.radians(50.0) * 0.5
    tile_w_fraction = min(
        0.75,
        math.tan(target_tile_fov) / max(math.tan(math.radians(full_fov_width) * 0.5), 1e-8),
    )
    tile_h_fraction = min(
        0.65,
        math.tan(target_tile_fov) / max(math.tan(math.radians(full_fov_height) * 0.5), 1e-8),
    )
    tile_w = min(solver_width, max(1200, int(round(solver_width * tile_w_fraction))))
    tile_h = min(valid_height, max(1200, int(round(valid_height * tile_h_fraction))))
    if tile_w < solver_width or tile_h < solver_height:
        def starts(length: int, tile: int) -> list[int]:
            if length <= tile:
                return [0]
            last = length - tile
            # Half-tile steps place each next tile center at the prior tile's
            # edge, where its WCS can still provide a local search seed.
            stride = max(1, int(tile * 0.5))
            values = list(range(0, last + 1, stride))
            values.append(last)
            unique = sorted(set(values))
            # Include both edges explicitly, even when the stride misses the
            # final tile center.
            return list(dict.fromkeys([0, last, *unique[1:-1]]))

        x_starts = starts(solver_width, tile_w)
        y_starts = starts(valid_height, tile_h)
        crop_boxes = [
            (x, y, min(solver_width, x + tile_w), min(valid_height, y + tile_h))
            for y in y_starts for x in x_starts
        ]
        for box in crop_boxes:
            if box not in boxes:
                boxes.append(box)

    # Keep one independently solvable patch centered on the optical axis.
    # On ultra-wide frames this smaller, lower-distortion view often provides
    # the reliable seed that the larger edge tiles cannot find blindly.
    if max(full_fov_width, full_fov_height) > 45.0:
        central_width = min(solver_width, max(900, int(round(solver_width * 0.42))))
        central_height = min(solver_height, max(900, int(round(solver_height * 0.42))))
        central_x0 = max(0, (solver_width - central_width) // 2)
        central_y0 = max(0, (solver_height - central_height) // 2)
        central_box = (
            central_x0,
            central_y0,
            central_x0 + central_width,
            central_y0 + central_height,
        )
        if central_box not in boxes:
            boxes.append(central_box)

    detector_stars_xy = np.asarray(
        [
            (float(getattr(star, "x")) * detector_scale_x,
             float(getattr(star, "y")) * detector_scale_y)
            for star in stars
        ],
        dtype=np.float64,
    ).reshape((-1, 2))
    if len(detector_stars_xy):
        valid_star_positions = np.isfinite(detector_stars_xy).all(axis=1)
        valid_star_positions &= (
            (detector_stars_xy[:, 0] >= 0)
            & (detector_stars_xy[:, 0] < solver_width)
            & (detector_stars_xy[:, 1] >= 0)
            & (detector_stars_xy[:, 1] < solver_height)
        )
        if sky_mask is not None and sky_mask.shape == detector.shape:
            finite_xy = np.nan_to_num(detector_stars_xy, nan=0.0, posinf=0.0, neginf=0.0)
            star_x = np.clip(
                np.floor(finite_xy[:, 0] / detector_scale_x).astype(np.intp),
                0,
                detector.shape[1] - 1,
            )
            star_y = np.clip(
                np.floor(finite_xy[:, 1] / detector_scale_y).astype(np.intp),
                0,
                detector.shape[0] - 1,
            )
            valid_star_positions &= ~np.asarray(sky_mask, dtype=bool)[star_y, star_x]
        detector_stars_xy = detector_stars_xy[valid_star_positions]

    def box_priority(box: tuple[int, int, int, int]) -> float:
        x0, y0, x1, y1 = box
        tile_height, tile_width = y1 - y0, x1 - x0
        if len(detector_stars_xy):
            in_tile = (
                (detector_stars_xy[:, 0] >= x0)
                & (detector_stars_xy[:, 0] < x1)
                & (detector_stars_xy[:, 1] >= y0)
                & (detector_stars_xy[:, 1] < y1)
            )
            star_count = int(np.count_nonzero(in_tile))
        else:
            star_count = 0
        sky_fraction = 1.0
        if sky_mask is not None and sky_mask.shape == detector.shape:
            dx0 = max(0, int(math.floor(x0 / detector_scale_x)))
            dy0 = max(0, int(math.floor(y0 / detector_scale_y)))
            dx1 = min(detector.shape[1], int(math.ceil(x1 / detector_scale_x)))
            dy1 = min(detector.shape[0], int(math.ceil(y1 / detector_scale_y)))
            if dx1 > dx0 and dy1 > dy0:
                sky_fraction = float(np.mean(~np.asarray(sky_mask[dy0:dy1, dx0:dx1], dtype=bool)))
        center_distance = math.hypot(
            ((x0 + x1) * 0.5 - solver_width * 0.5) / max(solver_width * 0.5, 1.0),
            ((y0 + y1) * 0.5 - solver_height * 0.5) / max(solver_height * 0.5, 1.0),
        )
        tile_fov_x, tile_fov_y = _local_box_fov_degrees(
            box, solver_width, solver_height, sensor_width_mm, sensor_height_mm, focal_mm
        )
        distortion_penalty = max(0.0, max(tile_fov_x, tile_fov_y) - 35.0) / 45.0
        return (
            (min(star_count, 500) + 5.0) * (0.2 + 0.8 * sky_fraction)
            / (1.0 + 0.6 * center_distance + 0.5 * distortion_penalty)
        )

    # Very wide frames can contain useful star fields through small openings or
    # between foreground silhouettes. Add a second, smaller tile scale so a
    # central field cannot hide an unsolved edge or an architectural window.
    compact_half_fov_x = math.radians(20.0) * 0.5
    compact_half_fov_y = math.radians(14.0) * 0.5
    compact_w_fraction = min(
        0.75,
        math.tan(compact_half_fov_x)
        / max(math.tan(math.radians(full_fov_width) * 0.5), 1e-8),
    )
    compact_h_fraction = min(
        0.65,
        math.tan(compact_half_fov_y)
        / max(math.tan(math.radians(full_fov_height) * 0.5), 1e-8),
    )
    compact_w = min(solver_width, max(960, int(round(solver_width * compact_w_fraction))))
    # Size crops from the full sensor geometry, then limit their lower edge to
    # the sky rows. Using the truncated sky height here shrinks fields and
    # changes their FOV, which leaves too few stars in tall portrait scenes.
    compact_h = min(valid_height, max(960, int(round(solver_height * compact_h_fraction))))

    def grid_starts(length: int, tile: int) -> list[int]:
        if length <= tile:
            return [0]
        last = length - tile
        stride = max(1, tile // 2)
        return sorted(set([0, last, *range(0, last + 1, stride)]))

    compact_boxes = [
        (x, y, min(solver_width, x + compact_w), min(valid_height, y + compact_h))
        for y in grid_starts(valid_height, compact_h)
        for x in grid_starts(solver_width, compact_w)
    ]
    # Rank local patches by detected point sources and sky coverage. Keep all
    # bounded compact fields; ASTAP's G05/W08 searches are fast on these crops.
    compact_boxes = sorted(
        {box for box in compact_boxes if box not in boxes},
        key=box_priority,
        reverse=True,
    )
    coarse_boxes = sorted(set(boxes), key=box_priority, reverse=True)
    boxes = coarse_boxes + compact_boxes

    successful: list[SolverTile] = []
    for tile in preliminary_local_tiles:
        if not any(
            (item.x0, item.y0, item.x1, item.y1) == (tile.x0, tile.y0, tile.x1, tile.y1)
            for item in successful
        ):
            successful.append(tile)
    strong_camera_model = bool(
        preliminary_camera_result is not None
        and preliminary_camera_result.camera_projection is not None
        and preliminary_camera_result.catalog_match_count >= 120
        and preliminary_camera_result.camera_projection.residual_p90_px <= 6.0
    )
    if strong_camera_model and not successful:
        # Seiza has already sampled distributed local patches and the global
        # pose passed hundreds of independent W08 checks. Repeating a large
        # ASTAP crop sweep on this case produced no trusted tile on 6627.NEF;
        # use the measured model residual for catalogue-guided image searches.
        boxes = []
        if progress:
            projection = preliminary_camera_result.camera_projection
            assert projection is not None
            progress(
                53,
                f"全幅坐标模型已通过 {preliminary_camera_result.catalog_match_count} 个分布式 W08 匹配，"
                f"P90 {projection.residual_p90_px:.2f} px；按实测误差回搜图像星点，"
                "未确认的星表位置仍不柔焦。",
            )
    wide_candidates: list[SolverTile] = []
    wide_camera_result: SeizaLocalSolveResult | None = None
    catalog_confirmed_boxes: set[tuple[int, int, int, int]] = set()
    diagnostics: list[str] = [seiza_local_message] if seiza_local_message else []
    solve_deadline = time.monotonic() + 150.0
    fast_deadline = solve_deadline - 20.0
    with tempfile.TemporaryDirectory(prefix="starsoft-solve-") as temporary:
        work_dir = Path(temporary)
        solver_source = tifffile.memmap(solver_image_path, mode="r")
        trial_specs: dict[
            tuple[int, int, int, int], tuple[Path, float, bool, float]
        ] = {}
        # Prepare every eligible crop before launching ASTAP. Each process gets
        # its own working directory and output prefix; crops and the catalog are
        # read-only inputs, so independent fields can be solved concurrently.
        for index, (x0, y0, x1, y1) in enumerate(boxes):
            tile_h, tile_w = y1 - y0, x1 - x0
            if tile_h < 900 or tile_w < 900:
                continue
            box = (x0, y0, x1, y1)
            fov_x, fov_y = _local_box_fov_degrees(
                box, solver_width, solver_height, sensor_width_mm, sensor_height_mm, focal_mm
            )
            if box == (0, 0, solver_width, solver_height):
                fov_y = min(fov_y, 80.0)
            if fov_y > 80.0 or min(fov_x, fov_y) < 0.15:
                continue
            expected_scale = 0.5 * (fov_x / tile_w + fov_y / tile_h) * 3600.0
            is_full_frame = (x0, y0, x1, y1) == (0, 0, solver_width, solver_height)
            # Always use the sky-masked, grayscale proxy when a horizon mask
            # exists; a source TIFF could otherwise put trees/buildings back
            # into the solver image.
            image_path = (
                primary_image_path
                if is_full_frame and primary_image_path is not None and sky_mask is None
                else solver_image_path
                if is_full_frame
                else work_dir / f"field_{index:02d}.tif"
            )
            if not is_full_frame:
                tifffile.imwrite(
                    str(image_path),
                    _solver_crop_from_map(
                        solver_source, box, sky_mask, detector_scale_xy
                    ),
                    compression=None, photometric="minisblack",
                )
            # ASTAP defines its field size from image height. Use the same axis
            # for database selection, tolerance, and tile trust rules.
            wide_field = fov_y > 20.0
            trial_specs[box] = (image_path, expected_scale, wide_field, fov_y)
        def build_command(
            box: tuple[int, int, int, int],
            index: int,
            seed_candidates: Sequence[SolverTile],
            output_base: Path,
        ) -> list[str]:
            x0, y0, x1, y1 = box
            image_path, _expected_scale, wide_field, fov_y = trial_specs[box]
            command = [
                str(executable), "-f", str(image_path), "-fov", f"{fov_y:.5f}",
                "-d", str(catalogs), "-z", "0",
                "-speed", "auto",
                "-s", "1000" if fov_y > 20.0 else "300",
                "-t", "0.015" if wide_field else "0.007",
                "-wcs", "-sip", "-log", "-o", str(output_base),
            ]
            # ASTAP chooses its star index from the image-height field.
            if fov_y > 20.0:
                command[command.index("-z"):command.index("-z")] = ["-D", "w08"]
            elif fov_y > 6.0:
                command[command.index("-z"):command.index("-z")] = ["-D", "g05"]
            if wide_field:
                seed_choices: list[tuple[float, SolverTile, np.ndarray]] = []
                for tile in seed_candidates:
                    tile_box = (tile.x0, tile.y0, tile.x1, tile.y1)
                    if not _wide_tile_is_strong(tile) and tile_box not in catalog_confirmed_boxes:
                        continue
                    orientation = _camera_orientation_from_tile(
                        tile, solver_width, solver_height,
                        sensor_width_mm, sensor_height_mm, focal_mm,
                    )
                    if orientation is None:
                        continue
                    optical_axis_distance = math.hypot(
                        (tile.x0 + tile.x1) * 0.5 - solver_width * 0.5,
                        (tile.y0 + tile.y1) * 0.5 - solver_height * 0.5,
                    )
                    seed_choices.append((optical_axis_distance, tile, orientation))
                if seed_choices:
                    _distance, _seed, orientation = min(seed_choices, key=lambda item: item[0])
                    try:
                        # Derive the camera's celestial orientation from the
                        # trusted WCS, then use the known lens projection to
                        # predict this crop's centre across the full wide field.
                        # Direct WCS extrapolation is inaccurate at the edges
                        # because ASTAP may not have enough matches for SIP.
                        camera_ray = _camera_ray(
                            (x0 + x1) * 0.5, (y0 + y1) * 0.5,
                            solver_width, solver_height,
                            sensor_width_mm, sensor_height_mm, focal_mm,
                        )
                        seed_world_vector = orientation @ camera_ray
                        seed_world_vector /= max(float(np.linalg.norm(seed_world_vector)), 1e-12)
                        seed_ra = math.degrees(math.atan2(seed_world_vector[1], seed_world_vector[0])) % 360.0
                        seed_dec = math.degrees(math.asin(float(np.clip(seed_world_vector[2], -1.0, 1.0))))
                        if math.isfinite(seed_ra) and math.isfinite(seed_dec) and -90 <= seed_dec <= 90:
                            sensor_angle = math.degrees(math.acos(float(np.clip(camera_ray[2], -1.0, 1.0))))
                            # Keep the edge search broad enough for modest
                            # residual lens distortion without opening the
                            # all-sky quad search to unrelated fields.
                            search_radius = max(8.0, min(22.0, 7.0 + 0.20 * sensor_angle))
                            seed_arguments = [
                                "-ra", f"{seed_ra / 15.0:.7f}",
                                "-spd", f"{seed_dec + 90.0:.7f}",
                                "-r", f"{search_radius:.2f}",
                            ]
                            output_option = command.index("-o")
                            command[output_option:output_option] = seed_arguments
                    except Exception:
                        pass
            return command

        def solve_fast_tile(
            box: tuple[int, int, int, int],
            index: int,
            seed_candidates: Sequence[SolverTile],
        ) -> tuple[SolverTile | None, str | None]:
            remaining_seconds = fast_deadline - time.monotonic()
            if remaining_seconds <= 0:
                return None, "快速图块搜索已达到时间预算。"
            image_path, expected_scale, wide_field, _fov_y = trial_specs[box]
            job_dir = work_dir / f"tile_{index:02d}"
            job_dir.mkdir(parents=True, exist_ok=True)
            output_base = job_dir / f"solution_{index:02d}"
            command = build_command(box, index, seed_candidates, output_base)
            try:
                completed = subprocess.run(
                    command,
                    cwd=str(job_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    errors="replace",
                    check=False,
                    timeout=min(22.0, remaining_seconds),
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except subprocess.TimeoutExpired:
                return None, f"{image_path.name}: ASTAP 并行快速搜索超时。"
            except OSError as error:
                return None, f"{image_path.name}: ASTAP 启动失败：{error}"

            solved_wcs = _extract_astap_wcs(
                output_base, expected_scale, allow_inaccurate_scale=wide_field
            )
            if solved_wcs is not None:
                ini_text = output_base.with_suffix(".ini").read_text(
                    encoding="utf-8", errors="replace"
                )
                scale_was_inaccurate = "scale was inaccurate" in ini_text.lower()
                matched_quads, total_quads = _astap_quad_counts(
                    output_base.with_suffix(".log"), completed.stdout
                )
                candidate = SolverTile(
                    *box, solved_wcs, expected_scale, wide_field,
                    matched_quads, total_quads, scale_was_inaccurate,
                )
                if wide_field and not _wide_tile_is_strong(candidate):
                    return candidate, (
                        f"{image_path.name}: 宽场候选匹配 {matched_quads}/{total_quads} 个四星组合，等待重叠图块确认。"
                    )
                return candidate, None

            if completed.stdout:
                useful_lines = [
                    line.strip() for line in completed.stdout.splitlines()
                    if line.strip() and any(token in line.lower() for token in (
                        "error", "fail", "no solution", "cannot", "not found", "inaccurate", "stars"
                    ))
                ]
                if useful_lines:
                    return None, f"{image_path.name}: {' | '.join(useful_lines[-4:])}"
            return None, f"{image_path.name}: ASTAP 未找到解。"

        eligible_boxes = list(trial_specs)
        worker_count = max(1, min(8, max(1, os.cpu_count() or 1), len(eligible_boxes)))
        completed_attempts = 0
        # Solve a bounded batch at a time. Later batches can use a strong WCS
        # from earlier fields as a local seed, while overlapping weak solutions
        # from the same batch remain independent for cross-checking.
        for batch_start in range(0, len(eligible_boxes), worker_count):
            if time.monotonic() >= fast_deadline:
                diagnostics.append("快速图块搜索达到时间预算，保留慢速补充搜索时间。")
                break
            batch = eligible_boxes[batch_start:batch_start + worker_count]
            seed_snapshot = tuple(wide_candidates)
            if progress:
                progress(
                    45 + int(8 * completed_attempts / max(len(eligible_boxes), 1)),
                    f"本机星空并行板解算 {completed_attempts + 1}–{completed_attempts + len(batch)}/{len(eligible_boxes)}（{len(batch)} 核）…",
                )
            batch_results: dict[int, tuple[SolverTile | None, str | None]] = {}
            with ThreadPoolExecutor(max_workers=len(batch), thread_name_prefix="astap-tile") as executor:
                future_indices = {
                    executor.submit(
                        solve_fast_tile,
                        box,
                        boxes.index(box),
                        seed_snapshot,
                    ): boxes.index(box)
                    for box in batch
                }
                for future in as_completed(future_indices):
                    index = future_indices[future]
                    try:
                        batch_results[index] = future.result()
                    except Exception as error:
                        batch_results[index] = (None, f"图块 {index + 1}: {type(error).__name__}: {error}")
                    if progress:
                        finished_in_batch = len(batch_results)
                        progress(
                            45 + int(8 * (completed_attempts + finished_in_batch) / max(len(eligible_boxes), 1)),
                            f"本机星空并行板解算已完成 {completed_attempts + finished_in_batch}/{len(eligible_boxes)} 个图块…",
                        )

            for index in sorted(batch_results):
                candidate, message = batch_results[index]
                if candidate is not None:
                    diagnostics.append(
                        f"图块 {index + 1} ({candidate.x0},{candidate.y0}–{candidate.x1},{candidate.y1}) "
                        f"匹配 {candidate.matched_quads}/{candidate.total_quads} 四星组合"
                    )
                    if candidate.wide_field:
                        wide_candidates.append(candidate)
                    else:
                        successful.append(candidate)
                if message:
                    diagnostics.append(message)
            completed_attempts += len(batch_results)
            if can_solve_full and successful and max(full_fov_width, full_fov_height) <= 30.0:
                break

        if wide_candidates:
            successful.extend(_trusted_wide_tiles(wide_candidates))
            confirmed = _catalog_confirmed_wide_tiles(
                wide_candidates, stars, detector, sky_mask, detector_scale_xy, catalogs
            )
            for tile, match_count, p90 in confirmed:
                catalog_confirmed_boxes.add((tile.x0, tile.y0, tile.x1, tile.y1))
                if not any(
                    (existing.x0, existing.y0, existing.x1, existing.y1)
                    == (tile.x0, tile.y0, tile.x1, tile.y1)
                    for existing in successful
                ):
                    successful.append(tile)
                diagnostics.append(
                    f"Gaia 复核图块 ({tile.x0},{tile.y0}–{tile.x1},{tile.y1}) "
                    f"确认 {match_count} 颗星，P90 {p90:.0f}″"
                )

        has_wide_trials = any(spec[2] for spec in trial_specs.values())
        if has_wide_trials:
            # A successful central WCS does not imply the outer field was
            # solved. Retry every uncovered crop with a camera-geometry seed
            # so one easy centre tile cannot suppress edge recovery.
            uncovered_boxes: list[tuple[int, int, int, int]] = []
            for box in boxes:
                if box not in trial_specs or not trial_specs[box][2]:
                    continue
                x0, y0, x1, y1 = box
                center_x, center_y = (x0 + x1) * 0.5, (y0 + y1) * 0.5
                if any(tile.x0 <= center_x < tile.x1 and tile.y0 <= center_y < tile.y1 for tile in successful):
                    continue
                uncovered_boxes.append(box)
            slow_boxes = sorted(uncovered_boxes, key=box_priority, reverse=True)
            remaining_seconds = solve_deadline - time.monotonic()
            if slow_boxes and remaining_seconds > 2.0:
                if progress:
                    progress(53, f"快速搜索后有 {len(slow_boxes)} 个边缘区域尚未覆盖，正在补做慢速星表解算…")

                def solve_slow_tile(
                    box: tuple[int, int, int, int], index: int, budget: float
                ) -> tuple[SolverTile | None, str | None]:
                    image_path, expected_scale, _wide_field, _fov_y = trial_specs[box]
                    output_base = work_dir / f"slow_solution_{index:02d}"
                    command = build_command(box, index, tuple(wide_candidates), output_base)
                    speed_index = command.index("-speed") + 1
                    command[speed_index] = "slow"
                    try:
                        completed = subprocess.run(
                            command,
                            cwd=str(work_dir),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            errors="replace",
                            check=False,
                            timeout=max(1.0, min(28.0, budget)),
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                        )
                    except subprocess.TimeoutExpired:
                        return None, f"{image_path.name}: 慢速补充搜索超时。"
                    solved_wcs = _extract_astap_wcs(
                        output_base, expected_scale, allow_inaccurate_scale=True
                    )
                    if solved_wcs is None:
                        return None, f"{image_path.name}: 慢速补充搜索未找到 WCS。"
                    ini_text = output_base.with_suffix(".ini").read_text(
                        encoding="utf-8", errors="replace"
                    )
                    scale_was_inaccurate = "scale was inaccurate" in ini_text.lower()
                    matched_quads, total_quads = _astap_quad_counts(
                        output_base.with_suffix(".log"), completed.stdout
                    )
                    return SolverTile(
                        *box, solved_wcs, expected_scale, True,
                        matched_quads, total_quads, scale_was_inaccurate,
                    ), None

                slow_worker_count = max(1, min(8, max(1, os.cpu_count() or 1), len(slow_boxes)))
                slow_results: list[tuple[SolverTile | None, str | None]] = []
                with ThreadPoolExecutor(max_workers=slow_worker_count, thread_name_prefix="astap-edge") as executor:
                    futures = [
                        executor.submit(solve_slow_tile, box, boxes.index(box), remaining_seconds)
                        for box in slow_boxes
                    ]
                    for future in as_completed(futures):
                        try:
                            slow_results.append(future.result())
                        except Exception as error:
                            slow_results.append((None, f"边缘慢速搜索异常：{type(error).__name__}: {error}"))
                for candidate, message in slow_results:
                    if candidate is not None:
                        wide_candidates.append(candidate)
                        diagnostics.append(
                            f"慢速图块 ({candidate.x0},{candidate.y0}–{candidate.x1},{candidate.y1}) "
                            f"匹配 {candidate.matched_quads}/{candidate.total_quads} 四星组合"
                        )
                    if message:
                        diagnostics.append(message)
                for tile in _trusted_wide_tiles(wide_candidates):
                    if not any(
                        (existing.x0, existing.y0, existing.x1, existing.y1)
                        == (tile.x0, tile.y0, tile.x1, tile.y1)
                        for existing in successful
                    ):
                        successful.append(tile)
                confirmed = _catalog_confirmed_wide_tiles(
                    wide_candidates, stars, detector, sky_mask, detector_scale_xy, catalogs
                )
                for tile, match_count, p90 in confirmed:
                    catalog_confirmed_boxes.add((tile.x0, tile.y0, tile.x1, tile.y1))
                    if not any(
                        (existing.x0, existing.y0, existing.x1, existing.y1)
                        == (tile.x0, tile.y0, tile.x1, tile.y1)
                        for existing in successful
                    ):
                        successful.append(tile)
                    diagnostics.append(
                        f"Gaia 复核图块 ({tile.x0},{tile.y0}–{tile.x1},{tile.y1}) "
                        f"确认 {match_count} 颗星，P90 {p90:.0f}″"
                    )

        if not successful:
            maximum_fov = max(full_fov_width, full_fov_height)
            if not seiza_attempted_locally:
                seiza_tile, seiza_message = _try_seiza_fallback_tile(
                    stars,
                    detector,
                    sky_mask,
                    info,
                    catalogs,
                    detector_scale_xy,
                    (solver_height, solver_width),
                    sensor_width_mm,
                    focal_mm,
                    progress,
                    allow_download=maximum_fov <= 90.0,
                )
                if seiza_tile is not None:
                    successful.append(seiza_tile)
                    diagnostics.append(seiza_message or "Seiza 盲解通过 W08 星表确认。")
                elif seiza_message:
                    diagnostics.append(seiza_message)
                if seiza_tile is None:
                    downloaded_seiza_data = _load_seiza_solver_data(None, allow_download=False)
                    if downloaded_seiza_data is not None:
                        wide_camera_result = _try_seiza_local_camera_fit(
                            stars,
                            detector,
                            sky_mask,
                            info,
                            catalogs,
                            detector_scale_xy,
                            (solver_height, solver_width),
                            sensor_width_mm,
                            sensor_height_mm,
                            focal_mm,
                            progress,
                            seiza_data=downloaded_seiza_data,
                        )
                        if wide_camera_result is not None:
                            for tile in wide_camera_result.verified_tiles:
                                if not any(
                                    (item.x0, item.y0, item.x1, item.y1)
                                    == (tile.x0, tile.y0, tile.x1, tile.y1)
                                    for item in successful
                                ):
                                    successful.append(tile)
            elif maximum_fov > 90.0 and seiza_local_message:
                diagnostics.append(
                    f"整幅视场约 {maximum_fov:.0f}°；本机 Seiza 候选未通过 W08 复核，"
                    "ASTAP 局部图块也未得到可信解。"
                )

            # Some crowded or low-contrast wide fields need fainter pattern
            # stars than the compact G<=15 catalog supplies. Escalate only
            # after ASTAP and the light Seiza catalog both fail, keeping the
            # larger G<=17 database out of ordinary first-run installs.
            if (
                not successful
                and wide_camera_result is None
                and maximum_fov <= 120.0
                and not strong_camera_model
            ):
                deep_seiza_data = _load_seiza_solver_data(
                    progress,
                    allow_download=True,
                    prefer_deep=True,
                )
                if deep_seiza_data is None:
                    diagnostics.append("未能取得 Seiza 深度 Gaia G≤17 盲解数据。")
                else:
                    deep_tile, deep_message = _try_seiza_fallback_tile(
                        stars,
                        detector,
                        sky_mask,
                        info,
                        catalogs,
                        detector_scale_xy,
                        (solver_height, solver_width),
                        sensor_width_mm,
                        focal_mm,
                        progress,
                        seiza_data=deep_seiza_data,
                        allow_download=False,
                    )
                    if deep_tile is not None:
                        successful.append(deep_tile)
                        diagnostics.append(deep_message or "Seiza 深度 Gaia 盲解通过 W08 星表确认。")
                    else:
                        if deep_message:
                            diagnostics.append(deep_message)
                        wide_camera_result = _try_seiza_local_camera_fit(
                            stars,
                            detector,
                            sky_mask,
                            info,
                            catalogs,
                            detector_scale_xy,
                            (solver_height, solver_width),
                            sensor_width_mm,
                            sensor_height_mm,
                            focal_mm,
                            progress,
                            seiza_data=deep_seiza_data,
                        )
                        if wide_camera_result is not None:
                            for tile in wide_camera_result.verified_tiles:
                                if not any(
                                    (item.x0, item.y0, item.x1, item.y1)
                                    == (tile.x0, tile.y0, tile.x1, tile.y1)
                                    for item in successful
                                ):
                                    successful.append(tile)

        if progress:
            tile_summary = "; ".join(
                f"({tile.x0},{tile.y0}–{tile.x1},{tile.y1})"
                for tile in successful
            ) or "无"
            progress(
                54,
                f"本机 WCS 接受 {len(successful)} 个局部图块：{tile_summary}。"
                + (f" 边缘补充 {len([item for item in diagnostics if '慢速图块' in item])} 个。" if diagnostics else ""),
            )
        solver_source._mmap.close()

    final_camera_result = next(
        (
            result for result in (wide_camera_result, preliminary_camera_result)
            if result is not None and result.camera_projection is not None
        ),
        None,
    )
    if final_camera_result is not None:
        camera_projection = final_camera_result.camera_projection
        assert camera_projection is not None
        catalog_match_count = final_camera_result.catalog_match_count
        positions, pixel_scale = _positions_from_camera_projection(
            stars,
            detector,
            sky_mask,
            detector_scale_xy,
            (sensor_width_mm, sensor_height_mm),
            camera_projection,
        )
        if successful:
            local_positions = _positions_from_solved_tiles(stars, successful, detector_scale_xy)
            positions_by_source = {index: (index, ra, dec) for index, ra, dec in positions}
            positions_by_source.update({index: (index, ra, dec) for index, ra, dec in local_positions})
            positions = list(positions_by_source.values())
        if progress:
            progress(
                55,
                f"广角相机模型通过分布式 Gaia/W08 校验：{catalog_match_count} 个匹配，"
                f"拟合 P90 {camera_projection.residual_p90_px:.2f} solver px；"
                f"局部 WCS 覆盖 {len(successful)} 个图块，图像点源另行统计。",
            )
        return positions, pixel_scale, full_fov_height, successful, camera_projection

    if not successful:
        diagnostic_suffix = ""
        if diagnostics:
            diagnostic_suffix = " ASTAP 信息：" + "；".join(diagnostics[-3:])
        raise RuntimeError(
            "本机板解算未能为这张照片取得可信 WCS，也未使用画面亮度代替星表星等。"
            "宽视场结果需要足够的四星几何匹配或重叠图块互相确认。请确认照片包含清晰星点，且镜头焦距与相机型号 EXIF 完整。"
            + diagnostic_suffix
        )

    source_positions = _positions_from_solved_tiles(stars, successful, detector_scale_xy)
    pixel_scale = float(np.median([tile.pixel_scale_arcsec for tile in successful]))
    return source_positions, pixel_scale, full_fov_height, successful, None


def _recover_catalog_source_near_pixel(
    detector: np.ndarray,
    sky_mask: np.ndarray | None,
    expected_x: float,
    expected_y: float,
    search_radius: float,
    psf_fwhm: float,
    sensitivity: float,
) -> tuple[float, float, float, float, float, float, float, float, float, float] | None:
    """Use the WCS prediction as a prior, then require a compact local SEP source."""
    height, width = detector.shape
    if not (math.isfinite(expected_x) and math.isfinite(expected_y)):
        return None
    center_x, center_y = int(round(expected_x)), int(round(expected_y))
    if not (0 <= center_x < width and 0 <= center_y < height):
        return None
    if sky_mask is not None and sky_mask.shape == detector.shape and sky_mask[center_y, center_x]:
        return None

    half = max(12, int(math.ceil(search_radius + 7.0)))
    x0, x1 = max(0, center_x - half), min(width, center_x + half + 1)
    y0, y1 = max(0, center_y - half), min(height, center_y + half + 1)
    patch = np.asarray(detector[y0:y1, x0:x1], dtype=np.float32)
    if patch.shape[0] < 15 or patch.shape[1] < 15:
        return None
    patch_mask = None
    if sky_mask is not None and sky_mask.shape == detector.shape:
        patch_mask = np.asarray(sky_mask[y0:y1, x0:x1], dtype=bool)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    radius = np.hypot(xx - expected_x, yy - expected_y)
    valid = np.isfinite(patch)
    if patch_mask is not None:
        valid &= ~patch_mask
    background_pixels = patch[(radius >= max(6.0, search_radius + 2.0)) & (radius <= half - 1.0) & valid]
    source_pixels = patch[(radius <= search_radius) & valid]
    if background_pixels.size < 24 or source_pixels.size < 4:
        return None
    local_background = float(np.median(background_pixels))
    mad = float(np.median(np.abs(background_pixels - local_background)))
    local_noise = max(1.4826 * mad, 1e-8)
    peak = float(np.max(source_pixels))
    if peak - local_background < max(2.5, min(float(sensitivity), 5.0) * 0.75) * local_noise:
        return None

    safe_patch = np.nan_to_num(patch, nan=local_background, posinf=local_background, neginf=local_background)
    block_x = max(4, min(16, safe_patch.shape[1] // 2))
    block_y = max(4, min(16, safe_patch.shape[0] // 2))
    try:
        background = sep.Background(safe_patch, mask=patch_mask, bw=block_x, bh=block_y, fw=3, fh=3)
        signal = np.ascontiguousarray(safe_patch - background.back(), dtype=np.float32)
        noise = np.ascontiguousarray(np.maximum(background.rms(), local_noise), dtype=np.float32)
        objects = sep.extract(
            signal,
            max(2.5, min(float(sensitivity), 5.0) * 0.75),
            err=noise,
            mask=patch_mask,
            minarea=3,
            deblend_nthresh=16,
            deblend_cont=0.005,
            clean=True,
            segmentation_map=False,
        )
    except Exception:
        return None
    if len(objects) == 0:
        return None

    fwhm = 2.354820045 * np.sqrt(np.maximum(objects["a"] * objects["b"], 0.0))
    major = np.maximum(objects["a"], objects["b"])
    minor = np.minimum(objects["a"], objects["b"])
    roundness = minor / np.maximum(major, 1e-8)
    distance = np.hypot(objects["x"] + x0 - expected_x, objects["y"] + y0 - expected_y)
    psf_sigma = max(float(psf_fwhm) / 2.354820045, 0.6)
    compact = (
        np.isfinite(distance)
        & (distance <= search_radius)
        & (minor >= 0.35)
        & (major <= max(16.0, 8.0 * psf_sigma))
        & (roundness >= 0.15)
        & np.isfinite(fwhm)
        & (fwhm >= 0.7)
    )
    indices = np.flatnonzero(compact)
    if not len(indices):
        return None
    aperture = np.clip(1.25 * fwhm[indices], 2.5, 12.0)
    flux, flux_error, flags = sep.sum_circle(
        signal,
        objects["x"][indices],
        objects["y"][indices],
        aperture,
        err=noise,
        mask=patch_mask,
        subpix=5,
    )
    snr = flux / np.maximum(flux_error, 1e-20)
    valid_measurements = (
        np.isfinite(flux)
        & (flux > 0)
        & np.isfinite(snr)
        & (snr >= max(4.0, float(sensitivity) * 0.75))
        & ((flags & sep.APER_TRUNC) == 0)
    )
    snr_values = snr[valid_measurements]
    flux_values = flux[valid_measurements]
    indices = indices[valid_measurements]
    if not len(indices):
        return None
    scores = distance[indices] / max(search_radius, 1.0) - 0.02 * np.log1p(np.maximum(snr_values, 0.0))
    measurement_index = int(np.argmin(scores))
    index = int(indices[measurement_index])
    x = float(objects["x"][index] + x0)
    y = float(objects["y"][index] + y0)
    return (
        x,
        y,
        float(flux_values[measurement_index]),
        float(objects["peak"][index]),
        float(fwhm[index]),
        float(objects["a"][index]),
        float(objects["b"][index]),
        float(objects["theta"][index]),
        float(snr_values[measurement_index]),
        float(distance[index]),
    )


def match_local_bright_stars(
    stars: Sequence[object],
    detector: np.ndarray,
    sky_mask: np.ndarray | None,
    info: object,
    solver_image_path: str | Path,
    detector_scale_xy: tuple[float, float],
    primary_image_path: str | Path | None = None,
    relative_magnitude_limit: float = 5.0,
    sensitivity: float = 4.8,
    progress: Progress | None = None,
) -> tuple[
    dict[int, CatalogMatch], int, list[RecoveredCatalogStar],
    tuple[CatalogPosition, ...], tuple[CatalogCoverageArea, ...],
]:
    """Solve locally and match against ASTAP's bundled Gaia-derived W08 bright-star index."""
    (
        positions,
        pixel_scale_arcsec,
        full_fov_height,
        successful_tiles,
        known_camera_projection,
    ) = _positions_to_sky(
        stars,
        detector,
        sky_mask,
        info,
        Path(solver_image_path),
        detector_scale_xy,
        Path(primary_image_path) if primary_image_path else None,
        progress,
    )
    detector_height, detector_width = detector.shape
    detector_scale_x, detector_scale_y = detector_scale_xy
    local_coverage_areas = tuple(
        CatalogCoverageArea(
            float(np.clip(tile.x0 / max(detector_scale_x, 1e-8), 0.0, detector_width)),
            float(np.clip(tile.y0 / max(detector_scale_y, 1e-8), 0.0, detector_height)),
            float(np.clip(tile.x1 / max(detector_scale_x, 1e-8), 0.0, detector_width)),
            float(np.clip(tile.y1 / max(detector_scale_y, 1e-8), 0.0, detector_height)),
            "local WCS tile",
        )
        for tile in successful_tiles
    )
    if known_camera_projection is not None:
        coverage_areas = (
            CatalogCoverageArea(0.0, 0.0, float(detector_width), float(detector_height), "camera projection"),
            *local_coverage_areas,
        )
    else:
        coverage_areas = local_coverage_areas
    if progress:
        progress(54, "正在本机读取 Gaia 亮星星表并匹配坐标…")
    _executable, catalogs = _executable_and_catalogs()
    catalog_path = next(iter(sorted(catalogs.glob("w08_*.001"))), None)
    if catalog_path is None:
        raise RuntimeError("程序包缺少 ASTAP W08 本地亮星索引；请重新下载完整版本并解压。")
    catalog_magnitudes, catalog_vectors = _load_w08_catalog(str(catalog_path.resolve()))
    if not len(catalog_magnitudes):
        return {}, len(positions), [], (), coverage_areas

    # W08 is an all-sky bright-star subset, complete to approximately G=8.
    # Its catalogue positions and rounded G magnitudes are bundled with the app;
    # matching therefore works offline and never sends the photograph or WCS.
    # ASTAP's W08 index is a bright-star reference for wide fields, where lens
    # distortion can leave a few detector-pixel residual even after SIP fit.
    # Use a pixel-scale-derived cone wide enough for that calibrated residual;
    # narrow fields retain a tighter tolerance to avoid ambiguous neighbours.
    radius_scale = 5.0 if full_fov_height > 20.0 else 2.0
    query_radius_arcsec = float(np.clip(pixel_scale_arcsec * radius_scale, 10.0, 240.0))
    if known_camera_projection is not None:
        # The rectilinear camera pose is validated by distributed catalogue
        # matches, but wide-lens residuals still grow toward the frame edges.
        # Use its measured P90 as a position prior for associating SEP sources
        # and recovering missed detections; cap the radius to limit ambiguous
        # neighbours in crowded fields. Image evidence is still mandatory.
        residual_guided_radius = (
            float(known_camera_projection.residual_p90_px) + 2.0
        ) * max(float(pixel_scale_arcsec), 1e-8)
        query_radius_arcsec = float(np.clip(
            max(query_radius_arcsec, residual_guided_radius), 10.0, 600.0
        ))
    tree = cKDTree(catalog_vectors)
    chord_limit = 2.0 * math.sin(math.radians(query_radius_arcsec / 3600.0) * 0.5)
    nearest_pairs: list[tuple[float, int, int]] = []
    if positions:
        source_vectors = _sky_unit_vectors(positions)
        distances, row_indices = tree.query(source_vectors, k=1, distance_upper_bound=chord_limit)
        for position, chord, row_index in zip(positions, distances, row_indices):
            row_index = int(row_index)
            if row_index >= len(catalog_magnitudes) or not math.isfinite(float(chord)):
                continue
            separation_arcsec = math.degrees(2.0 * math.asin(min(float(chord), 2.0) * 0.5)) * 3600.0
            nearest_pairs.append((separation_arcsec, int(position[0]), row_index))

    used_detections: set[int] = set()
    used_catalog_sources: set[int] = set()
    matches: dict[int, CatalogMatch] = {}
    for separation, det_id, row_index in sorted(nearest_pairs):
        if det_id in used_detections or row_index in used_catalog_sources:
            continue
        ra = math.degrees(math.atan2(catalog_vectors[row_index, 1], catalog_vectors[row_index, 0])) % 360.0
        dec = math.degrees(math.asin(float(np.clip(catalog_vectors[row_index, 2], -1.0, 1.0))))
        matches[det_id] = CatalogMatch(
            None, ra, dec, float(catalog_magnitudes[row_index]), None, None, separation,
            row_index, _named_star_label(ra, dec),
        )
        used_detections.add(det_id)
        used_catalog_sources.add(row_index)

    reference_g = min((match.g_mag for match in matches.values()), default=math.inf)
    maximum_g = min(8.0, reference_g + float(relative_magnitude_limit) + 0.2)
    if not math.isfinite(reference_g):
        maximum_g = 8.0
    detector_scale_x, detector_scale_y = detector_scale_xy
    # ASTAP and Seiza tiles use full-resolution solver-image pixels, while
    # SEP coordinates are binned detector pixels. One detector pixel spans
    # more solver pixels, so its angular scale must be multiplied here.
    detector_pixel_scale = pixel_scale_arcsec * max(
        math.sqrt(detector_scale_x * detector_scale_y), 1e-8
    )
    # Wide-angle lens distortion can leave bright edge stars several detector
    # pixels from the pinhole/WCS prior. Search farther around validated
    # projections, but still require an actual compact SEP source in the image.
    recovery_radius_px = float(np.clip(query_radius_arcsec / max(detector_pixel_scale, 1e-8), 3.0, 18.0))
    detected_xy = np.asarray(
        [(float(getattr(star, "x")), float(getattr(star, "y"))) for star in stars],
        dtype=np.float64,
    ).reshape((-1, 2))
    detection_tree = cKDTree(detected_xy) if len(detected_xy) else None
    psf_fwhm = float(np.median([float(getattr(star, "fwhm", 3.0)) for star in stars])) if stars else 3.0

    # In wide fields each accepted tile has its own distortion fit. Project the
    # catalogue back into every tile, prefer predictions furthest from a tile
    # edge, then inspect nearby image pixels for compact sources SEP missed.
    projected: dict[int, tuple[float, float, float]] = {}
    for tile in successful_tiles:
        tile_width, tile_height = tile.x1 - tile.x0, tile.y1 - tile.y0
        sample_pixels = np.asarray([
            [tile_width / 2.0, tile_height / 2.0],
            [-0.5, -0.5], [tile_width - 0.5, -0.5],
            [-0.5, tile_height - 0.5], [tile_width - 0.5, tile_height - 0.5],
            [tile_width / 2.0, -0.5], [tile_width / 2.0, tile_height - 0.5],
            [-0.5, tile_height / 2.0], [tile_width - 0.5, tile_height / 2.0],
        ], dtype=np.float64)
        try:
            sample_sky = _tile_pixel_to_world(tile, sample_pixels)
        except Exception:
            continue
        if sample_sky.shape[0] < 5 or not np.isfinite(sample_sky[0]).all():
            continue
        center_ra, center_dec = np.radians(sample_sky[0])
        center_vector = np.asarray([
            math.cos(center_dec) * math.cos(center_ra),
            math.cos(center_dec) * math.sin(center_ra),
            math.sin(center_dec),
        ])
        edge_vectors = _sky_unit_vectors([
            (i, float(world[0]) % 360.0, float(world[1]))
            for i, world in enumerate(sample_sky[1:])
            if np.isfinite(world).all() and -90.0 <= world[1] <= 90.0
        ])
        if not len(edge_vectors):
            continue
        edge_angles = np.degrees(np.arccos(np.clip(edge_vectors @ center_vector, -1.0, 1.0)))
        cone_radius = min(175.0, float(np.max(edge_angles)) + 4.0)
        cone_chord = 2.0 * math.sin(math.radians(cone_radius) * 0.5)
        catalog_rows = np.asarray(tree.query_ball_point(center_vector, cone_chord), dtype=np.int64)
        if not len(catalog_rows):
            continue
        catalog_rows = catalog_rows[catalog_magnitudes[catalog_rows] <= maximum_g]
        if not len(catalog_rows):
            continue
        vectors = catalog_vectors[catalog_rows]
        ras = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0])) % 360.0
        decs = np.degrees(np.arcsin(np.clip(vectors[:, 2], -1.0, 1.0)))
        world = np.column_stack((ras, decs))
        try:
            tile_pixels = _tile_world_to_pixel(tile, world)
        except Exception:
            continue
        finite = np.isfinite(tile_pixels).all(axis=1)
        inside = (
            finite
            & (tile_pixels[:, 0] >= -recovery_radius_px)
            & (tile_pixels[:, 0] <= tile_width + recovery_radius_px)
            & (tile_pixels[:, 1] >= -recovery_radius_px)
            & (tile_pixels[:, 1] <= tile_height + recovery_radius_px)
        )
        for local, row in enumerate(catalog_rows[inside]):
            px, py = tile_pixels[inside][local]
            edge_margin = min(
                px / max(tile_width, 1), (tile_width - px) / max(tile_width, 1),
                py / max(tile_height, 1), (tile_height - py) / max(tile_height, 1),
            )
            full_x, full_y = tile.x0 + float(px), tile.y0 + float(py)
            previous = projected.get(int(row))
            if previous is None or edge_margin > previous[0]:
                projected[int(row)] = (edge_margin, full_x / detector_scale_x, full_y / detector_scale_y)

    solver_width = max(1, int(round(detector.shape[1] * detector_scale_x)))
    solver_height = max(1, int(round(detector.shape[0] * detector_scale_y)))
    camera_projection = known_camera_projection or _fit_wide_camera_projection(
        stars, matches, successful_tiles, detector.shape, detector_scale_xy, info
    )
    if camera_projection is not None:
        sensor_width_mm, sensor_height_mm, _focal_mm = _sensor_dimensions(info, detector.shape)
        target_rows = np.flatnonzero(catalog_magnitudes <= maximum_g)
        if len(target_rows):
            projected_pixels, z = _project_camera_catalog(
                catalog_vectors[target_rows], camera_projection,
                solver_width, solver_height,
                sensor_width_mm, sensor_height_mm,
            )
            full_x, full_y = projected_pixels[:, 0], projected_pixels[:, 1]
            inside = (
                np.isfinite(projected_pixels).all(axis=1)
                & np.isfinite(z)
                & (z > 1e-6)
                & np.isfinite(full_x)
                & np.isfinite(full_y)
                & (full_x >= -recovery_radius_px * detector_scale_x)
                & (full_x <= solver_width + recovery_radius_px * detector_scale_x)
                & (full_y >= -recovery_radius_px * detector_scale_y)
                & (full_y <= solver_height + recovery_radius_px * detector_scale_y)
            )
            for row, x, y in zip(target_rows[inside], full_x[inside], full_y[inside]):
                # Local WCS projections always outrank this extrapolation.
                # It fills only the image area for which ASTAP found no local tile.
                projected.setdefault(
                    int(row),
                    (-2.0, float(x) / detector_scale_x, float(y) / detector_scale_y),
                )
            if progress:
                progress(
                    55,
                    f"可信本机相机模型拟合误差 P90 {camera_projection.residual_p90_px:.1f} px，"
                    f"正在按相机视场扩展 Gaia 星表位置…",
                )

    recovered: list[RecoveredCatalogStar] = []
    recovered_xy = np.empty((0, 2), dtype=np.float64)
    for row_index, (_edge_margin, expected_x, expected_y) in projected.items():
        if row_index in used_catalog_sources:
            continue
        ra = math.degrees(math.atan2(catalog_vectors[row_index, 1], catalog_vectors[row_index, 0])) % 360.0
        dec = math.degrees(math.asin(float(np.clip(catalog_vectors[row_index, 2], -1.0, 1.0))))
        if detection_tree is not None:
            distance, det_id = detection_tree.query([expected_x, expected_y], k=1)
            det_id = int(det_id)
            if float(distance) <= recovery_radius_px and det_id not in used_detections:
                matches[det_id] = CatalogMatch(
                    None, ra, dec, float(catalog_magnitudes[row_index]), None, None,
                    float(distance) * detector_pixel_scale, row_index,
                    _named_star_label(ra, dec), True,
                )
                used_detections.add(det_id)
                used_catalog_sources.add(row_index)
                continue
            if float(distance) <= recovery_radius_px and det_id in used_detections:
                continue

        local_source = _recover_catalog_source_near_pixel(
            detector,
            sky_mask,
            expected_x,
            expected_y,
            recovery_radius_px,
            psf_fwhm,
            sensitivity,
        )
        if local_source is None:
            continue
        x, y, flux, peak, fwhm, axis_a, axis_b, theta, snr, separation_px = local_source
        if len(recovered_xy):
            duplicate_radius = max(2.0, min(float(fwhm), 5.0))
            if float(np.min(np.hypot(recovered_xy[:, 0] - x, recovered_xy[:, 1] - y))) < duplicate_radius:
                continue
        recovered.append(RecoveredCatalogStar(
            float(catalog_magnitudes[row_index]),
            x,
            y,
            flux,
            peak,
            fwhm,
            axis_a,
            axis_b,
            theta,
            snr,
            separation_px * detector_pixel_scale,
            int(row_index),
            ra,
            dec,
            _named_star_label(ra, dec),
        ))
        recovered_xy = np.vstack((recovered_xy, [x, y]))
        used_catalog_sources.add(row_index)
    catalog_positions: dict[int, CatalogPosition] = {}
    for detector_id, match in matches.items():
        row_index = match.catalog_row_index
        if row_index is None or not (0 <= detector_id < len(stars)):
            continue
        source = stars[detector_id]
        catalog_positions[row_index] = CatalogPosition(
            float(getattr(source, "x")),
            float(getattr(source, "y")),
            match.g_mag,
            "recovered" if match.position_recovered else "detected",
            match.ra_deg,
            match.dec_deg,
            match.name or _named_star_label(match.ra_deg, match.dec_deg),
            "WCS-guided image source" if match.position_recovered else "image detection",
        )
    for source in recovered:
        catalog_positions[source.catalog_row_index] = CatalogPosition(
            source.x,
            source.y,
            source.g_mag,
            "recovered",
            source.ra_deg,
            source.dec_deg,
            source.name,
            "catalog-guided image recovery",
        )
    for row_index, (_edge_margin, expected_x, expected_y) in projected.items():
        if row_index in catalog_positions or not (0.0 <= expected_x < detector.shape[1] and 0.0 <= expected_y < detector.shape[0]):
            continue
        center_x = int(np.clip(round(expected_x), 0, detector.shape[1] - 1))
        center_y = int(np.clip(round(expected_y), 0, detector.shape[0] - 1))
        if sky_mask is not None and sky_mask.shape == detector.shape and sky_mask[center_y, center_x]:
            continue
        ra = math.degrees(math.atan2(catalog_vectors[row_index, 1], catalog_vectors[row_index, 0])) % 360.0
        dec = math.degrees(math.asin(float(np.clip(catalog_vectors[row_index, 2], -1.0, 1.0))))
        catalog_positions[row_index] = CatalogPosition(
            float(expected_x),
            float(expected_y),
            float(catalog_magnitudes[row_index]),
            "predicted",
            ra,
            dec,
            _named_star_label(ra, dec),
            "camera projection" if camera_projection is not None else "local WCS tile",
        )
    return (
        matches,
        len(positions),
        recovered,
        tuple(sorted(catalog_positions.values(), key=lambda item: (item.g_mag, item.y, item.x))),
        coverage_areas,
    )


def _sky_unit_vectors(positions: Sequence[tuple[int, float, float]]) -> np.ndarray:
    ra = np.radians(np.asarray([position[1] for position in positions], dtype=np.float64))
    dec = np.radians(np.asarray([position[2] for position in positions], dtype=np.float64))
    cos_dec = np.cos(dec)
    return np.column_stack((cos_dec * np.cos(ra), cos_dec * np.sin(ra), np.sin(dec)))


@lru_cache(maxsize=2)
def _load_w08_catalog(catalog_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load ASTAP W08 records: little-endian star count, then G*10, RA rad, Dec rad."""
    path = Path(catalog_path)
    with path.open("rb") as handle:
        header = np.fromfile(handle, dtype="<i4", count=1)
        records = np.fromfile(handle, dtype="<f4")
    if header.size != 1 or header[0] <= 0 or records.size != int(header[0]) * 3:
        raise RuntimeError("ASTAP W08 本地亮星索引格式无效；请重新下载完整版本并解压。")
    records = records.reshape((-1, 3))
    valid = (
        np.isfinite(records).all(axis=1)
        & (records[:, 0] <= 80.0)
        & (records[:, 1] >= 0.0)
        & (records[:, 1] <= (2.0 * math.pi + 1e-5))
        & (records[:, 2] >= -math.pi / 2.0)
        & (records[:, 2] <= math.pi / 2.0)
    )
    records = records[valid]
    magnitudes = np.asarray(records[:, 0] / 10.0, dtype=np.float64)
    ra = records[:, 1].astype(np.float64)
    dec = records[:, 2].astype(np.float64)
    cos_dec = np.cos(dec)
    vectors = np.column_stack((cos_dec * np.cos(ra), cos_dec * np.sin(ra), np.sin(dec)))
    return magnitudes, vectors
