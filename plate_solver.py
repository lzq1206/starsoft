"""Bundled ASTAP plate solving and local Gaia bright-star matching."""
from __future__ import annotations

import math
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import sep
import tifffile
from astropy.io import fits
from astropy.wcs import WCS
from scipy.spatial import cKDTree


Progress = Callable[[int, str], None]


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


@dataclass(frozen=True)
class CatalogMatch:
    source_id: int | None
    ra_deg: float
    dec_deg: float
    g_mag: float
    bp_mag: float | None
    rp_mag: float | None
    separation_arcsec: float


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
    sensor = next(
        (dimensions for model, dimensions in _SENSOR_SIZES_MM.items() if model in camera),
        None,
    )
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


def _extract_astap_wcs(path: Path, expected_scale_arcsec: float) -> WCS | None:
    ini_path = path.with_suffix(".ini")
    wcs_path = path.with_suffix(".wcs")
    if not ini_path.exists() or not wcs_path.exists():
        return None
    ini_text = ini_path.read_text(encoding="utf-8", errors="replace")
    if "PLTSOLVD=T" not in ini_text:
        return None
    if "scale was inaccurate" in ini_text.lower():
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
    if ratio < 0.55 or ratio > 1.8:
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
        tile.matched_quads >= 6
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


def _positions_to_sky(
    stars: Sequence[object],
    detector: np.ndarray,
    sky_mask: np.ndarray | None,
    info: object,
    solver_image_path: Path,
    detector_scale_xy: tuple[float, float],
    primary_image_path: Path | None,
    progress: Progress | None,
) -> tuple[list[tuple[int, float, float]], float, float, list[SolverTile]]:
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

    valid_height = solver_height
    if sky_mask is not None and sky_mask.shape == detector.shape:
        valid_rows = np.mean(~np.asarray(sky_mask, dtype=bool), axis=1)
        sky_rows = np.flatnonzero(valid_rows >= 0.03)
        if len(sky_rows):
            valid_height = min(solver_height, int(math.ceil((int(sky_rows[-1]) + 1) * detector_scale_y)))
    # ASTAP's -fov parameter is the image-height field, so a wide landscape
    # sensor may have a horizontal field above 80 degrees and still be a
    # supported single-frame solve.
    can_solve_full = full_fov_height <= 80.0
    boxes: list[tuple[int, int, int, int]] = []
    if can_solve_full:
        boxes.append((0, 0, solver_width, solver_height))

    # ASTAP W08 supports fields up to 80 degrees. Use overlapping full-detail
    # crops for very wide views, and as a fallback if the full image is too
    # distorted or contains too much foreground for one reliable solve.
    tile_w_fraction = min(0.75, 58.0 / max(full_fov_width, 1.0))
    tile_h_fraction = min(0.65, 58.0 / max(full_fov_height, 1.0))
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
        tile_fov_x = _axis_fov_degrees(x0, x1, solver_width, sensor_width_mm, focal_mm)
        tile_fov_y = _axis_fov_degrees(y0, y1, solver_height, sensor_height_mm, focal_mm)
        distortion_penalty = max(0.0, max(tile_fov_x, tile_fov_y) - 35.0) / 45.0
        return (
            (min(star_count, 500) + 5.0) * (0.2 + 0.8 * sky_fraction)
            / (1.0 + 0.6 * center_distance + 0.5 * distortion_penalty)
        )

    # Start with star-rich sky near the optical axis. Full-frame blind solves
    # are retained as candidates, but wide cropped fields generally have less
    # distortion and solve more quickly.
    boxes = sorted(set(boxes), key=box_priority, reverse=True)

    successful: list[SolverTile] = []
    wide_candidates: list[SolverTile] = []
    diagnostics: list[str] = []
    solve_deadline = time.monotonic() + 90.0
    fast_deadline = solve_deadline - 35.0
    with tempfile.TemporaryDirectory(prefix="starsoft-solve-") as temporary:
        work_dir = Path(temporary)
        solver_source = tifffile.memmap(solver_image_path, mode="r")
        trial_specs: dict[
            tuple[int, int, int, int], tuple[Path, float, bool]
        ] = {}
        # Prepare every eligible crop before launching ASTAP. Each process gets
        # its own working directory and output prefix; crops and the catalog are
        # read-only inputs, so independent fields can be solved concurrently.
        for index, (x0, y0, x1, y1) in enumerate(boxes):
            tile_h, tile_w = y1 - y0, x1 - x0
            if tile_h < 900 or tile_w < 900:
                continue
            fov_y = _axis_fov_degrees(y0, y1, solver_height, sensor_height_mm, focal_mm)
            fov_x = _axis_fov_degrees(x0, x1, solver_width, sensor_width_mm, focal_mm)
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
                    str(image_path), solver_source[y0:y1, x0:x1],
                    compression=None, photometric="minisblack",
                )
            wide_field = max(fov_x, fov_y) > 20.0
            trial_specs[(x0, y0, x1, y1)] = (image_path, expected_scale, wide_field)
        def build_command(
            box: tuple[int, int, int, int],
            index: int,
            seed_candidates: Sequence[SolverTile],
            output_base: Path,
        ) -> list[str]:
            x0, y0, x1, y1 = box
            image_path, _expected_scale, wide_field = trial_specs[box]
            fov_y = _axis_fov_degrees(y0, y1, solver_height, sensor_height_mm, focal_mm)
            command = [
                str(executable), "-f", str(image_path), "-fov", f"{fov_y:.5f}",
                "-d", str(catalogs), "-z", "0",
                "-speed", "auto",
                "-s", "500" if wide_field else "1000",
                "-t", "0.007", "-wcs", "-sip", "-log", "-o", str(output_base),
            ]
            if wide_field:
                command[command.index("-z"):command.index("-z")] = ["-D", "w08"]
                tile_center = ((x0 + x1) * 0.5, (y0 + y1) * 0.5)
                seed_tiles = [
                    tile for tile in seed_candidates
                    if _wide_tile_is_strong(tile)
                    and tile.x0 <= tile_center[0] <= tile.x1
                    and tile.y0 <= tile_center[1] <= tile.y1
                ]
                if seed_tiles:
                    seed = min(
                        seed_tiles,
                        key=lambda tile: (tile.x1 - tile.x0) * (tile.y1 - tile.y0),
                    )
                    try:
                        seed_pixel_x = float(np.clip(
                            tile_center[0] - seed.x0,
                            0.0,
                            max(seed.x1 - seed.x0 - 1, 0),
                        ))
                        seed_pixel_y = float(np.clip(
                            tile_center[1] - seed.y0,
                            0.0,
                            max(seed.y1 - seed.y0 - 1, 0),
                        ))
                        seed_world = _tile_pixel_to_world(
                            seed, np.asarray([[seed_pixel_x, seed_pixel_y]])
                        )[0]
                        seed_ra = float(seed_world[0]) % 360.0
                        seed_dec = float(seed_world[1])
                        if math.isfinite(seed_ra) and math.isfinite(seed_dec) and -90 <= seed_dec <= 90:
                            search_radius = max(3.0, min(15.0, fov_y * 0.25))
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
            image_path, expected_scale, wide_field = trial_specs[box]
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

            solved_wcs = _extract_astap_wcs(output_base, expected_scale)
            if solved_wcs is not None:
                matched_quads, total_quads = _astap_quad_counts(
                    output_base.with_suffix(".log"), completed.stdout
                )
                candidate = SolverTile(
                    *box, solved_wcs, expected_scale, wide_field,
                    matched_quads, total_quads,
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
        worker_count = max(1, min(4, max(1, os.cpu_count() or 1), len(eligible_boxes)))
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

        has_wide_trials = any(spec[2] for spec in trial_specs.values())
        if has_wide_trials and not _trusted_wide_tiles(wide_candidates):
            remaining_seconds = solve_deadline - time.monotonic()
            if progress and remaining_seconds > 0:
                progress(53, "快速宽场搜索未得到可信结果，正在补做一次重叠搜索…")
            if wide_candidates:
                best_candidate = max(
                    wide_candidates,
                    key=lambda tile: (
                        tile.matched_quads / max(tile.total_quads, 1),
                        tile.matched_quads,
                    ),
                )
                fallback_box = (best_candidate.x0, best_candidate.y0, best_candidate.x1, best_candidate.y1)
            else:
                fallback_box = next(
                    (box for box in boxes if box in trial_specs and trial_specs[box][2]),
                    None,
                )
            if remaining_seconds > 0 and fallback_box is not None:
                x0, y0, x1, y1 = fallback_box
                image_path, expected_scale, _wide_field = trial_specs[fallback_box]
                output_base = work_dir / "solution_slow_fallback"
                fov_y = _axis_fov_degrees(y0, y1, solver_height, sensor_height_mm, focal_mm)
                command = [
                    str(executable), "-f", str(image_path), "-fov", f"{fov_y:.5f}",
                    "-d", str(catalogs), "-D", "w08", "-z", "0",
                    "-speed", "slow", "-s", "500", "-t", "0.007",
                    "-wcs", "-sip", "-log", "-o", str(output_base),
                ]
                try:
                    completed = subprocess.run(
                        command,
                        cwd=str(work_dir),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        errors="replace",
                        check=False,
                        timeout=min(40.0, remaining_seconds),
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                except subprocess.TimeoutExpired:
                    diagnostics.append(f"{image_path.name}: ASTAP 慢速补充搜索超时。")
                else:
                    solved_wcs = _extract_astap_wcs(output_base, expected_scale)
                    if solved_wcs is not None:
                        matched_quads, total_quads = _astap_quad_counts(
                            output_base.with_suffix(".log"), completed.stdout
                        )
                        wide_candidates.append(SolverTile(
                            x0, y0, x1, y1, solved_wcs, expected_scale, True,
                            matched_quads, total_quads,
                        ))
                    if completed.stdout:
                        useful_lines = [
                            line.strip() for line in completed.stdout.splitlines()
                            if line.strip() and any(token in line.lower() for token in (
                                "error", "fail", "no solution", "cannot", "not found", "inaccurate", "stars"
                            ))
                        ]
                        if useful_lines:
                            diagnostics.append(f"{image_path.name} 慢速补充搜索: {' | '.join(useful_lines[-4:])}")
                    successful.extend(_trusted_wide_tiles(wide_candidates))
        solver_source._mmap.close()

    if not successful:
        diagnostic_suffix = ""
        if diagnostics:
            diagnostic_suffix = " ASTAP 信息：" + "；".join(diagnostics[-3:])
        raise RuntimeError(
            "本机板解算未能为这张照片取得可信 WCS，未调用 Gaia 星表，也未使用画面亮度代替星表星等。"
            "宽视场结果需要足够的四星几何匹配或重叠图块互相确认。请确认照片包含清晰星点，且镜头焦距与相机型号 EXIF 完整。"
            + diagnostic_suffix
        )

    source_positions: list[tuple[int, float, float]] = []
    for index, star in enumerate(stars):
        x = float(getattr(star, "x")) * detector_scale_x
        y = float(getattr(star, "y")) * detector_scale_y
        valid_tiles = [tile for tile in successful if tile.x0 <= x < tile.x1 and tile.y0 <= y < tile.y1]
        if not valid_tiles:
            continue
        tile = min(
            valid_tiles,
            key=lambda item: ((x - (item.x0 + item.x1) / 2.0) / max(item.x1 - item.x0, 1)) ** 2
                             + ((y - (item.y0 + item.y1) / 2.0) / max(item.y1 - item.y0, 1)) ** 2,
        )
        try:
            sky = _tile_pixel_to_world(
                tile, np.asarray([[x - tile.x0, y - tile.y0]])
            )[0]
            ra, dec = float(sky[0]) % 360.0, float(sky[1])
        except Exception:
            continue
        if math.isfinite(ra) and math.isfinite(dec) and -90.0 <= dec <= 90.0:
            source_positions.append((index, ra, dec))
    pixel_scale = float(np.median([tile.pixel_scale_arcsec for tile in successful]))
    return source_positions, pixel_scale, full_fov_height, successful


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
) -> tuple[dict[int, CatalogMatch], int, list[RecoveredCatalogStar]]:
    """Solve locally and match against ASTAP's bundled Gaia-derived W08 bright-star index."""
    positions, pixel_scale_arcsec, full_fov_height, successful_tiles = _positions_to_sky(
        stars,
        detector,
        sky_mask,
        info,
        Path(solver_image_path),
        detector_scale_xy,
        Path(primary_image_path) if primary_image_path else None,
        progress,
    )
    if progress:
        progress(54, "正在本机读取 Gaia 亮星星表并匹配坐标…")
    _executable, catalogs = _executable_and_catalogs()
    catalog_path = next(iter(sorted(catalogs.glob("w08_*.001"))), None)
    if catalog_path is None:
        raise RuntimeError("程序包缺少 ASTAP W08 本地亮星索引；请重新下载完整版本并解压。")
    catalog_magnitudes, catalog_vectors = _load_w08_catalog(str(catalog_path.resolve()))
    if not len(catalog_magnitudes):
        return {}, len(positions), []

    # W08 is an all-sky bright-star subset, complete to approximately G=8.
    # Its catalogue positions and rounded G magnitudes are bundled with the app;
    # matching therefore works offline and never sends the photograph or WCS.
    # ASTAP's W08 index is a bright-star reference for wide fields, where lens
    # distortion can leave a few detector-pixel residual even after SIP fit.
    # Use a pixel-scale-derived cone wide enough for that calibrated residual;
    # narrow fields retain a tighter tolerance to avoid ambiguous neighbours.
    radius_scale = 5.0 if full_fov_height > 20.0 else 2.0
    query_radius_arcsec = float(np.clip(pixel_scale_arcsec * radius_scale, 10.0, 240.0))
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
            None, ra, dec, float(catalog_magnitudes[row_index]), None, None, separation
        )
        used_detections.add(det_id)
        used_catalog_sources.add(row_index)

    reference_g = min((match.g_mag for match in matches.values()), default=math.inf)
    maximum_g = min(8.0, reference_g + float(relative_magnitude_limit) + 0.2)
    if not math.isfinite(reference_g):
        maximum_g = 8.0
    detector_scale_x, detector_scale_y = detector_scale_xy
    detector_pixel_scale = pixel_scale_arcsec / max(math.sqrt(detector_scale_x * detector_scale_y), 1e-8)
    recovery_radius_px = float(np.clip(query_radius_arcsec / max(detector_pixel_scale, 1e-8), 3.0, 10.0))
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
                    float(distance) * detector_pixel_scale,
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
        ))
        recovered_xy = np.vstack((recovered_xy, [x, y]))
        used_catalog_sources.add(row_index)
    return matches, len(positions), recovered


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
