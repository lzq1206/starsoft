"""Star detection and brightness-weighted soft-focus processing."""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
import xml.etree.ElementTree as ET
from io import BytesIO
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
import rawpy
import sep
import tifffile
from scipy.ndimage import gaussian_filter, zoom
from PIL import ExifTags, Image, ImageCms, ImageDraw, ImageFont, ImageOps

from version import APP_VERSION
from plate_solver import CatalogCoverageArea, CatalogPosition, match_local_bright_stars


Progress = Callable[[int, str], None]
# Measured by SEP globalback on the sky-masked LZQ_9331.tif detector image.
# Keep this on the same masked sky region used for per-image measurements.
REFERENCE_SKY_LEVEL = 0.17277


@dataclass(frozen=True)
class RawInfo:
    width: int
    height: int
    camera: str
    lens: str
    focal_length: float | None
    aperture: float | None
    preview: Image.Image | None
    focal_length_35mm: float | None = None
    sensor_format: str | None = None


MetadataCallback = Callable[[RawInfo], None]
_SEP_EXTRACT_LOCK = threading.Lock()


def _extract_sources(data: np.ndarray, threshold: float, **kwargs: object) -> np.ndarray:
    # SEP's default active-pixel stack is only 300,000 pixels. JPEG ringing,
    # sensor noise, or a dense field can exceed it; size it to the detector and
    # serialize the process-global SEP setting while extraction runs.
    with _SEP_EXTRACT_LOCK:
        required = int(data.size)
        if sep.get_extract_pixstack() < required:
            sep.set_extract_pixstack(required)
        return sep.extract(data, threshold, **kwargs)


@dataclass(frozen=True)
class Star:
    x: float
    y: float
    flux: float
    peak: float
    fwhm: float
    relative_flux_ratio: float = 1.0
    a: float = 1.0
    b: float = 1.0
    theta: float = 0.0
    signal_to_noise: float = 0.0
    crowded_field: bool = False
    image_flux: float | None = None
    catalog_g_mag: float | None = None
    catalog_bp_rp: float | None = None
    catalog_source_id: int | None = None
    catalog_delta_magnitude: float | None = None
    catalog_position_recovered: bool = False
    catalog_name: str | None = None
    catalog_ra_deg: float | None = None
    catalog_dec_deg: float | None = None
    catalog_position_state: str | None = None


@dataclass(frozen=True)
class ProcessResult:
    output_path: str
    star_count: int
    candidate_count: int
    selected_count: int
    catalog_match_count: int
    recovered_catalog_star_count: int
    brightness_source: str
    relative_magnitude_limit: float
    width: int
    height: int
    camera: str
    lens: str
    input_kind: str
    color_profile: str
    sky_background_level: float
    sky_adaptation_gain: float
    comparison_preview: bytes | None = None
    comparison_star_count: int = 0
    coverage_preview: bytes | None = None
    catalog_prediction_count: int = 0
    catalog_verified_count: int = 0
    catalog_unverified_count: int = 0
    catalog_guided_recovered_count: int = 0
    analysis: PreparedAnalysis | None = None


@dataclass(frozen=True)
class RasterInput:
    pixels: np.ndarray
    alpha: np.ndarray | None
    profile: bytes | None
    profile_description: str
    info: RawInfo
    is_rgb: bool
    input_kind: str
    linear_srgb: bool
    invert_gray: bool = False


@dataclass(frozen=True)
class PreparedAnalysis:
    """Recognition products and decoded scene data retained for parameter-only rerenders."""

    image_cache_path: str
    image_shape: tuple[int, ...]
    alpha_cache_path: str | None
    detector_shape: tuple[int, int]
    detector_stars: tuple[Star, ...]
    catalog_entries: tuple[tuple[Star, float, bool], ...]
    candidate_count: int
    catalog_position_count: int
    catalog_match_count: int
    info: RawInfo
    profile: bytes | None
    profile_description: str
    is_rgb: bool
    invert_gray: bool
    photometric: object
    input_kind: str
    encoding: str
    linear_srgb: bool
    sky_mask_info: dict[str, object]
    sky_background_level: float
    raw_preview_median: float | None
    raw_linear_median: float | None
    raw_preview_ev: float
    raw_xmp_exposure_ev: float | None
    raw_exposure_shift: float
    raw_acr_reference_median: float | None
    raw_acr_reference_name: str | None
    raw_brightness_calibration_method: str
    catalog_positions: tuple[CatalogPosition, ...] = ()
    catalog_coverage_areas: tuple[CatalogCoverageArea, ...] = ()


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").replace("\x00", " ").split())


def _raw_info(raw: rawpy.RawPy, preview: Image.Image | None = None) -> RawInfo:
    sizes = raw.sizes
    other = raw.other
    lens_data = raw.lens
    preview_exif = preview.getexif() if preview is not None else {}
    camera = " ".join(part for part in (
        _clean_text(preview_exif.get(271, "")),
        _clean_text(preview_exif.get(272, "")),
    ) if part) or "相机型号未读取"
    try:
        preview_exif_ifd = preview_exif.get_ifd(ExifTags.IFD.Exif) if preview_exif else {}
    except Exception:
        preview_exif_ifd = {}
    focal_35mm = _as_float(preview_exif_ifd.get(41989))
    lens_name = " ".join(
        part for part in (_clean_text(lens_data.make), _clean_text(lens_data.model)) if part
    ) or "未记录镜头"
    focal = float(other.focal_length or 0)
    aperture = float(other.aperture or 0)
    width = int(sizes.width)
    height = int(sizes.height)
    if int(sizes.flip) in (5, 6):
        width, height = height, width
    return RawInfo(
        width=width,
        height=height,
        camera=camera,
        lens=lens_name,
        focal_length=focal if math.isfinite(focal) and focal > 0 else None,
        aperture=aperture if math.isfinite(aperture) and aperture > 0 else None,
        preview=preview,
        focal_length_35mm=focal_35mm,
    )


def read_raw_info(path: str | Path) -> RawInfo:
    """Read camera/lens metadata and an embedded preview without full demosaicing."""
    with rawpy.imread(str(path)) as raw:
        preview = _extract_raw_preview(raw)
        return _raw_info(raw, preview)


def _apply_camera_overrides(
    info: RawInfo,
    focal_length_override_mm: float | None,
    sensor_format: str,
) -> RawInfo:
    sensor_format = str(sensor_format or "auto").strip().lower()
    if sensor_format not in {"auto", "full_frame", "aps_c", "medium_4433", "four_thirds", "one_inch"}:
        raise ValueError("相机画幅选项无效。")
    if focal_length_override_mm is not None:
        focal = float(focal_length_override_mm)
        if not math.isfinite(focal) or not 1.0 <= focal <= 1000.0:
            raise ValueError("手动焦距须在 1–1000 mm 范围内。")
        lens = f"{info.lens} · 手动焦距 {focal:g} mm" if info.lens else f"手动焦距 {focal:g} mm"
        info = replace(info, focal_length=focal, focal_length_35mm=None, lens=lens)
    return replace(info, sensor_format=None if sensor_format == "auto" else sensor_format)


def _extract_raw_preview(raw: rawpy.RawPy) -> Image.Image | None:
    """Return the embedded camera-rendered preview when the RAW contains one."""
    try:
        thumb = raw.extract_thumb()
    except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
        return None
    if thumb.format == rawpy.ThumbFormat.JPEG:
        preview = Image.open(BytesIO(thumb.data)).convert("RGB")
    elif thumb.format == rawpy.ThumbFormat.BITMAP:
        preview = Image.fromarray(thumb.data).convert("RGB")
    else:
        return None
    return ImageOps.exif_transpose(preview)


def _preview_linear_luminance_median(preview: Image.Image) -> float | None:
    """Measure an sRGB embedded preview without changing its ICC colour space."""
    profile = preview.info.get("icc_profile")
    if profile:
        try:
            description = ImageCms.getProfileDescription(
                ImageCms.ImageCmsProfile(BytesIO(profile))
            ).lower()
        except Exception:
            return None
        if "srgb" not in description:
            # An embedded preview is only an exposure reference. Avoid an
            # 8-bit ICC transform here; fall back to XMP exposure instead.
            return None
    rgb = np.asarray(preview.convert("RGB"), dtype=np.float32) / 255.0
    low = rgb <= 0.04045
    rgb[low] /= 12.92
    rgb[~low] = np.power((rgb[~low] + 0.055) / 1.055, 2.4)
    luminance = rgb @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return float(np.median(luminance))


def _linear_rgb_luminance_median(rgb: np.ndarray) -> float:
    """Measure median luminance of a linear sRGB buffer."""
    if rgb.ndim != 3 or rgb.shape[2] < 3 or rgb.size == 0:
        return 0.0
    luminance = rgb[..., :3].astype(np.float32, copy=False) @ np.asarray(
        [0.2126, 0.7152, 0.0722], dtype=np.float32
    )
    return float(np.median(luminance))


def _preview_exposure_ev(source_median: float, preview_median: float) -> float:
    """Return the linear exposure adjustment needed to match preview midtones."""
    if not math.isfinite(source_median) or not math.isfinite(preview_median):
        return 0.0
    if source_median <= 1e-8 or preview_median <= 1e-8:
        return 0.0
    return math.log2(preview_median / source_median)


def _read_xmp_exposure_ev(raw_path: Path) -> float | None:
    """Read Adobe Camera Raw's per-image Exposure2012 value from its sidecar."""
    xmp_path = raw_path.with_suffix(".xmp")
    if not xmp_path.is_file():
        xmp_path = raw_path.with_suffix(".XMP")
    if not xmp_path.is_file():
        return None
    try:
        root = ET.parse(xmp_path).getroot()
    except (OSError, ET.ParseError):
        return None

    exposure_tag = "{http://ns.adobe.com/camera-raw-settings/1.0/}Exposure2012"
    for description in root.iter("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description"):
        raw_value = description.attrib.get(exposure_tag)
        if raw_value is None:
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return None


def _raw_exposure_shift(preview_ev: float, xmp_ev: float | None) -> float:
    """Combine preview matching and Camera Raw exposure within LibRaw's range."""
    total_ev = preview_ev + (xmp_ev if xmp_ev is not None else 0.0)
    # LibRaw supports exp_shift from 0.25 (−2 EV) through 8 (+3 EV).
    return float(2.0 ** np.clip(total_ev, -2.0, 3.0))


def _acr_reference_luminance_median(raw_path: Path) -> tuple[float | None, str | None]:
    """Read a same-stem Adobe Camera Raw TIFF as an optional brightness reference."""
    reference_path = next(
        (candidate for suffix in (".tif", ".tiff", ".TIF", ".TIFF")
         if (candidate := raw_path.with_suffix(suffix)).is_file()),
        None,
    )
    if reference_path is None:
        return None, None
    try:
        with tifffile.TiffFile(reference_path) as tiff:
            page = tiff.pages[0]
            software_tag = page.tags.get("Software")
            software = str(software_tag.value if software_tag else "").lower()
            if "adobe" not in software or "camera raw" not in software:
                return None, None
            height, width = page.shape[:2]
            step = max(1, int(math.ceil(math.sqrt((height * width) / 500_000))))
            pixels = tifffile.memmap(reference_path, mode="r")[::step, ::step]
            if pixels.ndim == 2:
                encoded = np.repeat(pixels[..., None], 3, axis=2).astype(np.float32) / 65535.0
            elif pixels.ndim == 3 and pixels.shape[2] >= 3:
                encoded = pixels[..., :3].astype(np.float32) / 65535.0
            else:
                return None, None
            profile_tag = page.tags.get("InterColorProfile") or page.tags.get("ICCProfile")
            profile = bytes(profile_tag.value) if profile_tag else None
    except (OSError, ValueError, tifffile.TiffFileError):
        return None, None

    if profile:
        try:
            source_profile = ImageCms.ImageCmsProfile(BytesIO(profile))
            profile_description = ImageCms.getProfileDescription(source_profile).lower()
            if "srgb" not in profile_description:
                # Do not convert the reference into sRGB through an 8-bit
                # intermediary. This auxiliary brightness estimate is only
                # colorimetrically comparable when the source is sRGB.
                return None, None
        except Exception:
            return None, None
    low = encoded <= 0.04045
    encoded[low] /= 12.92
    encoded[~low] = np.power((encoded[~low] + 0.055) / 1.055, 2.4)
    luminance = encoded @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return float(np.median(luminance)), reference_path.name


def _sky_detection_mask(
    reference: Image.Image | np.ndarray,
    target_shape: tuple[int, int],
) -> tuple[np.ndarray, dict[str, object]]:
    """Mask ground and dark foreground silhouettes using the rendered scene preview."""
    if isinstance(reference, Image.Image):
        values = np.asarray(reference.convert("RGB"), dtype=np.float32) / 255.0
        values = values @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    else:
        original = np.asarray(reference)
        values = original.astype(np.float32, copy=True)
        if values.ndim == 3:
            values = values[..., :3].astype(np.float32, copy=False) @ np.asarray(
                [0.2126, 0.7152, 0.0722], dtype=np.float32
            )
        if np.issubdtype(original.dtype, np.integer):
            values /= float(np.iinfo(original.dtype).max)
        elif values.size and float(np.nanmax(values)) > 1.5:
            values /= 255.0
    gray = np.nan_to_num(values, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
    np.clip(gray, 0.0, 1.0, out=gray)
    max_edge = 1400
    if max(gray.shape) > max_edge:
        ratio = max_edge / max(gray.shape)
        small_shape = (
            max(1, int(round(gray.shape[0] * ratio))),
            max(1, int(round(gray.shape[1] * ratio))),
        )
        gray = zoom(
            gray,
            (small_shape[0] / gray.shape[0], small_shape[1] / gray.shape[1]),
            order=1,
            mode="nearest",
            prefilter=False,
        ).astype(np.float32, copy=False)
    height, width = gray.shape
    row_profile = np.median(gray, axis=1)
    smooth_span = max(3, int(round(height * 0.008)) | 1)
    smooth_profile = np.convolve(
        row_profile,
        np.ones(smooth_span, dtype=np.float32) / smooth_span,
        mode="same",
    )
    compare_span = max(4, int(round(height * 0.01)))
    search_start = max(compare_span, int(height * 0.15))
    search_end = min(height - compare_span, int(height * 0.94))
    best_row = height
    best_drop = 0.0
    if search_end > search_start:
        for row in range(search_start, search_end):
            before = float(np.median(smooth_profile[row - compare_span:row]))
            after = float(np.median(smooth_profile[row:row + compare_span]))
            drop = before - after
            if drop > best_drop:
                best_row, best_drop = row, drop
    sky_before = float(np.median(smooth_profile[max(0, best_row - compare_span):best_row])) if best_row < height else 0.0
    horizon_found = (
        best_row < height
        and sky_before > 0.015
        and best_drop >= max(0.015, sky_before * 0.22)
    )
    if horizon_found:
        boundary = max(1, best_row - max(2, int(round(height * 0.006))))
        softened = gaussian_filter(
            gray,
            sigma=max(2.0, width * 0.012),
            mode="nearest",
        )
        local_sky_level = np.median(softened, axis=1)
        local_floor = np.maximum(local_sky_level * 0.35, 0.008)
        mask_small = (np.arange(height)[:, None] < boundary) & (softened >= local_floor[:, None])
        # Keep the horizon cut if the silhouette threshold would remove most of
        # the visible sky; this protects dark-sky exposures from over-masking.
        if float(np.mean(mask_small)) < 0.15:
            mask_small = np.broadcast_to(np.arange(height)[:, None] < boundary, gray.shape).copy()
    else:
        boundary = height
        mask_small = np.ones(gray.shape, dtype=bool)

    target_height, target_width = target_shape
    mask = zoom(
        (~mask_small).astype(np.float32),
        (target_height / height, target_width / width),
        order=0,
        mode="nearest",
        prefilter=False,
    ) > 0.5
    mask = mask[:target_height, :target_width]
    if mask.shape != (target_height, target_width):
        padded = np.ones((target_height, target_width), dtype=bool)
        padded[:mask.shape[0], :mask.shape[1]] = mask
        mask = padded
    sky_fraction = float(np.mean(~mask))
    return mask, {
        "method": "preview horizon profile and low-pass foreground silhouette mask" if horizon_found else "full frame; no reliable horizon transition",
        "horizon_detected": bool(horizon_found),
        "horizon_fraction": round(boundary / max(height, 1), 5),
        "sky_fraction": round(sky_fraction, 5),
    }


def _nearby_compact_source_counts(
    source_x: np.ndarray,
    source_y: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Count compact detections near each target, using a spatial hash."""
    cell_size = max(float(radius), 1.0)
    radius_squared = cell_size * cell_size
    cells: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for x, y in zip(source_x, source_y):
        key = (int(float(x) // cell_size), int(float(y) // cell_size))
        cells.setdefault(key, []).append((float(x), float(y)))

    counts = np.zeros(len(target_x), dtype=np.int16)
    for index, (x, y) in enumerate(zip(target_x, target_y)):
        cell_x = int(float(x) // cell_size)
        cell_y = int(float(y) // cell_size)
        count = 0
        for offset_y in (-1, 0, 1):
            for offset_x in (-1, 0, 1):
                for point_x, point_y in cells.get((cell_x + offset_x, cell_y + offset_y), ()):
                    if (point_x - x) ** 2 + (point_y - y) ** 2 <= radius_squared:
                        count += 1
        counts[index] = min(count, np.iinfo(np.int16).max)
    return counts


def _detector_image(raw: rawpy.RawPy, focal_length_override_mm: float | None = None) -> tuple[np.ndarray, int]:
    """Build a binned, linear luminance proxy from the visible sensor area."""
    sensor = raw.raw_image_visible
    height, width = sensor.shape[:2]

    focal = float(focal_length_override_mm or raw.other.focal_length or 0)
    if not focal:
        focal = 70.0
    # Longer focal lengths give a larger stellar profile on the same sensor.
    # Retain more detector pixels for them; the Bayer-safe minimum bin is 2x2.
    target_edge = int(np.clip(4200 + focal * 55.0, 4600, 9000))
    factor = max(2, int(math.ceil(max(height, width) / target_edge)))
    if factor % 2:
        factor += 1  # keep Bayer color samples grouped together
    out_h = height // factor
    out_w = width // factor
    if out_h < 8 or out_w < 8:
        raise ValueError("RAW 图像尺寸过小，无法可靠识别星点。")

    usable = sensor[: out_h * factor, : out_w * factor]
    if usable.ndim == 3:
        usable = usable[..., : min(3, usable.shape[2])].astype(np.float32).mean(axis=2)
    binned = usable.astype(np.float32).reshape(out_h, factor, out_w, factor).mean(axis=(1, 3))

    black_levels = getattr(raw, "black_level_per_channel", None) or [0]
    black = float(np.mean(black_levels))
    white = float(raw.white_level)
    dynamic_range = max(white - black, 1.0)
    image = np.clip((binned - black) / dynamic_range, 0.0, 1.0).astype(np.float32, copy=False)
    # raw_image_visible is in sensor coordinates; LibRaw rotates the developed
    # RGB output from this same orientation tag, so align SEP coordinates first.
    flip = int(raw.sizes.flip)
    if flip == 3:
        image = np.rot90(image, 2)
    elif flip == 5:
        image = np.rot90(image, 1)
    elif flip == 6:
        image = np.rot90(image, 3)
    return image, factor


def _raster_detector_image(rgb: np.ndarray, focal_length: float | None) -> tuple[np.ndarray, int]:
    """Build a downsampled detector from rendered RGB or grayscale pixels."""
    height, width = rgb.shape[:2]
    focal = float(focal_length or 50.0)
    target_edge = int(np.clip(4200 + focal * 55.0, 4600, 9000))
    factor = max(1, int(math.ceil(max(height, width) / target_edge)))
    usable_h = height // factor * factor
    usable_w = width // factor * factor
    if usable_h < 8 or usable_w < 8:
        raise ValueError("图像尺寸过小，无法可靠识别星点。")

    if rgb.shape[2] >= 3:
        proxy = np.max(rgb[..., :3], axis=2)
    else:
        proxy = rgb[..., 0]
    if factor > 1:
        proxy = proxy[:usable_h, :usable_w].reshape(
            usable_h // factor, factor, usable_w // factor, factor
        ).mean(axis=(1, 3))
    return np.ascontiguousarray(proxy, dtype=np.float32), factor


def _write_solver_grayscale(
    image: np.ndarray,
    output_path: Path,
    sky_mask: np.ndarray | None = None,
) -> None:
    """Write a contrast-stretched, sky-only, full-resolution 16-bit solve proxy."""
    height, width = image.shape[:2]
    sample = np.asarray(image[::16, ::16], dtype=np.float32)
    if sample.ndim == 3:
        sample = np.max(sample[..., :3], axis=2)
    valid_reference = np.isfinite(sample)
    if sky_mask is not None and sky_mask.ndim == 2:
        sample_y = np.minimum(
            (np.arange(sample.shape[0], dtype=np.float64) * sky_mask.shape[0] / sample.shape[0]).astype(np.intp),
            sky_mask.shape[0] - 1,
        )
        sample_x = np.minimum(
            (np.arange(sample.shape[1], dtype=np.float64) * sky_mask.shape[1] / sample.shape[1]).astype(np.intp),
            sky_mask.shape[1] - 1,
        )
        sample_foreground = sky_mask[sample_y[:, None], sample_x[None, :]]
        sky_reference_mask = valid_reference & ~sample_foreground
        if np.any(sky_reference_mask):
            valid_reference = sky_reference_mask
    valid_reference = sample[valid_reference]
    if not valid_reference.size:
        raise ValueError("无法为本机板解算生成有效的天空灰度参考。")
    low, high = np.percentile(valid_reference, (0.2, 99.8))
    if not math.isfinite(float(low)) or not math.isfinite(float(high)) or high <= low:
        raise ValueError("天空亮度范围不足，无法生成本机板解算图像。")

    x_indices: np.ndarray | None = None
    if sky_mask is not None and sky_mask.ndim == 2:
        x_indices = np.minimum(
            (np.arange(width, dtype=np.float64) * sky_mask.shape[1] / width).astype(np.intp),
            sky_mask.shape[1] - 1,
        )
    target = tifffile.memmap(
        output_path,
        shape=(height, width),
        dtype=np.uint16,
        photometric="minisblack",
        metadata=None,
    )
    try:
        for y0 in range(0, height, 512):
            rows = np.array(image[y0:min(height, y0 + 512)], dtype=np.float32, copy=True)
            if rows.ndim == 3:
                rows = np.max(rows[..., :3], axis=2)
            np.nan_to_num(rows, copy=False, nan=float(low), posinf=float(high), neginf=float(low))
            if sky_mask is not None and x_indices is not None:
                y_indices = np.minimum(
                    (np.arange(y0, y0 + rows.shape[0], dtype=np.float64) * sky_mask.shape[0] / height).astype(np.intp),
                    sky_mask.shape[0] - 1,
                )
                foreground = sky_mask[y_indices[:, None], x_indices[None, :]]
                rows[foreground] = float(low)
            np.subtract(rows, low, out=rows)
            np.divide(rows, high - low, out=rows)
            np.clip(rows, 0.0, 1.0, out=rows)
            target[y0:y0 + rows.shape[0]] = np.rint(rows * 65535.0).astype(np.uint16)
        target.flush()
    finally:
        del target


def _orient_array(array: np.ndarray, orientation: int) -> np.ndarray:
    """Apply TIFF/EXIF orientation to pixels, returning orientation 1 data."""
    if orientation == 2:
        return np.fliplr(array)
    if orientation == 3:
        return np.rot90(array, 2)
    if orientation == 4:
        return np.flipud(array)
    if orientation == 5:
        return np.swapaxes(array, 0, 1)
    if orientation == 6:
        return np.rot90(array, 3)
    if orientation == 7:
        return np.flip(np.swapaxes(array, 0, 1), axis=(0, 1))
    if orientation == 8:
        return np.rot90(array, 1)
    return array


def _as_float(value: object) -> float | None:
    try:
        if hasattr(value, "numerator") and hasattr(value, "denominator"):
            numerator = float(value.numerator)
            denominator = float(value.denominator)
            number = numerator / denominator if denominator else 0.0
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            number = float(value[0]) / float(value[1]) if float(value[1]) else 0.0
        else:
            number = float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _profile_description(profile: bytes | None) -> str:
    if not profile:
        return "未嵌入 ICC"
    try:
        return _clean_text(ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(BytesIO(profile))))
    except Exception:
        return "ICC 色彩配置文件"


def _to_unit_float(pixels: np.ndarray, bits: int) -> np.ndarray:
    if np.issubdtype(pixels.dtype, np.integer):
        maximum = float((1 << min(max(int(bits), 1), 32)) - 1)
        return np.clip(pixels.astype(np.float32) / maximum, 0.0, 1.0)
    return np.clip(pixels.astype(np.float32), 0.0, 1.0)


def _inherit_sibling_raw_metadata(path: Path, info: RawInfo) -> RawInfo:
    if info.focal_length is not None and info.focal_length_35mm is not None:
        return info
    raw_extensions = (
        ".cr3", ".cr2", ".crw", ".nef", ".nrw", ".arw", ".sr2", ".srf", ".dng",
        ".orf", ".rw2", ".raf", ".pef", ".ptx", ".3fr", ".fff", ".iiq", ".kdc",
        ".dcr", ".mos", ".mrw", ".x3f",
    )
    for suffix in raw_extensions:
        candidate = path.with_suffix(suffix)
        if not candidate.is_file():
            candidate = path.with_suffix(suffix.upper())
        if not candidate.is_file():
            continue
        try:
            raw_info = read_raw_info(candidate)
        except Exception:
            continue
        return replace(
            info,
            camera=info.camera if info.camera and "未记录" not in info.camera else raw_info.camera,
            lens=info.lens if info.focal_length is not None and info.lens != "未记录镜头" else raw_info.lens,
            focal_length=info.focal_length or raw_info.focal_length,
            aperture=info.aperture or raw_info.aperture,
            focal_length_35mm=info.focal_length_35mm or raw_info.focal_length_35mm,
        )
    return info


def _read_raster(path: Path) -> RasterInput:
    suffix = path.suffix.lower()
    camera = ""
    lens = ""
    focal = None
    focal_35mm = None
    aperture = None
    alpha: np.ndarray | None = None
    invert_gray = False

    if suffix in {".tif", ".tiff"}:
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            photometric = int(page.photometric)
            if photometric not in {
                int(tifffile.PHOTOMETRIC.RGB),
                int(tifffile.PHOTOMETRIC.MINISBLACK),
                int(tifffile.PHOTOMETRIC.MINISWHITE),
            }:
                raise ValueError("目前支持 RGB 或灰度 TIFF；CMYK、调色板和其他 TIFF 色彩模式暂不支持。")
            raw_pixels = page.asarray()
            samples = int(page.samplesperpixel or 1)
            if page.planarconfig == tifffile.PLANARCONFIG.SEPARATE and raw_pixels.ndim == 3:
                raw_pixels = np.moveaxis(raw_pixels, 0, -1)
            if raw_pixels.ndim == 2:
                raw_pixels = raw_pixels[..., None]
            if raw_pixels.ndim != 3:
                raise ValueError("无法读取 TIFF 的像素通道布局。")
            if photometric == int(tifffile.PHOTOMETRIC.RGB):
                if raw_pixels.shape[-1] < 3 or raw_pixels.shape[-1] > 4:
                    raise ValueError("目前仅支持 3 通道 RGB 或带 Alpha 的 4 通道 TIFF。")
                if raw_pixels.shape[-1] == 4:
                    alpha = raw_pixels[..., 3]
                color_pixels = raw_pixels[..., :3]
                is_rgb = True
            else:
                if raw_pixels.shape[-1] not in (1, 2):
                    raise ValueError("灰度 TIFF 仅支持灰度通道或灰度加 Alpha。")
                if raw_pixels.shape[-1] == 2:
                    alpha = raw_pixels[..., 1]
                color_pixels = raw_pixels[..., :1]
                is_rgb = False
                invert_gray = photometric == int(tifffile.PHOTOMETRIC.MINISWHITE)

            bits_value = page.bitspersample
            bits = max(bits_value) if isinstance(bits_value, (tuple, list)) else int(bits_value or 16)
            if invert_gray:
                if np.issubdtype(raw_pixels.dtype, np.integer):
                    color_pixels = ((1 << bits) - 1) - color_pixels
                else:
                    color_pixels = 1.0 - color_pixels
            pixels = _to_unit_float(color_pixels, bits)
            alpha_float = _to_unit_float(alpha, bits) if alpha is not None else None
            profile_tag = page.tags.get("InterColorProfile")
            profile = bytes(profile_tag.value) if profile_tag is not None else None
            orientation_tag = page.tags.get("Orientation")
            orientation = int(orientation_tag.value) if orientation_tag is not None else 1
            pixels = _orient_array(pixels, orientation)
            if alpha_float is not None:
                alpha_float = _orient_array(alpha_float, orientation)

            make_tag = page.tags.get("Make")
            model_tag = page.tags.get("Model")
            lens_tag = page.tags.get("LensModel") or page.tags.get(42036)
            focal_tag = page.tags.get("FocalLength") or page.tags.get(37386)
            aperture_tag = page.tags.get("FNumber") or page.tags.get(33437)
            description_tag = page.tags.get("ImageDescription")
            try:
                stored_metadata = json.loads(str(description_tag.value)) if description_tag else {}
            except (TypeError, ValueError):
                stored_metadata = {}
            make = _clean_text(make_tag.value if make_tag is not None else "")
            model = _clean_text(model_tag.value if model_tag is not None else "")
            camera = " ".join(part for part in (make, model) if part) or _clean_text(stored_metadata.get("camera"))
            lens = _clean_text(lens_tag.value if lens_tag is not None else stored_metadata.get("lens"))
            focal = _as_float(focal_tag.value if focal_tag is not None else stored_metadata.get("focal_length_mm"))
            equivalent_tag = page.tags.get("FocalLengthIn35mmFilm") or page.tags.get(41989)
            focal_35mm = _as_float(equivalent_tag.value if equivalent_tag is not None else stored_metadata.get("focal_length_35mm_equivalent"))
            aperture = _as_float(aperture_tag.value if aperture_tag is not None else stored_metadata.get("aperture"))
            input_kind = "TIFF"
    elif suffix in {".jpg", ".jpeg"}:
        with Image.open(path) as image:
            if image.mode not in {"L", "LA", "RGB", "RGBA"}:
                raise ValueError("目前 JPG 仅支持 RGB 或灰度模式；请先将 CMYK/调色板 JPG 转为 RGB。")
            exif = image.getexif()
            exif_ifd = exif.get_ifd(ExifTags.IFD.Exif) if exif else {}
            profile = image.info.get("icc_profile")
            image = ImageOps.exif_transpose(image)
            raw_pixels = np.asarray(image)
            bits = 8
            if raw_pixels.ndim == 2:
                raw_pixels = raw_pixels[..., None]
            if raw_pixels.shape[-1] in (2, 4):
                alpha = raw_pixels[..., -1]
                color_pixels = raw_pixels[..., :-1]
            else:
                color_pixels = raw_pixels
            is_rgb = color_pixels.shape[-1] == 3
            pixels = _to_unit_float(color_pixels, bits)
            alpha_float = _to_unit_float(alpha, bits) if alpha is not None else None
            camera = " ".join(
                part for part in (
                    _clean_text(exif.get(271, "")),
                    _clean_text(exif.get(272, "")),
                ) if part
            )
            lens = _clean_text(exif_ifd.get(42036, ""))
            focal = _as_float(exif_ifd.get(37386))
            focal_35mm = _as_float(exif_ifd.get(41989))
            aperture = _as_float(exif_ifd.get(33437))
            input_kind = "JPG"
    else:
        raise ValueError("请选择相机 RAW、TIFF 或 JPG 文件。")

    description = _profile_description(profile)
    linear_srgb = not profile or "srgb" in description.lower()
    if linear_srgb:
        pixels = pixels.copy()
        low = pixels <= 0.04045
        pixels[low] /= 12.92
        pixels[~low] = np.power((pixels[~low] + 0.055) / 1.055, 2.4)

    info = RawInfo(
        width=int(pixels.shape[1]),
        height=int(pixels.shape[0]),
        camera=camera or "未记录相机型号",
        lens=lens or "未记录镜头",
        focal_length=focal,
        aperture=aperture,
        preview=None,
        focal_length_35mm=focal_35mm,
    )
    info = _inherit_sibling_raw_metadata(path, info)
    return RasterInput(
        pixels=pixels,
        alpha=alpha_float,
        profile=profile,
        profile_description=description,
        info=info,
        is_rgb=is_rgb,
        input_kind=input_kind,
        linear_srgb=linear_srgb,
        invert_gray=invert_gray,
    )


def detect_stars(
    detector: np.ndarray,
    sensitivity: float,
    detection_mask: np.ndarray | None = None,
    *,
    allow_widefield_elongation: bool = False,
) -> tuple[list[Star], int, int, float]:
    """Return SEP point-source candidates for image photometry or Gaia matching."""
    data = np.ascontiguousarray(detector, dtype=np.float32)
    mask = np.ascontiguousarray(detection_mask, dtype=bool) if detection_mask is not None else None
    if mask is not None and mask.shape != data.shape:
        raise ValueError("星空遮罩尺寸与星点检测图像不一致。")
    background = sep.Background(data, mask=mask, bw=64, bh=64, fw=3, fh=3)
    global_sky = float(np.clip(background.globalback, 0.0, 1.0))
    signal = np.ascontiguousarray(data - background.back(), dtype=np.float32)
    noise = np.ascontiguousarray(background.rms(), dtype=np.float32)
    first_pass = _extract_sources(
        signal,
        float(sensitivity),
        err=noise,
        mask=mask,
        minarea=4,
        deblend_nthresh=32,
        deblend_cont=0.005,
        clean=True,
    )
    if len(first_pass) == 0:
        return [], 0, 0, global_sky

    first_major = np.maximum(first_pass["a"], first_pass["b"])
    first_minor = np.minimum(first_pass["a"], first_pass["b"])
    first_size = np.sqrt(np.maximum(first_pass["a"] * first_pass["b"], 0.0))
    first_roundness = first_minor / np.maximum(first_major, 1e-8)
    first_noise = background.rms()
    global_rms = max(float(background.globalrms), 1e-12)
    first_good = (
        ((first_pass["flag"] & (sep.OBJ_TRUNC | sep.OBJ_SINGU)) == 0)
        & (first_roundness >= 0.45)
        & (first_size >= 0.45)
        & (first_size <= 6.0)
        & (first_pass["flux"] > 0)
    )
    if np.any(first_good):
        # Estimate the image's stellar width from its brighter compact sources,
        # then use SEP's documented PSF-shaped matched filter for a second pass.
        calibration_indices = np.flatnonzero(first_good)
        brightest = calibration_indices[
            np.argsort(first_pass["flux"][calibration_indices])[-200:]
        ]
        psf_sigma = float(np.median(first_size[brightest]))
    else:
        psf_sigma = 1.0
    kernel_radius = int(np.clip(math.ceil(2.5 * psf_sigma), 1, 6))
    axis = np.arange(-kernel_radius, kernel_radius + 1, dtype=np.float32)
    kernel = np.exp(
        -0.5 * (axis[:, None] ** 2 + axis[None, :] ** 2) / max(psf_sigma**2, 0.25)
    ).astype(np.float32)
    kernel /= float(kernel.sum())
    objects = _extract_sources(
        signal,
        float(sensitivity),
        err=noise,
        mask=mask,
        minarea=4,
        filter_kernel=kernel,
        filter_type="matched",
        deblend_nthresh=32,
        deblend_cont=0.005,
        clean=True,
        segmentation_map=False,
    )
    if len(objects) == 0:
        return [], 0, 0, global_sky

    major = np.maximum(objects["a"], objects["b"])
    minor = np.minimum(objects["a"], objects["b"])
    moment_size = np.sqrt(np.maximum(objects["a"] * objects["b"], 0.0))
    roundness = minor / np.maximum(major, 1e-8)
    bad_flags = (objects["flag"] & (sep.OBJ_TRUNC | sep.OBJ_SINGU)) != 0
    object_x = np.clip(np.rint(objects["x"]).astype(np.int64), 0, data.shape[1] - 1)
    object_y = np.clip(np.rint(objects["y"]).astype(np.int64), 0, data.shape[0] - 1)
    local_noise = np.asarray([
        np.median(
            first_noise[max(0, y - 4):min(data.shape[0], y + 5), max(0, x - 4):min(data.shape[1], x + 5)]
            if mask is None
            else first_noise[max(0, y - 4):min(data.shape[0], y + 5), max(0, x - 4):min(data.shape[1], x + 5)][
                ~mask[max(0, y - 4):min(data.shape[0], y + 5), max(0, x - 4):min(data.shape[1], x + 5)]
            ]
        )
        if mask is None or np.any(~mask[max(0, y - 4):min(data.shape[0], y + 5), max(0, x - 4):min(data.shape[1], x + 5)])
        else global_rms
        for x, y in zip(object_x, object_y)
    ], dtype=np.float32)
    compact_x = first_pass["x"][first_good]
    compact_y = first_pass["y"][first_good]
    crowding_radius = max(16.0, 6.0 * psf_sigma)
    nearby_compact = _nearby_compact_source_counts(
        compact_x, compact_y, objects["x"], objects["y"], crowding_radius
    )
    crowded_stellar_field = nearby_compact >= 4
    elevated_local_noise = local_noise > 4.0 * global_rms
    roundness_floor = 0.20 if allow_widefield_elongation else 0.35
    major_axis_limit = 18.0 if allow_widefield_elongation else min(12.0, max(6.0, 5.0 * psf_sigma))
    point_source = (
        np.isfinite(objects["x"])
        & np.isfinite(objects["y"])
        & np.isfinite(objects["flux"])
        & np.isfinite(moment_size)
        & (objects["flux"] > 0)
        & (minor >= 0.35)
        & (major <= major_axis_limit)
        & (roundness >= roundness_floor)
        & (~elevated_local_noise | crowded_stellar_field)
        & ~bad_flags
    )
    indices = np.flatnonzero(point_source)
    if len(indices) == 0:
        return [], 0, 0, global_sky

    # SEP's isophotal flux is useful for extraction. Circular aperture flux is
    # measured separately so the brightness ordering includes more of each PSF.
    fwhm = 2.354820045 * moment_size[indices]
    aperture_radius = np.clip(1.25 * fwhm, 2.5, 12.0)
    flux, flux_error, aperture_flags = sep.sum_circle(
        signal,
        objects["x"][indices],
        objects["y"][indices],
        aperture_radius,
        err=noise,
        mask=mask,
        subpix=5,
    )
    good = (
        np.isfinite(flux)
        & (flux > 0)
        & np.isfinite(flux_error)
        & (flux_error > 0)
        & ((aperture_flags & sep.APER_TRUNC) == 0)
    )
    indices = indices[good]
    flux = flux[good]
    flux_error = flux_error[good]
    local_noise = local_noise[indices]
    if len(indices) == 0:
        return [], 0, 0, global_sky

    candidate_count = int(len(indices))
    brightest_flux = float(np.max(flux))
    relative_flux = flux / max(brightest_flux, 1e-20)
    selected_count = candidate_count
    signal_to_noise = flux / np.maximum(flux_error, 1e-20)
    order = np.argsort(flux)[::-1]
    stars = [
        Star(
            float(objects["x"][indices[i]]),
            float(objects["y"][indices[i]]),
            float(flux[i]),
            float(objects["peak"][indices[i]]),
            float(2.354820045 * moment_size[indices[i]]),
            float(relative_flux[i]),
            float(objects["a"][indices[i]]),
            float(objects["b"][indices[i]]),
            float(objects["theta"][indices[i]]),
            float(signal_to_noise[i]),
            bool(elevated_local_noise[indices[i]] and crowded_stellar_field[indices[i]]),
        )
        for i in order
    ]
    return stars, candidate_count, selected_count, global_sky


def _soften_stars(
    image_data: np.ndarray,
    stars: list[Star],
    detector_shape: tuple[int, int],
    min_radius: float,
    max_radius: float,
    strength: float,
    opacity: float,
    sky_background_level: float,
    brightness_source: str,
    relative_magnitude_limit: float,
    progress: Progress | None = None,
) -> list[dict[str, object]]:
    if not stars or strength <= 0:
        return []
    out_h, out_w = image_data.shape[:2]
    det_h, det_w = detector_shape
    records: list[dict[str, object]] = []
    scale_x = out_w / det_w
    scale_y = out_h / det_h
    if max_radius < min_radius:
        min_radius, max_radius = max_radius, min_radius
    opacity_alpha = float(np.clip(opacity, 0.0, 100.0)) / 100.0
    # Weber contrast is the luminance difference divided by background
    # luminance. Scale peak-matched halo radiance by the measured sky level so
    # its visibility stays comparable across differently developed images.
    sky_gain = float(np.clip(sky_background_level / REFERENCE_SKY_LEVEL, 0.70, 1.50))
    prepared: list[dict[str, object]] = []

    # Capture each source profile before adding any halos. This scalar only
    # sizes the circular processing mask; no major/minor axis or angle is used.
    for star in stars:
        cx = (star.x + 0.5) * scale_x - 0.5
        cy = (star.y + 0.5) * scale_y - 0.5
        center_x = int(np.clip(round(cx), 0, out_w - 1))
        center_y = int(np.clip(round(cy), 0, out_h - 1))
        # SEP's scalar circularized FWHM sets only a radial support estimate.
        # The measured profile and every generated halo remain strictly round.
        source_sigma = max(
            float(star.fwhm) * (scale_x + scale_y) * 0.5 / 2.354820045,
            0.45,
        )
        widest_diffusion_sigma = (max_radius / 3.0) * math.sqrt(float(np.clip(strength, 0.0, 30.0)) / 40.0)
        probe_sigma = math.sqrt(source_sigma * source_sigma + widest_diffusion_sigma * widest_diffusion_sigma)
        source_support_radius = float(np.clip(
            2.0 * star.fwhm * (scale_x + scale_y) * 0.5,
            8.0,
            96.0,
        ))
        extent = max(
            8,
            int(math.ceil(4.0 * probe_sigma)),
            int(math.ceil(source_support_radius + 4.0 * widest_diffusion_sigma)),
        )
        x0, x1 = max(0, center_x - extent), min(out_w, center_x + extent + 1)
        y0, y1 = max(0, center_y - extent), min(out_h, center_y + extent + 1)
        source_patch = image_data[y0:y1, x0:x1, :]
        border = np.concatenate(
            (source_patch[0], source_patch[-1], source_patch[:, 0], source_patch[:, -1]),
            axis=0,
        )
        local_background = np.median(border, axis=0)
        # Measure a circular stellar profile directly from this developed
        # image. SEP moments can include asymmetric optical wings or nearby
        # diffuse structure, so using them as a single Gaussian source model
        # made the synthetic halo too narrow on the 9331 TIFF. Annular medians
        # retain the observed stellar PSF while rejecting non-radial context.
        yy_profile, xx_profile = np.ogrid[y0:y1, x0:x1]
        radius_profile = np.hypot(xx_profile - cx, yy_profile - cy)
        radius_bins = np.floor(radius_profile).astype(np.int32)
        profile_count = int(math.ceil(source_support_radius)) + 1
        source_signal = np.maximum(source_patch - local_background[None, None, :], 0.0)
        radial_profile = np.zeros((profile_count, image_data.shape[2]), dtype=np.float32)
        for radius_bin in range(profile_count):
            ring_pixels = source_signal[radius_bins == radius_bin]
            if ring_pixels.size:
                radial_profile[radius_bin] = np.median(ring_pixels, axis=0)
        populated_bins = np.flatnonzero(np.any(radial_profile > 0, axis=1))
        if populated_bins.size:
            all_bins = np.arange(profile_count)
            for channel in range(image_data.shape[2]):
                channel_bins = np.flatnonzero(radial_profile[:, channel] > 0)
                if channel_bins.size:
                    radial_profile[:, channel] = np.interp(
                        all_bins,
                        channel_bins,
                        radial_profile[channel_bins, channel],
                        left=radial_profile[channel_bins[0], channel],
                        right=0.0,
                    )
            fade_start = min(profile_count - 1, int(source_support_radius * 0.85))
            if fade_start < profile_count - 1:
                fade = np.linspace(1.0, 0.0, profile_count - fade_start, dtype=np.float32)
                fade = fade * fade * (3.0 - 2.0 * fade)
                radial_profile[fade_start:] *= fade[:, None]
        core = image_data[
            max(0, center_y - 2) : min(out_h, center_y + 3),
            max(0, center_x - 2) : min(out_w, center_x + 3),
            :,
        ]
        core_signal = np.maximum(np.max(core, axis=(0, 1)) - local_background, 0.0)
        aperture_radius = float(np.clip(star.fwhm * (scale_x + scale_y) * 0.625, 2.5, 24.0))
        aperture_extent = int(math.ceil(aperture_radius))
        ax0, ax1 = max(0, center_x - aperture_extent), min(out_w, center_x + aperture_extent + 1)
        ay0, ay1 = max(0, center_y - aperture_extent), min(out_h, center_y + aperture_extent + 1)
        ay, ax = np.ogrid[ay0:ay1, ax0:ax1]
        aperture_mask = (ax - cx) ** 2 + (ay - cy) ** 2 <= aperture_radius * aperture_radius
        aperture_signal = np.maximum(image_data[ay0:ay1, ax0:ax1, :] - local_background[None, None, :], 0.0)
        color_flux = np.sum(aperture_signal * aperture_mask[..., None], axis=(0, 1))
        if float(np.sum(color_flux)) <= 1e-12:
            color_flux = core_signal.copy()
        if float(np.sum(core_signal)) > 1e-12:
            core_color = core_signal / float(np.sum(core_signal))
        else:
            core_color = color_flux / max(float(np.sum(color_flux)), 1e-12)
        aperture_color = color_flux / max(float(np.sum(color_flux)), 1e-12)
        # Use aperture photometry for chroma while retaining some core color in
        # saturated or locally crowded stellar profiles.
        color_fraction = 0.7 * aperture_color + 0.3 * core_color
        color_fraction /= max(float(np.sum(color_fraction)), 1e-12)
        color_max = max(float(np.max(color_fraction)), 1e-12)
        source_peak = float(np.max(core_signal))
        prepared.append({
            "star": star,
            "cx": cx,
            "cy": cy,
            "x0": x0,
            "x1": x1,
            "y0": y0,
            "y1": y1,
            "source_sigma": source_sigma,
            "source_support_radius": source_support_radius,
            "radial_profile": radial_profile,
            "sigma": probe_sigma,
            "inner_mask_radius": 3.0 * probe_sigma,
            "outer_mask_radius": 4.0 * probe_sigma,
            "background": local_background,
            "source_peak": source_peak,
            "color_fraction": color_fraction,
        })

    # Catalogue mode maps its matched catalogue span to the radius range.
    # Image mode retains the 1.4.7 response against the selected Δm limit.
    flux_floor = min(float(item["star"].flux) for item in prepared)
    brightest_flux = max(float(item["star"].flux) for item in prepared)
    reference_peak = max(float(item["source_peak"]) for item in prepared)
    max_log_flux = max(
        math.log(max(brightest_flux, 1e-20) / max(flux_floor, 1e-20)),
        0.0,
    )
    for item in prepared:
        star = item["star"]
        if brightness_source == "image":
            delta_magnitude = -2.5 * math.log10(max(float(star.relative_flux_ratio), 1e-20))
            response = (
                float(np.clip(1.0 - delta_magnitude / relative_magnitude_limit, 0.0, 1.0))
                if relative_magnitude_limit > 1e-12 else 1.0
            )
        else:
            log_flux = max(math.log(max(float(star.flux), 1e-20) / max(flux_floor, 1e-20)), 0.0)
            response = float(np.clip(log_flux / max_log_flux, 0.0, 1.0)) if max_log_flux > 1e-12 else 1.0
        radius = min_radius + (max_radius - min_radius) * response
        diffusion_sigma = (radius / 3.0) * math.sqrt(float(np.clip(strength, 0.0, 30.0)) / 40.0)
        # Keep the measured radial PSF shape, but use it as a peak-matched
        # lightening layer. This preserves the observed star core and restores
        # the visible halo response of the earlier release. The sigma remains
        # isotropic, so neither brightness nor source ellipticity can stretch
        # the generated glow into an oval.
        flux_scale = float(np.clip(star.relative_flux_ratio, 0.0, 1.0))
        color_fraction = np.asarray(item["color_fraction"], dtype=np.float32)
        color_max = max(float(np.max(color_fraction)), 1e-12)
        target_peak = (
            reference_peak
            * flux_scale
            * (color_fraction / color_max)
            * sky_gain
        )
        source_sigma = max(float(item["source_sigma"]), 1e-6)
        output_sigma = math.sqrt(source_sigma**2 + diffusion_sigma**2)
        # Optical diffusion convolves the measured stellar PSF with a normalized
        # isotropic kernel. The resulting radial shape is peak-matched below;
        # the circular feather mask only limits the finite processing region.
        item["radius"] = radius
        item["diffusion_sigma"] = diffusion_sigma
        item["sigma"] = output_sigma
        item["inner_mask_radius"] = max(1.0, 3.0 * output_sigma)
        item["outer_mask_radius"] = max(float(item["inner_mask_radius"]) + 1.0, 4.0 * output_sigma)
        item["radius_response"] = response
        item["halo_flux_scale"] = flux_scale
        item["target_peak"] = target_peak
        cx, cy = float(item["cx"]), float(item["cy"])
        center_x, center_y = int(np.clip(round(cx), 0, out_w - 1)), int(np.clip(round(cy), 0, out_h - 1))
        extent = int(math.ceil(float(item["outer_mask_radius"])))
        item["x0"], item["x1"] = max(0, center_x - extent), min(out_w, center_x + extent + 1)
        item["y0"], item["y1"] = max(0, center_y - extent), min(out_h, center_y + extent + 1)

    for index, item in enumerate(prepared):
        star = item["star"]
        cx, cy = float(item["cx"]), float(item["cy"])
        x0, x1 = int(item["x0"]), int(item["x1"])
        y0, y1 = int(item["y0"]), int(item["y1"])
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dx, dy = xx - cx, yy - cy
        radial_squared = dx * dx + dy * dy
        radial_distance = np.sqrt(radial_squared)
        radial_profile = np.asarray(item["radial_profile"], dtype=np.float32)
        profile_radius = np.arange(radial_profile.shape[0], dtype=np.float32)
        source_model = np.stack(
            [
                np.interp(
                    radial_distance.ravel(),
                    profile_radius,
                    radial_profile[:, channel],
                    right=0.0,
                ).reshape(radial_distance.shape)
                for channel in range(image_data.shape[2])
            ],
            axis=-1,
        ).astype(np.float32, copy=False)
        target_model = gaussian_filter(
            source_model,
            sigma=(float(item["diffusion_sigma"]), float(item["diffusion_sigma"]), 0.0),
            mode="constant",
            cval=0.0,
            truncate=4.0,
        )
        inner_radius = float(item["inner_mask_radius"])
        outer_radius = float(item["outer_mask_radius"])
        feather = max(outer_radius - inner_radius, 1e-6)
        mask_t = np.clip((outer_radius - np.sqrt(radial_squared)) / feather, 0.0, 1.0)
        circular_mask = mask_t * mask_t * (3.0 - 2.0 * mask_t)
        profile_peak = np.max(target_model, axis=(0, 1))
        target_peak = np.asarray(item["target_peak"], dtype=np.float32)
        valid_channels = profile_peak > 1e-12
        channel_scale = np.ones_like(profile_peak, dtype=np.float32)
        np.divide(target_peak, profile_peak, out=channel_scale, where=valid_channels)
        target_model *= channel_scale[None, None, :]
        halo_peak = np.max(target_model, axis=(0, 1))
        patch = image_data[y0:y1, x0:x1, :]
        # Lighten only where the peak-matched radial target exceeds the source.
        # The stellar core therefore remains intact while the RGB wings grow
        # outward; opacity now controls that visible addition directly.
        target_model += np.asarray(item["background"], dtype=np.float32)[None, None, :]
        np.subtract(target_model, patch, out=target_model)
        np.maximum(target_model, 0.0, out=target_model)
        target_model *= circular_mask[..., None]
        patch += target_model * opacity_alpha
        records.append({
            "x_px": round(cx, 2),
            "y_px": round(cy, 2),
            "halo_geometry": "circle; isotropic x/y Gaussian convolution",
            "aperture_flux_image_units": round(float(star.image_flux if star.image_flux is not None else star.flux), 7),
            "catalog_source_id": star.catalog_source_id,
            "catalog_g_magnitude": round(float(star.catalog_g_mag), 1) if star.catalog_g_mag is not None else None,
            "catalog_name": star.catalog_name,
            "catalog_position_state": star.catalog_position_state,
            "catalog_ra_deg": round(float(star.catalog_ra_deg), 7) if star.catalog_ra_deg is not None else None,
            "catalog_dec_deg": round(float(star.catalog_dec_deg), 7) if star.catalog_dec_deg is not None else None,
            "catalog_bp_rp_colour_index": round(float(star.catalog_bp_rp), 5) if star.catalog_bp_rp is not None else None,
            "gaia_delta_g_from_brightest": round(float(star.catalog_delta_magnitude), 5) if star.catalog_delta_magnitude is not None else None,
            "catalog_position_recovered_from_wcs": bool(star.catalog_position_recovered),
            "image_delta_magnitude": round(float(-2.5 * math.log10(max(float(star.relative_flux_ratio), 1e-20))), 5) if brightness_source == "image" else None,
            "relative_flux_ratio": round(float(star.relative_flux_ratio), 6),
            "radius_curve": (
                "1.4.7 relative SEP aperture magnitude: clamp(1 - delta_m/selected_limit, 0, 1)"
                if brightness_source == "image"
                else "linear response to relative Gaia G magnitude; log(catalog_flux/faintest_selected_flux)"
            ),
            "source_signal_to_noise": round(float(star.signal_to_noise), 3),
            "radius_response": round(float(item["radius_response"]), 5),
            "halo_flux_scale": round(float(item["halo_flux_scale"]), 6),
            "fwhm_px": round(float(star.fwhm * (scale_x + scale_y) * 0.5), 3),
            "halo_radius_3sigma_px": round(3.0 * float(item["sigma"]), 2),
            "source_psf_sigma_px": round(float(item["source_sigma"]), 3),
            "effective_halo_sigma_px": round(float(item["sigma"]), 3),
            "diffusion_sigma_px": round(float(item["diffusion_sigma"]), 3),
            "circular_mask_outer_radius_px": round(float(item["outer_mask_radius"]), 2),
            "star_color_rgb_fraction": [round(float(value), 4) for value in item["color_fraction"]],
            "sky_adaptation_gain": round(sky_gain, 4),
            "opacity_percent": round(float(opacity), 1),
            "halo_peak_per_channel": [round(float(value), 6) for value in halo_peak],
        })
        if progress and (index == len(prepared) - 1 or index % max(1, len(prepared) // 20) == 0):
            progress(65 + int(25 * (index + 1) / len(prepared)), f"正在生成正圆柔光并羽化边缘… {index + 1}/{len(prepared)}")

    return records


def _linear_to_srgb_inplace(linear: np.ndarray) -> None:
    """Apply the sRGB transfer curve without allocating another full-size RGB image."""
    np.clip(linear, 0.0, 1.0, out=linear)
    high = np.greater(linear, 0.0031308)
    np.power(linear, 1.0 / 2.4, out=linear, where=high)
    np.multiply(linear, 1.055, out=linear, where=high)
    np.subtract(linear, 0.055, out=linear, where=high)
    np.logical_not(high, out=high)
    np.multiply(linear, 12.92, out=linear, where=high)


def _select_stars(
    detector_stars: list[Star],
    catalog_entries: list[tuple[Star, float, bool]],
    brightness_source: str,
    relative_magnitude_limit: float,
) -> tuple[list[Star], int, float | None, int]:
    if brightness_source == "catalog":
        if not catalog_entries:
            raise ValueError(
                "真实星表亮度模式没有可复用的星表匹配结果。请重新识别照片，或切换到图像解析星点亮度。"
            )
        reference_g_mag = min(g_mag for _star, g_mag, _recovered in catalog_entries)
        catalog_stars: list[Star] = []
        recovered_count = 0
        for star, g_mag, was_recovered in catalog_entries:
            delta_g = max(0.0, float(g_mag - reference_g_mag))
            if delta_g > relative_magnitude_limit:
                continue
            relative_catalog_flux = 10.0 ** (-0.4 * delta_g)
            catalog_stars.append(replace(
                star,
                flux=relative_catalog_flux,
                relative_flux_ratio=relative_catalog_flux,
                image_flux=star.image_flux if star.image_flux is not None else star.flux,
                catalog_g_mag=g_mag,
                catalog_delta_magnitude=delta_g,
                catalog_position_recovered=was_recovered,
            ))
            recovered_count += int(was_recovered)
        stars = sorted(catalog_stars, key=lambda star: star.catalog_g_mag if star.catalog_g_mag is not None else math.inf)
        if not stars:
            raise ValueError("本地星表匹配成功，但所选相对星等范围内没有星点；请增大亮度范围控制值。")
        return stars, len(stars), reference_g_mag, recovered_count

    brightest_image_flux = max((star.flux for star in detector_stars), default=0.0)
    if brightest_image_flux <= 0.0:
        raise ValueError("图像解析模式没有可用的星点测光结果。")
    image_stars: list[Star] = []
    for star in detector_stars:
        relative_flux = float(star.flux) / brightest_image_flux
        delta_m = -2.5 * math.log10(max(relative_flux, 1e-20))
        if delta_m <= relative_magnitude_limit + 1e-9:
            image_stars.append(replace(
                star,
                relative_flux_ratio=relative_flux,
                image_flux=star.flux,
            ))
    if not image_stars:
        raise ValueError("当前图像亮度范围内没有检出的星点，请增大 Δm 或降低星点识别灵敏度。")
    return image_stars, len(image_stars), None, 0


def _brightest_star_crops(
    image_data: np.ndarray,
    stars: list[Star],
    detector_shape: tuple[int, int],
    brightness_source: str,
    max_radius: float,
    limit: int = 3,
) -> list[tuple[tuple[int, int, int, int], np.ndarray, Star, str]]:
    if not stars:
        return []
    if brightness_source == "catalog":
        ranked_stars = sorted(
            stars,
            key=lambda star: (
                star.catalog_g_mag if star.catalog_g_mag is not None else math.inf,
                -star.flux,
                star.y,
                star.x,
            ),
        )
    else:
        ranked_stars = sorted(stars, key=lambda star: (-star.flux, star.y, star.x))
    height, width = image_data.shape[:2]
    scale_x, scale_y = width / detector_shape[1], height / detector_shape[0]
    half = int(np.clip(math.ceil(max(64.0, max_radius * 2.5)), 64, 256))
    crop_size = 2 * half + 1
    bright_count = min(max(1, int(limit)), len(ranked_stars))
    selected: list[tuple[Star, str, int]] = [
        (star, "BRIGHT", index + 1)
        for index, star in enumerate(ranked_stars[:bright_count])
    ]
    bright_keys = {(round(star.x, 3), round(star.y, 3)) for star in ranked_stars[:bright_count]}
    faint_stars = [
        star for star in reversed(ranked_stars[-bright_count:])
        if (round(star.x, 3), round(star.y, 3)) not in bright_keys
    ]
    selected.extend(
        (star, "FAINT", index + 1) for index, star in enumerate(faint_stars)
    )
    result: list[tuple[tuple[int, int, int, int], np.ndarray, Star, str]] = []
    for star, section, section_index in selected:
        center_x = int(round((star.x + 0.5) * scale_x - 0.5))
        center_y = int(round((star.y + 0.5) * scale_y - 0.5))
        center_x = int(np.clip(center_x, 0, width - 1))
        center_y = int(np.clip(center_y, 0, height - 1))
        x0, y0 = center_x - half, center_y - half
        x1, y1 = x0 + crop_size, y0 + crop_size
        source_x0, source_y0 = max(0, x0), max(0, y0)
        source_x1, source_y1 = min(width, x1), min(height, y1)
        crop = np.array(image_data[source_y0:source_y1, source_x0:source_x1], dtype=np.float32, copy=True)
        padding = (
            (source_y0 - y0, y1 - source_y1),
            (source_x0 - x0, x1 - source_x1),
        )
        if crop.ndim > 2:
            padding += ((0, 0),) * (crop.ndim - 2)
        if any(before or after for before, after in padding):
            crop = np.pad(crop, padding, mode="edge")
        if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
            continue
        if brightness_source == "catalog" and star.catalog_g_mag is not None:
            brightness = f"G={star.catalog_g_mag:.1f}"
            if star.catalog_delta_magnitude is not None:
                brightness += f" dG={star.catalog_delta_magnitude:.1f}"
        else:
            delta_m = -2.5 * math.log10(max(float(star.relative_flux_ratio), 1e-20))
            brightness = f"dm={delta_m:.1f} mag"
        name = star.catalog_name or f"SEP source {section_index}"
        source_kind = "W08-guided" if star.catalog_position_state == "recovered" else "image-confirmed"
        caption = f"{section} {section_index} | {name[:28]} | {brightness} | {source_kind}"
        result.append(((x0, y0, x1, y1), crop, star, caption))
    return result


def _comparison_jpeg(
    before_crops: list[np.ndarray],
    after_crops: list[np.ndarray],
    captions: list[str],
    profile: bytes | None,
) -> tuple[bytes | None, int]:
    if not before_crops or len(before_crops) != len(after_crops) or len(before_crops) != len(captions):
        return None, 0

    def to_rgb8(pixels: np.ndarray) -> np.ndarray:
        pixels = np.asarray(pixels, dtype=np.float32)
        if pixels.ndim == 2:
            pixels = pixels[..., None]
        if pixels.shape[2] == 1:
            pixels = np.repeat(pixels, 3, axis=2)
        pixels = np.clip(pixels[..., :3], 0.0, 1.0)
        return np.rint(pixels * 255.0).astype(np.uint8)

    panels: list[np.ndarray] = []
    label_height = 22
    for before, after, caption in zip(before_crops, after_crops, captions):
        left, right = to_rgb8(before), to_rgb8(after)
        height = min(left.shape[0], right.shape[0])
        width = min(left.shape[1], right.shape[1])
        if height < 8 or width < 8:
            continue
        left, right = left[:height, :width], right[:height, :width]
        panel = np.full((height + label_height, width * 2 + 5, 3), 9, dtype=np.uint8)
        panel[label_height:, :width] = left
        panel[label_height:, width + 5:] = right
        canvas = Image.fromarray(panel, mode="RGB")
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default(size=13)
        draw.text((5, 4), caption, fill=(230, 222, 246), font=font)
        panel = np.asarray(canvas, dtype=np.uint8)
        panels.append(panel)
    if not panels:
        return None, 0
    row_gap = 5
    panel_width = max(panel.shape[1] for panel in panels)
    panel_height = sum(panel.shape[0] for panel in panels) + row_gap * (len(panels) - 1)
    montage = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
    y = 0
    for panel in panels:
        montage[y:y + panel.shape[0], :panel.shape[1]] = panel
        y += panel.shape[0] + row_gap
    buffer = BytesIO()
    options: dict[str, object] = {"format": "JPEG", "quality": 92, "optimize": True}
    if profile:
        options["icc_profile"] = profile
    Image.fromarray(montage, mode="RGB").save(buffer, **options)
    return buffer.getvalue(), len(panels)


def _coverage_svg(
    detector_shape: tuple[int, int],
    detector_stars: list[Star],
    catalog_positions: tuple[CatalogPosition, ...],
    coverage_areas: tuple[CatalogCoverageArea, ...],
    relative_magnitude_limit: float,
    catalog_reference_g_mag: float | None,
) -> tuple[bytes, int, int, int]:
    """Draw image detections and catalog-projected candidates without conflating them."""
    height, width = detector_shape
    reference_g = catalog_reference_g_mag
    if reference_g is None:
        reference_g = min(
            (position.g_mag for position in catalog_positions if position.state in {"detected", "recovered"}),
            default=min((position.g_mag for position in catalog_positions), default=math.inf),
        )
    if math.isfinite(reference_g):
        magnitude_ceiling = reference_g + max(0.0, float(relative_magnitude_limit))
        catalog_positions = tuple(
            position for position in catalog_positions
            if position.g_mag <= magnitude_ceiling + 1e-9
        )
    else:
        catalog_positions = ()
    frame_width = 540.0
    frame_height = frame_width * height / max(width, 1)
    if frame_height > 310.0:
        frame_height = 310.0
        frame_width = frame_height * width / max(height, 1)
    frame_x = (600.0 - frame_width) * 0.5
    frame_y = 54.0 + (310.0 - frame_height) * 0.5
    columns, rows = 4, 3
    cell_width, cell_height = frame_width / columns, frame_height / rows
    bins: dict[tuple[int, int], list[int]] = {}
    for position in catalog_positions:
        if not (math.isfinite(position.x) and math.isfinite(position.y)):
            continue
        if not (0.0 <= position.x < width and 0.0 <= position.y < height):
            continue
        column = min(columns - 1, max(0, int(position.x * columns / max(width, 1))))
        row = min(rows - 1, max(0, int(position.y * rows / max(height, 1))))
        counts = bins.setdefault((column, row), [0, 0])
        counts[1] += 1
        counts[0] += int(position.state in {"detected", "recovered"})

    svg: list[str] = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="600" height="420" viewBox="0 0 600 420">',
        '<rect width="600" height="420" rx="12" fill="#0a0b10"/>',
        '<text x="18" y="24" fill="#d7c9f5" font-family="Segoe UI, sans-serif" font-size="12" font-weight="600">IMAGE FRAME · 4 × 3 COVERAGE</text>',
        f'<text x="18" y="42" fill="#98a0b4" font-family="Segoe UI, sans-serif" font-size="10">W08 G ≤ {magnitude_ceiling:.1f} | cell: confirmed / projected; amber projected points are not softened</text>' if math.isfinite(reference_g) else '<text x="18" y="42" fill="#98a0b4" font-family="Segoe UI, sans-serif" font-size="10">Image detections only | no validated catalog projection</text>',
        f'<rect x="{frame_x:.2f}" y="{frame_y:.2f}" width="{frame_width:.2f}" height="{frame_height:.2f}" rx="2" fill="#11151e" stroke="#68617f" stroke-width="1.4"/>',
    ]
    for area in coverage_areas:
        x0 = frame_x + np.clip(area.x0, 0.0, width) * frame_width / max(width, 1)
        y0 = frame_y + np.clip(area.y0, 0.0, height) * frame_height / max(height, 1)
        x1 = frame_x + np.clip(area.x1, 0.0, width) * frame_width / max(width, 1)
        y1 = frame_y + np.clip(area.y1, 0.0, height) * frame_height / max(height, 1)
        if x1 <= x0 or y1 <= y0:
            continue
        if area.source == "camera projection":
            style = 'fill="#7998e8" fill-opacity="0.08" stroke="#91aaff" stroke-opacity="0.75" stroke-dasharray="5 4"'
        else:
            style = 'fill="#67b9dc" fill-opacity="0.11" stroke="#74c7e6" stroke-opacity="0.85"'
        svg.append(f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{x1 - x0:.2f}" height="{y1 - y0:.2f}" {style}/>')
    for row in range(rows):
        for column in range(columns):
            confirmed, total = bins.get((column, row), (0, 0))
            if total == 0:
                fill, opacity, label = "#171c27", 0.9, "—"
            else:
                ratio = confirmed / total
                fill = "#54c89d" if ratio >= 0.8 else "#e3ae58" if ratio >= 0.4 else "#d56c74"
                opacity = 0.15 + 0.18 * ratio
                label = f"{confirmed}/{total}"
            x, y = frame_x + column * cell_width, frame_y + row * cell_height
            svg.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_width:.2f}" height="{cell_height:.2f}" fill="{fill}" fill-opacity="{opacity:.2f}" stroke="#43495a" stroke-width="0.8"/>'
            )
            svg.append(
                f'<text x="{x + 5:.2f}" y="{y + 14:.2f}" fill="#eef0f8" font-family="Segoe UI, sans-serif" font-size="10">{label}</text>'
            )

    def map_point(x: float, y: float) -> tuple[float, float] | None:
        if not (math.isfinite(x) and math.isfinite(y) and 0 <= x < width and 0 <= y < height):
            return None
        return frame_x + x * frame_width / max(width, 1), frame_y + y * frame_height / max(height, 1)

    # Faint image detections show the locations that SEP actually measured.
    for star in detector_stars:
        point = map_point(star.x, star.y)
        if point is not None:
            svg.append(f'<circle cx="{point[0]:.2f}" cy="{point[1]:.2f}" r="0.95" fill="#d7dbe8" fill-opacity="0.42"/>')
    for position in catalog_positions:
        point = map_point(position.x, position.y)
        if point is None:
            continue
        x, y = point
        if position.state == "detected":
            svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.25" fill="#c1a4ff" stroke="#090b11" stroke-width="0.7"/>')
        elif position.state == "recovered":
            svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.0" fill="#68e0b4" stroke="#090b11" stroke-width="0.8"/>')
        else:
            svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.2" fill="none" stroke="#f2bd66" stroke-width="1.5"/>')
    if not catalog_positions:
        svg.append(
            '<text x="18" y="405" fill="#d4be93" font-family="Segoe UI, sans-serif" font-size="10">No catalog projection; dots show image detections only.</text>'
        )
    else:
        svg.append(
            '<text x="18" y="405" fill="#98a0b4" font-family="Segoe UI, sans-serif" font-size="10">Green = WCS-guided image source recovered and softened; amber = projection only.</text>'
        )
    svg.append("</svg>")
    verified = sum(position.state in {"detected", "recovered"} for position in catalog_positions)
    unverified = sum(position.state == "predicted" for position in catalog_positions)
    return "".join(svg).encode("utf-8"), len(catalog_positions), verified, unverified


def process_raw(
    input_path: str | Path,
    output_path: str | Path,
    *,
    sensitivity: float = 4.8,
    strength: float = 10.0,
    opacity: float = 30.0,
    brightness_source: str = "catalog",
    relative_magnitude_limit: float = 5.0,
    min_radius: float = 3.0,
    max_radius: float = 42.0,
    focal_length_override_mm: float | None = None,
    sensor_format: str = "auto",
    analysis_cache_path: str | Path | None = None,
    prepared_analysis: PreparedAnalysis | None = None,
    progress: Progress | None = None,
    metadata_callback: MetadataCallback | None = None,
) -> ProcessResult:
    """Process a RAW/TIFF/JPG image and save a 16-bit TIFF in its color space."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("输出文件不能覆盖源文件。")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if max_radius < min_radius:
        min_radius, max_radius = max_radius, min_radius
    brightness_source = str(brightness_source).strip().lower()
    if brightness_source not in {"catalog", "image"}:
        raise ValueError("星点亮度来源无效，请选择真实星表亮度或图像解析星点亮度。")
    if focal_length_override_mm is not None:
        focal_length_override_mm = float(focal_length_override_mm)
        if not math.isfinite(focal_length_override_mm) or not 1.0 <= focal_length_override_mm <= 1000.0:
            raise ValueError("手动焦距须在 1–1000 mm 范围内。")
    sensor_format = str(sensor_format or "auto").strip().lower()
    if sensor_format not in {"auto", "full_frame", "aps_c", "medium_4433", "four_thirds", "one_inch"}:
        raise ValueError("相机画幅选项无效。")
    relative_magnitude_limit = float(np.clip(relative_magnitude_limit, 0.0, 10.0))
    strength = float(np.clip(strength, 0.0, 30.0))
    opacity = float(np.clip(opacity, 0.0, 100.0))

    def report(percent: int, message: str) -> None:
        if progress:
            progress(percent, message)

    report(2, "正在读取图像与镜头信息…")
    raw_preview_median: float | None = None
    raw_linear_median: float | None = None
    raw_preview_ev = 0.0
    raw_xmp_exposure_ev: float | None = None
    raw_exposure_shift = 1.0
    raw_acr_reference_median: float | None = None
    raw_acr_reference_name: str | None = None
    raw_brightness_calibration_method = "no calibration reference available"
    sky_mask_info: dict[str, object] = {}
    raster: RasterInput | None = None
    linear_srgb = False
    if prepared_analysis is None:
        if input_path.suffix.lower() in {".tif", ".tiff", ".jpg", ".jpeg"}:
            raster = _read_raster(input_path)
            info = _apply_camera_overrides(raster.info, focal_length_override_mm, sensor_format)
            raster = replace(raster, info=info)
            if metadata_callback:
                metadata_callback(info)
            detector, _factor = _raster_detector_image(raster.pixels, info.focal_length)
            sky_mask, sky_mask_info = _sky_detection_mask(raster.pixels, detector.shape)
            report(12, "正在识别星点与测量亮度…")
            stars, candidate_count, selected_count, sky_background_level = detect_stars(
                detector, sensitivity, sky_mask,
                allow_widefield_elongation=brightness_source == "catalog",
            )
            image_data = raster.pixels
            profile = raster.profile
            profile_description = raster.profile_description
            is_rgb = raster.is_rgb
            alpha = raster.alpha
            invert_gray = raster.invert_gray
            photometric = tifffile.PHOTOMETRIC.RGB if is_rgb else (
                tifffile.PHOTOMETRIC.MINISWHITE if invert_gray else tifffile.PHOTOMETRIC.MINISBLACK
            )
            input_kind = raster.input_kind
            encoding = (
                "linear-light sRGB processing; original ICC profile bytes preserved"
                if raster.linear_srgb and profile
                else "linear-light sRGB processing; input was untagged and remains untagged"
                if raster.linear_srgb
                else "native source channel encoding retained; original ICC profile bytes preserved"
            )
            report(36, f"天空区域找到 {candidate_count:,} 个点源候选，准备按亮度来源筛选…")
        else:
            raw_extensions = {
                ".cr3", ".cr2", ".crw", ".nef", ".nrw", ".arw", ".sr2", ".srf", ".dng",
                ".orf", ".rw2", ".raf", ".pef", ".ptx", ".3fr", ".fff", ".iiq", ".kdc",
                ".dcr", ".mos", ".mrw", ".x3f",
            }
            if input_path.suffix.lower() not in raw_extensions:
                raise ValueError("请选择相机 RAW、TIFF 或 JPG 文件。")
            with rawpy.imread(str(input_path)) as raw:
                raw_xmp_exposure_ev = _read_xmp_exposure_ev(input_path)
                preview = _extract_raw_preview(raw)
                info = _apply_camera_overrides(_raw_info(raw, preview), focal_length_override_mm, sensor_format)
                if metadata_callback:
                    metadata_callback(info)
                detector, _factor = _detector_image(raw, focal_length_override_mm)
                sky_mask, sky_mask_info = _sky_detection_mask(
                    preview if preview is not None else detector, detector.shape
                )
                report(12, "正在识别星点与测量亮度…")
                stars, candidate_count, selected_count, sky_background_level = detect_stars(
                    detector, sensitivity, sky_mask,
                    allow_widefield_elongation=brightness_source == "catalog",
                )
                raw_acr_reference_median, raw_acr_reference_name = _acr_reference_luminance_median(input_path)
                if preview is not None:
                    raw_preview_median = _preview_linear_luminance_median(preview)
                if preview is not None or raw_acr_reference_median is not None:
                    report(36, "正在校准 RAW 曝光与同名参考图…")
                    calibration_rgb = raw.postprocess(
                        gamma=(1, 1),
                        no_auto_bright=True,
                        exp_shift=1.0,
                        exp_preserve_highlights=0.0,
                        output_bps=16,
                        output_color=rawpy.ColorSpace.sRGB,
                        use_camera_wb=True,
                        demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                        half_size=True,
                    )
                    calibration_float = calibration_rgb.astype(np.float32)
                    np.divide(calibration_float, 65535.0, out=calibration_float)
                    raw_linear_median = _linear_rgb_luminance_median(calibration_float)
                    del calibration_rgb, calibration_float
                if preview is not None and raw_linear_median is not None and raw_preview_median is not None:
                    raw_preview_ev = _preview_exposure_ev(
                        raw_linear_median, raw_preview_median
                    )
                raw_exposure_shift = _raw_exposure_shift(
                    raw_preview_ev, raw_xmp_exposure_ev
                )
                if raw_acr_reference_median is not None and raw_linear_median is not None and raw_linear_median > 1e-8:
                    # A same-stem Adobe Camera Raw TIFF has already rendered the
                    # sidecar settings through Adobe's profile and tone pipeline.
                    # Match its scene median directly instead of stacking XMP EV a
                    # second time on top of the embedded camera JPEG.
                    raw_exposure_shift = float(np.clip(
                        raw_acr_reference_median / raw_linear_median, 0.25, 8.0
                    ))
                    raw_brightness_calibration_method = "same-stem Adobe Camera Raw TIFF median; XMP exposure already reflected in reference"
                elif raw_preview_median is not None:
                    raw_brightness_calibration_method = "embedded preview median combined with XMP Exposure2012 when present"
                elif preview is not None and raw_xmp_exposure_ev is not None:
                    raw_brightness_calibration_method = "non-sRGB embedded preview left in its original colour space; XMP Exposure2012 only"
                elif raw_xmp_exposure_ev is not None:
                    raw_brightness_calibration_method = "XMP Exposure2012 only; no embedded preview or Adobe TIFF reference"
                report(43, f"天空区域找到 {candidate_count:,} 个点源候选，正在解码 RAW…")
                rgb = raw.postprocess(
                    gamma=(1, 1),
                    no_auto_bright=True,
                    exp_shift=raw_exposure_shift,
                    exp_preserve_highlights=1.0,
                    output_bps=16,
                    output_color=rawpy.ColorSpace.sRGB,
                    use_camera_wb=True,
                    demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                )

            if rgb.ndim != 3 or rgb.shape[2] < 3:
                raise ValueError("RAW 解码后未得到 RGB 图像。")
            image_data = np.empty(rgb.shape[:2] + (3,), dtype=np.float32)
            np.divide(rgb[..., :3], 65535.0, out=image_data, casting="unsafe")
            del rgb
            profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
            profile_description = "sRGB IEC61966-2.1"
            is_rgb = True
            alpha = None
            invert_gray = False
            photometric = tifffile.PHOTOMETRIC.RGB
            input_kind = "RAW"
            encoding = "RAW developed to 16-bit sRGB with XMP/sRGB-preview or matching Adobe Camera Raw TIFF exposure calibration and highlight preservation; halo processing uses float32 linear-light values; sRGB ICC profile embedded"

        detector_shape = detector.shape
        linear_srgb = bool(raster.linear_srgb) if raster is not None else False
    else:
        report(4, "正在复用本机已识别的星点与 WCS…")
        prepared = prepared_analysis
        mapped = np.memmap(prepared.image_cache_path, mode="r", dtype=np.float32, shape=prepared.image_shape)
        try:
            image_data = np.array(mapped, dtype=np.float32, copy=True)
        finally:
            mapped._mmap.close()
        alpha = None
        if prepared.alpha_cache_path:
            alpha_shape = prepared.image_shape[:2]
            alpha_map = np.memmap(prepared.alpha_cache_path, mode="r", dtype=np.float32, shape=alpha_shape)
            try:
                alpha = np.array(alpha_map, dtype=np.float32, copy=True)
            finally:
                alpha_map._mmap.close()
        detector_shape = prepared.detector_shape
        stars = list(prepared.detector_stars)
        candidate_count = prepared.candidate_count
        selected_count = len(stars)
        sky_background_level = prepared.sky_background_level
        sky_mask_info = dict(prepared.sky_mask_info)
        info = prepared.info
        profile = prepared.profile
        profile_description = prepared.profile_description
        is_rgb = prepared.is_rgb
        invert_gray = prepared.invert_gray
        photometric = prepared.photometric
        input_kind = prepared.input_kind
        encoding = prepared.encoding
        linear_srgb = prepared.linear_srgb
        raw_preview_median = prepared.raw_preview_median
        raw_linear_median = prepared.raw_linear_median
        raw_preview_ev = prepared.raw_preview_ev
        raw_xmp_exposure_ev = prepared.raw_xmp_exposure_ev
        raw_exposure_shift = prepared.raw_exposure_shift
        raw_acr_reference_median = prepared.raw_acr_reference_median
        raw_acr_reference_name = prepared.raw_acr_reference_name
        raw_brightness_calibration_method = prepared.raw_brightness_calibration_method
        if metadata_callback:
            metadata_callback(info)
    detector_stars = list(prepared_analysis.detector_stars) if prepared_analysis is not None else list(stars)
    catalog_entries: list[tuple[Star, float, bool]] = []
    catalog_positions: tuple[CatalogPosition, ...] = ()
    catalog_coverage_areas: tuple[CatalogCoverageArea, ...] = ()
    catalog_position_count = 0
    catalog_match_count = 0
    if prepared_analysis is not None:
        catalog_entries = list(prepared_analysis.catalog_entries)
        catalog_positions = prepared_analysis.catalog_positions
        catalog_coverage_areas = prepared_analysis.catalog_coverage_areas
        catalog_position_count = prepared_analysis.catalog_position_count
        catalog_match_count = prepared_analysis.catalog_match_count
    elif brightness_source == "catalog":
        report(45, "图像解码完成，正在本机解算星空坐标…")
        solver_scale_xy = (
            image_data.shape[1] / detector.shape[1],
            image_data.shape[0] / detector.shape[0],
        )
        primary_solver_path: Path | None = None
        if input_path.suffix.lower() in {".tif", ".tiff"}:
            try:
                with tifffile.TiffFile(input_path) as source_tiff:
                    source_page = source_tiff.pages[0]
                    orientation_tag = source_page.tags.get("Orientation")
                    source_orientation = int(orientation_tag.value) if orientation_tag is not None else 1
                    if (
                        source_orientation == 1
                        and source_page.imagelength == image_data.shape[0]
                        and source_page.imagewidth == image_data.shape[1]
                    ):
                        primary_solver_path = input_path
            except (OSError, ValueError, tifffile.TiffFileError):
                primary_solver_path = None
        with tempfile.TemporaryDirectory(prefix="starsoft-solver-image-") as solver_directory:
            solver_image_path = Path(solver_directory) / "solver_16bit.tif"
            _write_solver_grayscale(image_data, solver_image_path, sky_mask)
            (
                catalog_matches,
                catalog_position_count,
                recovered_sources,
                catalog_positions,
                catalog_coverage_areas,
            ) = match_local_bright_stars(
                detector_stars,
                detector,
                sky_mask,
                info,
                solver_image_path,
                solver_scale_xy,
                primary_solver_path,
                # Cache all available W08 sources so changing Δm only reselects
                # the existing matches instead of solving and matching again.
                relative_magnitude_limit=10.0,
                sensitivity=sensitivity,
                progress=report,
            )
        catalog_match_count = len(catalog_matches) + len(recovered_sources)
        catalog_entries = [
            (replace(
                detector_stars[index],
                catalog_g_mag=match.g_mag,
                catalog_name=match.name,
                catalog_ra_deg=match.ra_deg,
                catalog_dec_deg=match.dec_deg,
                catalog_position_state="recovered" if match.position_recovered else "detected",
                catalog_position_recovered=match.position_recovered,
            ), match.g_mag, match.position_recovered)
            for index, match in catalog_matches.items()
            if 0 <= index < len(detector_stars)
        ]
        catalog_entries.extend((
            Star(
                source.x,
                source.y,
                source.flux,
                source.peak,
                source.fwhm,
                1.0,
                source.a,
                source.b,
                source.theta,
                source.signal_to_noise,
                False,
                source.flux,
                source.g_mag,
                None,
                None,
                None,
                None,
                False,
                source.name,
                source.ra_deg,
                source.dec_deg,
                catalog_position_state="recovered",
            ),
            source.g_mag,
            True,
        ) for source in recovered_sources)
    stars, selected_count, catalog_reference_g_mag, recovered_catalog_star_count = _select_stars(
        detector_stars, catalog_entries, brightness_source, relative_magnitude_limit
    )
    coverage_reference_g_mag = catalog_reference_g_mag
    if coverage_reference_g_mag is None:
        coverage_reference_g_mag = min(
            (position.g_mag for position in catalog_positions if position.state in {"detected", "recovered"}),
            default=None,
        )
    coverage_magnitude_ceiling = (
        coverage_reference_g_mag + relative_magnitude_limit
        if coverage_reference_g_mag is not None else None
    )
    in_range_catalog_positions = tuple(
        position for position in catalog_positions
        if coverage_magnitude_ceiling is not None
        and position.g_mag <= coverage_magnitude_ceiling + 1e-9
    )
    catalog_verified_count = sum(position.state in {"detected", "recovered"} for position in in_range_catalog_positions)
    catalog_unverified_count = sum(position.state == "predicted" for position in in_range_catalog_positions)
    catalog_guided_recovered_count = sum(
        star.catalog_position_state == "recovered" for star in stars
    ) if brightness_source == "catalog" else 0
    if brightness_source == "catalog":
        report(
            61,
            f"SEP 实测 {candidate_count:,} 个点源候选；ΔG≤{relative_magnitude_limit:.1f} 的星表候选 "
            f"{len(in_range_catalog_positions):,} 个，"
            f"图像确认 {catalog_verified_count:,} 个，其中星表引导回搜补回 {catalog_guided_recovered_count:,} 个；"
            f"另 {catalog_unverified_count:,} 个无点源证据，仅保留为推算候选、不柔焦；"
            f"本次柔焦 {selected_count:,} 个…",
        )
    else:
        report(61, f"SEP 图像实测 {candidate_count:,} 个点源候选，其中 {selected_count:,} 个符合亮度范围，正在柔焦…")
    detector_shape = prepared_analysis.detector_shape if prepared_analysis is not None else detector.shape

    analysis = prepared_analysis
    if analysis is None and analysis_cache_path is not None:
        cache_path = Path(analysis_cache_path)
        alpha_cache_path: Path | None = None
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_map = np.memmap(cache_path, mode="w+", dtype=np.float32, shape=image_data.shape)
            try:
                for y0 in range(0, image_data.shape[0], 512):
                    cache_map[y0:y0 + 512] = image_data[y0:y0 + 512]
                cache_map.flush()
            finally:
                cache_map._mmap.close()
            if alpha is not None:
                alpha_cache_path = cache_path.with_name(cache_path.stem + "_alpha.f32")
                alpha_map = np.memmap(alpha_cache_path, mode="w+", dtype=np.float32, shape=alpha.shape)
                try:
                    for y0 in range(0, alpha.shape[0], 512):
                        alpha_map[y0:y0 + 512] = alpha[y0:y0 + 512]
                    alpha_map.flush()
                finally:
                    alpha_map._mmap.close()
            analysis = PreparedAnalysis(
                image_cache_path=str(cache_path),
                image_shape=tuple(image_data.shape),
                alpha_cache_path=str(alpha_cache_path) if alpha_cache_path else None,
                detector_shape=tuple(detector_shape),
                detector_stars=tuple(detector_stars),
                catalog_entries=tuple(catalog_entries),
                candidate_count=int(candidate_count),
                catalog_position_count=int(catalog_position_count),
                catalog_match_count=int(catalog_match_count),
                info=info,
                profile=profile,
                profile_description=profile_description,
                is_rgb=is_rgb,
                invert_gray=invert_gray,
                photometric=photometric,
                input_kind=input_kind,
                encoding=encoding,
                linear_srgb=linear_srgb,
                sky_mask_info=dict(sky_mask_info),
                sky_background_level=float(sky_background_level),
                raw_preview_median=raw_preview_median,
                raw_linear_median=raw_linear_median,
                raw_preview_ev=float(raw_preview_ev),
                raw_xmp_exposure_ev=raw_xmp_exposure_ev,
                raw_exposure_shift=float(raw_exposure_shift),
                raw_acr_reference_median=raw_acr_reference_median,
                raw_acr_reference_name=raw_acr_reference_name,
                raw_brightness_calibration_method=raw_brightness_calibration_method,
                catalog_positions=tuple(catalog_positions),
                catalog_coverage_areas=tuple(catalog_coverage_areas),
            )
        except (OSError, ValueError, MemoryError):
            analysis = None
            cache_path.unlink(missing_ok=True)
            if alpha_cache_path is not None:
                alpha_cache_path.unlink(missing_ok=True)

    brightest_crops = _brightest_star_crops(
        image_data, stars, detector_shape, brightness_source, max_radius
    )
    before_crops = [item[1] for item in brightest_crops]
    comparison_captions = [item[3] for item in brightest_crops]
    star_measurements = _soften_stars(
        image_data,
        stars,
        detector_shape,
        min_radius,
        max_radius,
        strength,
        opacity,
        sky_background_level,
        brightness_source,
        relative_magnitude_limit,
        report,
    )
    report(92, "正在写入 16 位 TIFF…")
    if input_kind == "RAW" or linear_srgb:
        _linear_to_srgb_inplace(image_data)
        for before_crop in before_crops:
            _linear_to_srgb_inplace(before_crop)
    if invert_gray:
        np.subtract(1.0, image_data, out=image_data)
        for before_crop in before_crops:
            np.subtract(1.0, before_crop, out=before_crop)
    after_crops: list[np.ndarray] = []
    image_height, image_width = image_data.shape[:2]
    for crop_box, _before_crop, _star, _caption in brightest_crops:
        x0, y0, x1, y1 = crop_box
        source_x0, source_y0 = max(0, x0), max(0, y0)
        source_x1, source_y1 = min(image_width, x1), min(image_height, y1)
        crop = np.array(image_data[source_y0:source_y1, source_x0:source_x1], dtype=np.float32, copy=True)
        padding = ((source_y0 - y0, y1 - source_y1), (source_x0 - x0, x1 - source_x1))
        if crop.ndim > 2:
            padding += ((0, 0),) * (crop.ndim - 2)
        if any(before or after for before, after in padding):
            crop = np.pad(crop, padding, mode="edge")
        after_crops.append(crop)
    comparison_preview, comparison_star_count = _comparison_jpeg(
        before_crops, after_crops, comparison_captions, profile
    )
    coverage_positions = analysis.catalog_positions if analysis is not None else tuple(catalog_positions)
    coverage_areas = analysis.catalog_coverage_areas if analysis is not None else tuple(catalog_coverage_areas)
    coverage_preview, catalog_prediction_count, catalog_verified_count, catalog_unverified_count = _coverage_svg(
        detector_shape,
        detector_stars,
        coverage_positions,
        coverage_areas,
        relative_magnitude_limit,
        catalog_reference_g_mag,
    )
    # The working buffer remains float32 until this final 16-bit export step.
    # Clamp all formats here so additive halos cannot wrap uint16 highlights.
    np.clip(image_data, 0.0, 1.0, out=image_data)
    np.multiply(image_data, 65535.0, out=image_data)
    np.rint(image_data, out=image_data)
    pixels16 = image_data.astype(np.uint16)
    del image_data
    if alpha is not None:
        alpha16 = np.rint(np.clip(alpha, 0.0, 1.0) * 65535.0).astype(np.uint16)
        pixels16 = np.concatenate((pixels16, alpha16[..., None]), axis=2)
        del alpha16

    description = {
        "software": f"星点柔焦 {APP_VERSION}",
        "camera": info.camera,
        "lens": info.lens,
        "focal_length_mm": info.focal_length,
        "focal_length_manual_override_mm": focal_length_override_mm,
        "focal_length_35mm_equivalent": info.focal_length_35mm,
        "sensor_format": info.sensor_format or "auto",
        "aperture": info.aperture,
        "detected_stars": sum(star.catalog_position_state in {"detected", "recovered"} for star in stars) if brightness_source == "catalog" else len(stars),
        "soft_focus_center_count": len(stars),
        "point_source_candidates": candidate_count,
        "brightness_source": brightness_source,
        "brightness_source_label": "真实星表亮度" if brightness_source == "catalog" else "图像解析星点亮度",
        "star_detection": "SEP local background/RMS, PSF matched extraction, circular aperture flux",
        "star_selection": (
            "ASTAP W08 Gaia-derived G magnitudes rounded to 0.1 mag; select image-confirmed and WCS-guided recovered point sources within delta G of the brightest selected in-frame source"
            if brightness_source == "catalog"
            else "1.4.7 SEP circular-aperture image photometry; select point sources within delta m of the brightest detected sky source"
        ),
        "catalog": (
            "Bundled ASTAP W08 Gaia-derived all-sky bright-star index (approximately complete through G=8); WCS solving and coordinate cross-match are local and offline"
            if brightness_source == "catalog" else None
        ),
        "catalog_match_count": catalog_match_count,
        "catalog_position_recovered_count": recovered_catalog_star_count,
        "catalog_position_count_checked_locally": catalog_position_count,
        "catalog_projected_candidate_count": catalog_prediction_count,
        "catalog_image_confirmed_candidate_count": catalog_verified_count,
        "catalog_unconfirmed_projected_candidate_count_not_softened": catalog_unverified_count,
        "catalog_wcs_guided_recovered_soft_focus_center_count": catalog_guided_recovered_count,
        "catalog_reference_g_magnitude": round(catalog_reference_g_mag, 5) if catalog_reference_g_mag is not None else None,
        "catalog_photometry_fields": ["Gaia-derived G magnitude (W08, 0.1 mag resolution)" ] if brightness_source == "catalog" else [],
        "catalog_name_source": "HYG v4.1 common names/Bayer-Flamsteed designations matched to W08 sky coordinates; RA/Dec fallback where no name matches" if brightness_source == "catalog" else None,
        "partial_coverage_policy": "Coverage counts image-confirmed and catalog-projected W08 positions separately. Validated WCS projections guide a wider local search; only a compact image source found by SEP at or near a projected position is added as a recovered soft-focus center. Projections without compact-source image evidence remain visible in the map and are not softened. Sky-mask foreground positions are excluded.",
        "relative_magnitude_limit": relative_magnitude_limit,
        "relative_flux_floor_ratio": round(10.0 ** (-0.4 * relative_magnitude_limit), 8),
        "relative_magnitude_difference_definition": (
            "delta_G=local W08 G magnitude minus the brightest image-confirmed or WCS-guided recovered source; include sources where delta_G is at most the selected limit"
            if brightness_source == "catalog"
            else "delta_m=-2.5*log10(SEP circular-aperture flux / brightest detected SEP aperture flux); include sources where delta_m is at most the selected limit"
        ),
        "soft_focus": "strictly circular isotropic Gaussian diffusion convolved with an RGB stellar radial profile sampled by circular-annulus medians; x and y sigma are equal and SEP ellipticity/angle never shapes the halo; the convolved profile is peak-matched to the historical relative-flux and color response, then only positive lightening difference is blended over the source with a circular smoothstep feather mask",
        "halo_geometry": "strictly circular; Euclidean radial bins and equal x/y Gaussian sigma; source ellipticity and position angle are not used to shape the halo",
        "brightness_to_radius_curve": (
            "linear response to relative local W08 G magnitude rounded to 0.1 mag: log(catalog_flux/faintest_selected_flux) normalized to the brightest selected catalogue source, matching 18ffa10 mapping"
            if brightness_source == "catalog"
            else "1.4.7 linear response to SEP relative aperture magnitude: clamp(1-delta_m/selected_limit, 0, 1)"
        ),
        "brightness_to_halo_strength_curve": (
            "each selected star's measured RGB radial profile is scattered by an isotropic Gaussian kernel and peak-matched to local W08 relative G-band flux; per-channel halo colour comes from the source image RGB profile"
            if brightness_source == "catalog"
            else "each selected star's measured RGB radial profile is scattered by an isotropic Gaussian kernel and peak-matched to 1.4.7 SEP relative aperture flux; per-channel halo colour comes from the source image RGB profile"
        ),
        "soft_focus_strength": round(strength, 1),
        "soft_focus_strength_note": "Gaussian scattering-kernel sigma=(radius/3)*sqrt(strength/40); default strength 10, maximum 30; opacity controls the blend amount independently",
        "soft_focus_opacity_percent": round(opacity, 1),
        "soft_focus_color": "background-subtracted RGB aperture chromaticity blended with core chromaticity; per-channel halo and original ICC retained",
        "sky_background_level": round(sky_background_level, 7),
        "sky_reference_level": REFERENCE_SKY_LEVEL,
        "sky_adaptation_gain": round(float(np.clip(sky_background_level / REFERENCE_SKY_LEVEL, 0.70, 1.50)), 4),
        "sky_adaptation": "SEP global background measured within the detected sky mask; peak-matched halo radiance is scaled in proportion to sky luminance with a 0.70x–1.50x clamp (Weber contrast adaptation)",
        "source_rejection": "sky-region mask; SEP PSF matched detection; reject sources with roundness below 0.35, minor-axis size below 0.35 px, major-axis size above max(6 px, 5x estimated PSF sigma) capped at 12 px, or local RMS above 4x sky-only global RMS except compact sources in dense stellar fields",
        "star_photometry_and_halo_parameters": star_measurements,
        "crowded_stellar_field_sources": sum(star.crowded_field for star in stars),
        "crowded_stellar_field_rule": "allow compact, round PSF detections above the local RMS threshold only when at least four compact detections lie within max(16 detector pixels, 6 x estimated PSF sigma); extended diffuse structure remains rejected",
        "sky_region": sky_mask_info,
        "encoding": encoding,
        "input_color_profile": profile_description,
        "raw_preview_available": input_kind == "RAW" and info.preview is not None,
        "raw_embedded_preview_median_luminance_linear": round(raw_preview_median, 7) if raw_preview_median is not None else None,
        "raw_unadjusted_median_luminance_linear": round(raw_linear_median, 7) if raw_linear_median is not None else None,
        "raw_preview_exposure_ev": round(raw_preview_ev, 5) if input_kind == "RAW" else None,
        "raw_xmp_exposure_ev": round(raw_xmp_exposure_ev, 5) if raw_xmp_exposure_ev is not None else None,
        "raw_acr_reference_tiff": raw_acr_reference_name if input_kind == "RAW" else None,
        "raw_acr_reference_median_luminance_linear": round(raw_acr_reference_median, 7) if raw_acr_reference_median is not None else None,
        "raw_brightness_calibration_method": raw_brightness_calibration_method if input_kind == "RAW" else None,
        "raw_final_exposure_shift": round(raw_exposure_shift, 5) if input_kind == "RAW" else None,
    }
    temporary = output_path.with_name(output_path.stem + ".writing" + output_path.suffix)
    try:
        write_options = {
            "photometric": photometric,
            "metadata": {"axes": "YXS" if is_rgb or alpha is not None else "YX", **description},
            "software": f"StarSoftFocus {APP_VERSION}",
        }
        if is_rgb:
            write_options["planarconfig"] = "contig"
        if alpha is not None:
            write_options["extrasamples"] = "unassalpha"
        if profile:
            write_options["iccprofile"] = profile
        tifffile.imwrite(
            str(temporary),
            pixels16[..., 0] if not is_rgb and alpha is None else pixels16,
            **write_options,
        )
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    report(100, "完成")
    return ProcessResult(
        output_path=str(output_path),
        star_count=len(stars),
        candidate_count=candidate_count,
        selected_count=selected_count,
        catalog_match_count=catalog_match_count,
        recovered_catalog_star_count=recovered_catalog_star_count,
        brightness_source=brightness_source,
        relative_magnitude_limit=relative_magnitude_limit,
        width=int(pixels16.shape[1]),
        height=int(pixels16.shape[0]),
        camera=info.camera,
        lens=info.lens,
        input_kind=input_kind,
        color_profile=profile_description,
        sky_background_level=sky_background_level,
        sky_adaptation_gain=float(np.clip(sky_background_level / REFERENCE_SKY_LEVEL, 0.70, 1.50)),
        comparison_preview=comparison_preview,
        comparison_star_count=comparison_star_count,
        coverage_preview=coverage_preview,
        catalog_prediction_count=catalog_prediction_count,
        catalog_verified_count=catalog_verified_count,
        catalog_unverified_count=catalog_unverified_count,
        catalog_guided_recovered_count=catalog_guided_recovered_count,
        analysis=analysis,
    )
