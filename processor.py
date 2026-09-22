"""RAW star detection and brightness-weighted soft-focus processing."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import rawpy
import tifffile
from PIL import Image, ImageCms


Progress = Callable[[int, str], None]


@dataclass(frozen=True)
class RawInfo:
    width: int
    height: int
    camera: str
    lens: str
    focal_length: float | None
    aperture: float | None
    preview: Image.Image | None


MetadataCallback = Callable[[RawInfo], None]


@dataclass(frozen=True)
class Star:
    x: float
    y: float
    flux: float
    peak: float


@dataclass(frozen=True)
class ProcessResult:
    output_path: str
    star_count: int
    width: int
    height: int
    camera: str
    lens: str


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").replace("\x00", " ").split())


def _raw_info(raw: rawpy.RawPy, preview: Image.Image | None = None) -> RawInfo:
    sizes = raw.sizes
    other = raw.other
    lens_data = raw.lens
    # rawpy exposes lens and exposure fields but not the camera body strings.
    # Keep that distinction visible rather than presenting the lens as the body.
    camera = "相机型号未读取"
    lens_name = " ".join(
        part for part in (_clean_text(lens_data.make), _clean_text(lens_data.model)) if part
    ) or "未记录镜头"
    focal = float(other.focal_length or 0)
    aperture = float(other.aperture or 0)
    return RawInfo(
        width=int(sizes.width),
        height=int(sizes.height),
        camera=camera,
        lens=lens_name,
        focal_length=focal if math.isfinite(focal) and focal > 0 else None,
        aperture=aperture if math.isfinite(aperture) and aperture > 0 else None,
        preview=preview,
    )


def read_raw_info(path: str | Path) -> RawInfo:
    """Read camera/lens metadata and an embedded preview without full demosaicing."""
    with rawpy.imread(str(path)) as raw:
        preview: Image.Image | None = None
        try:
            thumb = raw.extract_thumb()
            if thumb.format == rawpy.ThumbFormat.JPEG:
                from io import BytesIO

                preview = Image.open(BytesIO(thumb.data)).convert("RGB")
            elif thumb.format == rawpy.ThumbFormat.BITMAP:
                preview = Image.fromarray(thumb.data).convert("RGB")
        except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
            pass
        return _raw_info(raw, preview)


def _detector_image(raw: rawpy.RawPy) -> tuple[np.ndarray, int]:
    """Build a small linear luminance proxy from the visible sensor area."""
    sensor = raw.raw_image_visible
    height, width = sensor.shape[:2]

    focal = float(raw.other.focal_length or 0)
    if not focal:
        focal = 70.0
    # Longer focal lengths resolve a star to more sensor pixels; retain more
    # detector detail for them. Short lenses use a smaller, faster search image.
    target_edge = int(np.clip(2200 + focal * 5.0, 2300, 4200))
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
    return image, factor


def _background_and_noise(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape
    tile_size = 112
    rows = math.ceil(height / tile_size)
    cols = math.ceil(width / tile_size)
    bg_tiles = np.empty((rows, cols), dtype=np.float32)
    noise_tiles = np.empty((rows, cols), dtype=np.float32)

    for ty in range(rows):
        y0 = ty * tile_size
        y1 = min(height, y0 + tile_size)
        for tx in range(cols):
            x0 = tx * tile_size
            x1 = min(width, x0 + tile_size)
            patch = image[y0:y1, x0:x1]
            median = float(np.median(patch))
            mad = float(np.median(np.abs(patch - median))) * 1.4826
            bg_tiles[ty, tx] = median
            noise_tiles[ty, tx] = mad

    # Nearest-tile interpolation keeps the full detector image allocation small.
    bg = np.repeat(np.repeat(bg_tiles, tile_size, axis=0), tile_size, axis=1)[:height, :width]
    noise = np.repeat(np.repeat(noise_tiles, tile_size, axis=0), tile_size, axis=1)[:height, :width]
    global_mad = float(np.median(np.abs(image - np.median(image)))) * 1.4826
    noise_floor = max(global_mad * 0.25, 1.0 / 65535.0)
    np.maximum(noise, noise_floor, out=noise)
    return bg, noise


def _star_shape_ok(signal: np.ndarray, x: int, y: int, threshold: float) -> bool:
    height, width = signal.shape
    if x < 3 or y < 3 or x >= width - 3 or y >= height - 3:
        return False
    patch = signal[y - 3 : y + 4, x - 3 : x + 4]
    peak = float(signal[y, x])
    # A hot/dead sensor pixel has no surrounding point-spread profile.
    if np.count_nonzero(patch >= max(peak * 0.045, threshold * 0.18)) < 2:
        return False
    weights = np.maximum(patch - max(threshold * 0.08, 0.0), 0.0)
    total = float(weights.sum())
    if total <= 0:
        return False
    yy, xx = np.mgrid[-3:4, -3:4]
    cx = float((weights * xx).sum() / total)
    cy = float((weights * yy).sum() / total)
    if abs(cx) > 1.15 or abs(cy) > 1.15:
        return False
    xx = xx - cx
    yy = yy - cy
    mxx = float((weights * xx * xx).sum() / total)
    myy = float((weights * yy * yy).sum() / total)
    mxy = float((weights * xx * yy).sum() / total)
    discriminant = math.sqrt(max((mxx - myy) ** 2 + 4.0 * mxy * mxy, 0.0))
    major = max((mxx + myy + discriminant) * 0.5, 1e-8)
    minor = max((mxx + myy - discriminant) * 0.5, 0.0)
    roundness = math.sqrt(minor / major)
    return roundness >= 0.19 and major < 18.0


def detect_stars(
    detector: np.ndarray,
    focal_length: float | None,
    sensitivity: float,
    max_stars: int = 12000,
) -> list[Star]:
    """Find compact, round local maxima with an adaptive local-noise threshold."""
    background, noise = _background_and_noise(detector)
    signal = detector - background
    threshold_map = noise * float(sensitivity)
    center = signal[1:-1, 1:-1]
    candidates = center >= threshold_map[1:-1, 1:-1]
    neighbors = (
        signal[:-2, :-2], signal[:-2, 1:-1], signal[:-2, 2:],
        signal[1:-1, :-2], signal[1:-1, 2:],
        signal[2:, :-2], signal[2:, 1:-1], signal[2:, 2:],
    )
    for neighbor in neighbors:
        candidates &= center >= neighbor
    # Break plateaus deterministically and avoid duplicate equal-valued maxima.
    strict = np.zeros_like(candidates)
    for neighbor in neighbors:
        strict |= center > neighbor
    candidates &= strict

    ys, xs = np.nonzero(candidates)
    if len(xs) == 0:
        return []
    ys = ys + 1
    xs = xs + 1
    peaks = signal[ys, xs]
    order = np.argsort(peaks)[::-1]
    if len(order) > 60000:
        order = order[:60000]

    # The focal-length prior narrows duplicate suppression for wide lenses and
    # allows slightly wider separation for long lenses. Missing EXIF uses 70 mm.
    focal = focal_length or 70.0
    min_separation = float(np.clip(1.45 + focal / 700.0, 1.5, 2.5))
    cell_size = min_separation
    buckets: dict[tuple[int, int], list[tuple[float, float]]] = {}
    stars: list[Star] = []

    for idx in order:
        x = int(xs[idx])
        y = int(ys[idx])
        peak = float(peaks[idx])
        threshold = float(threshold_map[y, x])
        if not _star_shape_ok(signal, x, y, threshold):
            continue
        cell = (int(x / cell_size), int(y / cell_size))
        duplicate = False
        for by in range(cell[1] - 2, cell[1] + 3):
            for bx in range(cell[0] - 2, cell[0] + 3):
                for ox, oy in buckets.get((bx, by), ()):
                    if (ox - x) ** 2 + (oy - y) ** 2 < min_separation**2:
                        duplicate = True
                        break
                if duplicate:
                    break
            if duplicate:
                break
        if duplicate:
            continue
        patch = signal[y - 2 : y + 3, x - 2 : x + 3]
        flux = float(np.maximum(patch, 0).sum())
        if flux <= 0:
            continue
        buckets.setdefault(cell, []).append((x, y))
        stars.append(Star(float(x), float(y), flux, peak))
        if len(stars) >= max_stars:
            break
    return stars


def _soften_stars(
    linear_rgb: np.ndarray,
    stars: list[Star],
    detector_shape: tuple[int, int],
    min_radius: float,
    max_radius: float,
    strength: float,
    progress: Progress | None = None,
) -> None:
    if not stars or strength <= 0:
        return
    out_h, out_w = linear_rgb.shape[:2]
    det_h, det_w = detector_shape
    xs = np.asarray([star.flux for star in stars], dtype=np.float64)
    log_flux = np.log10(np.maximum(xs, 1e-20))
    low, high = np.percentile(log_flux, [10, 99])
    if high <= low:
        levels = np.linspace(0.0, 1.0, len(stars), dtype=np.float32)
    else:
        levels = np.clip((log_flux - low) / (high - low), 0.0, 1.0).astype(np.float32)

    order = np.argsort(xs)  # process dim stars first so bright-star glow remains visible
    for index, position in enumerate(order):
        star = stars[int(position)]
        brightness = float(levels[int(position)]) ** 0.85
        radius = float(min_radius + brightness * (max_radius - min_radius))
        sigma = max(radius / 3.0, 0.65)
        cx = int(round((star.x + 0.5) * out_w / det_w))
        cy = int(round((star.y + 0.5) * out_h / det_h))
        extent = max(2, int(math.ceil(radius)))
        x0 = max(0, cx - extent)
        x1 = min(out_w, cx + extent + 1)
        y0 = max(0, cy - extent)
        y1 = min(out_h, cy + extent + 1)
        patch = linear_rgb[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        source = patch.copy()
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = np.exp(-0.5 * ((xx - cx) ** 2 + (yy - cy) ** 2) / max((radius / 2.4) ** 2, 0.2))
        mask = (mask * strength).astype(np.float32)[..., None]
        blurred = _blur_float_rgb(source, sigma)
        patch[...] = source + (blurred - source) * mask
        if progress and (index == len(order) - 1 or index % max(1, len(order) // 20) == 0):
            progress(65 + int(25 * (index + 1) / len(order)), f"正在柔化星点… {index + 1}/{len(order)}")


def _box_blur_float(image: np.ndarray, radius: int, axis: int) -> np.ndarray:
    """Fast edge-extended box blur for float RGB arrays using cumulative sums."""
    if radius <= 0:
        return image
    pad = [(0, 0)] * image.ndim
    pad[axis] = (radius, radius)
    padded = np.pad(image, pad, mode="edge")
    cumulative = np.cumsum(padded, axis=axis, dtype=np.float32)
    zero_shape = list(cumulative.shape)
    zero_shape[axis] = 1
    cumulative = np.concatenate((np.zeros(zero_shape, dtype=np.float32), cumulative), axis=axis)
    width = radius * 2 + 1
    start = [slice(None)] * image.ndim
    end = [slice(None)] * image.ndim
    start[axis] = slice(0, image.shape[axis])
    end[axis] = slice(width, width + image.shape[axis])
    return (cumulative[tuple(end)] - cumulative[tuple(start)]) / width


def _blur_float_rgb(image: np.ndarray, sigma: float) -> np.ndarray:
    """Approximate a Gaussian with three separable box passes in float32."""
    radius = max(1, int(round(sigma)))
    blurred = image
    for _ in range(3):
        blurred = _box_blur_float(blurred, radius, axis=1)
    for _ in range(3):
        blurred = _box_blur_float(blurred, radius, axis=0)
    return blurred


def _linear_to_srgb_inplace(linear: np.ndarray) -> None:
    """Apply the sRGB transfer curve without allocating another full-size RGB image."""
    np.clip(linear, 0.0, 1.0, out=linear)
    high = np.greater(linear, 0.0031308)
    np.power(linear, 1.0 / 2.4, out=linear, where=high)
    np.multiply(linear, 1.055, out=linear, where=high)
    np.subtract(linear, 0.055, out=linear, where=high)
    np.logical_not(high, out=high)
    np.multiply(linear, 12.92, out=linear, where=high)


def process_raw(
    input_path: str | Path,
    output_path: str | Path,
    *,
    sensitivity: float = 4.8,
    strength: float = 0.55,
    min_radius: float = 3.0,
    max_radius: float = 32.0,
    max_stars: int = 12000,
    progress: Progress | None = None,
    metadata_callback: MetadataCallback | None = None,
) -> ProcessResult:
    """Develop one RAW, soften detected stars, and save a 16-bit RGB TIFF."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("输出文件不能覆盖 RAW 源文件。")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if max_radius < min_radius:
        min_radius, max_radius = max_radius, min_radius

    def report(percent: int, message: str) -> None:
        if progress:
            progress(percent, message)

    report(2, "正在读取 RAW 与镜头信息…")
    with rawpy.imread(str(input_path)) as raw:
        info = _raw_info(raw)
        if metadata_callback:
            metadata_callback(info)
        detector, _factor = _detector_image(raw)
        report(12, "正在识别星点与测量亮度…")
        stars = detect_stars(detector, info.focal_length, sensitivity, max_stars)
        report(35, f"识别到 {len(stars):,} 个候选星点，正在解码 RAW…")
        rgb = raw.postprocess(
            gamma=(1, 1),
            no_auto_bright=True,
            output_bps=16,
            output_color=rawpy.ColorSpace.sRGB,
            use_camera_wb=True,
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
        )

    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("RAW 解码后未得到 RGB 图像。")
    linear_rgb = np.empty(rgb.shape[:2] + (3,), dtype=np.float32)
    np.divide(rgb[..., :3], 65535.0, out=linear_rgb, casting="unsafe")
    del rgb
    report(63, "RAW 解码完成，开始按星点亮度柔化…")
    _soften_stars(
        linear_rgb,
        stars,
        detector.shape,
        min_radius,
        max_radius,
        float(np.clip(strength, 0.0, 1.0)),
        report,
    )
    report(92, "正在写入 16 位 TIFF…")
    _linear_to_srgb_inplace(linear_rgb)
    np.multiply(linear_rgb, 65535.0, out=linear_rgb)
    np.rint(linear_rgb, out=linear_rgb)
    rgb16 = linear_rgb.astype(np.uint16)
    del linear_rgb
    description = {
        "software": "星点柔焦 1.0",
        "camera": info.camera,
        "lens": info.lens,
        "focal_length_mm": info.focal_length,
        "aperture": info.aperture,
        "detected_stars": len(stars),
        "encoding": "sRGB transfer, 16-bit RGB",
    }
    srgb_profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    temporary = output_path.with_name(output_path.stem + ".writing" + output_path.suffix)
    try:
        tifffile.imwrite(
            str(temporary),
            rgb16,
            photometric="rgb",
            planarconfig="contig",
            metadata={"axes": "YXS", **description},
            iccprofile=srgb_profile,
            software="StarSoftFocus 1.0",
        )
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    report(100, "完成")
    return ProcessResult(
        output_path=str(output_path),
        star_count=len(stars),
        width=int(rgb16.shape[1]),
        height=int(rgb16.shape[0]),
        camera=info.camera,
        lens=info.lens,
    )
