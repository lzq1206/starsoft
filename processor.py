"""RAW star detection and brightness-weighted soft-focus processing."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import rawpy
import sep
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
    fwhm: float


@dataclass(frozen=True)
class ProcessResult:
    output_path: str
    star_count: int
    candidate_count: int
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
    """Build a binned, linear luminance proxy from the visible sensor area."""
    sensor = raw.raw_image_visible
    height, width = sensor.shape[:2]

    focal = float(raw.other.focal_length or 0)
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


def detect_stars(
    detector: np.ndarray,
    sensitivity: float,
    max_stars: int = 50,
) -> tuple[list[Star], int]:
    """Use SEP extraction and aperture photometry, returning the brightest point sources."""
    data = np.ascontiguousarray(detector, dtype=np.float32)
    background = sep.Background(data, bw=64, bh=64, fw=3, fh=3)
    signal = np.ascontiguousarray(data - background.back(), dtype=np.float32)
    noise = np.ascontiguousarray(background.rms(), dtype=np.float32)
    first_pass = sep.extract(
        signal,
        float(sensitivity),
        err=noise,
        minarea=4,
        deblend_nthresh=32,
        deblend_cont=0.005,
        clean=True,
    )
    if len(first_pass) == 0:
        return [], 0

    first_major = np.maximum(first_pass["a"], first_pass["b"])
    first_minor = np.minimum(first_pass["a"], first_pass["b"])
    first_size = np.sqrt(np.maximum(first_pass["a"] * first_pass["b"], 0.0))
    first_roundness = first_minor / np.maximum(first_major, 1e-8)
    first_good = (
        (first_pass["flag"] == 0)
        & (first_roundness >= 0.5)
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
    objects = sep.extract(
        signal,
        float(sensitivity),
        err=noise,
        minarea=4,
        filter_kernel=kernel,
        filter_type="matched",
        deblend_nthresh=32,
        deblend_cont=0.005,
        clean=True,
        segmentation_map=False,
    )
    if len(objects) == 0:
        return [], 0

    major = np.maximum(objects["a"], objects["b"])
    minor = np.minimum(objects["a"], objects["b"])
    moment_size = np.sqrt(np.maximum(objects["a"] * objects["b"], 0.0))
    roundness = minor / np.maximum(major, 1e-8)
    bad_flags = (objects["flag"] & (sep.OBJ_TRUNC | sep.OBJ_SINGU)) != 0
    point_source = (
        np.isfinite(objects["x"])
        & np.isfinite(objects["y"])
        & np.isfinite(objects["flux"])
        & np.isfinite(moment_size)
        & (objects["flux"] > 0)
        & (minor >= 0.45)
        & (major <= 12.0)
        & (roundness >= 0.35)
        & ~bad_flags
    )
    indices = np.flatnonzero(point_source)
    if len(indices) == 0:
        return [], 0

    # SEP's isophotal flux is useful for extraction. Circular aperture flux is
    # measured separately so the brightness ordering includes more of each PSF.
    fwhm = 2.354820045 * moment_size[indices]
    aperture_radius = np.clip(1.25 * fwhm, 2.5, 12.0)
    flux, _flux_error, aperture_flags = sep.sum_circle(
        signal,
        objects["x"][indices],
        objects["y"][indices],
        aperture_radius,
        err=noise,
        subpix=5,
    )
    good = np.isfinite(flux) & (flux > 0) & ((aperture_flags & sep.APER_TRUNC) == 0)
    indices = indices[good]
    flux = flux[good]
    if len(indices) == 0:
        return [], 0

    candidate_count = int(len(indices))
    order = np.argsort(flux)[::-1][: max(1, int(max_stars))]
    stars = [
        Star(
            float(objects["x"][indices[i]]),
            float(objects["y"][indices[i]]),
            float(flux[i]),
            float(objects["peak"][indices[i]]),
            float(2.354820045 * moment_size[indices[i]]),
        )
        for i in order
    ]
    return stars, candidate_count


def _soften_stars(
    linear_rgb: np.ndarray,
    stars: list[Star],
    detector_shape: tuple[int, int],
    min_radius: float,
    max_radius: float,
    strength: float,
    progress: Progress | None = None,
) -> list[dict[str, float]]:
    if not stars or strength <= 0:
        return []
    out_h, out_w = linear_rgb.shape[:2]
    det_h, det_w = detector_shape
    xs = np.asarray([star.flux for star in stars], dtype=np.float64)
    log_flux = np.log10(np.maximum(xs, 1e-20))
    low, high = np.percentile(log_flux, [10, 99])
    if high <= low:
        levels = np.linspace(0.0, 1.0, len(stars), dtype=np.float32)
    else:
        levels = np.clip((log_flux - low) / (high - low), 0.0, 1.0).astype(np.float32)

    order = np.argsort(xs)[::-1]
    records: list[dict[str, float]] = []
    scale_x = out_w / det_w
    scale_y = out_h / det_h
    prepared: list[tuple[Star, float, float, np.ndarray, np.ndarray]] = []
    for star in stars:
        cx = (star.x + 0.5) * scale_x - 0.5
        cy = (star.y + 0.5) * scale_y - 0.5
        center_x = int(np.clip(round(cx), 0, out_w - 1))
        center_y = int(np.clip(round(cy), 0, out_h - 1))
        core = linear_rgb[
            max(0, center_y - 2) : min(out_h, center_y + 3),
            max(0, center_x - 2) : min(out_w, center_x + 3),
            :,
        ]
        color_peak = np.max(core.reshape(-1, 3), axis=0)
        center_rgb = linear_rgb[center_y, center_x, :].copy()
        prepared.append((star, cx, cy, color_peak, center_rgb))

    for index, position in enumerate(order):
        star, cx, cy, color_peak, center_rgb = prepared[int(position)]
        brightness = float(levels[int(position)]) ** 0.85
        radius = float(min_radius + brightness * (max_radius - min_radius))
        sigma = max(radius / 3.0, 0.65)
        extent = max(2, int(math.ceil(3.0 * sigma)))
        center_x = int(np.clip(round(cx), 0, out_w - 1))
        center_y = int(np.clip(round(cy), 0, out_h - 1))
        x0 = max(0, center_x - extent)
        x1 = min(out_w, center_x + extent + 1)
        y0 = max(0, center_y - extent)
        y1 = min(out_h, center_y + extent + 1)
        if x1 <= x0 or y1 <= y0:
            continue

        # The Gaussian profile is a separate light layer. Lighten compositing
        # keeps brighter source pixels exactly as captured, including the core,
        # while the wider Gaussian wings replace darker surrounding pixels.
        peak = float(np.max(color_peak))
        if peak > 0:
            yy, xx = np.ogrid[y0:y1, x0:x1]
            distance2 = (xx - cx) ** 2 + (yy - cy) ** 2
            gaussian = np.exp(-0.5 * distance2 / (sigma * sigma)).astype(np.float32)
            halo_scale = float(np.clip(0.55 + 0.35 * brightness, 0.0, 0.9)) * strength
            # Ensure the Gaussian layer cannot lift the centroid pixel itself.
            halo_peak = np.minimum(color_peak, center_rgb / max(halo_scale, 1e-8)) * halo_scale
            halo = gaussian[..., None] * halo_peak.astype(np.float32)[None, None, :]
            patch = linear_rgb[y0:y1, x0:x1, :]
            np.maximum(patch, halo, out=patch)
            records.append(
                {
                    "x_px": round(float(cx), 2),
                    "y_px": round(float(cy), 2),
                    "aperture_flux": round(float(star.flux), 7),
                    "fwhm_px": round(float(star.fwhm * (scale_x + scale_y) * 0.5), 3),
                    "halo_radius_px": round(radius, 2),
                }
            )
        if progress and (index == len(order) - 1 or index % max(1, len(order) // 20) == 0):
            progress(65 + int(25 * (index + 1) / len(order)), f"正在生成逐星 Gaussian 光晕… {index + 1}/{len(order)}")

    # Preserve every measured centroid sample exactly, including where two
    # neighboring Gaussian layers overlap.
    for _star, cx, cy, _color_peak, center_rgb in prepared:
        center_x = int(np.clip(round(cx), 0, out_w - 1))
        center_y = int(np.clip(round(cy), 0, out_h - 1))
        linear_rgb[center_y, center_x, :] = center_rgb
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


def process_raw(
    input_path: str | Path,
    output_path: str | Path,
    *,
    sensitivity: float = 4.8,
    strength: float = 0.55,
    min_radius: float = 3.0,
    max_radius: float = 32.0,
    max_stars: int = 50,
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
        stars, candidate_count = detect_stars(detector, sensitivity, max_stars)
        report(35, f"SEP 找到 {candidate_count:,} 个点源候选，选取最亮的 {len(stars):,} 颗，正在解码 RAW…")
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
    star_measurements = _soften_stars(
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
        "software": "星点柔焦 1.1",
        "camera": info.camera,
        "lens": info.lens,
        "focal_length_mm": info.focal_length,
        "aperture": info.aperture,
        "detected_stars": len(stars),
        "point_source_candidates": candidate_count,
        "star_detection": "SEP local background/RMS, PSF matched extraction, circular aperture flux",
        "soft_focus": "Gaussian PSF layer, Lighten blend, sigma=radius/3, original bright core retained",
        "star_photometry_and_halo_parameters": star_measurements,
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
            software="StarSoftFocus 1.1",
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
        width=int(rgb16.shape[1]),
        height=int(rgb16.shape[0]),
        camera=info.camera,
        lens=info.lens,
    )
