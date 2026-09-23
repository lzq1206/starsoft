"""Star detection and brightness-weighted soft-focus processing."""

from __future__ import annotations

import math
import os
import threading
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import rawpy
import sep
import tifffile
from PIL import ExifTags, Image, ImageCms, ImageOps

from version import APP_VERSION


Progress = Callable[[int, str], None]
REFERENCE_SKY_LEVEL = 0.05134


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


@dataclass(frozen=True)
class ProcessResult:
    output_path: str
    star_count: int
    candidate_count: int
    selected_count: int
    star_count_limit: int
    width: int
    height: int
    camera: str
    lens: str
    input_kind: str
    color_profile: str
    sky_background_level: float
    sky_adaptation_gain: float


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
        preview = _extract_raw_preview(raw)
        return _raw_info(raw, preview)


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


def _preview_linear_luminance_median(preview: Image.Image) -> float:
    """Measure the embedded preview's median luminance in linear sRGB."""
    profile = preview.info.get("icc_profile")
    if profile:
        try:
            preview = ImageCms.profileToProfile(
                preview,
                ImageCms.ImageCmsProfile(BytesIO(profile)),
                ImageCms.createProfile("sRGB"),
                outputMode="RGB",
            )
        except Exception:
            preview = preview.convert("RGB")
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


def _preview_exposure_shift(source_median: float, preview_median: float) -> float:
    """Return LibRaw's supported exposure multiplier that matches preview midtones."""
    if not math.isfinite(source_median) or not math.isfinite(preview_median):
        return 1.0
    if source_median <= 1e-8 or preview_median <= 1e-8:
        return 1.0
    # LibRaw's exp_shift range is 0.25 (−2 EV) through 8 (＋3 EV).
    return float(np.clip(preview_median / source_median, 0.25, 8.0))


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


def _read_raster(path: Path) -> RasterInput:
    suffix = path.suffix.lower()
    camera = ""
    lens = ""
    focal = None
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
            make = _clean_text(make_tag.value if make_tag is not None else "")
            model = _clean_text(model_tag.value if model_tag is not None else "")
            camera = " ".join(part for part in (make, model) if part)
            lens = _clean_text(lens_tag.value if lens_tag is not None else "")
            focal = _as_float(focal_tag.value if focal_tag is not None else None)
            aperture = _as_float(aperture_tag.value if aperture_tag is not None else None)
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
    )
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
    star_count_limit: int = 200,
) -> tuple[list[Star], int, int, float]:
    """Return the requested number of brightest SEP point sources."""
    data = np.ascontiguousarray(detector, dtype=np.float32)
    background = sep.Background(data, bw=64, bh=64, fw=3, fh=3)
    global_sky = float(np.clip(background.globalback, 0.0, 1.0))
    signal = np.ascontiguousarray(data - background.back(), dtype=np.float32)
    noise = np.ascontiguousarray(background.rms(), dtype=np.float32)
    first_pass = _extract_sources(
        signal,
        float(sensitivity),
        err=noise,
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
        (first_pass["flag"] == 0)
        & (first_roundness >= 0.68)
        & (first_size >= 0.45)
        & (first_size <= 4.5)
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
        np.median(first_noise[max(0, y - 4):min(data.shape[0], y + 5), max(0, x - 4):min(data.shape[1], x + 5)])
        for x, y in zip(object_x, object_y)
    ], dtype=np.float32)
    point_source = (
        np.isfinite(objects["x"])
        & np.isfinite(objects["y"])
        & np.isfinite(objects["flux"])
        & np.isfinite(moment_size)
        & (objects["flux"] > 0)
        & (minor >= 0.45)
        & (major <= min(6.0, max(3.5, 2.25 * psf_sigma)))
        & (roundness >= 0.68)
        & (local_noise <= 4.0 * global_rms)
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
    signal_to_noise = flux / np.maximum(flux_error, 1e-20)
    order = np.argsort(flux)[::-1][: int(np.clip(star_count_limit, 0, 500))]
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
        )
        for i in order
    ]
    return stars, candidate_count, len(stars), global_sky


def _soften_stars(
    image_data: np.ndarray,
    stars: list[Star],
    detector_shape: tuple[int, int],
    min_radius: float,
    max_radius: float,
    strength: float,
    opacity: float,
    sky_background_level: float,
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
    # luminance. Scale the added wings in proportion to the measured sky level
    # so the halo keeps a similar relationship to the sky across exposures.
    sky_gain = float(np.clip(sky_background_level / REFERENCE_SKY_LEVEL, 0.70, 1.50))
    prepared: list[dict[str, object]] = []

    # Capture each source profile before adding any halos. SEP's second moments
    # define a Gaussian approximation to the measured point-spread function.
    for star in stars:
        cx = (star.x + 0.5) * scale_x - 0.5
        cy = (star.y + 0.5) * scale_y - 0.5
        center_x = int(np.clip(round(cx), 0, out_w - 1))
        center_y = int(np.clip(round(cy), 0, out_h - 1))
        a = max(float(star.a) * scale_x, 0.45)
        b = max(float(star.b) * scale_y, 0.45)
        # SEP shape moments size the local measurement patch; the generated
        # scattering wing itself is always a circular, separate Gaussian.
        source_sigma = math.sqrt((a * a + b * b) * 0.5)
        widest_diffusion_sigma = (max_radius / 3.0) * math.sqrt(float(np.clip(strength, 0.0, 30.0)) / 40.0)
        probe_sigma = math.sqrt(source_sigma * source_sigma + widest_diffusion_sigma * widest_diffusion_sigma)
        extent = max(8, int(math.ceil(4.0 * probe_sigma)))
        x0, x1 = max(0, center_x - extent), min(out_w, center_x + extent + 1)
        y0, y1 = max(0, center_y - extent), min(out_h, center_y + extent + 1)
        source_patch = image_data[y0:y1, x0:x1, :]
        border = np.concatenate(
            (source_patch[0], source_patch[-1], source_patch[:, 0], source_patch[:, -1]),
            axis=0,
        )
        local_background = np.median(border, axis=0)
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
            "sigma": probe_sigma,
            "inner_mask_radius": 3.0 * probe_sigma,
            "outer_mask_radius": 4.0 * probe_sigma,
            "background": local_background,
            "source_peak": source_peak,
            "color_fraction": color_fraction,
        })

    # Match the magnitude-based response used by the earlier 18ffa10 build:
    # delta magnitude is logarithmic in measured flux, then maps linearly to
    # radius. A square-root response made most stars cluster near max_radius.
    flux_floor = min(float(item["star"].flux) for item in prepared)
    brightest_flux = max(float(item["star"].flux) for item in prepared)
    max_log_flux = max(
        math.log(max(brightest_flux, 1e-20) / max(flux_floor, 1e-20)),
        0.0,
    )
    reference_peak = max(float(item["source_peak"]) for item in prepared)
    for item in prepared:
        star = item["star"]
        log_flux = max(math.log(max(float(star.flux), 1e-20) / max(flux_floor, 1e-20)), 0.0)
        response = float(np.clip(log_flux / max_log_flux, 0.0, 1.0)) if max_log_flux > 1e-12 else 1.0
        radius = min_radius + (max_radius - min_radius) * response
        diffusion_sigma = (radius / 3.0) * math.sqrt(float(np.clip(strength, 0.0, 30.0)) / 40.0)
        # A star's halo peak follows its measured SEP aperture flux. Using the
        # brightest selected core as the radiometric reference keeps the halo
        # in the developed image's units while reducing faint-star intensity.
        flux_scale = float(np.clip(star.relative_flux_ratio, 0.0, 1.0))
        color_fraction = np.asarray(item["color_fraction"])
        color_max = max(float(np.max(color_fraction)), 1e-12)
        item["amplitude"] = (
            reference_peak * flux_scale * (color_fraction / color_max) * sky_gain
        )
        # The added scattering wing has its own circular Gaussian width. The
        # original, possibly imperfect stellar core remains unmodified below.
        output_sigma = diffusion_sigma
        item["radius"] = radius
        item["diffusion_sigma"] = diffusion_sigma
        item["sigma"] = output_sigma
        item["inner_mask_radius"] = max(1.0, 3.0 * output_sigma)
        item["outer_mask_radius"] = max(float(item["inner_mask_radius"]) + 1.0, 4.0 * output_sigma)
        item["radius_response"] = response
        item["halo_flux_scale"] = flux_scale
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
        sigma = float(item["sigma"])
        gaussian = np.exp(-0.5 * radial_squared / max(sigma * sigma, 1e-12)).astype(np.float32)
        inner_radius = float(item["inner_mask_radius"])
        outer_radius = float(item["outer_mask_radius"])
        feather = max(outer_radius - inner_radius, 1e-6)
        mask_t = np.clip((outer_radius - np.sqrt(radial_squared)) / feather, 0.0, 1.0)
        circular_mask = mask_t * mask_t * (3.0 - 2.0 * mask_t)
        halo_peak = np.asarray(item["amplitude"])
        halo = np.asarray(item["background"])[None, None, :] + (
            gaussian * circular_mask
        )[..., None] * halo_peak[None, None, :]
        patch = image_data[y0:y1, x0:x1, :]
        # Blend only the added wing over the original image; the photographed
        # star core remains untouched and opacity is independent of halo width.
        np.maximum(patch, halo, out=halo)
        np.subtract(halo, patch, out=halo)
        patch += halo * opacity_alpha
        records.append({
            "x_px": round(cx, 2),
            "y_px": round(cy, 2),
            "aperture_flux": round(float(star.flux), 7),
            "relative_flux_ratio": round(float(star.relative_flux_ratio), 6),
            "radius_curve": "linear response to relative stellar magnitude; log(aperture_flux/faintest_selected_flux)",
            "source_signal_to_noise": round(float(star.signal_to_noise), 3),
            "radius_response": round(float(item["radius_response"]), 5),
            "halo_flux_scale": round(float(item["halo_flux_scale"]), 6),
            "fwhm_px": round(float(star.fwhm * (scale_x + scale_y) * 0.5), 3),
            "halo_radius_3sigma_px": round(3.0 * float(item["sigma"]), 2),
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


def process_raw(
    input_path: str | Path,
    output_path: str | Path,
    *,
    sensitivity: float = 4.8,
    strength: float = 10.0,
    opacity: float = 30.0,
    star_count_limit: int = 200,
    min_radius: float = 3.0,
    max_radius: float = 42.0,
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
    star_count_limit = int(np.clip(star_count_limit, 0, 500))
    strength = float(np.clip(strength, 0.0, 30.0))
    opacity = float(np.clip(opacity, 0.0, 100.0))

    def report(percent: int, message: str) -> None:
        if progress:
            progress(percent, message)

    report(2, "正在读取图像与镜头信息…")
    raw_preview_median: float | None = None
    raw_linear_median: float | None = None
    raw_exposure_shift = 1.0
    if input_path.suffix.lower() in {".tif", ".tiff", ".jpg", ".jpeg"}:
        raster = _read_raster(input_path)
        info = raster.info
        if metadata_callback:
            metadata_callback(info)
        detector, _factor = _raster_detector_image(raster.pixels, info.focal_length)
        report(12, "正在识别星点与测量亮度…")
        stars, candidate_count, selected_count, sky_background_level = detect_stars(detector, sensitivity, star_count_limit)
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
        report(36, f"SEP 找到 {candidate_count:,} 个点源，选取最亮的 {selected_count:,} 个，开始柔焦…")
    else:
        raw_extensions = {
            ".cr3", ".cr2", ".crw", ".nef", ".nrw", ".arw", ".sr2", ".srf", ".dng",
            ".orf", ".rw2", ".raf", ".pef", ".ptx", ".3fr", ".fff", ".iiq", ".kdc",
            ".dcr", ".mos", ".mrw", ".x3f",
        }
        if input_path.suffix.lower() not in raw_extensions:
            raise ValueError("请选择相机 RAW、TIFF 或 JPG 文件。")
        with rawpy.imread(str(input_path)) as raw:
            preview = _extract_raw_preview(raw)
            info = _raw_info(raw, preview)
            if metadata_callback:
                metadata_callback(info)
            detector, _factor = _detector_image(raw)
            report(12, "正在识别星点与测量亮度…")
            stars, candidate_count, selected_count, sky_background_level = detect_stars(detector, sensitivity, star_count_limit)
            if preview is not None:
                raw_preview_median = _preview_linear_luminance_median(preview)
                report(36, "正在按相机内嵌预览校准 RAW 曝光…")
                calibration_rgb = raw.postprocess(
                    gamma=(1, 1),
                    no_auto_bright=True,
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
                raw_exposure_shift = _preview_exposure_shift(
                    raw_linear_median, raw_preview_median
                )
            report(43, f"SEP 找到 {candidate_count:,} 个点源，选取最亮的 {selected_count:,} 个，正在解码 RAW…")
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
        encoding = "RAW developed to linear sRGB with embedded-preview median exposure matching and highlight preservation; sRGB ICC profile embedded"

    report(63, "图像解码完成，按星点亮度与 PSF 形状柔化…")
    star_measurements = _soften_stars(
        image_data,
        stars,
        detector.shape,
        min_radius,
        max_radius,
        strength,
        opacity,
        sky_background_level,
        report,
    )
    report(92, "正在写入 16 位 TIFF…")
    if input_kind == "RAW" or (input_path.suffix.lower() in {".tif", ".tiff", ".jpg", ".jpeg"} and raster.linear_srgb):
        _linear_to_srgb_inplace(image_data)
    if invert_gray:
        np.subtract(1.0, image_data, out=image_data)
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
        "aperture": info.aperture,
        "detected_stars": len(stars),
        "point_source_candidates": candidate_count,
        "requested_brightest_star_count": star_count_limit,
        "star_detection": "SEP local background/RMS, PSF matched extraction, circular aperture flux",
        "star_selection": "top N by SEP circular aperture flux; no catalog magnitude calibration",
        "soft_focus": "circular isotropic Gaussian wings with circular smoothstep feather mask; blend wing opacity over original pixels while retaining original star cores",
        "brightness_to_radius_curve": "linear response to relative stellar magnitude: log(aperture_flux/faintest_selected_flux) normalized to brightest selected star, matching 18ffa10 mapping",
        "brightness_to_halo_strength_curve": "halo peak scales linearly with SEP aperture-flux ratio to the brightest selected star; faint stars receive smaller halos as well as smaller radii",
        "soft_focus_strength": round(strength, 1),
        "soft_focus_strength_note": "diffusion variance ratio relative to strength 40; default strength 10, maximum 30; diffusion sigma scales as sqrt(strength/40)",
        "soft_focus_opacity_percent": round(opacity, 1),
        "soft_focus_color": "background-subtracted RGB aperture chromaticity blended with core chromaticity; per-channel halo and original ICC retained",
        "sky_background_level": round(sky_background_level, 7),
        "sky_reference_level": REFERENCE_SKY_LEVEL,
        "sky_adaptation_gain": round(float(np.clip(sky_background_level / REFERENCE_SKY_LEVEL, 0.70, 1.50)), 4),
        "sky_adaptation": "SEP global background; halo wing amplitude scaled in proportion to background luminance with a 0.70x–1.50x clamp (Weber contrast adaptation)",
        "source_rejection": "SEP PSF matched detection; reject sources with nonstellar roundness, excessive size relative to estimated PSF, or local RMS above 4x image global RMS",
        "star_photometry_and_halo_parameters": star_measurements,
        "encoding": encoding,
        "input_color_profile": profile_description,
        "raw_preview_available": input_kind == "RAW" and info.preview is not None,
        "raw_embedded_preview_median_luminance_linear": round(raw_preview_median, 7) if raw_preview_median is not None else None,
        "raw_unadjusted_median_luminance_linear": round(raw_linear_median, 7) if raw_linear_median is not None else None,
        "raw_exposure_shift_from_embedded_preview": round(raw_exposure_shift, 5) if input_kind == "RAW" else None,
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
        star_count_limit=star_count_limit,
        width=int(pixels16.shape[1]),
        height=int(pixels16.shape[0]),
        camera=info.camera,
        lens=info.lens,
        input_kind=input_kind,
        color_profile=profile_description,
        sky_background_level=sky_background_level,
        sky_adaptation_gain=float(np.clip(sky_background_level / REFERENCE_SKY_LEVEL, 0.70, 1.50)),
    )
