"""Offline command-line companion to tools/image_compare.html.

Requires NumPy and Pillow. No Blender or browser is required. Metric definitions,
JSON metric keys, ROI rules and difference gain match the standalone viewer.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import sys


MAX_PIXELS = 40_000_000
MARGIN_KEYS = ("top", "right", "bottom", "left")


def compare_arrays(reference, candidate, *, margins=None, tolerance=0, gain=8,
                   make_difference=False):
    """Compare equal-sized uint8 RGBA arrays using the viewer's scoring rules.

    Return (metrics, optional full-size opaque RGBA difference array). Process
    eight rows at a time to avoid full-size float64 intermediate images.
    """
    import numpy as np

    a, b = np.asarray(reference), np.asarray(candidate)
    if (a.shape != b.shape or a.ndim != 3 or a.shape[2] != 4
            or a.dtype != np.uint8 or b.dtype != np.uint8
            or min(a.shape[:2]) < 1):
        raise ValueError("Comparison requires equally sized uint8 RGBA images; no resizing is applied.")
    h, w = a.shape[:2]
    if w * h > MAX_PIXELS:
        raise ValueError("Each image is limited to 40 million pixels.")
    m = dict.fromkeys(MARGIN_KEYS, 0) if margins is None else dict(margins)
    if (set(m) != set(MARGIN_KEYS)
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in m.values())
            or m["left"] + m["right"] >= w or m["top"] + m["bottom"] >= h):
        raise ValueError("Margins must be non-negative whole pixels and leave at least one pixel to compare.")
    if (isinstance(tolerance, bool) or not isinstance(tolerance, int)
            or not 0 <= tolerance <= 255):
        raise ValueError("RGB tolerance must be a whole number from 0 to 255.")
    if (isinstance(gain, bool) or not isinstance(gain, (int, float))
            or not math.isfinite(gain) or not 1 <= gain <= 32):
        raise ValueError("Difference gain must be from 1 to 32.")

    x0, x1 = m["left"], w - m["right"]
    y0, y1 = m["top"], h - m["bottom"]
    pixel_count = (x1 - x0) * (y1 - y0)
    difference = None
    if make_difference:
        difference = np.zeros((h, w, 4), dtype=np.uint8)
        difference[:, :, 3] = 255
    abs_sum = square_sum = alpha_sum = changed = alpha_changed = max_error = 0
    ssim_sum = 0.0
    starts = np.arange(0, x1 - x0, 8)
    widths = np.minimum(8, x1 - x0 - starts)
    weights = np.array([.2126, .7152, .0722], dtype=np.float64)
    c1, c2 = 2.55 ** 2, 7.65 ** 2

    def over_black(rgba):
        # Equal to JS Math.round(channel * alpha / 255) on integer byte input.
        rgb = rgba[:, :, :3].astype(np.uint16)
        alpha = rgba[:, :, 3:4].astype(np.uint16)
        return ((rgb * alpha + 127) // 255).astype(np.int16)

    def block_sums(values):
        return np.add.reduceat(values.sum(axis=0, dtype=np.float64), starts)

    for y in range(y0, y1, 8):
        end = min(y + 8, y1)
        ar, br = a[y:end, x0:x1], b[y:end, x0:x1]
        rgb_a, rgb_b = over_black(ar), over_black(br)
        error = np.abs(rgb_a - rgb_b)
        abs_sum += int(error.sum(dtype=np.int64))
        square_sum += int(np.square(error.astype(np.int32)).sum(dtype=np.int64))
        per_pixel = error.max(axis=2)
        changed += int(np.count_nonzero(per_pixel > tolerance))
        max_error = max(max_error, int(per_pixel.max()))
        alpha_error = np.abs(ar[:, :, 3].astype(np.int16) - br[:, :, 3].astype(np.int16))
        alpha_sum += int(alpha_error.sum(dtype=np.int64))
        alpha_changed += int(np.count_nonzero(alpha_error))
        if difference is not None:
            # NumPy rint uses ties-to-even, as Uint8ClampedArray does after gain.
            difference[y:end, x0:x1, :3] = np.rint(np.minimum(255, error * float(gain))).astype(np.uint8)

        u = (rgb_a * weights).sum(axis=2)
        v = (rgb_b * weights).sum(axis=2)
        counts = widths * (end - y)
        mx, my = block_sums(u) / counts, block_sums(v) / counts
        vx = np.maximum(0, block_sums(u * u) / counts - mx * mx)
        vy = np.maximum(0, block_sums(v * v) / counts - my * my)
        cov = block_sums(u * v) / counts - mx * my
        scores = ((2 * mx * my + c1) * (2 * cov + c2)
                  / ((mx * mx + my * my + c1) * (vx + vy + c2)))
        ssim_sum += float((np.clip(scores, -1, 1) * counts).sum())

    mse = square_sum / (pixel_count * 3)
    metrics = {
        "pixel_count": pixel_count,
        "rgb_mae_255": abs_sum / (pixel_count * 3),
        "rgb_rmse_255": math.sqrt(mse),
        "rgb_psnr_db": None if mse == 0 else 10 * math.log10(255 ** 2 / mse),
        "rgb_identical": mse == 0,
        "ssim_luma_8x8": ssim_sum / pixel_count,
        "rgb_max_channel_error_255": max_error,
        "rgb_changed_pixels": changed,
        "rgb_changed_percent": 100 * changed / pixel_count,
        "alpha_mae_255": alpha_sum / pixel_count,
        "alpha_changed_pixels": alpha_changed,
    }
    return metrics, difference


def read_image(path):
    """Read static display-image input, orient it, and honor embedded ICC profiles.

    Pillow and browsers can differ in JPEG decoding or color conversion. The
    report records the decoder; matching formulas do not erase decoder variation.
    """
    import numpy as np
    from PIL import Image, ImageCms, ImageOps

    path = Path(path)
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise ValueError(f"{path}: choose PNG, JPEG or WebP. Convert EXR/HDR to matched display PNGs first.")
    with Image.open(path) as source:
        if source.format not in {"PNG", "JPEG", "WEBP"}:
            raise ValueError(f"{path}: unsupported actual image format {source.format}.")
        if getattr(source, "is_animated", False):
            raise ValueError(f"{path}: animated images are outside this tool's scope.")
        if source.width * source.height > MAX_PIXELS:
            raise ValueError(f"{path}: image exceeds the 40-million-pixel limit.")
        if source.mode in {"I", "F"} or source.mode.startswith("I;16"):
            raise ValueError(f"{path}: export high-depth data to an 8-bit sRGB display PNG first.")
        profile = source.info.get("icc_profile")
        has_gamma = "gamma" in source.info or "chromaticity" in source.info
        image = ImageOps.exif_transpose(source)
        if profile:
            if image.mode not in {"RGB", "RGBA", "CMYK", "L", "LAB"}:
                image = image.convert("RGBA")
            try:
                image = ImageCms.profileToProfile(
                    image, ImageCms.ImageCmsProfile(BytesIO(profile)),
                    ImageCms.createProfile("sRGB"), outputMode="RGBA")
            except ImageCms.PyCMSError as exc:
                raise ValueError(f"{path}: cannot convert the embedded ICC profile to sRGB: {exc}") from exc
        pixels = np.array(image.convert("RGBA"), dtype=np.uint8)
    info = {
        "name": path.name,
        "path": str(path.resolve()),
        "width": pixels.shape[1],
        "height": pixels.shape[0],
        "file_bytes": path.stat().st_size,
        "mime_type": {".png": "image/png", ".webp": "image/webp"}.get(path.suffix.lower(), "image/jpeg"),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "icc_to_srgb": bool(profile),
        "png_gamma_or_chromaticity_metadata": has_gamma,
    }
    return pixels, info


def make_report(a_info, b_info, metrics, margins, tolerance, gain):
    import PIL
    import numpy as np

    return {
        "tool": "Image Compare CLI",
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "reference": a_info,
        "candidate": b_info,
        "scored_region_margins_px": margins,
        "rgb_changed_tolerance_255": tolerance,
        "difference_gain": gain,
        "metrics": metrics,
        "method": {
            "rgb": "Decoded 8-bit sRGB composited over black; rounded to integer channels. Alpha scored separately.",
            "psnr": "10 log10(255^2 / RGB MSE); null with rgb_identical=true means positive infinity.",
            "ssim": "Rec.709 luma, non-overlapping 8x8 population-statistic blocks, C1=2.55^2 C2=7.65^2; partial edge blocks weighted by pixel count.",
            "resampling": "None",
            "difference_png": "Full source dimensions, absolute RGB error times gain; black outside scored region.",
            "interpretation": "No automatic verdict. Noise, color management and alignment must be controlled; scores do not identify a codec or establish renderer correctness.",
        },
        "decoder": {
            "name": "Pillow", "version": PIL.__version__, "numpy_version": np.__version__,
            "color": "Embedded ICC profiles converted to sRGB with LittleCMS. Other inputs assumed sRGB; bare PNG gAMA/cHRM metadata is not transformed. EXIF orientation applied.",
            "browser_parity": "Same scoring formulas as image_compare.html. Matching decoded pixels produce matching numbers (floating-point precision applies). Browser/Pillow JPEG, ICC, high-depth and alpha decoding can differ; use untagged opaque 8-bit sRGB PNGs for reproducible cross-tool comparisons.",
        },
    }


def print_metrics(report):
    m = report["metrics"]
    print(f"A: {report['reference']['path']}")
    print(f"B: {report['candidate']['path']}")
    print(f"Scored pixels: {m['pixel_count']:,} | margins: {report['scored_region_margins_px']}")
    print(f"RGB MAE /255:          {m['rgb_mae_255']:.3f}")
    print(f"RGB RMSE /255:         {m['rgb_rmse_255']:.3f}")
    print("RGB PSNR (dB):        " + (" inf (identical RGB)" if m["rgb_identical"] else f" {m['rgb_psnr_db']:.2f}"))
    print(f"Luma SSIM (8x8):       {m['ssim_luma_8x8']:.6f}")
    print(f"Changed RGB pixels:    {m['rgb_changed_percent']:.2f}% ({m['rgb_changed_pixels']:,}; tolerance {report['rgb_changed_tolerance_255']})")
    print(f"Max RGB channel error: {m['rgb_max_channel_error_255']}/255")
    print(f"Alpha MAE /255:        {m['alpha_mae_255']:.3f}")
    print(f"Changed alpha pixels:  {m['alpha_changed_pixels']:,}")
    print("Display RGB over black; no resampling; no automatic pass/fail.")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare two display images using the same metric definitions as Image Compare.",
        epilog="Matching opaque 8-bit sRGB PNGs give reproducible browser/CLI results. Decoder differences may affect other inputs. Exit 0 = comparison completed, 2 = error; image differences are not a failing exit code.")
    parser.add_argument("reference", type=Path, help="Image A: approved reference PNG/JPEG/WebP")
    parser.add_argument("candidate", type=Path, help="Image B: candidate PNG/JPEG/WebP")
    for name in MARGIN_KEYS:
        parser.add_argument(f"--{name}", type=int, default=0, help=f"Exclude {name} margin in source pixels (default: 0)")
    parser.add_argument("--tolerance", "--threshold", dest="tolerance", type=int, default=0,
                        help="Count RGB pixels with any channel error greater than this 0-255 value (default: 0)")
    parser.add_argument("--gain", type=int, choices=[1, 4, 8, 16, 32], default=8,
                        help="Difference PNG amplification; never changes metrics (default: 8)")
    parser.add_argument("--json", dest="json_path", help="Save metrics JSON, or use '-' for JSON-only stdout")
    parser.add_argument("--diff", type=Path, help="Save a full-size difference PNG")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files (source images are always protected)")
    args = parser.parse_args(argv)
    margins = {k: getattr(args, k) for k in MARGIN_KEYS}
    try:
        outputs = ([Path(args.json_path)] if args.json_path and args.json_path != "-" else [])
        if args.diff:
            if args.diff.suffix.lower() != ".png":
                raise ValueError("The --diff output must have a .png extension.")
            outputs.append(args.diff)
        resolved = [p.resolve() for p in outputs]
        if len(set(resolved)) != len(resolved):
            raise ValueError("JSON and difference output paths must be different.")
        if set(resolved) & {args.reference.resolve(), args.candidate.resolve()}:
            raise ValueError("Output paths must not overwrite either source image.")
        for output in outputs:
            if output.exists() and not args.force:
                raise ValueError(f"Output already exists: {output}. Use --force to overwrite it.")
        a, a_info = read_image(args.reference)
        b, b_info = read_image(args.candidate)
        metrics, diff = compare_arrays(a, b, margins=margins, tolerance=args.tolerance,
                                      gain=args.gain, make_difference=bool(args.diff))
        report = make_report(a_info, b_info, metrics, margins, args.tolerance, args.gain)
        text = json.dumps(report, indent=2, allow_nan=False) + "\n"
        for output in outputs:
            output.parent.mkdir(parents=True, exist_ok=True)
        if args.diff:
            from PIL import Image
            Image.fromarray(diff).save(args.diff, format="PNG")
        if args.json_path and args.json_path != "-":
            Path(args.json_path).write_text(text, encoding="utf-8")
        if args.json_path == "-":
            print(text, end="")
        else:
            print_metrics(report)
            if args.json_path:
                print(f"Metrics JSON: {Path(args.json_path).resolve()}")
            if args.diff:
                print(f"Difference PNG: {args.diff.resolve()} ({args.gain}x gain)")
        return 0
    except ImportError as exc:
        print(f"error: {exc}. Install dependencies: python -m pip install numpy Pillow", file=sys.stderr)
    except (OSError, ValueError, RuntimeError, MemoryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
