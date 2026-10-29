# Standalone Image Compare

Double-click **image_compare.html** to open it in Chrome or Edge. You can copy that
one file to another machine. No installation, Blender, Python, server or internet
connection is required. Images stay on the local machine.

1. Choose an approved image as **A · Reference** and a test image as **B · Candidate**.
   You can also drop files on the two cards, or drop two files on the viewer.
2. Choose **Wipe slider**, **Side by side**, **Difference**, **Blend** or **Blink**.
3. Use **100%** for source-pixel inspection. Scroll to zoom and drag to pan;
   side-by-side views share the same transform. **Fit** restores the whole view.
4. Set top/right/bottom/left margins in source pixels to exclude overlays or borders.
5. Set **RGB tolerance** to count pixels whose maximum channel error exceeds that
   value. The tolerance does not modify the other metrics. In Difference view,
   **Gain** amplifies visible errors without changing the scores.
6. Save a full-size difference PNG and a JSON record with settings and metrics.
   The difference is black outside the scored region. The JSON includes source
   names/dimensions, not the source images; retain the originals with the record.

## Scope and metric definitions

- PNG, JPEG and WebP; at most 40 million pixels per image. Animated images are
  outside scope. Images with different dimensions can be inspected, but are never
  resized or scored against each other.
- Metrics compare browser-decoded **8-bit sRGB composited over black**. Alpha MAE
  and changed-alpha pixel counts are reported separately. Transparent pixels can
  lose hidden RGB information during canvas decoding. This is not a raw-byte,
  linear-light or HDR comparison. Export EXR/HDR/high-depth renders to matched
  display PNGs for this tool; use a high-depth workflow to validate raw values.
- RGB MAE, RMSE and maximum channel error use 0–255 units. RGB PSNR uses a peak of
  255 and all three channels. Identical RGB yields infinity in the UI and `null`
  in JSON, accompanied by `rgb_identical: true`.
- SSIM uses Rec.709 luminance weights, non-overlapping 8×8 blocks, population
  variance/covariance, C1 = 2.55² and C2 = 7.65². Partial edge blocks are included
  and weighted by their pixel count. It is not a sliding Gaussian-window SSIM.
  Unlike `viewport_quality.py`, this tool includes incomplete edge blocks.
- Difference PNG values are `min(255, abs(A−B) × gain)` for each composited RGB
  channel, with opaque alpha. Alpha differences are recorded in JSON, not encoded
  into the RGB difference image.

Use the same camera, frame, render settings, view transform and output encoding.
Noise seeds and denoisers can affect similarity. Metrics do not identify the
source of pixelation, prove correctness or set a universal pass/fail threshold.
Compare saved renders separately from viewport screenshots.

## Matching command-line tool

From the repository root, using Python with NumPy and Pillow installed:

```powershell
python -m pip install numpy Pillow
python compare_images.py "reference.png" "candidate.png"
python compare_images.py "reference.png" "candidate.png" --json "comparison.json" --diff "difference.png"
```

Use the same settings as the viewer to reproduce a scored region:

```powershell
python compare_images.py "reference.png" "candidate.png" --top 64 --right 8 --bottom 8 --left 8 --tolerance 2 --gain 8 --json "comparison.json" --diff "difference.png"
```

- All margins default to zero, RGB tolerance to zero, and difference gain to 8×,
  matching the viewer. `--threshold` is an alias for `--tolerance`.
- Use `--json -` for JSON-only stdout. Full-precision metrics use the same names
  as the viewer's JSON export. Human-readable values use the viewer's precision.
- Existing outputs require `--force`. The tool never overwrites a source image.
  Exit code 0 means the comparison completed; 2 means invalid input or an error.
  Image differences alone do not cause a failing exit code.
- The CLI is a self-contained Python file and does not require Blender or Node.
  In this repository it lives at `compare_images.py`; the viewer is `tools/image_compare.html`.

The **scoring formulas match**. Given the same decoded pixels, the two tools
agree within floating-point precision, including partial SSIM edge blocks,
transparency, tolerance and difference pixels. The CLI uses Pillow instead of the
browser's image decoder, applies EXIF orientation and converts embedded ICC
profiles to sRGB. Untagged images are assumed sRGB; bare PNG gAMA/cHRM metadata is
not transformed. JPEG decoding, ICC conversion, high-depth conversion and alpha
rounding can differ between decoders. For reproducible cross-tool results, use
untagged, opaque **8-bit sRGB PNGs**, as used in the saved render checks. Do not
expect raw HDR/EXR or arbitrary color-profile conversions to be bit-identical.

## Developer validation

```powershell
python -m unittest discover -s tests -p test_image_compare_cli.py -v
node tests/test_image_compare.cjs
node tests/test_image_compare_cli_parity.cjs python
```

The parity test executes the actual HTML worker against the Python implementation
with deterministic RGBA images and non-aligned regions. It checks every metric to
1e-11 and every difference byte exactly. Browser QA should also cover file
selection/drop, dimension mismatch, region validation, all view modes, zoom/pan,
and both downloads.
