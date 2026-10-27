"""
validate_ui_renders.py
======================
Image validation for ui_render_test.py / run_ui_render_suite.py output. Reads the
rows of one session from results/ui_render*.csv and checks:

  sanity      every OK render exists, has the CSV resolution, is not blank / flat,
              and has no large magenta area (Cycles' missing-texture colour)
  display     the image shown in the UI (cropped from the window screenshot at the
              recorded image_rect) matches the saved render -> the progressive /
              interop display path showed the final result, not black / stale / noisy
  device      Cycles on CUDA and OptiX vs Cycles on CPU, same denoiser, same build
  denoiser    OIDN on GPU vs OIDN on CPU, same compute device, same build
  reference   vs a golden image saved earlier with --save-reference (same model,
              mode, build, perm) -> regression check across runs / driver updates
  timing      render time vs the golden time saved with that image (times.json);
              WARN beyond +-25 %. References saved under one --label are never
              replaced by a run with another label unless --force-reference
  version     each perm vs the same perm on the newest build in the session --
              INFO only (3.x renders with Filmic, 4.x+ with AgX; engines change)

Metrics: PSNR (dB, RGB), SSIM (luma, 7x7 box window), mean abs error and % of
pixels off by more than 16/255. Failing comparisons write an amplified diff image
to results/diffs/. Verdicts go to results/ui_validation_<session>.csv; exit code 1
if anything FAILs.

Usage (system Python with numpy + Pillow):
  python validate_ui_renders.py                       # latest session
  python validate_ui_renders.py --session 20261014_201500
  python validate_ui_renders.py --save-reference      # bless passing renders as golden
"""

import argparse
import csv
import glob
import json
import math
import os
import shutil
import sys

import numpy as np
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
REF_DIR = os.path.join(RESULTS_DIR, "references")
DIFF_DIR = os.path.join(RESULTS_DIR, "diffs")

# Pass thresholds: (min PSNR dB, min SSIM), calibrated on 4.5.9 / Classroom / test mode
# (measured: display 30-35 dB SSIM >= 0.985 -- the UI's GPU color management differs
# slightly from save_render's; CPU vs GPU 37-50 dB; OIDN GPU vs CPU ~61 dB). A black,
# stale or still-noisy display, or a wrong device result, lands far below these.
THRESHOLDS = {
    "display": (26.0, 0.97),
    "device": (33.0, 0.97),
    "device-noisy": (32.0, 0.97),     # no denoiser: CPU and GPU draw the same samples
    "denoiser": (45.0, 0.995),
    "reference": (38.0, 0.98),
}
TIMING_WARN_PCT = 25.0    # render time this far from the golden time (either way) = WARN
BLANK_STD = 2.0           # luma std-dev below this = flat / blank image
MAGENTA_FRAC = 0.005      # > 0.5 % pure magenta pixels = missing textures
VERSION_WARN_PSNR = 18.0  # cross-version below this is worth a look


# ---------------------------------------------------------------------------
# Image metrics
# ---------------------------------------------------------------------------
def load_rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32)


def _box(a, k=7):
    c = np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def _luma(img):
    return img @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def ssim(a, b):
    # float64 + centring: box sums of x*x over a 1080p frame reach ~1e11, where float32
    # cancellation in E[x^2] - E[x]^2 turns SSIM into noise (0.97 for identical-looking frames).
    x, y = _luma(a).astype(np.float64), _luma(b).astype(np.float64)
    offset = (x.mean() + y.mean()) / 2
    x, y = x - offset, y - offset
    if min(x.shape) < 8:
        return 1.0 if np.array_equal(x, y) else 0.0
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mx, my = _box(x), _box(y)
    sxx = _box(x * x) - mx * mx
    syy = _box(y * y) - my * my
    sxy = _box(x * y) - mx * my
    m = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return float(m.mean())


def compare(a, b):
    d = np.abs(a - b)
    mse = float((d * d).mean())
    return {
        "psnr": 99.0 if mse == 0 else 10 * math.log10(255.0 ** 2 / mse),
        "ssim": ssim(a, b),
        "mae": float(d.mean()),
        "bad_px_pct": float((d.max(axis=2) > 16).mean() * 100),
    }


ALIGN_PX = 3   # view2d_to_region rounding puts the recorded rect up to ~1 px off on some builds


def _best_offset(window, ref, x0, y0):
    """(dx, dy) within +-ALIGN_PX that best aligns ref with the window screenshot,
    searched on 4x-subsampled luma (cheap; a 1 px shift alone costs ~10 dB PSNR)."""
    h, w = ref.shape[:2]
    r = _luma(ref)[::4, ::4]
    wl = _luma(window)
    best = (float("inf"), 0, 0)
    for dy in range(-ALIGN_PX, ALIGN_PX + 1):
        for dx in range(-ALIGN_PX, ALIGN_PX + 1):
            ys, xs = y0 + dy, x0 + dx
            if ys < 0 or xs < 0 or ys + h > wl.shape[0] or xs + w > wl.shape[1]:
                continue
            e = float(((wl[ys:ys + h:4, xs:xs + w:4] - r) ** 2).mean())
            if e < best[0]:
                best = (e, dx, dy)
    return best[1], best[2]


def _load_times(build_dir):
    try:
        with open(os.path.join(build_dir, "times.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _line_fallback(build_dir):
    """Newest sibling reference build of the same x.y line (the golden default), or None."""
    parent, build = os.path.split(build_dir)
    line = ".".join(build.split(".")[:2]) + "."
    try:
        cands = [d for d in os.listdir(parent)
                 if d.startswith(line) and d != build and os.path.isdir(os.path.join(parent, d))]
    except OSError:
        return None
    return os.path.join(parent, max(cands, key=_build_key)) if cands else None


def save_diff(a, b, name):
    os.makedirs(DIFF_DIR, exist_ok=True)
    d = np.clip(np.abs(a - b).max(axis=2) * 8, 0, 255).astype(np.uint8)
    path = os.path.join(DIFF_DIR, name + ".png")
    Image.fromarray(d, "L").save(path)
    return os.path.relpath(path, SCRIPT_DIR)


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------
def load_rows():
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "ui_render*.csv"))):
        with open(path, newline="", encoding="utf-8") as f:
            rows += list(csv.DictReader(f))
    return rows


def _abs(rel):
    return os.path.join(SCRIPT_DIR, rel) if rel and not os.path.isabs(rel) else rel


def _build_key(v):
    try:
        return [int(x) for x in v.split()[0].split(".")]
    except ValueError:
        return [0]


def pick_renders(rows):
    """{(build, perm): row} -- the last OK run of each perm (warm if --runs > 1)."""
    out = {}
    for r in rows:
        if r.get("status") == "OK" and r.get("render_png"):
            out[(r["blender"], r["perm"])] = r
    return out


def _parse_perm(perm):
    if perm.startswith("cycles-"):
        _, compute, denoise = perm.split("-", 2)
        return compute, denoise
    return None, None


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
class Validator:
    def __init__(self, session):
        self.session = session
        self.results = []
        self.cache = {}
        self.refused = []

    def img(self, row):
        p = _abs(row["render_png"])
        if p not in self.cache:
            self.cache[p] = load_rgb(p)
        return self.cache[p]

    def add(self, check, build, perm, against, verdict, detail="", metrics=None, diff=""):
        m = metrics or {}
        self.results.append({
            "session": self.session, "check": check, "build": build, "perm": perm,
            "against": against, "verdict": verdict,
            "psnr": f"{m['psnr']:.2f}" if "psnr" in m else "",
            "ssim": f"{m['ssim']:.4f}" if "ssim" in m else "",
            "mae": f"{m['mae']:.2f}" if "mae" in m else "",
            "bad_px_pct": f"{m['bad_px_pct']:.2f}" if "bad_px_pct" in m else "",
            "delta_pct": f"{m['delta_pct']:.1f}" if "delta_pct" in m else "",
            "detail": detail, "diff": diff,
        })

    def judge(self, check, kind, build, perm, against, a, b, tag):
        if a.shape != b.shape:
            self.add(check, build, perm, against, "FAIL",
                     f"size {a.shape[1]}x{a.shape[0]} vs {b.shape[1]}x{b.shape[0]}")
            return
        m = compare(a, b)
        min_psnr, min_ssim = THRESHOLDS[kind]
        ok = m["psnr"] >= min_psnr and m["ssim"] >= min_ssim
        diff = "" if ok else save_diff(a, b, f"{self.session}_{tag}")
        self.add(check, build, perm, against, "PASS" if ok else "FAIL",
                 f"needs PSNR>={min_psnr:g} SSIM>={min_ssim:g}", m, diff)

    def sanity(self, row):
        build, perm = row["blender"], row["perm"]
        path = _abs(row["render_png"])
        if not os.path.isfile(path):
            self.add("sanity", build, perm, "", "FAIL", f"missing {row['render_png']}")
            return False
        a = self.img(row)
        problems = []
        want = row.get("resolution", "")
        if want and want != f"{a.shape[1]}x{a.shape[0]}":
            problems.append(f"size {a.shape[1]}x{a.shape[0]}, expected {want}")
        std = float(_luma(a).std())
        if std < BLANK_STD:
            problems.append(f"blank/flat (luma std {std:.2f})")
        magenta = float(((a[..., 0] > 240) & (a[..., 1] < 15) & (a[..., 2] > 240)).mean())
        if magenta > MAGENTA_FRAC:
            problems.append(f"{magenta * 100:.1f}% magenta (missing textures?)")
        self.add("sanity", build, perm, "", "FAIL" if problems else "PASS",
                 "; ".join(problems) or f"luma std {std:.1f}")
        return not problems

    def display(self, row):
        build, perm = row["blender"], row["perm"]
        rect, shot = row.get("image_rect", ""), _abs(row.get("screenshot", ""))
        if not rect or not shot or not os.path.isfile(shot):
            self.add("display", build, perm, "screenshot", "N/A", "no screenshot / image_rect")
            return
        x0, y0, x1, y1 = (int(round(float(v))) for v in rect.split(","))
        with Image.open(shot) as im:
            win = im.convert("RGB")
            wpx = row.get("window_px", "")
            if wpx and wpx != f"{win.width}x{win.height}":
                w, h = (int(v) for v in wpx.split("x"))
                sx, sy = win.width / w, win.height / h
                x0, x1, y0, y1 = int(x0 * sx), int(x1 * sx), int(y0 * sy), int(y1 * sy)
            if x0 < 0 or y0 < 0 or x1 > win.width or y1 > win.height or x1 <= x0 or y1 <= y0:
                self.add("display", build, perm, "screenshot", "FAIL",
                         f"image rect {rect} outside the {win.width}x{win.height} screenshot")
                return
            ref = self.img(row)
            if (x1 - x0, y1 - y0) == (ref.shape[1], ref.shape[0]):
                dx, dy = _best_offset(np.asarray(win, dtype=np.float32), ref, x0, y0)
                x0, x1, y0, y1 = x0 + dx, x1 + dx, y0 + dy, y1 + dy
            crop = win.crop((x0, y0, x1, y1))
            if crop.size != (ref.shape[1], ref.shape[0]):
                crop = crop.resize((ref.shape[1], ref.shape[0]), Image.BILINEAR)
            shown = np.asarray(crop, dtype=np.float32)
        self.judge("display", "display", build, perm, "screenshot", shown, ref,
                   f"{build}_{perm}_display")

    def run(self, renders, save_reference, ref_dir, force=False):
        builds = sorted({b for b, _ in renders}, key=_build_key)
        good = {}
        for key, row in sorted(renders.items(), key=lambda kv: (_build_key(kv[0][0]), kv[0][1])):
            if self.sanity(row):
                good[key] = row
                self.display(row)

        for build in builds:
            perms = {p: r for (b, p), r in good.items() if b == build}
            for perm, row in perms.items():
                compute, denoise = _parse_perm(perm)
                if compute in ("cuda", "optix") and f"cycles-cpu-{denoise}" in perms:
                    ref = perms[f"cycles-cpu-{denoise}"]
                    kind = "device-noisy" if denoise == "none" else "device"
                    self.judge("device", kind, build, perm, ref["perm"], self.img(row),
                               self.img(ref), f"{build}_{perm}_vs_cpu")
                if denoise == "oidn-gpu" and f"cycles-{compute}-oidn-cpu" in perms:
                    ref = perms[f"cycles-{compute}-oidn-cpu"]
                    self.judge("denoiser", "denoiser", build, perm, ref["perm"],
                               self.img(row), self.img(ref), f"{build}_{perm}_vs_oidncpu")

        times_cache = {}
        for (build, perm), row in sorted(good.items(), key=lambda kv: (_build_key(kv[0][0]), kv[0][1])):
            mode = "test" if row.get("test_mode") == "True" else "full"
            build_dir = os.path.join(ref_dir, row.get("scene", "scene"), mode, build.split()[0])
            ref_path = os.path.join(build_dir, f"{perm}.png")
            if build_dir not in times_cache:
                times_cache[build_dir] = _load_times(build_dir)
            times = times_cache[build_dir]
            cmp_path, cmp_times = ref_path, times
            if not os.path.isfile(ref_path) and not save_reference:
                alt = _line_fallback(build_dir)
                if alt:
                    cmp_path = os.path.join(alt, f"{perm}.png")
                    cmp_times = times_cache.setdefault(alt, _load_times(alt))
            if os.path.isfile(cmp_path):
                self.judge("reference", "reference", build, perm,
                           os.path.relpath(cmp_path, SCRIPT_DIR), self.img(row),
                           load_rgb(cmp_path), f"{build}_{perm}_vs_ref")
            ref_t = cmp_times.get(perm)
            if ref_t and cmp_path != ref_path:
                ref_t = dict(ref_t, label=f"{ref_t.get('label') or ref_t['session']} "
                                          f"@{os.path.basename(os.path.dirname(cmp_path))}")
            if ref_t and row.get("render_s"):
                cur, base = float(row["render_s"]), float(ref_t["render_s"])
                delta = (cur / base - 1) * 100 if base > 0 else 0.0
                verdict = "WARN" if abs(delta) > TIMING_WARN_PCT else "INFO"
                self.add("timing", build, perm, f"{ref_t['label'] or ref_t['session']}", verdict,
                         f"{cur:.1f}s vs {base:.1f}s ({delta:+.1f}%)",
                         {"delta_pct": delta, "cur_s": cur, "ref_s": base})
            if save_reference:
                owner = next((t.get("label") for t in times.values()), None)
                if owner is not None and owner != row.get("label") and not force:
                    if build_dir not in self.refused:
                        self.refused.append(build_dir)
                    continue
                os.makedirs(build_dir, exist_ok=True)
                shutil.copyfile(_abs(row["render_png"]), ref_path)
                times[perm] = {"render_s": row["render_s"], "session": row["session"],
                               "label": row.get("label", ""), "completed_at": row["completed_at"]}
                with open(os.path.join(build_dir, "times.json"), "w", encoding="utf-8") as f:
                    json.dump(times, f, indent=2)

        if len(builds) > 1:
            newest = builds[-1]
            for (build, perm), row in good.items():
                if build == newest or (newest, perm) not in good:
                    continue
                a, b = self.img(row), self.img(good[(newest, perm)])
                if a.shape != b.shape:
                    continue
                m = compare(a, b)
                _, denoise = _parse_perm(perm)
                comparable = denoise not in (None, "none")   # noise / EEVEE generations differ anyway
                verdict = "WARN" if comparable and m["psnr"] < VERSION_WARN_PSNR else "INFO"
                self.add("version", build, perm, newest, verdict,
                         "informational: color pipeline / engine differ between versions", m)


def write_report(results, session):
    path = os.path.join(RESULTS_DIR, f"ui_validation_{session}.csv")
    cols = ["session", "check", "build", "perm", "against", "verdict", "psnr", "ssim",
            "mae", "bad_px_pct", "delta_pct", "detail", "diff"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(results)
    return path


def print_report(results, path):
    counts = {}
    for r in results:
        counts.setdefault(r["check"], {}).setdefault(r["verdict"], 0)
        counts[r["check"]][r["verdict"]] += 1
    print("\n=== Image validation ===")
    for check in ("sanity", "display", "device", "denoiser", "reference", "timing", "version"):
        if check in counts:
            print(f"  {check:<10} " + "  ".join(f"{v} {n}" for v, n in sorted(counts[check].items())))
    deltas = [float(r["delta_pct"]) for r in results if r["check"] == "timing"]
    if deltas:
        deltas.sort()
        print(f"  timing vs golden: median {deltas[len(deltas) // 2]:+.1f}%  "
              f"range {deltas[0]:+.1f}% .. {deltas[-1]:+.1f}%  (+ = slower)")
    flagged = [r for r in results if r["verdict"] in ("FAIL", "WARN")]
    for r in flagged:
        vs = f" vs {r['against']}" if r["against"] else ""
        metrics = f" PSNR {r['psnr']} SSIM {r['ssim']}" if r["psnr"] else ""
        if r["check"] == "timing":
            vs = f" vs golden '{r['against']}'"
        diff = f" diff {r['diff']}" if r["diff"] else ""
        print(f"  {r['verdict']:<4} {r['check']:<9} {r['build']:<10} {r['perm']:<22}{vs}"
              f"{metrics} -- {r['detail']}{diff}")
    print(f"  report: {os.path.relpath(path, SCRIPT_DIR)}")


def validate(session=None, save_reference=False, ref_dir=REF_DIR, force=False):
    """Validate one session; returns the number of FAILs (None if nothing to check)."""
    rows = load_rows()
    sessions = sorted({r.get("session", "") for r in rows if r.get("session")})
    session = session or (sessions[-1] if sessions else None)
    rows = [r for r in rows if r.get("session") == session]
    renders = pick_renders(rows)
    if not renders:
        print(f"[validate] no OK renders for session {session}")
        return None
    v = Validator(session)
    v.run(renders, save_reference, ref_dir, force)
    path = write_report(v.results, session)
    print_report(v.results, path)
    if save_reference:
        print(f"  references saved under {os.path.relpath(ref_dir, SCRIPT_DIR)}")
        for d in v.refused:
            print(f"  [kept] {os.path.relpath(d, SCRIPT_DIR)}: references belong to another label "
                  f"-- not overwritten (--force-reference to replace)")
    return sum(r["verdict"] == "FAIL" for r in v.results)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", default=None, help="session id (default: latest)")
    p.add_argument("--save-reference", action="store_true",
                   help="store this session's sane renders as the golden references")
    p.add_argument("--ref-dir", default=REF_DIR)
    p.add_argument("--force-reference", action="store_true",
                   help="let --save-reference replace references saved under another label")
    a = p.parse_args()
    fails = validate(a.session, a.save_reference, a.ref_dir, a.force_reference)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
