"""
interactive_benchmark.py
========================

SCOPE  (Tier 1 -- headless interactivity)
-----------------------------------------
Measures how expensive it is to *interact* with a scene, using only operations
that force a dependency-graph re-evaluation -- NO GL window required. These are
the numbers behind "the timeline scrubs smoothly" / "dragging an object lags":

  * FRAME SCRUB    : sequential scene.frame_set() across the timeline -> per-frame
                     re-eval cost (animation + physics/particle sims). Derived
                     playback FPS = 1 / median-frame. This is the eval-only proxy
                     for playback; it excludes viewport DRAW (that is Tier 3, needs
                     a display -- see the project README).
  * OBJECT XFORM   : nudge one object, tag it, depsgraph.update() -> the latency of
                     a single edit propagating through the graph.
  * SUBSURF SWEEP  : if the scene has a Subdivision-Surface modifier, raise its
                     viewport level and time the re-eval (cage rebuild cost).
                     Skipped cleanly when the scene has no subsurf.

WHY headless is valid here: frame_set() and depsgraph.update() evaluate the graph
(modifiers, constraints, drivers, sims) the same way whether or not a window is
open. What headless CANNOT measure is the viewport *draw* / CUDA-GL interop -- that
is deliberately out of scope for this Tier-1 harness.

RELATIONSHIP TO THE REST OF THE SUITE
-------------------------------------
Same shape as engine_benchmark.py: it IMPORTS the invariants from
qa_standardize_render_settings.py (single source of truth, via engine_benchmark)
so determinism/color settings can't drift, reports median + min-max spread over
WARM_RUNS, and skips-with-a-reason instead of aborting. It does NOT render and does
NOT touch the render engine -- depsgraph eval cost is engine-independent, so this
runs once per model (no CUDA/OptiX/EEVEE dimension).

HOW TO RUN
----------
  blender --enable-autoexec -b scene.blend -P interactive_benchmark.py -- [options]

  Options:
    --test          fast smoke check: cap the scrub to 24 frames, warm=1.
    --warm N        warm passes to take the median + spread over (default 3).
    --frames N      cap the frame-scrub window to N frames (default: whole range,
                    bounded by MAX_FRAMES so a 20k-frame file can't run forever).
    --xform-iters N object-transform iterations to median over (default 30).
    -h, --help      show this help and exit.

  Batch across models into a spreadsheet (plain python, NOT inside Blender):
    python run_benchmark_suite.py --suite interactive --help
"""

import bpy
import os
import sys
import time
import statistics

# ---------------------------------------------------------------------------
# Reuse engine_benchmark's provenance/reporting helpers so the header line, the
# invariants single-source story, and GPU/version detection stay identical
# across the two harnesses (import has no side effects beyond loading INVARIANTS).
# ---------------------------------------------------------------------------
def _script_dir():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()

if _script_dir() not in sys.path:
    sys.path.insert(0, _script_dir())

try:
    from engine_benchmark import (
        apply_invariants, _extract_version, _extract_gpu, _extract_model,
        INVARIANTS_IMPORTED,
    )
    _HELPERS_OK = True
except Exception as _e:                       # standalone / relocated -> degrade, don't crash
    print(f"[interactive] could not import engine_benchmark helpers ({_e}); "
          f"using local fallbacks")
    _HELPERS_OK = False
    INVARIANTS_IMPORTED = False

    def apply_invariants(scene):
        pass

    def _extract_version():
        try:
            return bpy.app.version_string.split()[0]
        except Exception:
            return "?"

    def _extract_gpu():
        try:
            import gpu
            return gpu.platform.renderer_get() or "unknown GPU"
        except Exception:
            return "unknown GPU"

    def _extract_model():
        fp = bpy.data.filepath
        return os.path.splitext(os.path.basename(fp))[0] if fp else bpy.context.scene.name


# ---------------------------------------------------------------------------
# Controls (CLI-overridable in main())
# ---------------------------------------------------------------------------
WARM_RUNS = 3            # warm passes to median + spread over
MAX_FRAMES = 120         # hard cap on the scrub window in FULL mode (bounds runtime)
XFORM_ITERS = 30         # object-transform iterations

TEST_MODE = False
TEST_MAX_FRAMES = 24     # --test scrub window
FRAMES_OVERRIDE = None   # --frames N


# ---------------------------------------------------------------------------
# Metric 1: FRAME SCRUB  (animation + sim re-eval; derived playback FPS)
# ---------------------------------------------------------------------------
def _scrub_pass(scene, frames):
    """One sequential pass frame_start..frame_start+frames-1. Returns per-frame
    seconds. Sequential (not random) so physics/particle sims step correctly, the
    way real playback advances them. frame_set() forces a synchronous depsgraph
    evaluation, so the time captured IS the re-eval cost."""
    fs = scene.frame_start
    per_frame = []
    for i in range(frames):
        t0 = time.perf_counter()
        scene.frame_set(fs + i)
        per_frame.append(time.perf_counter() - t0)
    return per_frame


def measure_frame_scrub(scene):
    span = scene.frame_end - scene.frame_start + 1
    if span <= 1:
        return {"status": "SKIPPED",
                "reason": f"single-frame scene (range {scene.frame_start}-{scene.frame_end})"}
    cap = TEST_MAX_FRAMES if TEST_MODE else (FRAMES_OVERRIDE or MAX_FRAMES)
    frames = min(span, cap)

    _scrub_pass(scene, frames)                 # COLD pass (lazy load / first eval) -- discarded
    warm = []
    for _ in range(WARM_RUNS):
        warm.extend(_scrub_pass(scene, frames))
    med = statistics.median(warm)
    # Derived playback fps = 1/median-frame. Below a 0.1ms/frame floor the timeline
    # carries no depsgraph work (nothing animated on these frames) -- report that
    # plainly instead of an absurd six-figure fps that reads like a bug.
    if med <= 0 or med < 1e-4:
        derived = "no eval cost"
    else:
        derived = f"{1.0 / med:.1f} fps"
    return {"status": "OK",
            "frames": frames, "span": span,
            "median_s": med, "min_s": min(warm), "max_s": max(warm),
            "derived": derived,
            "scene_fps": scene.render.fps}


# ---------------------------------------------------------------------------
# Metric 2: OBJECT TRANSFORM  (single-edit propagation latency)
# ---------------------------------------------------------------------------
def _pick_transform_target(scene):
    """A representative object to nudge: prefer a visible mesh (its transform
    forces the most re-eval), else the camera, else any object."""
    for ob in scene.objects:
        if ob.type == 'MESH' and ob.visible_get():
            return ob
    return scene.camera or (scene.objects[0] if scene.objects else None)


def measure_transform(scene):
    ob = _pick_transform_target(scene)
    if ob is None:
        return {"status": "SKIPPED", "reason": "scene has no objects"}
    deps = bpy.context.evaluated_depsgraph_get()
    orig = tuple(ob.location)
    times = []
    try:
        for i in range(XFORM_ITERS):
            d = 0.001 if (i % 2 == 0) else -0.001    # jitter +/- so it doesn't drift away
            t0 = time.perf_counter()
            ob.location.x += d
            ob.update_tag()
            deps.update()
            times.append(time.perf_counter() - t0)
    finally:
        ob.location = orig
        ob.update_tag()
        deps.update()
    med = statistics.median(times)
    return {"status": "OK", "target": ob.name,
            "median_s": med, "min_s": min(times), "max_s": max(times),
            "derived": "-"}


# ---------------------------------------------------------------------------
# Metric 3: SUBSURF SWEEP  (cage rebuild cost; skipped if no subsurf)
# ---------------------------------------------------------------------------
def measure_subsurf(scene):
    target = None
    for ob in scene.objects:
        for m in getattr(ob, "modifiers", []):
            if m.type == 'SUBSURF':
                target = (ob, m)
                break
        if target:
            break
    if target is None:
        return {"status": "SKIPPED", "reason": "no Subdivision-Surface modifier in scene"}
    ob, mod = target
    deps = bpy.context.evaluated_depsgraph_get()
    orig_level = mod.levels
    sweep_to = 1 if TEST_MODE else 3
    times = []
    try:
        for _ in range(WARM_RUNS):
            for lvl in range(0, sweep_to + 1):
                t0 = time.perf_counter()
                mod.levels = lvl
                ob.update_tag()
                deps.update()
                times.append(time.perf_counter() - t0)
    finally:
        mod.levels = orig_level
        ob.update_tag()
        deps.update()
    med = statistics.median(times)
    return {"status": "OK", "target": f"{ob.name} L0-L{sweep_to}",
            "median_s": med, "min_s": min(times), "max_s": max(times),
            "derived": "-"}


# ---------------------------------------------------------------------------
# Reporting  (mirror engine_benchmark's table so benchmark_parse can scrape it)
# ---------------------------------------------------------------------------
_METRIC_LABELS = [
    ("scrub",   "Frame scrub"),
    ("xform",   "Object transform"),
    ("subsurf", "Subsurf sweep"),
]


def print_table(results):
    tag = "  [TEST MODE - NOT REPRESENTATIVE]" if TEST_MODE else ""
    scrub = results.get("scrub", {})
    frames_note = (f"frames={scrub.get('frames')}/{scrub.get('span')}"
                   if scrub.get("status") == "OK" else "frames=n/a")
    print("\n" + "=" * 92)
    print(f"  Blender {_extract_version()} INTERACTIVE   GPU={_extract_gpu()}   "
          f"profile={_extract_model()}   {frames_note}{tag}")
    print(f"  invariants source: "
          f"{'qa_standardize_render_settings.py' if INVARIANTS_IMPORTED else 'LOCAL FALLBACK (out of sync risk)'}")
    print("  metrics are depsgraph re-eval only (no viewport draw / GL interop)")
    print("=" * 92)
    print(f"{'Metric':<20} | {'Status':<8} | {'Median(ms)':>11} | {'Min(ms)':>9} | "
          f"{'Max(ms)':>9} | {'Derived':>10}")
    print("-" * 92)
    for key, label in _METRIC_LABELS:
        m = results.get(key, {"status": "SKIPPED", "reason": "not run"})
        if m["status"] != "OK":
            print(f"{label:<20} | {'SKIP':<8} | {'':>11} | {'':>9} | {'':>9} | "
                  f"{'':>10}   ({m.get('reason', '')})")
            continue
        med = f"{m['median_s'] * 1000:.3f}"
        mn = f"{m['min_s'] * 1000:.3f}"
        mx = f"{m['max_s'] * 1000:.3f}"
        print(f"{label:<20} | {'OK':<8} | {med:>11} | {mn:>9} | {mx:>9} | {m['derived']:>10}")
    print("-" * 92)
    print("  Frame scrub = per-frame animation+sim re-eval (median of WARM passes).")
    print("  Derived fps = 1 / median-frame (eval only; excludes viewport draw).")
    print(f"  Object transform = single edit -> update_tag -> depsgraph.update. "
          f"N warm = {WARM_RUNS}.")
    print("=" * 92 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
_USAGE = """\
interactive_benchmark.py -- Tier-1 headless interactivity benchmark (runs INSIDE Blender).

  Invoke:  blender --enable-autoexec -b <scene.blend> -P interactive_benchmark.py -- [options]

  Measures depsgraph RE-EVAL cost (no GL window): frame scrub (-> playback fps),
  object-transform latency, and subsurf-level re-eval. Viewport DRAW / CUDA-GL
  interop is out of scope here (needs a display -- Tier 3).

  Options:
    --test           fast smoke check: 24-frame scrub, warm=1. Stamped NOT REPRESENTATIVE.
    --warm N         warm passes to median over (default 3)
    --frames N       cap the frame-scrub window (default: whole range, max 120)
    --xform-iters N  object-transform iterations (default 30)
    -h, --help       show this help and exit
"""


def _parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    if "--help" in argv or "-h" in argv:
        return {"help": True}

    def _val(flag, default=None):
        if flag in argv:
            i = argv.index(flag) + 1
            if i < len(argv):
                return argv[i]
        return default

    def _int(flag, default):
        v = _val(flag)
        if v is None:
            return default, False
        try:
            return int(v), True
        except ValueError:
            return default, False

    warm, warm_explicit = _int("--warm", WARM_RUNS)
    frames, _ = _int("--frames", None)
    xform, _ = _int("--xform-iters", XFORM_ITERS)
    return {"test": "--test" in argv, "warm": warm, "warm_explicit": warm_explicit,
            "frames": frames, "xform_iters": xform}


def main():
    global WARM_RUNS, TEST_MODE, FRAMES_OVERRIDE, XFORM_ITERS
    args = _parse_args()
    if args.get("help"):
        print(_USAGE)
        return

    WARM_RUNS = args["warm"]
    FRAMES_OVERRIDE = args["frames"]
    XFORM_ITERS = args["xform_iters"]
    TEST_MODE = args["test"]
    if TEST_MODE and not args["warm_explicit"]:
        WARM_RUNS = 1
    if TEST_MODE:
        print("=" * 92)
        print(f"  *** TEST MODE *** interactivity smoke check -- warm={WARM_RUNS}, "
              f"scrub<= {TEST_MAX_FRAMES} frames. NUMBERS ARE NOT REPRESENTATIVE.")
        print("=" * 92)

    scene = bpy.context.scene
    apply_invariants(scene)          # keep determinism identical to the render harness

    results = {}
    results["scrub"] = measure_frame_scrub(scene)
    results["xform"] = measure_transform(scene)
    results["subsurf"] = measure_subsurf(scene)
    print_table(results)


if __name__ == "__main__":
    main()
