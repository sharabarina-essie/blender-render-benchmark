"""
viewport_benchmark_gui.py  --  TIER-3 interactive viewport benchmark.
=====================================================================

Runs INSIDE a running Blender GUI (needs a live GL context + a CAMERA). Measures the
viewport DRAW / CUDA-GL interop path the headless harness cannot see.

PHASES (pick which run via RUN_PHASES below):
  "offscreen" : render the camera view into a FIXED-res off-screen buffer (default
                1920x1080, layout-independent) -> DRAW per shading mode, a short
                ORBIT, and PLAYBACK. Isolates render+interop (no window swap).
  "onscreen"  : redraw the REAL window with buffer swap -> DRAW + short ORBIT (the
                full present path the user sees / Boost forwards). Size = the on-screen
                3D view.
  "orbit"     : the INTERACTIVE test -> continuously orbits the RENDERED viewport with
                a live on-screen FPS overlay for SUSTAINED_S seconds and records
                average + 1%/5% LOWS (stutter) + min/max. Writes its OWN per-run
                timestamped file (see RESULTS below), because it is the interactive one.
  "converge"  : time a full-sample Cycles render to completion (e.g. 1024/1024),
                F12-style so the UI stays responsive.
  "coldstart" : COLD time-to-first-frame (kernel compile + BVH build + texture upload).
                Clears the render session (frees BVH + GPU/remote uploads) before EACH
                repeat so every run is a true cold start, and records per-run + medians.
                Over a remote GPU this is the cache-populate-across-the-link cost.
  "flythrough": MODAL, callback/event-driven camera flythrough for FLYTHROUGH_S seconds
                (default 120). Flies the scene's animated camera if it has one (e.g.
                temple), else a generated orbit. Drives position by WALL-CLOCK so late
                ticks show up as long/dropped frames -- records avg + 1%/5% low + mean/max
                frame ms + dropped-frame count. Stays interactive (Esc to stop).

RUN JUST THE INTERACTIVE ORBIT:  set  RUN_PHASES = {"orbit"}  below.
RUN JUST THE COLD-START TEST:    set  RUN_PHASES = {"coldstart"}  below.
RUN JUST THE FLYTHROUGH:         set  RUN_PHASES = {"flythrough"}  below.

RESULTS (written into the project's results/ folder, next to the suite's results):
  * the "orbit" interactive test -> results/viewport_orbit.csv  (ONE shared file, a
    row APPENDED each run: timestamp, scene, blender, engine, gpu, resolution,
    duration, frames, avg + median + 1%/5% low + min/max fps, stutter).
  * every other phase          -> results/results_viewport.csv (appended)

HOW TO RUN
----------
  1. Open your .blend in the GUI, engine = Cycles for the interop numbers.
  2. Scripting workspace -> Text Editor -> Open this file (Text > Reload if changed).
  3. Run Script (Alt+P). Watch the System Console (Window > Toggle System Console).

Note: a script runs on Blender's single UI thread, so the short DRAW/ORBIT/PLAYBACK
loops briefly pause the UI. The "orbit" test visibly spins the viewport (with the FPS
overlay) for SUSTAINED_S seconds; the CONVERGE render runs responsively (F12-style).
"""

import bpy
import os
import time
import math
import csv
import datetime

import gpu
import blf
from mathutils import Matrix, Quaternion

# Collect everything printed so we can always write a log file (the GUI System
# Console is easy to miss / may be closed).
_LOG = []


def _log(*args, **_kwargs):     # tolerate a stray flush=... from the print->_log rewrite
    msg = " ".join(str(a) for a in args)
    print(msg, flush=True)
    _LOG.append(msg)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WIDTH, HEIGHT = 1920, 1080     # FIXED off-screen measurement resolution (edit for 4K)
DRAW_ITER = 20                 # static redraws per shading mode
ORBIT_ITER = 20                # redraws in the SHORT orbit throughput sample
ORBIT_STEP_DEG = 2.0           # view rotation per orbit frame
PLAYBACK_MAX = 120             # cap playback steps
CONVERGE_SAMPLES = 1024        # sample budget the CONVERGE render runs to (e.g. 1024/1024)
SUSTAINED_S = 20.0             # INTERACTIVE "orbit" test duration in seconds
COLDSTART_RUNS = 5             # "coldstart" test: repeats, each with caches cleared first
FLYTHROUGH_S = 120.0           # "flythrough" test duration in seconds (wall-clock driven)
FLY_INTERVAL = 1.0 / 120.0     # requested tick interval; actual cadence is gated by render

# Which phases to run. Edit this to a subset, e.g. RUN_PHASES = {"orbit"} for just the
# interactive orbit FPS test.
#RUN_PHASES = {"offscreen", "onscreen", "orbit", "converge"}
RUN_PHASES = {"coldstart", "flythrough"}

# Where result files go. Empty = auto-detect the project results/ folder. Set an
# ABSOLUTE path if auto-detect picks the wrong place -- e.g. when you launch via the
# Python Console (exec), the script can't find its own location and falls back to the
# .blend's folder. Recommended: point this at the project results/ folder.
RESULTS_DIR = r""      # e.g. r"C:\Projects\blender-render-benchmark\results"

SHADING = ['SOLID', 'MATERIAL', 'RENDERED']
CSV_HEADER = ['metric', 'target', 'shading', 'engine', 'value', 'unit',
              'resolution', 'scene', 'blender']
ORBIT_FILE = "viewport_orbit.csv"   # single shared file: one appended row per run
ORBIT_HEADER = ['timestamp', 'scene', 'blender', 'engine', 'gpu', 'resolution',
                'startup_s', 'duration_s', 'frames', 'avg_fps', 'median_fps',
                '1pct_low_fps', '5pct_low_fps', 'min_fps', 'max_fps', 'orbit_deg_per_frame']
COLD_FILE = "viewport_coldstart.csv"   # cold time-to-first-frame, one row per repeat
COLD_HEADER = ['timestamp', 'scene', 'blender', 'engine', 'gpu', 'resolution',
               'run_index', 'cold_s', 'cleared']
FLY_FILE = "viewport_flythrough.csv"   # camera-path flythrough, one row per run
FLY_HEADER = ['timestamp', 'scene', 'blender', 'engine', 'gpu', 'resolution', 'mode',
              'startup_s', 'duration_s', 'ticks', 'avg_fps', 'median_fps', '1pct_low_fps',
              '5pct_low_fps', 'min_fps', 'max_fps', 'mean_frame_ms', 'max_frame_ms',
              'dropped_frames']


# ---------------------------------------------------------------------------
# Output locations -- put results in the project's results/ folder (same place the
# suite writes), NOT next to the .blend.
# ---------------------------------------------------------------------------
def _script_dir():
    """Directory this script lives in = the project root. Robust to how it's launched:
    __file__ if set, else the Text-editor datablock's on-disk path, else the .blend dir."""
    p = globals().get("__file__")
    if p:
        try:
            return os.path.dirname(os.path.abspath(p))
        except Exception:
            pass
    for t in bpy.data.texts:
        fp = getattr(t, "filepath", "")
        if fp and os.path.basename(fp) == "viewport_benchmark_gui.py":
            return os.path.dirname(os.path.abspath(bpy.path.abspath(fp)))
    return os.path.dirname(bpy.data.filepath) or bpy.app.tempdir


def _results_dir():
    d = RESULTS_DIR if RESULTS_DIR else os.path.join(_script_dir(), "results")
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        d = os.path.dirname(bpy.data.filepath) or bpy.app.tempdir
    return d


def _append_rows(out, engine, scene_name, rows):
    """Append per-metric rows to the shared viewport CSV (header on first create)."""
    new = not os.path.exists(out)
    with open(out, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if new:
            w.writerow(CSV_HEADER)
        for metric, target, shading, value, unit, res_r in rows:
            w.writerow([metric, target, shading, engine, f"{value:.3f}", unit,
                        res_r, scene_name, bpy.app.version_string])


def _gpu_name():
    """The live viewport GL/Vulkan renderer -- the GPU that actually draws the
    interactive frames (needs a live GL context, present in the GUI)."""
    try:
        return gpu.platform.renderer_get() or "unknown GPU"
    except Exception:
        return "unknown GPU"


def _write_orbit_result(results_dir, scene_name, engine, px, s, startup_s=0.0):
    """APPEND one summary row per run to the shared orbit results file (timestamp +
    gpu + startup/warm-up time + fps/stutter/low detail). Header written once."""
    out = os.path.join(results_dir, ORBIT_FILE)
    now = datetime.datetime.now()
    new = not os.path.exists(out)
    with open(out, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if new:
            w.writerow(ORBIT_HEADER)
        w.writerow([now.strftime("%Y-%m-%d %H:%M:%S"), scene_name, bpy.app.version_string,
                    engine, _gpu_name(), px, f"{startup_s:.2f}", f"{s['total']:.2f}", s['n'],
                    f"{s['avg']:.2f}", f"{s['median']:.2f}", f"{s['low1']:.2f}",
                    f"{s['low5']:.2f}", f"{s['min']:.2f}", f"{s['max']:.2f}",
                    f"{ORBIT_STEP_DEG:.2f}"])
    return out


def _write_cold_result(results_dir, scene_name, engine, px, run_index, cold_s):
    """APPEND one cold-start row per repeat to the shared cold-start file."""
    out = os.path.join(results_dir, COLD_FILE)
    now = datetime.datetime.now()
    new = not os.path.exists(out)
    with open(out, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if new:
            w.writerow(COLD_HEADER)
        w.writerow([now.strftime("%Y-%m-%d %H:%M:%S"), scene_name, bpy.app.version_string,
                    engine, _gpu_name(), px, run_index, f"{cold_s:.3f}",
                    "session_freed(BVH+upload); kernels warm after run1"])
    return out


def _camera_path_mode(scene):
    """'existing' if the scene camera actually MOVES over the timeline (own fcurves,
    a follow/clamp-to-path constraint, or an animated parent -- e.g. temple), else
    'generated' (we synthesize an orbit path)."""
    cam = scene.camera
    if cam is None or (scene.frame_end - scene.frame_start) <= 0:
        return 'generated'
    ad = cam.animation_data
    if ad and ad.action and ad.action.fcurves:
        return 'existing'
    if any(c.type in {'FOLLOW_PATH', 'CLAMP_TO'} for c in cam.constraints):
        return 'existing'
    par = cam.parent
    if par and par.animation_data and par.animation_data.action and par.animation_data.action.fcurves:
        return 'existing'
    return 'generated'


def _write_fly_result(results_dir, scene_name, engine, px, mode, s, startup_s,
                      mean_ms, max_ms, dropped, ticks):
    """APPEND one flythrough summary row (fps/lows + frame-time drift/dropped)."""
    out = os.path.join(results_dir, FLY_FILE)
    now = datetime.datetime.now()
    new = not os.path.exists(out)
    with open(out, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if new:
            w.writerow(FLY_HEADER)
        w.writerow([now.strftime("%Y-%m-%d %H:%M:%S"), scene_name, bpy.app.version_string,
                    engine, _gpu_name(), px, mode, f"{startup_s:.2f}", f"{FLYTHROUGH_S:.1f}",
                    ticks, f"{s['avg']:.2f}", f"{s['median']:.2f}", f"{s['low1']:.2f}",
                    f"{s['low5']:.2f}", f"{s['min']:.2f}", f"{s['max']:.2f}",
                    f"{mean_ms:.2f}", f"{max_ms:.2f}", dropped])
    return out


# ---------------------------------------------------------------------------
# Live on-screen FPS overlay (painted over the viewport while the orbit runs).
# ---------------------------------------------------------------------------
_HUD = {"label": ""}


def _draw_hud():
    if not _HUD["label"]:
        return
    try:
        font_id = 0
        try:
            blf.size(font_id, 30)              # Blender 4.0+ : (font_id, size)
        except TypeError:
            blf.size(font_id, 30, 72)          # Blender 3.x  : (font_id, size, dpi)
        blf.enable(font_id, blf.SHADOW)
        blf.shadow(font_id, 5, 0.0, 0.0, 0.0, 1.0)
        blf.color(font_id, 1.0, 1.0, 1.0, 1.0)
        blf.position(font_id, 24, 60, 0)
        blf.draw(font_id, _HUD["label"])
    except Exception:
        pass


def _stats_fps(frame_times):
    """Game-style FPS stats: average, median, 1%/5% LOWS (mean of the slowest 1%/5%
    frames -> fps; the stutter metric), and min/max. A long capture makes lows real."""
    n = len(frame_times)
    if n == 0:
        return {"avg": 0.0, "median": 0.0, "low1": 0.0, "low5": 0.0,
                "min": 0.0, "max": 0.0, "n": 0, "total": 0.0}
    total = sum(frame_times)
    ordered = sorted(frame_times)
    slowest = ordered[::-1]

    def _low(p):
        k = max(1, n * p // 100)
        worst = slowest[:k]
        m = sum(worst) / len(worst)
        return 1.0 / m if m else 0.0

    return {"avg": n / total if total else 0.0,
            "median": 1.0 / ordered[n // 2] if ordered[n // 2] else 0.0,
            "low1": _low(1), "low5": _low(5),
            "min": 1.0 / max(frame_times), "max": 1.0 / min(frame_times),
            "n": n, "total": total}


def _find_view3d():
    best, best_px = (None, None, None, None), -1
    for win in bpy.context.window_manager.windows:
        for area in win.screen.areas:
            if area.type != 'VIEW_3D':
                continue
            region = next((r for r in area.regions if r.type == 'WINDOW'), None)
            if region and region.width * region.height > best_px:
                best_px = region.width * region.height
                best = (win, area, area.spaces.active, region)
    return best


def _start_converge_render(scene, out, engine_name, scene_name, res_off):
    """Full-sample Cycles render, F12-STYLE (non-blocking) with a render_complete
    handler recording wall-clock -> UI stays responsive."""
    cy = getattr(scene, "cycles", None)
    r = scene.render
    st = {"engine": r.engine,
          "res": (r.resolution_x, r.resolution_y, r.resolution_percentage),
          "samples": getattr(cy, "samples", None) if cy is not None else None,
          "adaptive": getattr(cy, "use_adaptive_sampling", None) if cy is not None else None,
          "t0": None}
    r.engine = 'CYCLES'
    if cy is not None:
        cy.samples = CONVERGE_SAMPLES
        if st["adaptive"] is not None:
            cy.use_adaptive_sampling = False
    r.resolution_x, r.resolution_y, r.resolution_percentage = WIDTH, HEIGHT, 100

    def _restore():
        r.engine = st["engine"]
        r.resolution_x, r.resolution_y, r.resolution_percentage = st["res"]
        if cy is not None:
            if st["samples"] is not None:
                cy.samples = st["samples"]
            if st["adaptive"] is not None:
                cy.use_adaptive_sampling = st["adaptive"]

    def _cleanup():
        for hlist, fn in ((bpy.app.handlers.render_complete, _done),
                          (bpy.app.handlers.render_cancel, _cancelled)):
            if fn in hlist:
                hlist.remove(fn)

    def _done(*_a):
        secs = time.perf_counter() - st["t0"]
        _restore()
        _log(f"  converge CYCLES    [render {res_off}] {secs:8.2f} s  to {CONVERGE_SAMPLES} samples",
              flush=True)
        _append_rows(out, engine_name, scene_name,
                     [('converge', 'render', 'CYCLES', secs, 's', res_off)])
        _log(f"  -> converge row appended to {out}\n")
        _cleanup()

    def _cancelled(*_a):
        _restore()
        _log("  converge CYCLES    render cancelled -- no row written")
        _cleanup()

    bpy.app.handlers.render_complete.append(_done)
    bpy.app.handlers.render_cancel.append(_cancelled)
    _log(f"  converge CYCLES    rendering {CONVERGE_SAMPLES} samples at {res_off} "
          f"F12-style -- UI stays responsive, watch the progress bar...")
    st["t0"] = time.perf_counter()
    try:
        bpy.ops.render.render('INVOKE_DEFAULT', write_still=False)
    except Exception as e:
        _restore()
        _cleanup()
        _log(f"  converge CYCLES    could not start render: {e}")


# ---------------------------------------------------------------------------
# FLYTHROUGH: a MODAL operator (callback/event-driven, like an Unreal tick) that
# flies the camera along its path for FLYTHROUGH_S seconds. It drives position by
# WALL-CLOCK (perf_counter) -- so the path always takes exactly N real seconds even
# when a tick fires late -- and records the per-tick wall-clock frame time, so late
# callbacks show up as long frames / dropped frames (the drift you measure in Unreal).
# Stays interactive (PASS_THROUGH); Esc stops early.
# ---------------------------------------------------------------------------
class VIEWBENCH_OT_flythrough(bpy.types.Operator):
    bl_idname = "viewbench.flythrough"
    bl_label = "Viewport Benchmark: Flythrough"

    def execute(self, context):
        win, area, space, region = _find_view3d()
        if area is None:
            self.report({'ERROR'}, "no 3D viewport"); return {'CANCELLED'}
        self.win, self.area, self.space, self.region = win, area, space, region
        self.scene = context.scene
        self.rv3d = space.region_3d
        self.results_dir = _results_dir()
        self.scene_name = os.path.basename(bpy.data.filepath) or self.scene.name
        self.engine = self.scene.render.engine
        self.px = f"{region.width}x{region.height}"
        self.fs, self.fe = self.scene.frame_start, self.scene.frame_end
        self.mode = _camera_path_mode(self.scene)

        # save state to restore on finish
        self.saved_shading = space.shading.type
        self.saved_rot = self.rv3d.view_rotation.copy()
        self.saved_persp = self.rv3d.view_perspective
        self.saved_frame = self.scene.frame_current
        self.cy = getattr(self.scene, "cycles", None)
        self.saved_pause = getattr(self.cy, "preview_pause", None) if self.cy is not None else None
        if self.saved_pause is not None:
            self.cy.preview_pause = False

        space.shading.type = 'RENDERED'
        if self.mode == 'generated' and self.rv3d.view_perspective == 'CAMERA':
            self.rv3d.view_perspective = 'PERSP'
        self.base_rot = self.rv3d.view_rotation.copy()

        _HUD["label"] = f"flythrough ({self.mode}) warming up..."
        self.hud = bpy.types.SpaceView3D.draw_handler_add(_draw_hud, (), 'WINDOW', 'POST_PIXEL')

        # startup / first frame (warm-up), timed separately
        t_setup = time.perf_counter()
        with bpy.context.temp_override(window=win, area=area, region=region):
            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
        self.startup_s = time.perf_counter() - t_setup

        self.t0 = self.last = time.perf_counter()
        self.samples = []
        self.timer = context.window_manager.event_timer_add(FLY_INTERVAL, window=win)
        context.window_manager.modal_handler_add(self)
        _log(f"  flythrough {self.mode} [onscreen {self.px}] startup {self.startup_s:.2f}s; "
             f"flying camera ~{FLYTHROUGH_S:.0f}s by wall-clock (UI responsive; Esc to stop)...")
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'ESC':
            return self._finish(context, cancelled=True)
        if event.type == 'TIMER':
            now = time.perf_counter()
            t = now - self.t0
            frac = min(t / FLYTHROUGH_S, 1.0)
            if self.mode == 'existing':
                ff = self.fs + frac * (self.fe - self.fs)    # position by WALL-CLOCK, not tick count
                fi = int(ff)
                try:
                    self.scene.frame_set(fi, subframe=ff - fi)
                except TypeError:
                    self.scene.frame_set(fi)
            else:
                ang = math.radians(frac * 360.0)             # one full turn over the run
                self.rv3d.view_rotation = Quaternion((0.0, 0.0, 1.0), ang) @ self.base_rot
            self.area.tag_redraw()
            dt = now - self.last
            self.last = now
            self.samples.append(dt)
            recent = self.samples[-15:]
            inst = len(recent) / sum(recent) if sum(recent) else 0.0
            _HUD["label"] = f"FLY {inst:5.1f} fps   {t:5.1f}/{FLYTHROUGH_S:.0f}s   ({self.mode})"
            if t >= FLYTHROUGH_S:
                return self._finish(context, cancelled=False)
        return {'PASS_THROUGH'}

    def _finish(self, context, cancelled):
        try:
            context.window_manager.event_timer_remove(self.timer)
        except Exception:
            pass
        try:
            bpy.types.SpaceView3D.draw_handler_remove(self.hud, 'WINDOW')
        except Exception:
            pass
        _HUD["label"] = ""
        # restore scene state
        self.space.shading.type = self.saved_shading
        self.rv3d.view_rotation = self.saved_rot
        self.rv3d.view_perspective = self.saved_persp
        try:
            self.scene.frame_set(self.saved_frame)
        except Exception:
            pass
        if self.saved_pause is not None:
            self.cy.preview_pause = self.saved_pause

        samples = self.samples[1:] if len(self.samples) > 1 else self.samples   # drop first (warm) tick
        s = _stats_fps(samples)
        if samples:
            med_dt = sorted(samples)[len(samples) // 2]
            mean_ms = sum(samples) / len(samples) * 1000.0
            max_ms = max(samples) * 1000.0
            dropped = sum(1 for d in samples if d > 2 * med_dt)   # frames >2x the median = a hitch
        else:
            mean_ms = max_ms = 0.0
            dropped = 0
        tag = "CANCELLED " if cancelled else ""
        _log(f"  flythrough {self.mode} [onscreen {self.px}] {tag}avg {s['avg']:6.1f} | "
             f"1% low {s['low1']:6.1f} | 5% low {s['low5']:6.1f} | min {s['min']:6.1f} | "
             f"max {s['max']:6.1f} fps | mean {mean_ms:6.1f}ms | max {max_ms:7.1f}ms | "
             f"dropped {dropped} | startup {self.startup_s:.2f}s ({len(samples)} ticks)")
        out = _write_fly_result(self.results_dir, self.scene_name, self.engine, self.px,
                                self.mode, s, self.startup_s, mean_ms, max_ms, dropped, len(samples))
        _log(f"  -> flythrough result -> {out}")
        return {'CANCELLED'} if cancelled else {'FINISHED'}


def _register_flythrough():
    try:
        bpy.utils.unregister_class(VIEWBENCH_OT_flythrough)
    except Exception:
        pass
    bpy.utils.register_class(VIEWBENCH_OT_flythrough)


def _run():
    win, area, space, region = _find_view3d()
    if area is None:
        _log("[viewport-bench] No 3D Viewport found -- keep a 3D view open and rerun.")
        return
    scene = bpy.context.scene
    view_layer = bpy.context.view_layer
    engine = scene.render.engine
    scene_name = os.path.basename(bpy.data.filepath) or scene.name
    cam = scene.camera
    res_off = f"{WIDTH}x{HEIGHT}"
    results_dir = _results_dir()
    shared_csv = os.path.join(results_dir, "results_viewport.csv")

    _log(f"\n=== VIEWPORT BENCH  Blender {bpy.app.version_string}  engine={engine}  "
          f"scene={scene_name}  phases={sorted(RUN_PHASES)} ===")
    _log(f"  results -> {results_dir}")

    rv3d = space.region_3d
    onscreen_px = f"{region.width}x{region.height}"
    delta = Quaternion((0.0, 0.0, 1.0), math.radians(ORBIT_STEP_DEG))

    def _swap(iterations):
        with bpy.context.temp_override(window=win, area=area, region=region):
            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=iterations)

    # Off-screen buffer + camera matrices only needed for the "offscreen" phase.
    offscreen = None
    base_view = proj = None
    if "offscreen" in RUN_PHASES:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        if cam is not None:
            base_view = cam.matrix_world.inverted()
            proj = cam.calc_matrix_camera(depsgraph, x=WIDTH, y=HEIGHT)
        else:
            base_view = rv3d.view_matrix.copy()
            proj = rv3d.window_matrix.copy()
            _log("  [warn] no camera -- using current view for off-screen framing")
        try:
            offscreen = gpu.types.GPUOffScreen(WIDTH, HEIGHT)
        except Exception as e:
            _log(f"  [warn] off-screen buffer failed ({e}); skipping offscreen phase")
            offscreen = None

    def _draw(view_matrix):
        with bpy.context.temp_override(window=win, area=area, region=region):
            try:
                offscreen.draw_view3d(scene, view_layer, space, region,
                                      view_matrix, proj, do_color_management=True)
            except TypeError:
                offscreen.draw_view3d(scene, view_layer, space, region, view_matrix, proj)

    cy = getattr(scene, "cycles", None)
    prev_pause = getattr(cy, "preview_pause", None) if cy is not None else None
    if prev_pause is not None:
        cy.preview_pause = False

    rows = []                                      # shared per-metric rows
    orig_shading = space.shading.type
    saved_rot = rv3d.view_rotation.copy()
    saved_persp = rv3d.view_perspective
    try:
        # ---------------- OFF-SCREEN (fixed res) --------------------------
        if "offscreen" in RUN_PHASES and offscreen is not None:
            for mode in SHADING:
                try:
                    space.shading.type = mode
                except Exception as e:
                    _log(f"  draw {mode:<9} cannot set shading ({e})")
                    continue
                _log(f"  draw     {mode:<9} [offscreen {res_off}] measuring...")
                _draw(base_view)
                t0 = time.perf_counter()
                for _ in range(DRAW_ITER):
                    _draw(base_view)
                dt = time.perf_counter() - t0
                fps, ms = (DRAW_ITER / dt if dt else 0.0), (dt / DRAW_ITER * 1000 if DRAW_ITER else 0.0)
                _log(f"  draw     {mode:<9} [offscreen {res_off}] {fps:8.1f} fps   {ms:8.2f} ms/frame",
                      flush=True)
                rows.append(('draw', 'offscreen', mode, fps, 'fps', res_off))

            space.shading.type = 'RENDERED'
            _log(f"  orbit    RENDERED  [offscreen {res_off}] measuring...")
            _draw(base_view)
            step = math.radians(ORBIT_STEP_DEG)
            t0 = time.perf_counter()
            for i in range(ORBIT_ITER):
                _draw(base_view @ Matrix.Rotation(step * i, 4, 'Z'))
            dt = time.perf_counter() - t0
            fps, ms = (ORBIT_ITER / dt if dt else 0.0), (dt / ORBIT_ITER * 1000 if ORBIT_ITER else 0.0)
            _log(f"  orbit    RENDERED  [offscreen {res_off}] {fps:8.1f} fps   {ms:8.2f} ms/frame",
                  flush=True)
            rows.append(('orbit', 'offscreen', 'RENDERED', fps, 'fps', res_off))

            span = scene.frame_end - scene.frame_start + 1
            if span > 1:
                n = min(span, PLAYBACK_MAX)
                _log(f"  playback RENDERED  [offscreen {res_off}] measuring {n} frames...")
                t0 = time.perf_counter()
                for i in range(n):
                    scene.frame_set(scene.frame_start + i)
                    _draw(base_view)
                dt = time.perf_counter() - t0
                fps, ms = (n / dt if dt else 0.0), (dt / n * 1000 if n else 0.0)
                _log(f"  playback RENDERED  [offscreen {res_off}] {fps:8.1f} fps   {ms:8.2f} ms/frame "
                      f"({n} frames)")
                rows.append(('playback', 'offscreen', 'RENDERED', fps, 'fps', res_off))

        # ---------------- ON-SCREEN (real window + swap) ------------------
        if "onscreen" in RUN_PHASES:
            space.shading.type = 'RENDERED'
            _log(f"  draw     RENDERED  [onscreen  {onscreen_px}] measuring...")
            _swap(1)
            t0 = time.perf_counter()
            _swap(DRAW_ITER)
            dt = time.perf_counter() - t0
            fps, ms = (DRAW_ITER / dt if dt else 0.0), (dt / DRAW_ITER * 1000 if DRAW_ITER else 0.0)
            _log(f"  draw     RENDERED  [onscreen  {onscreen_px}] {fps:8.1f} fps   {ms:8.2f} ms/frame",
                  flush=True)
            rows.append(('draw', 'onscreen', 'RENDERED', fps, 'fps', onscreen_px))

            if rv3d.view_perspective == 'CAMERA':
                rv3d.view_perspective = 'PERSP'
            _log(f"  orbit    RENDERED  [onscreen  {onscreen_px}] measuring (camera moves)...")
            _swap(1)
            t0 = time.perf_counter()
            for _ in range(ORBIT_ITER):
                rv3d.view_rotation = delta @ rv3d.view_rotation
                _swap(1)
            dt = time.perf_counter() - t0
            fps, ms = (ORBIT_ITER / dt if dt else 0.0), (dt / ORBIT_ITER * 1000 if ORBIT_ITER else 0.0)
            _log(f"  orbit    RENDERED  [onscreen  {onscreen_px}] {fps:8.1f} fps   {ms:8.2f} ms/frame",
                  flush=True)
            rows.append(('orbit', 'onscreen', 'RENDERED', fps, 'fps', onscreen_px))

        # ---------------- ORBIT: interactive sustained FPS ----------------
        if ("orbit" in RUN_PHASES or "sustained" in RUN_PHASES) and SUSTAINED_S > 0:
            if rv3d.view_perspective == 'CAMERA':
                rv3d.view_perspective = 'PERSP'
            hud_handle = bpy.types.SpaceView3D.draw_handler_add(_draw_hud, (), 'WINDOW', 'POST_PIXEL')
            frame_times = []
            startup_s = 0.0
            try:
                # STARTUP: entering RENDERED + the FIRST frame. This is where the engine
                # compiles kernels, builds the BVH, and uploads geometry/textures to the
                # GPU -- over a remote GPU this is the cache/geometry populating across the
                # link, i.e. the long unresponsive pause before you can interact. Time it
                # separately so it never contaminates the orbit fps.
                _HUD["label"] = "warming up (first frame / cache)..."
                _log(f"  orbit    RENDERED  [onscreen {onscreen_px}] warming up: first RENDERED "
                     f"frame (kernel/BVH/upload; a long pause here over a remote GPU = cache "
                     f"populating)...")
                t_setup = time.perf_counter()
                space.shading.type = 'RENDERED'
                _swap(1)                           # COLD first frame -> the startup cost
                startup_s = time.perf_counter() - t_setup
                _log(f"  orbit    RENDERED  [onscreen {onscreen_px}] startup {startup_s:8.2f} s  "
                     f"(time to first frame BEFORE orbiting -- warm-up / cache populate)")
                _swap(2)                           # settle a couple more frames (not timed)

                _log(f"  orbit    RENDERED  [onscreen {onscreen_px}] now orbiting ~{SUSTAINED_S:.0f}s "
                     f"with live FPS overlay (UI busy until done)...")
                _HUD["label"] = "FPS   --.-"
                t_start = time.perf_counter()
                while time.perf_counter() - t_start < SUSTAINED_S:
                    rv3d.view_rotation = delta @ rv3d.view_rotation
                    f0 = time.perf_counter()
                    _swap(1)
                    frame_times.append(time.perf_counter() - f0)
                    recent = frame_times[-15:]
                    inst = len(recent) / sum(recent) if sum(recent) else 0.0
                    avg = len(frame_times) / sum(frame_times) if sum(frame_times) else 0.0
                    _HUD["label"] = (f"FPS {inst:5.1f}   avg {avg:5.1f}   "
                                     f"{time.perf_counter() - t_start:4.1f}/{SUSTAINED_S:.0f}s")
            finally:
                _HUD["label"] = ""
                bpy.types.SpaceView3D.draw_handler_remove(hud_handle, 'WINDOW')
            s = _stats_fps(frame_times)
            _log(f"  orbit    RENDERED  [onscreen {onscreen_px}] startup {startup_s:.2f}s | "
                  f"avg {s['avg']:6.1f} | 1% low {s['low1']:6.1f} | 5% low {s['low5']:6.1f} | "
                  f"min {s['min']:6.1f} | max {s['max']:6.1f} fps  "
                  f"({s['n']} frames in {s['total']:.1f}s)")
            orbit_file = _write_orbit_result(results_dir, scene_name, engine, onscreen_px, s, startup_s)
            _log(f"  -> interactive orbit result -> {orbit_file}")

        # ---------------- COLDSTART: kernel compile / BVH build / upload -------
        # Cold time-to-first-frame, cleared before EVERY repeat so each is a true cold
        # start. "Clear" = tear down the viewport render session (exit RENDERED + bounce
        # the render engine) so the BVH and GPU/remote-uploaded textures are freed, then
        # re-tag geometry to force a rebuild. Over a remote GPU this evicts the uploaded
        # scene so each run re-populates the cache across the link -- the pause you saw.
        # NOTE: kernel COMPILE is a one-time-per-Blender-session cost: run 1 pays it,
        # runs 2+ don't. So run1 - median(runs2+) ~= kernel compile; median(runs2+) ~=
        # BVH build + (re-)upload. For a fully-cold kernel number, restart Blender and
        # read run 1.
        if "coldstart" in RUN_PHASES:
            engine_id = scene.render.engine
            saved_engine_cs = engine_id
            _log(f"  coldstart {engine_id} [onscreen {onscreen_px}] cold time-to-first-frame "
                 f"(kernel/BVH/upload), clearing session+GPU caches before each of "
                 f"{COLDSTART_RUNS} runs...")
            cold_times = []
            try:
                for i in range(1, COLDSTART_RUNS + 1):
                    # ---- CLEAR: free the render session (BVH + GPU/remote uploads) ----
                    space.shading.type = 'SOLID'
                    _swap(2)
                    scene.render.engine = 'BLENDER_WORKBENCH'
                    _swap(1)
                    scene.render.engine = engine_id
                    for ob in scene.objects:
                        try:
                            ob.update_tag()
                        except Exception:
                            pass
                    try:
                        bpy.context.evaluated_depsgraph_get().update()
                    except Exception:
                        pass
                    _swap(1)                       # settle in SOLID (still cold for RENDERED)
                    # ---- MEASURE: enter RENDERED + first frame (the sync/upload pause) ----
                    t0 = time.perf_counter()
                    space.shading.type = 'RENDERED'
                    _swap(1)
                    cold = time.perf_counter() - t0
                    cold_times.append(cold)
                    _log(f"  coldstart run {i}/{COLDSTART_RUNS}: {cold:8.2f} s")
                    _write_cold_result(results_dir, scene_name, engine_id, onscreen_px, i, cold)
            finally:
                scene.render.engine = saved_engine_cs
            if cold_times:
                srt = sorted(cold_times)
                med = srt[len(srt) // 2]
                run1 = cold_times[0]
                rest = cold_times[1:]
                rest_med = sorted(rest)[len(rest) // 2] if rest else med
                _log(f"  coldstart {engine_id} [onscreen {onscreen_px}] median {med:.2f}s | "
                     f"run1 {run1:.2f}s (incl. kernel compile) | runs2+ median {rest_med:.2f}s "
                     f"(BVH+upload / re-populate)  ->  kernel compile ~= {run1 - rest_med:.2f}s")
    finally:
        rv3d.view_rotation = saved_rot
        rv3d.view_perspective = saved_persp
        space.shading.type = orig_shading
        if prev_pause is not None:
            cy.preview_pause = prev_pause
        if offscreen is not None:
            offscreen.free()

    if rows:
        _append_rows(shared_csv, engine, scene_name, rows)
        _log(f"  -> appended {len(rows)} row(s) to {shared_csv}")

    if "converge" in RUN_PHASES:
        _start_converge_render(scene, shared_csv, 'CYCLES', scene_name, res_off)

    if "flythrough" in RUN_PHASES:
        # Modal + wall-clock (callback-driven); runs async so the UI stays responsive.
        _register_flythrough()
        try:
            bpy.ops.viewbench.flythrough('INVOKE_DEFAULT')
        except Exception as e:
            _log(f"  flythrough could not start: {e}")


def main():
    """Wrapper: always leave an artifact -- a log file in results/ and a popup with
    the status + paths -- so a run is never silent even if the System Console is closed
    or the body raises before writing anything."""
    import traceback
    _LOG.clear()
    results_dir = _results_dir()
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logpath = os.path.join(results_dir, f"viewport_run_{stamp}.log")
    status = "OK"
    try:
        _run()
    except Exception:
        status = "ERROR"
        _log("[ERROR] viewport benchmark failed:")
        _log(traceback.format_exc())
    try:
        with open(logpath, "w", encoding="utf-8") as f:
            f.write("\n".join(_LOG) + "\n")
    except Exception as e:
        print(f"[viewport-bench] could not write log {logpath}: {e}", flush=True)
    print(f"[viewport-bench] {status}  log -> {logpath}", flush=True)

    # Popup so the result is visible without the System Console.
    try:
        lines = [f"Viewport benchmark: {status}",
                 f"results dir: {results_dir}",
                 f"log: {os.path.basename(logpath)}"]

        def _draw(self, _ctx):
            for ln in lines:
                self.layout.label(text=ln)
        bpy.context.window_manager.popup_menu(
            _draw, title="Viewport Benchmark", icon=('INFO' if status == "OK" else 'ERROR'))
    except Exception:
        pass


if __name__ == "__main__":
    main()
