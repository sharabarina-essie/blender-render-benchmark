"""
ui_render_test.py
=================
F12 render IN THE UI. Unlike engine_benchmark.py (headless, -b), a render started
from a Blender window gets Cycles' display driver, which shows the image as it
renders through graphics interop (CUDA/OptiX <-> OpenGL/Vulkan). Headless
renders skip that path entirely; this test exercises it.

In a full-screen Blender window, for each requested permutation ("perm"):
  1. applies the benchmark invariants + customer profile (same as engine_benchmark,
     so UI and headless numbers are comparable);
  2. sets the perm's engine, compute device and denoiser;
  3. renders F12-style (INVOKE_DEFAULT -> render job, progressive display), with
     the result shown full screen;
  4. on completion: records the time, screenshots the whole window, saves the
     render, appends a row to results/ui_render.csv.
Then quits. run_ui_render_suite.py drives this across Blender builds.

Perms (--perms, comma-separated ids or groups):
  cycles-<compute>-<denoise>   compute: optix | cuda | cpu
                               denoise: none | oidn-cpu | oidn-gpu | optix
  eevee, workbench
  groups: all (default), cycles, gpu (Cycles on GPU), cpu (Cycles on CPU), raster
A CPU render still denoises on the GPU for oidn-gpu / optix: Cycles takes the
denoise device from the GPUs enabled in Preferences, not from the scene device.
Where a combination is unsupported Cycles falls back to OIDN on the CPU silently,
so oidn-gpu is SKIPPED on builds without the "denoise on GPU" option (< 4.1).

Run (from the repo root; -W = full-screen window):
  blender\\4.5.9\\blender.exe -W --enable-autoexec models\\archviz\\classroom\\classroom\\main.blend ^
      --python ui_render_test.py -- --perms all --test --label "smoke"

Options (after --):
  --perms LIST          perms / groups to run (default all)
  --device OPTIX|CUDA   GPU backend set in Preferences for CPU-compute perms (default OPTIX)
  --gpu "SUBSTR"        only present devices whose name contains SUBSTR
  --runs N              renders per perm; run 1 is cold (kernels, BVH) (default 1)
  --samples N           override the profile's Cycles samples
  --test                smoke settings: 25% res (480x270), 16 Cycles / 8 EEVEE samples,
                        Workbench AA off -- not representative
  --label "..."         free text recorded in the CSV (e.g. "native", "workstation")
  --session ID          groups rows from one suite run (default: this run's timestamp)
  --stamp               burn date / render time / perm into the pixels (always in the
                        PNG metadata; burned-in text defeats validate_ui_renders.py)
  --keep-open           leave Blender open at the end
  --display MODE        render display: SCREEN (default) | WINDOW | AREA | NONE
  --no-override         start the render with only the window in context (no screen/area)
Outputs go to results/: ui_render.csv, ui_render_<ts>_<perm>_run<N>_{screen,render}.png,
ui_render_<ts>.log. Progress lines "[ui-render] PERM_DONE <perm>" let the driver resume.
"""

import csv
import ctypes
import os
import sys
import time

import bpy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
import engine_benchmark as eb   # noqa: E402  (invariants, profile, device selection)

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
CSV_PATH = os.path.join(RESULTS_DIR, "ui_render.csv")
START_DELAY_S = 3.0      # let the window open, go full screen and draw before rendering
SETTLE_S = 1.5           # after completion, let the final (denoised) image reach the screen
TEST_CYCLES_SAMPLES = 16
TEST_EEVEE_SAMPLES = 8
CSV_COLUMNS = [
    "session", "completed_at", "host", "label", "blender", "scene", "perm", "engine", "compute",
    "denoiser", "denoise_device", "test_mode", "run", "status", "render_s", "devices",
    "gl_renderer", "resolution", "samples", "window_px", "screen_px", "screenshot",
    "render_png", "image_rect",
]

COMPUTES = ("optix", "cuda", "cpu")
DENOISES = ("none", "oidn-cpu", "oidn-gpu", "optix")
# GPU Cycles first, rasterizers next, CPU Cycles (slowest) last.
PERMS = ([f"cycles-{c}-{d}" for c in COMPUTES if c != "cpu" for d in DENOISES]
         + ["eevee", "workbench"] + [f"cycles-cpu-{d}" for d in DENOISES])
GROUPS = {
    "all": PERMS,
    "cycles": [p for p in PERMS if p.startswith("cycles-")],
    "gpu": [p for p in PERMS if p.startswith(("cycles-optix-", "cycles-cuda-"))],
    "cpu": [p for p in PERMS if p.startswith("cycles-cpu-")],
    "raster": ["eevee", "workbench"],
}

STAMP = time.strftime("%Y%m%d_%H%M%S")
LOG_PATH = os.path.join(RESULTS_DIR, f"ui_render_{STAMP}.log")


def _log(msg):
    line = f"[ui-render] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def expand_perms(spec):
    out = []
    for tok in (t.strip().lower() for t in spec.split(",") if t.strip()):
        for p in GROUPS.get(tok, [tok]):
            if p not in PERMS:
                raise ValueError(f"unknown perm '{tok}'. Perms: {', '.join(PERMS)}; "
                                 f"groups: {', '.join(GROUPS)}")
            if p not in out:
                out.append(p)
    return out


def _parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []

    def val(flag, default=None):
        if flag in argv and argv.index(flag) + 1 < len(argv):
            return argv[argv.index(flag) + 1]
        return default

    return {
        "perms": expand_perms(val("--perms", "all")),
        "device": (val("--device", "OPTIX") or "OPTIX").upper(),
        "gpu": val("--gpu"),
        "runs": max(1, int(val("--runs", "1"))),
        "samples": int(val("--samples")) if val("--samples") else None,
        "test": "--test" in argv,
        "label": val("--label", ""),
        "session": val("--session", STAMP),
        "stamp": "--stamp" in argv,
        "display": (val("--display", "SCREEN") or "SCREEN").upper(),
        "override": "--no-override" not in argv,
        "keep_open": "--keep-open" in argv,
    }


def _screen_px():
    try:
        user32 = ctypes.windll.user32
        return f"{user32.GetSystemMetrics(0)}x{user32.GetSystemMetrics(1)}"
    except Exception:
        return "?"


def _cpu_name():
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    except Exception:
        return f"CPU ({os.cpu_count()} threads)"


def _largest_area(window):
    return max(window.screen.areas, key=lambda a: a.width * a.height)


def _quit():
    bpy.ops.wm.quit_blender()
    return None


class UIRenderTest:
    def __init__(self, args):
        self.args = args
        self.scene = bpy.context.scene
        self.window = bpy.context.window_manager.windows[0]
        self.queue = list(args["perms"])
        self.perm = None
        self.info = {}
        self.run = 0
        self.t0 = None
        self.done_at = None
        self.status = None
        self.prefs = bpy.context.preferences.view
        self.saved_display_type = self.prefs.render_display_type
        eb.TEST_MODE = args["test"]
        eb.GPU_SUBSTR = args["gpu"]
        if args["test"]:
            eb.TEST_CYCLES_SAMPLES = TEST_CYCLES_SAMPLES
            eb.TEST_EEVEE_SAMPLES = TEST_EEVEE_SAMPLES
        self.prefs.render_display_type = args["display"]   # SCREEN: result fills the window
        _log(f"Blender {bpy.app.version_string} | scene {eb._extract_model()} | "
             f"perms: {', '.join(self.queue)} | test={args['test']} | "
             f"screen {_screen_px()}")

    # -- per-perm setup ------------------------------------------------------
    def configure(self, perm):
        """Point the scene at one perm. Returns (ok, reason); fills self.info."""
        scene, a = self.scene, self.args
        eb.apply_invariants(scene)
        eb.apply_scene_controls(scene)
        r = scene.render

        if perm.startswith("cycles-"):
            _, compute, denoise = perm.split("-", 2)
            gpu_backend = compute.upper() if compute != "cpu" else a["device"]
            ok, gpus, reason = eb.configure_backend(gpu_backend)
            needs_gpu = compute != "cpu" or denoise in ("oidn-gpu", "optix")
            if not ok and needs_gpu:
                return False, reason
            cy = scene.cycles
            if compute == "cpu":
                cy.device = 'CPU'
            eb.apply_customer_profile(scene, "CYCLES")
            if a["samples"]:
                cy.samples = a["samples"]

            cy.use_denoising = denoise != "none"
            for layer in scene.view_layers:
                layer.cycles.use_denoising = denoise != "none"
            has_gpu_oidn = hasattr(cy, "denoising_use_gpu")
            denoise_device = "-"
            if denoise == "optix":
                try:
                    cy.denoiser = 'OPTIX'
                except TypeError:
                    return False, "OptiX denoiser not available on this build"
                denoise_device = "GPU"
            elif denoise in ("oidn-cpu", "oidn-gpu"):
                if denoise == "oidn-gpu" and not has_gpu_oidn:
                    return False, "OIDN on GPU needs Blender 4.1+"
                cy.denoiser = 'OPENIMAGEDENOISE'
                denoise_device = "GPU" if denoise == "oidn-gpu" else "CPU"
            if has_gpu_oidn:
                cy.denoising_use_gpu = denoise in ("oidn-gpu", "optix")

            devices = [_cpu_name()] if compute == "cpu" else gpus
            self.info = {
                "engine": "CYCLES", "compute": compute.upper(),
                "denoiser": {"none": "NONE", "optix": "OPTIX"}.get(denoise, "OPENIMAGEDENOISE"),
                "denoise_device": denoise_device, "devices": "; ".join(devices),
                "samples": cy.samples,
            }
        else:
            logical = "BLENDER_EEVEE_NEXT" if perm == "eevee" else "BLENDER_WORKBENCH"
            r.engine = eb._resolve_engine_id(logical, scene)
            eb.apply_customer_profile(scene, logical)
            ev = scene.eevee
            samples = ev.taa_render_samples if perm == "eevee" else \
                getattr(scene.display, "render_aa", "-")
            self.info = {
                "engine": r.engine, "compute": "GPU (GL)", "denoiser": "-",
                "denoise_device": "-", "devices": "", "samples": samples,
            }

        r.image_settings.file_format = 'PNG'           # still format for save_render
        r.dither_intensity = 0.0                        # dither noise would differ run to run
        # Date / render time / note always go into the PNG metadata; burning them into
        # the pixels (--stamp) makes renders differ by their text, so validation masks nothing.
        r.use_stamp = a["stamp"]
        r.use_stamp_date = r.use_stamp_time = r.use_stamp_render_time = True
        r.use_stamp_note = True
        r.stamp_note_text = (f"{a['label'] or 'ui_render_test'} | {perm} | "
                             f"{self.info['devices'] or eb._gl_renderer_string()}")
        return True, "ok"

    def _image_rect(self):
        """Where the render sits in the window screenshot: 'x0,y0,x1,y1', top-left origin."""
        for area in self.window.screen.areas:
            if area.type != 'IMAGE_EDITOR':
                continue
            for region in area.regions:
                if region.type == 'WINDOW':
                    v2d = region.view2d
                    x0, y0 = v2d.view_to_region(0.0, 0.0, clip=False)
                    x1, y1 = v2d.view_to_region(1.0, 1.0, clip=False)
                    h = self.window.height
                    return (f"{region.x + x0},{h - (region.y + y1)},"
                            f"{region.x + x1},{h - (region.y + y0)}")
        return ""

    def _res(self):
        r = self.scene.render
        return f"{r.resolution_x * r.resolution_percentage // 100}x" \
               f"{r.resolution_y * r.resolution_percentage // 100}"

    def _window_px(self):
        return f"{self.window.width}x{self.window.height}"

    # -- perm loop -----------------------------------------------------------
    def next_perm(self):
        if not self.queue:
            self.finish()
            return None
        self.perm = self.queue.pop(0)
        self.run = 0
        self.info = {}
        try:
            ok, reason = self.configure(self.perm)
        except Exception as e:
            ok, reason = False, f"setup error: {eb._one_line(e)}"
        if not ok:
            _log(f"{self.perm}: SKIPPED ({reason})")
            self._write_row({"status": f"SKIPPED: {reason}", "render_s": ""})
            _log(f"RESULT {self.perm} run0 SKIPPED -")
            _log(f"PERM_DONE {self.perm}")
            bpy.app.timers.register(self.next_perm, first_interval=0.1)
            return None
        _log(f"{self.perm}: {self.info['engine']} | compute {self.info['compute']} | "
             f"denoise {self.info['denoiser']} ({self.info['denoise_device']}) | "
             f"{self._res()} | samples {self.info['samples']} | "
             f"{self.info['devices'] or eb._gl_renderer_string()}")
        self.start_run()
        return None

    # -- one render ----------------------------------------------------------
    def start_run(self):
        self.run += 1
        self.done_at, self.status = None, None
        bpy.app.handlers.render_complete.append(self._on_complete)
        bpy.app.handlers.render_cancel.append(self._on_cancel)
        _log(f"{self.perm} run {self.run}/{self.args['runs']}: F12 render started")
        area = _largest_area(self.window)
        self.t0 = time.perf_counter()
        try:
            if self.args["override"]:
                with bpy.context.temp_override(window=self.window, screen=self.window.screen,
                                               area=area):
                    bpy.ops.render.render('INVOKE_DEFAULT', write_still=False)
            else:
                with bpy.context.temp_override(window=self.window):
                    bpy.ops.render.render('INVOKE_DEFAULT', write_still=False)
        except Exception as e:
            self.status, self.done_at = f"FAILED: {eb._one_line(e)}", time.perf_counter()
        bpy.app.timers.register(self._poll, first_interval=0.25)

    def _on_complete(self, *_):
        self.done_at, self.status = time.perf_counter(), "OK"

    def _on_cancel(self, *_):
        self.done_at, self.status = time.perf_counter(), "CANCELLED"

    def _poll(self):
        if self.done_at is None:
            return 0.25
        for hlist, fn in ((bpy.app.handlers.render_complete, self._on_complete),
                          (bpy.app.handlers.render_cancel, self._on_cancel)):
            if fn in hlist:
                hlist.remove(fn)
        bpy.app.timers.register(self._capture, first_interval=SETTLE_S)
        return None

    def _capture(self):
        render_s = self.done_at - self.t0
        base = os.path.join(RESULTS_DIR, f"ui_render_{STAMP}_{self.perm}_run{self.run}")
        shot = render_png = ""
        area = _largest_area(self.window)
        try:
            with bpy.context.temp_override(window=self.window, screen=self.window.screen,
                                           area=area):
                bpy.ops.screen.screenshot(filepath=base + "_screen.png")
            shot = base + "_screen.png"
        except Exception as e:
            _log(f"[warn] screenshot failed: {e}")
        img = bpy.data.images.get("Render Result")
        if self.status == "OK" and img is not None:
            try:
                img.save_render(filepath=base + "_render.png", scene=self.scene)
                render_png = base + "_render.png"
            except Exception as e:
                _log(f"[warn] saving render failed: {e}")

        _log(f"{self.perm} run {self.run}: {self.status} in {render_s:.2f}s "
             f"| window {self._window_px()}")
        self._write_row({
            "status": self.status, "render_s": f"{render_s:.3f}",
            "screenshot": os.path.relpath(shot, SCRIPT_DIR) if shot else "",
            "render_png": os.path.relpath(render_png, SCRIPT_DIR) if render_png else "",
            "image_rect": self._image_rect(),
        })
        _log(f"RESULT {self.perm} run{self.run} {self.status.split(':')[0]} {render_s:.3f}")

        if self.status == "OK" and self.run < self.args["runs"]:
            bpy.app.timers.register(self._again, first_interval=0.5)
        else:
            _log(f"PERM_DONE {self.perm}")
            bpy.app.timers.register(self.next_perm, first_interval=0.5)
        return None

    def _again(self):
        self.start_run()
        return None

    def _write_row(self, fields):
        row = {
            "session": self.args["session"], "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "host": os.environ.get("COMPUTERNAME", ""), "label": self.args["label"], "blender": bpy.app.version_string,
            "scene": eb._extract_model(), "perm": self.perm, "test_mode": self.args["test"],
            "run": self.run, "gl_renderer": eb._gl_renderer_string(),
            "resolution": self._res(), "window_px": self._window_px(),
            "screen_px": _screen_px(), **self.info, **fields,
        }
        write_csv_row(row)

    def finish(self):
        self.prefs.render_display_type = self.saved_display_type
        _log(f"ALL_DONE -> {CSV_PATH}")
        if not self.args["keep_open"]:
            bpy.app.timers.register(_quit, first_interval=0.5)


def write_csv_row(row):
    """Append to results/ui_render.csv; a file with an older header is rotated aside."""
    if os.path.isfile(CSV_PATH):
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), [])
        if header != CSV_COLUMNS:
            os.replace(CSV_PATH, CSV_PATH.replace(".csv", f"_old_{STAMP}.csv"))
    new = not os.path.isfile(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    if bpy.app.background:
        print("[ui-render] needs the Blender UI -- run without -b (see the docstring)")
        return
    os.makedirs(RESULTS_DIR, exist_ok=True)
    try:
        args = _parse_args()
    except ValueError as e:
        _log(f"ERROR: {e}")
        bpy.app.timers.register(_quit, first_interval=0.5)
        return
    test = UIRenderTest(args)
    bpy.app.timers.register(test.next_perm, first_interval=START_DELAY_S)


if __name__ == "__main__":
    main()
