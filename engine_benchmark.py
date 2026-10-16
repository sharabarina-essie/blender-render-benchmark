"""
engine_benchmark.py
===================

SCOPE
-----
This measures OFFLINE (F12) render time per engine/backend.  It does
NOT test the viewport/CUDA-GL interop path or the video-encode readback path;

RELATIONSHIP TO qa_standardize_render_settings.py
-------------------------------------------------
Division of labour (do not overlap them):

  * qa_standardize_render_settings.py = ENVIRONMENT SANITIZER.
    Run it ONCE to put a QA box in a known-clean state: it resets to factory
    defaults then applies the INVARIANTS that must be identical across every
    test regardless of engine -- color management/view (AgX, sRGB, exposure,
    gamma), determinism (fixed seed, animated-seed off, simplify off), output,
    compositor device. It owns machine reproducibility.

  * engine_benchmark.py (this file) = MEASUREMENT HARNESS.
    It does NOT reset the machine and does NOT re-litigate color management.
    It inherits the clean state, IMPORTS the invariants from the standardizer
    (single source of truth), then applies the CUSTOMER-WORKFLOW render values
    (PROFILE) and times the engines. It owns only what varies for the run:
    engine, device/backend, and the customer sampling profile.

So: standardizer sets the invariants -> benchmark sets the workflow variables
and measures. If the standardizer file is alongside this one, its CONFIG is
imported and the color/determinism settings cannot drift between the two.

DECISIONS BAKED IN 
-----------------------------------
  * Unavailable backend  -> SKIP that config, log why, continue the others.
  * Engine race sampling -> FIXED samples, adaptive sampling OFF (apples-to-apples).
  * Denoising            -> measure BOTH raw and denoised, reported separately.
  * Backend confirmation -> a device of the requested type must be present AND
    active, else the config is skipped -- a run is never mislabeled as OptiX/
    CUDA while silently falling back to CPU.

HOW TO RUN
----------
  Cleanest (one process per engine, no thermal/cache carryover):
    blender --enable-autoexec -b scene.blend --python engine_benchmark.py -- --engine OPTIX
    blender --enable-autoexec -b scene.blend --python engine_benchmark.py -- --engine CUDA
    blender --enable-autoexec -b scene.blend --python engine_benchmark.py -- --engine EEVEE

  Cycles Boost add-on (Blender 4.5, must be installed + enabled as an extension):
    ... --engine BOOST_TCP_OPTIX      render_on=TCP: renderer on THIS machine behind a
                                      local socket (the protocol control)
    ... --engine BOOST_REMOTE_OPTIX   render_on=BOOST: renderer reached through the
                                      Boost tunnel; needs Blender running under Boost
    (also BOOST_TCP_CUDA / BOOST_REMOTE_CUDA; BOOST = all four in one process)
    It reads the scene's own Cycles settings, so the same profile applies. Every
    render opens a fresh session (no persistent data), so warm ~= cold minus
    kernel compile. --gpu is ignored: the add-on picks its device by type only.

  All engines in one process (adds cooldown between them; less isolated):
    blender --enable-autoexec -b scene.blend --python engine_benchmark.py

  NOTE: --enable-autoexec lets the .blend's own Python run (expression drivers, e.g.
  the camera drivers in the version splash scenes). It is OFF by default in -b/-P,
  so without it those scenes render wrong ("restricted access disallows name
  'scene'"). The run_*_suite.py runners add it automatically.

  Options:
    -- --test              FAST harness smoke check: 1 warm iteration + slashed
                           samples (Cycles 64 / Eevee 16) + 25% resolution
                           (480x270). Renders finish in seconds; every table is
                           stamped NOT REPRESENTATIVE.
    -- --warm 5            warm iterations to median over (overrides --test's 1).
    DEVICE SELECTION (default, no flag): PRESENCE-AWARE. The harness probes which
    Cycles devices are physically reachable right now (enable-alone +
    has_active_device), enables ONLY those, and ignores the userpref 'ghost' list
    Blender never prunes -- devices it remembers but that are not present now. So
    whatever GPU(s) the machine actually has right now is what gets used; no flag
    is needed.
    -- --gpu "RTX 5080"    OPTIONAL manual override -- narrow to PRESENT devices
                           whose name CONTAINS this substring (case-insensitive).
                           Use only to force a specific card when several are
                           present at once (e.g. a multi-GPU machine).
    -- --compositor CPU|GPU
                           pin scene.render.compositor_device for the engine race
                           (default CPU). The compositor runs inside every F12
                           render when the scene composites, so leaving this at the
                           scene's shipped value (often GPU) silently taxes the numbers.
    -- --compositor-ab     run the COMPOSITOR CPU-vs-GPU A/B instead of the engine
                           race: same engine, times render+compositor with the
                           compositor on CPU then GPU and reports the delta (the
                           GPU-compositor cost vs CPU). Combine with --engine to pick
                           the engine; skipped if the scene has no compositor tree.
    -- --save-renders      write the COLD frame's render to disk as a real image
                           file per engine, for visual A/B comparison later (across
                           engines / versions / models). This is an
                           extra, UNTIMED render pass that runs immediately after
                           the timed cold measurement (same cached kernels/BVH,
                           so it doesn't pay compile cost again) -- it never
                           affects any timed number. Does not touch file_format or
                           color management (those stay whatever the scene/
                           standardizer already has); only sets a distinguishing
                           filename so different engines/models don't overwrite
                           each other, then restores the scene's own filepath.
    -- --render-dir DIR    where --save-renders writes files (default: a
                           'cold_renders' folder next to this script).
    -- --render-tag STR    filename base for saved renders (the suite passes
                           "<version>_<model>" so renders match their log/xlsx
                           row and don't collide across versions). Omit for a
                           direct run -> base is "<scene-file>__<blender-build>".
"""

import bpy
import sys
import os
import re
import time
import statistics


def _script_dir():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()


# ---------------------------------------------------------------------------
# Import invariants from the standardizer (single source of truth).
# Falls back to a local copy (with a warning) if run standalone.
# ---------------------------------------------------------------------------
def _load_invariants():
    script_dir = _script_dir()
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    try:
        import qa_standardize_render_settings as qa
        c = qa.CONFIG
        return {
            "display_device": c["display_device"],
            "view_transform": c["view_transform"],
            "look": c["look"],
            "exposure": c["exposure"],
            "gamma": c["gamma"],
            "sequencer_colorspace": c["sequencer_colorspace"],
            "seed": c["seed"],
            "animated_seed": c["animated_seed"],
            "use_simplify": c["use_simplify"],
        }, True
    except Exception as e:
        print(f"[invariants] Could not import qa_standardize_render_settings "
              f"({e}); using local fallback. Keep the two files in sync!")
        return {
            "display_device": "sRGB", "view_transform": "AgX", "look": "None",
            "exposure": 0.0, "gamma": 1.0, "sequencer_colorspace": "sRGB",
            "seed": 0, "animated_seed": False, "use_simplify": False,
        }, False

INVARIANTS, INVARIANTS_IMPORTED = _load_invariants()

# ---------------------------------------------------------------------------
# CUSTOMER WORKFLOW PROFILE  -- every render uses these (the point of the run).
# This consolidates the intent of the old load_cycles(): real customer target
# resolution, sample budget, and the glass/glTF-safe bounce limits.
# ---------------------------------------------------------------------------
PROFILE = {
    "name": "ARCHVIZ_PRODUCTION",
    "res_x": 1920,
    "res_y": 1080,
    "res_percentage": 100,
    "cycles_samples": 512,          # fixed; adaptive is forced OFF
    "eevee_samples": 64,
    "denoiser": "OPENIMAGEDENOISE", # vendor-neutral; consistent across QA boxes
    # Customer light-path workflow (glass/transmission-safe for imported assets)
    "max_bounces": 16,
    "diffuse_bounces": 4,
    "glossy_bounces": 4,
    "transmission_bounces": 12,
    "volume_bounces": 0,
    "transparent_max_bounces": 16,
    "use_motion_blur": False,
    "eevee_raytracing": True,       # customer expects RT reflections/AO in Eevee
    "workbench_aa": "8",            # Workbench final-render AA (pinned for apples-to-apples)
}

# ---------------------------------------------------------------------------
# Benchmark controls
# ---------------------------------------------------------------------------
WARM_RUNS = 3          # warm iterations to take median + spread over
COLD_FRAME = 1
WARM_FRAME = 2
COOLDOWN_SEC = 5.0     # pause between engines in single-process mode

# Set from CLI in main(). GPU_SUBSTR (from --gpu) optionally narrows the present
# devices to those whose name contains it; None = use all present devices.
# COMPOSITOR_DEVICE pins scene.render.compositor_device for the engine race.
GPU_SUBSTR = None
COMPOSITOR_DEVICE = "CPU"                             # safe default

# --save-renders: write the cold frame to disk per engine (see module docstring).
SAVE_RENDERS = False
RENDER_DIR = None   # resolved in main() -> defaults to '<script_dir>/cold_renders'
RENDER_TAG = ""     # --render-tag: filename base (the suite passes "<version>_<model>")

# TEST MODE (--test): fast smoke check of the harness plumbing. Forces 1 warm
# iteration and slashes samples so renders finish in seconds. Numbers are NOT
# representative -- every table is stamped so they can't be mistaken for real.
TEST_MODE = False
TEST_CYCLES_SAMPLES = 64
TEST_EEVEE_SAMPLES = 16
TEST_RES_PERCENTAGE = 25   # 25% of profile res (1920x1080 -> 480x270): fast, still 16:9
TEST_WORKBENCH_AA = "OFF"  # Workbench anti-aliasing off in test mode

# Cycles Boost add-on: its own RenderEngine, but it renders from scene.cycles, so
# it gets the Cycles profile and the raw/denoised split like stock Cycles.
BOOST_ENGINE_ID = "CYCLES_BOOST"
CYCLES_FAMILY = ("CYCLES", BOOST_ENGINE_ID)

# ---------------------------------------------------------------------------
# Setting application
# ---------------------------------------------------------------------------
def _try_set(obj, attr, value, fallbacks=()):
    """Set obj.attr = value; if the enum value is invalid on THIS Blender version,
    try each fallback, else warn and leave the default. Keeps the harness runnable
    across versions -- e.g. the 'AgX' view transform does not exist before 4.0
    (3.x only has Filmic/Standard), so it falls back instead of crashing."""
    for candidate in (value, *fallbacks):
        try:
            setattr(obj, attr, candidate)
            if candidate != value:
                print(f"    [invariants] {attr}='{value}' unavailable here; using '{candidate}'")
            return candidate
        except (TypeError, ValueError):
            continue
    print(f"    [invariants] could not set {attr}='{value}' (no fallback); leaving default")
    return None


def apply_invariants(scene):
    """Color management + determinism, engine-independent, applied once.
    Defensive across Blender versions: color-management enums differ between
    releases, so an unavailable value falls back rather than aborting the run."""
    _try_set(scene.display_settings, "display_device", INVARIANTS["display_device"], ("sRGB", "None"))
    _try_set(scene.view_settings, "view_transform", INVARIANTS["view_transform"], ("Filmic", "Standard"))
    _try_set(scene.view_settings, "look", INVARIANTS["look"], ("None",))
    try:
        scene.view_settings.exposure = INVARIANTS["exposure"]
        scene.view_settings.gamma = INVARIANTS["gamma"]
    except (AttributeError, TypeError, ValueError) as e:
        print(f"    [invariants] exposure/gamma not set: {e}")
    if hasattr(scene, "sequencer_colorspace_settings"):
        _try_set(scene.sequencer_colorspace_settings, "name",
                 INVARIANTS["sequencer_colorspace"], ("sRGB",))
    try:
        scene.render.use_simplify = INVARIANTS["use_simplify"]
    except (AttributeError, TypeError):
        pass
    cy = getattr(scene, "cycles", None)
    if cy is not None:
        try:
            cy.seed = INVARIANTS["seed"]
            cy.use_animated_seed = INVARIANTS["animated_seed"]
        except (AttributeError, TypeError):
            pass


def apply_customer_profile(scene, engine):
    """Customer-workflow render values applied to EVERY benchmark render."""
    r = scene.render
    r.resolution_x = PROFILE["res_x"]
    r.resolution_y = PROFILE["res_y"]
    r.resolution_percentage = TEST_RES_PERCENTAGE if TEST_MODE else PROFILE["res_percentage"]
    r.use_motion_blur = PROFILE["use_motion_blur"]
    r.use_persistent_data = True   # required for a meaningful warm run

    if engine in CYCLES_FAMILY:
        cy = scene.cycles
        # Fixed samples, adaptive OFF -> honest apples-to-apples engine race
        cy.use_adaptive_sampling = False
        cy.samples = TEST_CYCLES_SAMPLES if TEST_MODE else PROFILE["cycles_samples"]
        cy.denoiser = PROFILE["denoiser"]
        cy.max_bounces = PROFILE["max_bounces"]
        cy.diffuse_bounces = PROFILE["diffuse_bounces"]
        cy.glossy_bounces = PROFILE["glossy_bounces"]
        cy.transmission_bounces = PROFILE["transmission_bounces"]
        cy.volume_bounces = PROFILE["volume_bounces"]
        cy.transparent_max_bounces = PROFILE["transparent_max_bounces"]
    elif engine == "BLENDER_EEVEE_NEXT":
        ev = scene.eevee
        if hasattr(ev, "taa_render_samples"):
            ev.taa_render_samples = TEST_EEVEE_SAMPLES if TEST_MODE else PROFILE["eevee_samples"]
        if hasattr(ev, "use_raytracing"):
            ev.use_raytracing = PROFILE["eevee_raytracing"]
    elif engine == "BLENDER_WORKBENCH":
        # Workbench is a solid-shading GL rasterizer: no samples/bounces/denoise.
        # Its only timing-relevant knob is final-render anti-aliasing -- pin it so
        # the number is stable (this run is a clean GL-forwarding baseline).
        d = getattr(scene, "display", None)
        if d is not None and hasattr(d, "render_aa"):
            d.render_aa = TEST_WORKBENCH_AA if TEST_MODE else PROFILE["workbench_aa"]


def apply_scene_controls(scene):
    """Engine-independent controls the benchmark owns beyond invariants+profile.
    These are settings the standardizer would reset but that this file must also
    guarantee so a single run is honest on an UN-standardized model:
      * clear any stray render region/border (would time only a crop);
      * PIN compositor_device -- the compositor runs inside every F12 render when
        the scene composites, and models often ship with GPU. Default CPU;
        override with --compositor / --compositor-ab."""
    r = scene.render
    r.use_border = False
    r.use_crop_to_border = False
    try:
        r.compositor_device = COMPOSITOR_DEVICE
    except (TypeError, AttributeError) as e:
        print(f"    [warn] could not set compositor_device={COMPOSITOR_DEVICE}: {e}")


def _compositor_node_tree(scene):
    """The compositor node tree, across API versions:
      * Blender 5.0+ : scene.compositing_node_group (use_nodes/node_tree removed)
      * Blender <=4.x: scene.node_tree when scene.use_nodes is on
    Returns the node tree or None. Checks the 5.0+ attribute FIRST so we never
    touch the deprecated use_nodes/node_tree on newer builds."""
    if hasattr(scene, "compositing_node_group"):
        return scene.compositing_node_group
    if getattr(scene, "use_nodes", False):
        return getattr(scene, "node_tree", None)
    return None


def has_compositor(scene):
    """True only if the scene has a compositor that will actually RUN and do
    something beyond the default Render Layers -> Composite passthrough."""
    if not getattr(scene.render, "use_compositing", True):
        return False
    nt = _compositor_node_tree(scene)
    if nt is None:
        return False
    meaningful = [n for n in nt.nodes
                  if n.type not in {'R_LAYERS', 'COMPOSITE', 'REROUTE'}]
    return len(meaningful) > 0


# ---------------------------------------------------------------------------
# Backend selection + CONFIRMATION (never silently mismeasure)
# ---------------------------------------------------------------------------
def _get_devices(cprefs):
    try:
        cprefs.get_devices()
    except TypeError:
        cprefs.get_devices(bpy.context)


def _present_device_ids(cprefs, backend_type):
    """IDs of backend_type devices that are PHYSICALLY PRESENT right now,
    determined empirically rather than by name: enable each device ALONE and ask
    Cycles has_active_device(). Confirmed reliable -- a non-present ghost returns
    False even though it sits in the persisted list with use=True.

    WHY THIS IS NEEDED: Blender remembers every Cycles device it has ever seen in
    userpref.blend (keyed by name+PCI = the .id) and never prunes ones that later
    vanish, so GPUs the machine no longer has still appear, all flagged use=True.
    The live has_active_device() probe is the only trustworthy present-vs-ghost test.

    Returns a set of .id strings (not device objects): bpy hands back a fresh
    Python wrapper on every collection access, so object identity is unstable --
    the stable .id (e.g. 'CUDA_NVIDIA GeForce RTX 4070 Laptop GPU_0000:64:00') is
    the only safe key. Restores every device's original use flag before returning."""
    type_ids = [d.id for d in cprefs.devices if d.type == backend_type]
    if not hasattr(cprefs, "has_active_device"):
        return set(type_ids)                  # can't probe on this build; assume all
    saved = {d.id: d.use for d in cprefs.devices}
    present = set()
    for tid in type_ids:
        for d in cprefs.devices:
            d.use = (d.type == backend_type and d.id == tid)   # isolate this one id
        try:
            if cprefs.has_active_device():
                present.add(tid)
        except Exception as exc:
            print(
                f"Warning: has_active_device() probe failed for backend={backend_type}, device_id={tid}: {exc}",
                file=sys.stderr,
            )
    for d in cprefs.devices:
        if d.id in saved:
            d.use = saved[d.id]               # restore caller's selection
    return present


def configure_backend(backend_type):
    """
    Select a Cycles compute backend and CONFIRM a device is present and active.
    Enables ONLY devices that are PHYSICALLY PRESENT right now (probed via
    has_active_device), so a run never fans out across the userpref 'ghost' list
    of remembered-but-unreachable GPUs -- it uses whatever the machine actually has
    now. --gpu optionally narrows the present set to a named card. Returns
    (ok, device_names, reason); ok=False => skip.
    """
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'

    addon = bpy.context.preferences.addons.get('cycles')
    if addon is None:
        return False, [], "Cycles add-on not enabled"
    cprefs = addon.preferences

    # 1) Is this backend even a valid compute type on this build?
    try:
        cprefs.compute_device_type = backend_type
    except TypeError:
        return False, [], f"'{backend_type}' not a valid device type on this build"

    # 2) Enable ONLY physically-present devices (optionally narrowed by --gpu).
    #    Ghosts get use=False, so the renderer only ever touches real hardware and
    #    the reported GPU list is honest. CPU is never force-enabled.
    _get_devices(cprefs)
    type_devs = [d for d in cprefs.devices if d.type == backend_type]
    type_present = len(type_devs)
    if type_present == 0:
        return False, [], f"no {backend_type} device present on this machine"

    present_ids = _present_device_ids(cprefs, backend_type)
    matched = []
    for dev in type_devs:
        sel = dev.id in present_ids
        if GPU_SUBSTR:
            sel = sel and (GPU_SUBSTR.lower() in dev.name.lower())
        dev.use = sel
        if sel:
            matched.append(dev.name)
    if not matched:
        if GPU_SUBSTR:
            return False, [], (f"no PRESENT {backend_type} device matches --gpu "
                               f"'{GPU_SUBSTR}'")
        return False, [], (f"{type_present} {backend_type} device(s) in prefs but none "
                           f"physically present now (only ghosts?)")

    # 3) Confirm Cycles actually has an active device for GPU rendering.
    if hasattr(cprefs, "has_active_device") and not cprefs.has_active_device():
        return False, [], f"{backend_type} present but no active device (driver/kernel?)"

    return True, matched, "ok"


def _boost_local_devices(backend_type):
    """Names of backend_type devices the Cycles Boost core sees in THIS process.
    Under a Boost session that is whatever GPU Boost is presenting."""
    from cycles_boost_core import _cycles_boost
    names = []
    for d in _cycles_boost.devices():
        if d["type"] == backend_type and d["description"] not in names:
            names.append(d["description"])
    return names


def _boost_device_labels(settings):
    """Device label(s) for a configured Cycles Boost scene. In BOOST mode the GPU
    is on the far side of the tunnel and this process cannot name it."""
    if settings.render_on == "BOOST":
        return [f"remote via Boost plugin ({settings.device})"]
    try:
        names = _boost_local_devices(settings.device)
    except Exception:
        names = []
    return [f"{n} ({settings.device}, Boost TCP)" for n in names] or \
           [f"unknown GPU ({settings.device}, Boost TCP)"]


def configure_boost(backend_type, render_on):
    """Select the Cycles Boost engine, its device type and where it renders
    (render_on = 'TCP' | 'BOOST'). Returns (ok, device_names, reason); ok=False => skip."""
    scene = bpy.context.scene
    if not hasattr(scene, "cycles_boost"):
        return False, [], "Cycles Boost add-on not enabled"
    try:
        scene.render.engine = BOOST_ENGINE_ID
    except TypeError:
        return False, [], "Cycles Boost engine not registered"
    try:
        _boost_local_devices(backend_type)
    except Exception as e:
        return False, [], f"Cycles Boost core not loadable: {_one_line(e)}"

    settings = scene.cycles_boost
    settings.render_on = render_on
    try:
        settings.device = backend_type
    except TypeError:
        return False, [], f"no {backend_type} device visible to Cycles Boost"
    if GPU_SUBSTR:
        print(f"    [warn] --gpu '{GPU_SUBSTR}' ignored: Cycles Boost picks its device by type only")
    return True, _boost_device_labels(settings), "ok"


def _configure_setup(setup):
    """Point the scene at a setup's engine/device. Returns (ok, devices, detail)."""
    engine, backend = setup["engine"], setup["backend"]
    if engine == "CYCLES":
        return configure_backend(backend)
    if engine == BOOST_ENGINE_ID:
        return configure_boost(backend, setup["render_on"])
    eng_id = _resolve_engine_id(engine, bpy.context.scene)   # Eevee id varies by version
    bpy.context.scene.render.engine = eng_id                 # EEVEE (Next/legacy) or Workbench
    return True, [], f"engine={eng_id}"


def _one_line(exc):
    """An exception as one table-safe line: parentheses would break the parser's
    '(reason)' match, and newlines the row."""
    text = re.sub(r"\s+", " ", str(exc)).strip()
    text = re.sub(r"^Error:\s*", "", text)
    return text.replace("(", "[").replace(")", "]") or type(exc).__name__


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def _timed_render():
    t0 = time.perf_counter()
    bpy.ops.render.render(write_still=False)   # no disk write -> no I/O in timing
    return time.perf_counter() - t0


def _warm_series(n):
    times = []
    for _ in range(n):
        times.append(_timed_render())
    return times


def _gl_renderer_string():
    """Live GL/Vulkan renderer string = the GPU EEVEE actually rasterizes on.
    EEVEE ignores Cycles device prefs entirely and renders on whatever GPU backs
    Blender's display/GL context, so this is the only way to report EEVEE's device
    (and to verify it matches the Cycles compute GPU for a valid cross-engine A/B).
    Needs a live GL context -- present during/after an EEVEE render, but empty in a
    headless Cycles-only process. E.g. 'NVIDIA GeForce RTX 4070 Laptop GPU/PCIe/SSE2'."""
    try:
        import gpu
        return gpu.platform.renderer_get() or ""
    except Exception:
        return ""


SAVE_RENDER_FORMAT = "PNG"   # deterministic, viewable, and -- critically -- a still-image
                             # format; some scenes ship image_settings.file_format set to
                             # a movie container (FFMPEG/AVI_*), and write_still=True always
                             # fails against those ("Cannot write a single file with an
                             # animation format selected"), so this pass always forces PNG
                             # regardless of the scene's own configured format.


def _save_cold_render(scene, setup_name):
    """Untimed extra render pass (see --save-renders in the module docstring).
    Writes the cold frame just measured by _timed_render() to a distinguishable
    PNG. Must run BEFORE frame_set(WARM_FRAME) so it captures the cold frame,
    not a warm one. Restores scene.render.filepath/use_file_extension/
    image_settings.file_format afterward so nothing about the scene's own
    (standardizer-owned) output config is left mutated."""
    engine_tag = re.sub(r"[^A-Za-z0-9]+", "_", setup_name).strip("_")
    mode_tag = "test" if TEST_MODE else "full"
    ts = time.strftime("%Y%m%d_%H%M%S")
    # Base carries the version so renders of the same scene on different Blender
    # versions never collide and are trivially comparable. The suite passes
    # --render-tag "<version>_<model>" (matching the log/xlsx naming); a direct run
    # (no tag) falls back to "<scene-file>__<blender-build>".
    base = RENDER_TAG or f"{_extract_model()}__{_extract_version()}"
    filename = f"{base}__{engine_tag}__{_res_str()}__{mode_tag}__{ts}.png"
    out_path = os.path.join(RENDER_DIR, filename)

    prev_filepath = scene.render.filepath
    prev_use_ext = scene.render.use_file_extension
    prev_format = scene.render.image_settings.file_format
    scene.render.filepath = out_path
    scene.render.use_file_extension = False   # write exactly out_path, no auto frame/ext suffix
    scene.render.image_settings.file_format = SAVE_RENDER_FORMAT
    try:
        bpy.ops.render.render(write_still=True)
        print(f"    Saved cold render -> {out_path}")
    except Exception as e:
        print(f"    [warn] --save-renders failed: {e}")
        out_path = None
    finally:
        scene.render.filepath = prev_filepath
        scene.render.use_file_extension = prev_use_ext
        scene.render.image_settings.file_format = prev_format
    return out_path


def _eevee_id(scene=None):
    """The Eevee render-engine id valid on THIS Blender build: 'BLENDER_EEVEE_NEXT'
    on 4.2+ (Eevee Next), 'BLENDER_EEVEE' on older releases. Detected from the
    build's own engine enum, not the version number, so it survives future renames."""
    try:
        rs = scene.render if scene is not None else None
        prop = (rs.bl_rna.properties["engine"] if rs is not None
                else bpy.types.RenderSettings.bl_rna.properties["engine"])
        ids = {e.identifier for e in prop.enum_items}
    except Exception:
        ids = set()
    for candidate in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        if candidate in ids:
            return candidate
    return "BLENDER_EEVEE_NEXT"   # fall back to the modern id if detection fails


def _resolve_engine_id(engine, scene=None):
    """Map a setup's logical engine token to the id this build accepts. Only Eevee
    needs remapping (BLENDER_EEVEE_NEXT <-> BLENDER_EEVEE across versions); Cycles
    and Workbench ids are stable."""
    if engine in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        return _eevee_id(scene)
    return engine


def benchmark_engine(setup):
    """Run one engine config: cold + warm(raw) + warm(denoised, Cycles only)."""
    scene = bpy.context.scene
    name, engine, backend = setup["name"], setup["engine"], setup["backend"]
    print(f"\n>>> {name} <<<")

    is_cycles = engine in CYCLES_FAMILY
    ok, devices, detail = _configure_setup(setup)
    if not ok:
        print(f"    SKIPPED: {detail}")
        return {"status": "SKIPPED", "reason": detail}
    if is_cycles:
        print(f"    Active {backend} device(s): {', '.join(devices)}")
    else:
        print(f"    {name} (GPU rasterizer, {detail})")

    apply_customer_profile(scene, engine)

    # A render that errors (e.g. Cycles Boost with no renderer to reach) raises
    # from bpy.ops; record it as this engine's row instead of losing the table.
    try:
        return _measure_engine(scene, name, is_cycles, devices)
    except RuntimeError as e:
        reason = _one_line(e)
        print(f"    FAILED: {reason}")
        return {"status": "FAILED", "reason": reason}


def _measure_engine(scene, name, is_cycles, devices):
    # --- RAW (denoise off) ---
    if is_cycles:
        scene.cycles.use_denoising = False

    scene.frame_set(COLD_FRAME)
    cold_raw = _timed_render()                 # cold: parse + BVH/kernel compile
    if not is_cycles:                          # EEVEE/Workbench run on the GL-context GPU,
        gl = _gl_renderer_string()             # not a Cycles device -- record which one
        if gl:
            devices = [gl]
            print(f"    {name} GL renderer: {gl}")
        else:
            print(f"    [warn] {name} GL renderer unavailable (no readable GL context)")
    cold_render_path = _save_cold_render(scene, name) if SAVE_RENDERS else None
    scene.frame_set(WARM_FRAME)
    warm_raw = _warm_series(WARM_RUNS)          # warm: cached steady-state

    result = {
        "status": "OK",
        "devices": devices,
        "cold_raw": cold_raw,
        "cold_render_path": cold_render_path,
        "warm_raw_med": statistics.median(warm_raw),
        "warm_raw_min": min(warm_raw),
        "warm_raw_max": max(warm_raw),
        "warm_den_med": None,
    }

    # --- DENOISED (Cycles only; Eevee has no equivalent render denoise) ---
    if is_cycles:
        scene.cycles.use_denoising = True
        warm_den = _warm_series(WARM_RUNS)      # cold already paid; measure warm
        result["warm_den_med"] = statistics.median(warm_den)
        result["warm_den_min"] = min(warm_den)
        result["warm_den_max"] = max(warm_den)

    return result


# ---------------------------------------------------------------------------
# Compositor CPU-vs-GPU A/B (isolates the GPU-compositor cost vs CPU)
# ---------------------------------------------------------------------------
def _warm_with_compositor(scene, device):
    """Switch compositor device, pay a one-time re-init render (not timed), then
    return N warm samples of render+compositor at that device."""
    scene.render.compositor_device = device
    _timed_render()                    # throwaway: compositor backend init
    return _warm_series(WARM_RUNS)


def benchmark_compositor(setup):
    """Same engine, two compositor devices. Delta = the compositor's contribution
    difference (CPU vs a forwarded GPU-compute path). Requires a real compositor."""
    scene = bpy.context.scene
    name, engine, backend = setup["name"], setup["engine"], setup["backend"]
    print(f"\n>>> COMPOSITOR A/B: {name} <<<")

    if not has_compositor(scene):
        print("    SKIPPED: no meaningful compositor tree (nothing to forward)")
        return {"status": "SKIPPED", "reason": "no compositor tree"}

    # The CPU/GPU compositor A/B needs scene.render.compositor_device (Blender 4.0+).
    # Older builds have no GPU compositor at all -> nothing to compare; skip cleanly.
    if not hasattr(scene.render, "compositor_device"):
        print("    SKIPPED: compositor_device not available (Blender < 4.0)")
        return {"status": "SKIPPED", "reason": "no compositor_device (<4.0)"}

    is_cycles = engine in CYCLES_FAMILY
    ok, devices, detail = _configure_setup(setup)
    if not ok:
        print(f"    SKIPPED: {detail}")
        return {"status": "SKIPPED", "reason": detail}
    if is_cycles:
        print(f"    Active {backend} device(s): {', '.join(devices)}")
    else:
        print(f"    {name} (GPU rasterizer, {detail})")

    apply_customer_profile(scene, engine)
    if is_cycles:
        scene.cycles.use_denoising = False   # isolate the compositor, not denoise

    scene.frame_set(WARM_FRAME)
    try:
        _timed_render()                      # reach render steady-state first
        if not is_cycles:                    # EEVEE: record the actual GL GPU used
            gl = _gl_renderer_string()
            if gl:
                devices = [gl]
        cpu = _warm_with_compositor(scene, "CPU")
    except RuntimeError as e:
        reason = _one_line(e)
        print(f"    FAILED: {reason}")
        return {"status": "FAILED", "reason": reason}

    result = {"status": "OK", "devices": devices}

    # CPU compositor
    result.update(cpu_med=statistics.median(cpu), cpu_min=min(cpu), cpu_max=max(cpu))
    print(f"    CPU compositor: {statistics.median(cpu):.3f}s (median)")

    # GPU compositor -- may be unsupported headless / on this build; that itself
    # is a finding, so catch it rather than aborting the whole run.
    try:
        gpu = _warm_with_compositor(scene, "GPU")
        result.update(gpu_med=statistics.median(gpu), gpu_min=min(gpu),
                      gpu_max=max(gpu), gpu_error=None)
        print(f"    GPU compositor: {statistics.median(gpu):.3f}s (median)")
    except Exception as e:
        result.update(gpu_med=None, gpu_error=str(e))
        print(f"    GPU compositor FAILED: {e}")

    # Restore the pinned default so nothing leaks into a later engine's timing.
    scene.render.compositor_device = COMPOSITOR_DEVICE
    return result


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
ALL_SETUPS = [
    {"name": "Cycles (CUDA)",  "engine": "CYCLES", "backend": "CUDA"},
    {"name": "Cycles (OptiX)", "engine": "CYCLES", "backend": "OPTIX"},
    {"name": "EEVEE Next",     "engine": "BLENDER_EEVEE_NEXT", "backend": None},
    {"name": "Workbench",      "engine": "BLENDER_WORKBENCH",  "backend": None},
]

_ENGINE_ALIASES = {
    "CUDA":  [{"name": "Cycles (CUDA)",  "engine": "CYCLES", "backend": "CUDA"}],
    "OPTIX": [{"name": "Cycles (OptiX)", "engine": "CYCLES", "backend": "OPTIX"}],
    "EEVEE": [{"name": "EEVEE Next", "engine": "BLENDER_EEVEE_NEXT", "backend": None}],
    "WORKBENCH": [{"name": "Workbench", "engine": "BLENDER_WORKBENCH", "backend": None}],
    "WB":  [{"name": "Workbench", "engine": "BLENDER_WORKBENCH", "backend": None}],
}

# Cycles Boost: render_on TCP = local renderer behind a socket (control),
# BOOST = renderer through the Boost tunnel.
BOOST_SETUPS = {
    "BOOST_TCP_OPTIX":    {"name": "Boost TCP (OptiX)",    "engine": BOOST_ENGINE_ID,
                           "backend": "OPTIX", "render_on": "TCP"},
    "BOOST_TCP_CUDA":     {"name": "Boost TCP (CUDA)",     "engine": BOOST_ENGINE_ID,
                           "backend": "CUDA",  "render_on": "TCP"},
    "BOOST_REMOTE_OPTIX": {"name": "Boost Remote (OptiX)", "engine": BOOST_ENGINE_ID,
                           "backend": "OPTIX", "render_on": "BOOST"},
    "BOOST_REMOTE_CUDA":  {"name": "Boost Remote (CUDA)",  "engine": BOOST_ENGINE_ID,
                           "backend": "CUDA",  "render_on": "BOOST"},
}
for _token, _setup in BOOST_SETUPS.items():
    _ENGINE_ALIASES[_token] = [_setup]
_ENGINE_ALIASES["BOOST"] = list(BOOST_SETUPS.values())


_USAGE = """\
engine_benchmark.py -- OFFLINE per-engine render benchmark (runs INSIDE Blender).

  Invoke:  blender -b <scene.blend> -P engine_benchmark.py -- [options]
           (the '--' is required; everything after it is passed to this script)

  Options:
    --engine CUDA|OPTIX|EEVEE|WORKBENCH
                                run ONE engine (omit = race them all, one process).
                                Workbench is a solid-shading GL rasterizer -- a
                                diagnostic GL-forwarding baseline, not a production
                                render path.
    --engine BOOST_TCP_OPTIX|BOOST_TCP_CUDA|BOOST_REMOTE_OPTIX|BOOST_REMOTE_CUDA|BOOST
                                Cycles Boost add-on (Blender 4.5). TCP = renderer on
                                this machine behind a socket; REMOTE = through the
                                Boost tunnel (Blender must run under Boost).
    --test                      fast smoke check: warm=1, 64/16 samples, 25% res
                                (480x270). Every table stamped NOT REPRESENTATIVE.
    --warm N                    warm iterations to median over (overrides --test's 1)
    --gpu "SUBSTR"              OPTIONAL. Default: every
                                device Blender can INITIALIZE -- i.e. that yields a
                                live CUDA/OptiX context (has_active_device probe) --
                                which drops the stale userpref 'ghost' entries
                                automatically.
    --compositor CPU|GPU        pin scene.render.compositor_device (default CPU)
    --compositor-ab             run the compositor CPU-vs-GPU A/B instead of the race
    --save-renders              write each engine's COLD frame to disk (untimed pass)
    --render-dir DIR            where --save-renders writes (default: ./cold_renders)
    -h, --help                  show this help and exit

  Batch many models into a spreadsheet (plain python, NOT inside Blender):
    python run_benchmark_suite.py  --help     (engine-race suite)
    python run_compositor_suite.py --help     (compositor A/B suite)
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

    engine = _val("--engine")
    warm = WARM_RUNS
    w = _val("--warm")
    if w is not None:
        try:
            warm = int(w)
        except ValueError:
            # Ignore invalid --warm values and keep the default WARM_RUNS.
            pass
    return {
        "engine": engine.upper() if engine else None,
        "warm": warm,
        "warm_explicit": "--warm" in argv,
        "gpu": _val("--gpu"),
        "compositor": (_val("--compositor", "CPU") or "CPU").upper(),
        "compositor_ab": "--compositor-ab" in argv,
        "test": "--test" in argv,
        "save_renders": "--save-renders" in argv,
        "render_dir": _val("--render-dir"),
        "render_tag": _val("--render-tag"),
    }


def _extract_version():
    """Blender version, e.g. '4.5.3'."""
    try:
        return bpy.app.version_string.split()[0]  # strip a trailing ' LTS' etc.
    except Exception:
        return "?"


def _extract_gpu():
    """ACTIVE GPU(s) for this run: only devices PHYSICALLY PRESENT right now,
    probed via has_active_device (enable-alone) -- NOT the persisted catalog.
    Blender's device list can be full of remembered GPUs it never prunes (cards the
    machine no longer has); reporting all of them would be misleading, so we report
    only what is actually reachable. Falls back to the live GL renderer (EEVEE-only
    runs, where no Cycles device is active), then 'unknown GPU' if nothing is
    determinable."""
    # EEVEE rasterizes on the GL-context GPU, not a Cycles device -- report that
    # directly so an EEVEE run's header isn't mislabeled with a persisted Cycles
    # backend (e.g. '(OPTIX)') it never used.
    try:
        if bpy.context.scene.render.engine == BOOST_ENGINE_ID:
            return ", ".join(_boost_device_labels(bpy.context.scene.cycles_boost))
        if bpy.context.scene.render.engine != 'CYCLES':   # EEVEE / Workbench: GL device
            gl = _gl_renderer_string()
            if gl:
                return gl
    except Exception:
        # Best-effort probe: if scene/render context is unavailable, continue
        # to Cycles/GL fallback detection below.
        return None
    try:
        addon = bpy.context.preferences.addons.get('cycles')
        if addon is not None:
            cprefs = addon.preferences
            _get_devices(cprefs)
            btype = getattr(cprefs, "compute_device_type", "") or ""
            if btype not in ("", "NONE"):
                present_ids = _present_device_ids(cprefs, btype)
                seen = []
                for d in cprefs.devices:
                    if d.type != 'CPU' and d.id in present_ids:
                        label = f"{d.name} ({d.type})"
                        if label not in seen:
                            seen.append(label)
                if seen:
                    return ", ".join(seen)
    except Exception:
        # Best-effort device probing: ignore Cycles API/preference errors and
        # fall back to the live GL renderer string (or "unknown GPU" below).
        pass
    gl = _gl_renderer_string()   # EEVEE-only fallback: the live GL/Vulkan GPU
    return gl if gl else "unknown GPU"


def _res_str():
    """Effective rendered resolution, accounting for the test-mode downscale."""
    p = PROFILE
    pct = TEST_RES_PERCENTAGE if TEST_MODE else p["res_percentage"]
    return f"{p['res_x'] * pct // 100}x{p['res_y'] * pct // 100}"


def _extract_model():
    """The model under test = the .blend filename (sans extension), or its folder
    for a generic 'main.blend' (benchmark scenes), or the scene name if unsaved."""
    try:
        fp = bpy.data.filepath
        if fp:
            name = os.path.splitext(os.path.basename(fp))[0]
            if name.lower() == "main":
                name = os.path.basename(os.path.dirname(fp)) or name
            return name
    except Exception as exc:
        # Best-effort metadata extraction: if filepath access fails, fall back
        # to the active scene name instead of aborting benchmark reporting.
        print(f"[warn] Could not extract model name from filepath: {exc}", file=sys.stderr)
    return bpy.context.scene.name


def _status_cell(m):
    """Table status for a non-OK result: FAIL = the render errored, SKIP = the
    engine/device was never available."""
    return "FAIL" if m["status"] == "FAILED" else "SKIP"


def print_table(results):
    p = PROFILE
    samples = TEST_CYCLES_SAMPLES if TEST_MODE else p['cycles_samples']
    tag = "  [TEST MODE - NOT REPRESENTATIVE]" if TEST_MODE else ""
    print("\n" + "=" * 92)
    print(f"  Blender {_extract_version()} BENCHMARK   GPU={_extract_gpu()}   "
          f"profile={_extract_model()}   "
          f"{_res_str()}   cycles_samples={samples} (adaptive OFF){tag}")
    print(f"  invariants source: "
          f"{'qa_standardize_render_settings.py' if INVARIANTS_IMPORTED else 'LOCAL FALLBACK (out of sync risk)'}")
    scene = bpy.context.scene
    if has_compositor(scene):
        # compositor_device only exists on Blender 4.0+; older builds have no selector.
        dev = getattr(scene.render, "compositor_device", "n/a (<4.0)")
        print(f"  compositor: ON, device={dev} "
              f"(runs inside every render below -> included in these times)")
    else:
        print("  compositor: none (render times are pure engine)")
    print("=" * 92)
    print(f"{'Engine':<20} | {'Status':<8} | {'Cold(s)':>9} | "
          f"{'Warm raw med':>13} | {'Warm den med':>13} | {'Denoise+':>9}")
    print("-" * 92)
    for name, m in results.items():
        if m["status"] != "OK":
            print(f"{name:<20} | {_status_cell(m):<8} | {'':>9} | {'':>13} | {'':>13} | "
                  f"{'':>9}   ({m['reason']})")
            continue
        cold = f"{m['cold_raw']:.3f}"
        wraw = f"{m['warm_raw_med']:.3f}"
        if m["warm_den_med"] is not None:
            wden = f"{m['warm_den_med']:.3f}"
            dover = f"+{m['warm_den_med'] - m['warm_raw_med']:.3f}"
        else:
            wden, dover = "  n/a", "  n/a"
        print(f"{name:<20} | {'OK':<8} | {cold:>9} | {wraw:>13} | {wden:>13} | {dover:>9}")
    print("-" * 92)
    print("  Cold = parse + BVH/kernel compile (one-shot).  "
          "Warm = cached steady-state (median of N).")
    print("  Denoise+ = warm denoised - warm raw = denoise pass cost.  "
          f"N warm runs = {WARM_RUNS}.")
    # spread footnote
    for name, m in results.items():
        if m["status"] == "OK":
            print(f"    {name}: warm raw spread {m['warm_raw_min']:.3f}-{m['warm_raw_max']:.3f}s"
                  + (f", devices [{', '.join(m['devices'])}]" if m["devices"] else ""))
    print("=" * 92 + "\n")


def print_compositor_table(results):
    tag = "   [TEST MODE - NOT REPRESENTATIVE]" if TEST_MODE else ""
    print("\n" + "=" * 92)
    print(f"  Blender {_extract_version()} COMPOSITOR A/B   GPU={_extract_gpu()}   "
          f"profile={_extract_model()}   {_res_str()}{tag}")
    print("=" * 92)
    print(f"{'Engine':<20} | {'Status':<8} | {'CPU comp med':>13} | "
          f"{'GPU comp med':>13} | {'GPU-CPU':>9}")
    print("-" * 92)
    for name, m in results.items():
        if m["status"] != "OK":
            print(f"{name:<20} | {_status_cell(m):<8} | {'':>13} | {'':>13} | "
                  f"{'':>9}   ({m['reason']})")
            continue
        cpu = f"{m['cpu_med']:.3f}"
        if m.get("gpu_med") is not None:
            gpu = f"{m['gpu_med']:.3f}"
            delta = f"{m['gpu_med'] - m['cpu_med']:+.3f}"
        else:
            gpu = "  ERR"
            delta = f"({(m.get('gpu_error') or '?')[:18]})"
        print(f"{name:<20} | {'OK':<8} | {cpu:>13} | {gpu:>13} | {delta:>9}")
    print("-" * 92)
    print("  Full warm render time (render + compositor), median of N. Delta")
    print("  isolates compositor_device: GPU-CPU > 0 means GPU compositing is the")
    print(f"  slower option here. N = {WARM_RUNS}.")
    for name, m in results.items():
        if m["status"] == "OK" and m["devices"]:
            print(f"    {name}: devices [{', '.join(m['devices'])}]")
    print("=" * 92 + "\n")


def main():
    global WARM_RUNS, COMPOSITOR_DEVICE, TEST_MODE, SAVE_RENDERS, RENDER_DIR, GPU_SUBSTR, RENDER_TAG
    args = _parse_args()
    if args.get("help"):
        print(_USAGE)
        return
    WARM_RUNS = args["warm"]

    SAVE_RENDERS = args["save_renders"]
    RENDER_TAG = args.get("render_tag") or ""
    RENDER_DIR = args["render_dir"] or os.path.join(_script_dir(), "cold_renders")
    if SAVE_RENDERS:
        os.makedirs(RENDER_DIR, exist_ok=True)
        print(f"[args] --save-renders on -> cold frame renders will be written to {RENDER_DIR}")

    # Test mode: 1 warm run (unless --warm was given explicitly) + reduced samples.
    TEST_MODE = args["test"]
    if TEST_MODE:
        if not args["warm_explicit"]:
            WARM_RUNS = 1
        print("=" * 92)
        print(f"  *** TEST MODE *** harness smoke check -- warm={WARM_RUNS}, "
              f"res={_res_str()} ({TEST_RES_PERCENTAGE}%), "
              f"samples cycles={TEST_CYCLES_SAMPLES}/eevee={TEST_EEVEE_SAMPLES}. "
              f"NUMBERS ARE NOT REPRESENTATIVE.")
        print("=" * 92)

    # Optional --gpu substring narrowing (default: all present devices)
    GPU_SUBSTR = args["gpu"]

    # Compositor device pin (--compositor)
    comp = args["compositor"]
    if comp not in ("CPU", "GPU"):
        print(f"[args] unknown --compositor '{comp}', using 'CPU'")
        comp = "CPU"
    COMPOSITOR_DEVICE = comp

    if GPU_SUBSTR:
        print(f"[args] --gpu filter: '{GPU_SUBSTR}'")

    scene = bpy.context.scene
    apply_invariants(scene)      # once; engine-independent
    apply_scene_controls(scene)  # clear stray region + pin compositor device

    engine_arg = args["engine"]
    if engine_arg and engine_arg not in _ENGINE_ALIASES:
        print(f"[args] unknown --engine '{engine_arg}'. Valid: {', '.join(_ENGINE_ALIASES)}")
        return
    setups = _ENGINE_ALIASES.get(engine_arg, ALL_SETUPS) if engine_arg else ALL_SETUPS
    single_process = engine_arg is not None

    # --- Compositor CPU-vs-GPU A/B mode ---
    if args["compositor_ab"]:
        results = {}
        for i, setup in enumerate(setups):
            results[setup["name"]] = benchmark_compositor(setup)
            if not single_process and i < len(setups) - 1 and COOLDOWN_SEC > 0:
                time.sleep(COOLDOWN_SEC)
        print_compositor_table(results)
        return

    # --- Default: native engine race ---
    results = {}
    for i, setup in enumerate(setups):
        results[setup["name"]] = benchmark_engine(setup)
        # Cooldown between engines only when several run in one process
        if not single_process and i < len(setups) - 1 and COOLDOWN_SEC > 0:
            time.sleep(COOLDOWN_SEC)

    print_table(results)
    if not single_process:
        print("Note: for the least thermal/cache carryover, run one engine per "
              "process with `-- --engine OPTIX|CUDA|EEVEE`.")


if __name__ == "__main__":
    main()