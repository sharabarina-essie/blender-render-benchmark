"""
qa_standardize_render_settings.py
=================================
Standardize Blender render / viewport / color-management (view) / output settings
for QA machines, so no leftover "random" settings from a previous session can
influence a test.

WHAT IT DOES
------------
1. RESET: for every scene, every relevant settings group (render, cycles, eevee,
   image output, color management, sequencer color space, per-view-layer denoise)
   is reset to Blender's factory defaults via `property_unset`. This clears stray
   render regions, simplify toggles, odd sampling values, changed sequencer color
   spaces, etc. -- WITHOUT wiping the loaded scene / geometry.
2. APPLY: a single known-good CONFIG (below) is then written on top, including
   SEPARATE Cycles noise thresholds for the viewport and the final render.
3. REPORT: prints exactly what was reset and applied, per scene, so QA has a log.

WHAT IT INTENTIONALLY DOES NOT TOUCH
------------------------------------
- GPU device SELECTION (Preferences > System > Cycles Render Devices). That is
  per-machine, hardware-dependent, and must stay whatever each QA box has set.
  This script only flips the *scene* compute toggle (cycles.device = 'GPU'/'CPU').
- Anything in User Preferences.

HOW TO RUN
----------
- Interactive: open the Scripting tab, paste/open this file, press Run.
- Headless   : blender --background yourfile.blend --python qa_standardize_render_settings.py
  (Headless changes are IN-MEMORY only unless saved: uncomment the save line at the bottom.)
Tested against the Blender 4.5 LTS Python API. Setters are guarded so a renamed
property on a different version logs a warning instead of crashing the whole run.
"""

import bpy

# ---------------------------------------------------------------------------
# CONFIG  -- edit these to define your QA standard, then hand the file to QA.
# ---------------------------------------------------------------------------
CONFIG = {
    # --- Engine ---------------------------------------------------------
    # 'CYCLES' | 'BLENDER_EEVEE_NEXT' (Eevee in 4.5) | 'BLENDER_WORKBENCH'
    "engine": "CYCLES",

    # --- Cycles device toggle (scene-level; real GPU pick is in Prefs) ---
    "cycles_device": "GPU",            # 'GPU' or 'CPU'

    # --- Cycles FINAL render sampling -----------------------------------
    "render_samples": 1024,            # max samples (ceiling)
    "render_adaptive": True,
    "render_noise_threshold": 0.01,    # <-- FINAL render noise threshold
    "render_min_samples": 0,           # 0 = auto

    # --- Cycles VIEWPORT sampling ---------------------------------------
    "viewport_samples": 1024,          # viewport max samples
    "viewport_noise_threshold": 0.10,  # <-- VIEWPORT noise threshold
    "viewport_min_samples": 0,

    # --- Determinism (critical for QA / reproducible image diffing) -----
    "seed": 0,
    "animated_seed": False,            # fixed noise pattern across frames
    "use_simplify": False,

    # --- Denoising ------------------------------------------------------
    # 'OPENIMAGEDENOISE' works on any vendor (best for cross-machine QA);
    # 'OPTIX' is NVIDIA-only. Keep OIDN unless every QA box is NVIDIA.
    "denoise_render": True,
    "denoiser": "OPENIMAGEDENOISE",
    "denoise_viewport": True,
    "viewport_denoiser": "OPENIMAGEDENOISE",

    # --- Light paths / clamping ----------------------------------------
    "max_bounces": 12,
    "diffuse_bounces": 4,
    "glossy_bounces": 4,
    "transmission_bounces": 12,
    "volume_bounces": 0,
    "transparent_bounces": 8,
    "clamp_direct": 0.0,
    "clamp_indirect": 10.0,
    "persistent_data": True,

    # --- Color management (the "view" settings) -------------------------
    "display_device": "sRGB",          # match most monitors / deliverables
    "view_transform": "AgX",
    "look": "None",
    "exposure": 0.0,
    "gamma": 1.0,
    "sequencer_colorspace": "sRGB",    # resets the VSE color space too

    # --- Output ---------------------------------------------------------
    "resolution_x": 1920,
    "resolution_y": 1080,
    "resolution_percentage": 100,
    "fps": 24,
    "frame_start": 1,
    "frame_end": 1,                    # single-frame default; QA sets per test
    "frame_step": 1,
    "film_transparent": False,
    "filepath": "//render/qa_",        # blend-relative ./render/ folder
    "file_format": "PNG",              # deterministic + directly viewable
    "color_mode": "RGBA",
    "color_depth": "16",
    "png_compression": 15,

    # --- Compositor device (default = CPU) ------------------------------
    # GPU compositing has been unstable in some setups; keep it CPU unless
    # it has been validated. Guarded.
    "compositor_device": "CPU",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_log = []

def _set(obj, attr, value):
    """Defensive setattr: skips (with a note) if the property doesn't exist
    on this Blender version instead of aborting the whole run."""
    if obj is None:
        return
    if not hasattr(obj, attr):
        _log.append(f"  [skip] {obj!r}.{attr} not present on this version")
        return
    try:
        setattr(obj, attr, value)
    except Exception as e:  # enum value not available, read-only, etc.
        _log.append(f"  [warn] {attr} = {value!r} failed: {e}")


def _reset_group(group):
    """Reset every writable property in a group to its factory default,
    preserving scene geometry. Uses property_unset (the API equivalent of
    right-click > Reset to Default Value)."""
    if group is None:
        return
    try:
        props = group.bl_rna.properties
    except AttributeError:
        return
    for prop in props:
        pid = prop.identifier
        if pid == "rna_type" or prop.is_readonly:
            continue
        try:
            group.property_unset(pid)
        except Exception:
            pass  # pointer/collection props etc. -- safe to ignore


# ---------------------------------------------------------------------------
# Reset pass
# ---------------------------------------------------------------------------
def reset_scene(scene):
    groups = [
        scene.render,
        getattr(scene.render, "image_settings", None),
        getattr(scene.render, "ffmpeg", None),
        getattr(scene, "cycles", None),
        getattr(scene, "eevee", None),
        scene.view_settings,
        scene.display_settings,
        getattr(scene, "sequencer_colorspace_settings", None),
    ]
    for g in groups:
        _reset_group(g)
    # Per-view-layer denoise / passes
    for vl in scene.view_layers:
        _reset_group(getattr(vl, "cycles", None))
        _reset_group(getattr(vl, "eevee", None))
    _log.append(f"  reset groups on scene '{scene.name}' + {len(scene.view_layers)} view layer(s)")


# ---------------------------------------------------------------------------
# Apply pass
# ---------------------------------------------------------------------------
def apply_scene(scene, c):
    r = scene.render

    # Engine
    _set(r, "engine", c["engine"])

    # ---- Cycles ----
    cy = getattr(scene, "cycles", None)
    if cy is not None and c["engine"] == "CYCLES":
        _set(cy, "device", c["cycles_device"])

        # Final render sampling + noise threshold
        _set(cy, "use_adaptive_sampling", c["render_adaptive"])
        _set(cy, "samples", c["render_samples"])
        _set(cy, "adaptive_threshold", c["render_noise_threshold"])
        _set(cy, "adaptive_min_samples", c["render_min_samples"])

        # Viewport sampling + noise threshold
        _set(cy, "preview_samples", c["viewport_samples"])
        _set(cy, "preview_adaptive_threshold", c["viewport_noise_threshold"])
        _set(cy, "preview_adaptive_min_samples", c["viewport_min_samples"])
        # (viewport adaptive sampling is always on in 4.5; guarded set anyway)
        _set(cy, "use_preview_adaptive_sampling", True)

        # Determinism
        _set(cy, "seed", c["seed"])
        _set(cy, "use_animated_seed", c["animated_seed"])

        # Denoising
        _set(cy, "use_denoising", c["denoise_render"])
        _set(cy, "denoiser", c["denoiser"])
        _set(cy, "use_preview_denoising", c["denoise_viewport"])
        _set(cy, "preview_denoiser", c["viewport_denoiser"])

        # Light paths + clamping
        _set(cy, "max_bounces", c["max_bounces"])
        _set(cy, "diffuse_bounces", c["diffuse_bounces"])
        _set(cy, "glossy_bounces", c["glossy_bounces"])
        _set(cy, "transmission_bounces", c["transmission_bounces"])
        _set(cy, "volume_bounces", c["volume_bounces"])
        _set(cy, "transparent_max_bounces", c["transparent_bounces"])
        _set(cy, "sample_clamp_direct", c["clamp_direct"])
        _set(cy, "sample_clamp_indirect", c["clamp_indirect"])

        # Enable denoise per view layer too
        for vl in scene.view_layers:
            _set(getattr(vl, "cycles", None), "use_denoising", c["denoise_render"])

    # ---- Eevee (only if selected; noise threshold is a Cycles concept) ----
    if c["engine"] == "BLENDER_EEVEE_NEXT":
        ev = getattr(scene, "eevee", None)
        _set(ev, "taa_render_samples", c["render_samples"])
        _set(ev, "taa_samples", c["viewport_samples"])
        _set(ev, "use_raytracing", True)

    # ---- Determinism / simplify (engine-agnostic) ----
    _set(r, "use_simplify", c["use_simplify"])
    _set(r, "use_persistent_data", c["persistent_data"])
    _set(r, "use_border", False)          # clear stray render region
    _set(r, "use_crop_to_border", False)

    # ---- Color management (the "view") ----
    _set(scene.display_settings, "display_device", c["display_device"])
    _set(scene.view_settings, "view_transform", c["view_transform"])
    _set(scene.view_settings, "look", c["look"])
    _set(scene.view_settings, "exposure", c["exposure"])
    _set(scene.view_settings, "gamma", c["gamma"])
    _set(scene.view_settings, "use_curve_mapping", False)
    _set(getattr(scene, "sequencer_colorspace_settings", None),
         "name", c["sequencer_colorspace"])

    # ---- Output ----
    _set(r, "resolution_x", c["resolution_x"])
    _set(r, "resolution_y", c["resolution_y"])
    _set(r, "resolution_percentage", c["resolution_percentage"])
    _set(r, "fps", c["fps"])
    _set(r, "fps_base", 1.0)
    _set(scene, "frame_start", c["frame_start"])
    _set(scene, "frame_end", c["frame_end"])
    _set(scene, "frame_step", c["frame_step"])
    _set(r, "film_transparent", c["film_transparent"])
    _set(r, "filepath", c["filepath"])

    imgs = r.image_settings
    _set(imgs, "file_format", c["file_format"])
    _set(imgs, "color_mode", c["color_mode"])
    _set(imgs, "color_depth", c["color_depth"])
    _set(imgs, "compression", c["png_compression"])  # PNG compression %

    # ---- Compositor device (guarded; CPU default) ----
    _set(r, "compositor_device", c["compositor_device"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(reset_first=True):
    scenes = list(bpy.data.scenes)
    _log.append(f"Standardizing {len(scenes)} scene(s). reset_first={reset_first}")
    for scene in scenes:
        _log.append(f"\nScene: {scene.name}")
        if reset_first:
            reset_scene(scene)
        apply_scene(scene, CONFIG)
        _log.append(f"  applied QA config (engine={CONFIG['engine']}, "
                    f"render_noise={CONFIG['render_noise_threshold']}, "
                    f"viewport_noise={CONFIG['viewport_noise_threshold']})")

    print("\n".join(_log))
    print("\nQA render settings standardized. "
          "Remember: GPU DEVICE selection is per-machine (Preferences > System).")


_USAGE = """\
qa_standardize_render_settings.py -- ENVIRONMENT SANITIZER (runs INSIDE Blender).

  Invoke:  blender -b <scene.blend> -P qa_standardize_render_settings.py [-- --help]

  Resets every relevant settings group to factory defaults, then applies the QA
  CONFIG: color management (AgX/sRGB), determinism (seed 0, animated-seed off,
  simplify off), sampling/light-paths, output format, compositor=CPU. Edit the
  CONFIG dict at the TOP of this file to change the QA standard.

  Options:
    -h, --help    show this help and exit

  Takes no other CLI options. Headless changes are IN-MEMORY only unless saved --
  uncomment bpy.ops.wm.save_mainfile() below, or run in the GUI and save.
"""

if __name__ == "__main__":
    import sys
    _argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if "--help" in _argv or "-h" in _argv:
        print(_USAGE)
    else:
        main(reset_first=True)
        # Headless save (uncomment to overwrite the file after standardizing):
        # bpy.ops.wm.save_mainfile()