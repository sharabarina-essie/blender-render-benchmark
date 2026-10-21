# Blender Render Benchmark

A multi-version **Blender** render benchmark. Times **Cycles** (CUDA + OptiX) and
**EEVEE** render performance across several Blender versions, plus a **compositor
CPU-vs-GPU A/B**. The git repo is **code only** — a `bootstrap` step downloads the
Blender builds and test models from official blender.org URLs, so
`clone → bootstrap → run` works on any Windows machine with a GPU + drivers.

Two suites:
- **cycles** — CUDA + OptiX on each version's Cycles/splash scene.
- **eevee** — EEVEE on EEVEE demo scenes.

Plus a compositor CPU-vs-GPU A/B (one Cycles + one EEVEE model per version).

---

## Clone & bootstrap

```powershell
git clone <this-repo> blender-render-benchmark
cd blender-render-benchmark

# 1) download Blender builds + models into blender/ and models/ (double-click, or:)
bootstrap.cmd                     # add --Force to re-download

# 2) host-Python deps for the runners
pip install -r requirements.txt

# 3) verify
python run_benchmark_suite.py --list-versions
python run_benchmark_suite.py --all-versions --suite cycles --dry-run
```

`bootstrap` prints the `.blend` files it extracted and validates each download
(a URL that returns an HTML page instead of a file fails loudly).

> The machine needs a **GPU + drivers**; `blender/` and `models/` are git-ignored
> (downloaded, never committed) — only code + config + bootstrap are committed.

---

## Layout

```
engine_benchmark.py                 # in-Blender measurement harness
qa_standardize_render_settings.py   # in-Blender environment sanitizer
benchmark_parse.py                  # shared stdout parser
run_benchmark_suite.py              # host: cycles/eevee render-time runner -> xlsx
run_compositor_suite.py             # host: compositor CPU-vs-GPU runner   -> xlsx
benchmark_config.toml               # versions, model catalog, per-suite model lists
bootstrap.ps1 / bootstrap.cmd       # download Blender + models into place
requirements.txt
blender/<version>/blender.exe       # downloaded (git-ignored)
models/…                            # downloaded (git-ignored)
results/  logs/                     # generated (git-ignored)
```

---

## Configuration (`benchmark_config.toml`)

- **`[models]`** — `root = "models"` plus one key per scene. Splash scenes (Cycles)
  and `eevee/*` demo scenes (EEVEE).
- **`[blender.versions."X"]`** — each version's `exe` plus:
  - `cycles_models` — keys the **cycles** suite runs (version-matched splash).
  - `eevee_models` — keys the **eevee** suite runs.
  - `compositor_cycles` / `compositor_eevee` — the one scene each compositor A/B uses.
- **`[blender]`** — `default_version`, `cycles_engines` (`CUDA, OPTIX`), `eevee_engines` (`EEVEE`).
- **`[run]`** — `mode`, `suite` (default when `--suite` omitted), timeouts, output dirs.

Model URLs live in `bootstrap.ps1`; the config points at where they land under `models/`.

---

## Running

```powershell
# discover / preview
python run_benchmark_suite.py --list-versions
python run_benchmark_suite.py --all-versions --suite all --dry-run

# CYCLES suite (CUDA + OptiX on each version's cycles_models)
python run_benchmark_suite.py --all-versions --suite cycles --mode test
python run_benchmark_suite.py --blender-version 4.5 --suite cycles --mode full

# EEVEE suite (EEVEE on each version's eevee_models)
python run_benchmark_suite.py --all-versions --suite eevee --mode test

# both suites, every version, one spreadsheet
python run_benchmark_suite.py --all-versions --suite all --mode full

# compositor CPU-vs-GPU A/B (one cycles + one eevee model per version)
python run_compositor_suite.py --all-versions --mode test

# overrides
python run_benchmark_suite.py --blender-version 5.2 --suite cycles --engines OPTIX
python run_benchmark_suite.py --blender-version 4.5 --suite eevee --models eevee_temple
python run_benchmark_suite.py --all-versions --suite cycles --save-renders
```

Every script supports `--help` (the two Blender scripts need `-- --help`).

---

## Outputs (inside the project)

- `results/results_<ts>.xlsx` — one row per `(version, suite→engine, model)`:
  **Version · Model · Mode · Engine · Status · Cold · Warm(raw/den) · Blender Build · Reported GPU · Reason · Process Status · Log File**.
- `results/compositor_results_<ts>.xlsx` — compositor A/B (CPU vs GPU vs delta).
- `logs/<version>_[suite_]<model>_<engine>_<mode>_<ts>.log` — full Blender stdout.
- `results/renders/<version>_<model>__…png` — cold frames (only with `--save-renders`).

The `Version` column is always filled (even on a crash); rows are written
incrementally, and a non-clean exit is a definitive `CRASH`/`TIMEOUT` row.

---

## Key concepts

- **Two suites, one config.** `--suite cycles|eevee|all` selects engines + the
  matching per-version model list; the compositor suite picks one model of each.
- **Presence-aware GPU selection.** Enables only the Cycles devices Blender can
  actually initialize right now (`has_active_device`), ignoring stale userpref
  "ghost" entries for GPUs the machine no longer has. `--gpu "<substr>"` forces one.
- **Cross-version robustness.** Eevee id auto-detected (`BLENDER_EEVEE_NEXT` 4.2+,
  `BLENDER_EEVEE` 3.x); AgX → Filmic → Standard fallback; `compositor_device`
  treated as 4.0+ only. `--enable-autoexec` is passed so scene Python (driver-driven
  cameras in the splash scenes) runs, as the GUI does.
- **Cold vs warm**, **test mode** (`--test`: warm 1, 64/16 samples, 25% res, stamped
  NOT REPRESENTATIVE) — pipeline validation, not real numbers.

---

## Notes & caveats

- **GPU compositor defaults to CPU.** `compositor_device` is pinned to CPU because
  GPU compositing has been unstable in some setups; use `--compositor GPU` or
  `--compositor-ab` to measure/test it.
- **Model ↔ version compatibility** in `eevee_models` / splash lists is a best-effort
  default — tune per version once you see what opens (newer Blender opens older
  `.blend`, not vice-versa). Anything that fails is recorded as an isolated
  `CRASH`/skip row, not a broken run.


## Interactive viewport FPS and image quality

The test starts the **full Blender GUI**, selects **Layout**, maximizes the 3D
Viewport and orbits without playing the timeline. It requires **OpenGL** and an
actual **1920x1080 viewport**, Pixel Size 1x. It calibrates only its own window;
keep it visible on a desktop large enough for the viewport plus window chrome.
Scenes and user preferences are not saved.

### Prepare the native test machine

Use Windows, the NVIDIA driver and Python **3.10 or later**. Disconnect Boost on
the target machine for a native baseline. The tests never start or stop sessions.

```powershell
git fetch origin
git switch --track origin/interactive-viewport-tests
python -m pip install -r requirements.txt
python setup_viewport_test.py --builds 4.5.14,5.2.1 --scene smoke
python -m unittest discover -s tests -v
```

If already on this branch, use `git pull --ff-only`. Setup downloads selected
portable Blender builds and Classroom with its linked assets. Existing builds
are reused. Use `--builds all` to install all 13 builds used by the viewport
matrix. Existing installations can be selected with `--blender PATH` for a single
run or `--blender-dir DIRECTORY` for the matrix.

**All models stay out of Git.** Download Medieval Hut, Temple and other manually
supplied scenes yourself, then place them under `models/`:

| Scene | Path under the repository |
|---|---|
| Classroom smoke | `models/archviz/classroom/classroom/main.blend` plus its complete `assets/` directory |
| Medieval Fisher Hut full | `models/medieval_fisher_hut.blend` |
| Temple | `models/eevee/temple.blend` |

[Medieval Fisher Hut](https://blendswap.com/blend/30099) is by DHCG_1492973,
CC-BY (see source page); download requires sign-in. Preserve the original .blend
without resaving. `viewport_assets.json` pins the scene hashes from the existing
measurements. After placing Medieval Hut, verify it:

```powershell
python setup_viewport_test.py --scene full --verify-only
# Or import an existing original into models/ and verify it:
python setup_viewport_test.py --scene full --fisher-hut-source 'D:\Downloads\Medieval Fisher Hut.blend'
```

### Run the native baseline

Find the device name with `nvidia-smi --query-gpu=name --format=csv,noheader`.
The examples use the substring `RTX 600`, which matches RTX 600 Ada / RTX 6000 Ada
names; select a unique substring if multiple cards are installed. Both Cycles
selection and the OpenGL renderer must match. A mismatched integrated GPU fails
validation; select NVIDIA for that executable in Windows Graphics settings before
retrying. `--label` records a description, not proof that Boost is disconnected.

```powershell
# Classroom smoke: all six shading permutations, one 5-second trial each.
python run_viewport_performance.py --blender-version 5.2.1 --smoke --cases all --gpu 'RTX 600' --label native-rtx-ada --timeout 900

# Medieval Hut: reproduce the two quality cases with three 30-second FPS trials.
python run_viewport_performance.py --blender-version 5.2.1 --cases rendered-cycles-cuda --gpu 'RTX 600' --quality-capture --label native-rtx-ada --timeout 900
python run_viewport_performance.py --blender-version 4.5.14 --cases rendered-cycles-optix --gpu 'RTX 600' --quality-capture --label native-rtx-ada --timeout 900

# Full matrix: every installed build, all six permutations, three 30-second trials.
python run_viewport_matrix.py --gpu 'RTX 600' --label native-rtx-ada --quality-capture --timeout 900

# Other manually supplied scenes can be selected by catalog key or .blend path.
python run_viewport_performance.py --blender-version 5.2.1 --model eevee_temple --cases solid,material,rendered-eevee --gpu 'RTX 600' --label native-temple
```

Permutations: Solid (Workbench), Material Preview with studio HDRI, Material
Preview with Scene Lights/World, Rendered EEVEE, Rendered Cycles OptiX, Rendered
Cycles CUDA. The matrix uses a fresh Blender process per build/permutation and
continues after crashes, recording failures. Narrow it with `--builds
4.5.14,5.2.1` or `--cases ...`; `--smoke` selects Classroom. Native runs need no
Boost CLI. Optional `--capture-boost-state` takes read-only session snapshots.

### Settings, metrics and evidence

`opengl-1080p-v1` fixes 64 viewport samples, seed 0, adaptive sampling off,
Cycles OptiX viewport denoising from sample 1, motion blur/DOF/compositor off,
and a stationary timeline. EEVEE Next uses Screen-Trace raytracing, half-resolution
traces, spatial/temporal/bilateral denoising, Fast GI off and temporal reprojection
off. Older EEVEE uses legacy SSR with compatibility notes. Effective settings,
viewport dimensions, scene/build/harness hashes and the command are saved.

Draw FPS counts distinct requested views observed by POST_PIXEL callbacks,
**not GPU completion or client presentation FPS**. Repeated redraws and timer ticks
are excluded. Reports include p95/p99 intervals, 1% low FPS and gap CSVs. A relative
stutter is an interval above `max(100ms, 3 * run median interval)`; initial and tail
waits are separate. PASS confirms complete measurements and image sanity; there
is no minimum FPS threshold.

`--quality-capture` adds a **separate workload after the FPS trials**: moving PNGs
at quarter-orbit checkpoints, then the frozen requested pose at 0, 0.25, 0.5, 1,
2, 5, 10 and 15 seconds. Readbacks never enter the timed FPS loop. Actual capture
times and readback overhead are recorded. `quality.html` provides a refinement
slider; `quality.json` records 8x8 luma SSIM, RGB MAE and relative edge detail
against the last stationary image, excluding the top 64 pixels and 8-pixel borders.
Different moving poses are not compared with that reference. Readbacks can perturb
capture cadence; displayed content can lag the requested pose.

Cycles may lower internal resolution during navigation, even at Pixel Size 1x.
A 1920x1080 PNG does not establish internal render dimensions or completed samples.
The final stationary image is an observed reference, not guaranteed convergence
or ground truth; unchanged degraded images can still match it. These metrics do
not establish compression as the cause. Compare native/remote runs with the same
scene, build, settings and comparable machine conditions.

Results stay under ignored `results/viewport_performance_<timestamp>/` and
`results/viewport_matrix_<timestamp>/`: HTML, JSON, frame/gap CSVs, PNGs and logs.
Copy the whole result directory back for comparison. A temporary Windows awake
request is released on exit without changing the power plan.
