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
