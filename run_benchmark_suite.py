"""
run_benchmark_suite.py
=======================
Orchestrates engine_benchmark.py across multiple .blend models and collects
every result into one .xlsx table. Does not change what is measured -- that
is engine_benchmark.py's job (and, upstream of it, qa_standardize_render_
settings.py's). This file only drives repeated invocations and harvests
their printed output.

WHY one Blender process per (model, engine)
--------------------------------------------
engine_benchmark.py's main() only prints its results table ONCE, after every
engine in that process has finished (see print_table() call at the end of
main()). If a later engine crashes the process (this has happened with EEVEE
Next on a large model), the numbers for engines that already finished
successfully in that same process are lost -- they were only ever held in
memory, never printed. Running one engine per process (which the script's
own docstring recommends for thermal/cache isolation anyway) means a crash
in one engine only voids that single (model, engine) row; every other
combination is unaffected and already safely on disk in its own log file.

CONFIG
------
All machine paths and the model list live in benchmark_config.toml (next to
this file), not in code -- see that file for the schema. CLI flags override
individual config values for one-off runs without editing the file:

  --config PATH   use a different config file entirely (e.g. per-machine)
  --mode / --models / --engines / --warm / --timeout / --blender / --script /
  --log-dir / --out   override the matching config value for this invocation

USAGE
-----
  Validate the pipeline (fast, smallest scene, non-representative numbers):
    python run_benchmark_suite.py --mode test --models hero

  Full run across all default models from the config (slow -- real render times):
    python run_benchmark_suite.py --mode full

  Dry run (print the commands, execute nothing):
    python run_benchmark_suite.py --mode test --dry-run

Each invocation writes its own timestamped file by default (e.g.
results_20260804_201228.xlsx next to results.xlsx from config), so re-running
never overwrites or appends onto a previous run. Pass --out PATH to opt back
into accumulating multiple runs into one fixed file instead. Within a single
run, results are still written incrementally after every (model, engine)
result, so a crash partway through loses nothing already measured.
"""

import argparse
import datetime
import os
import subprocess
import sys
import time

try:
    import tomllib
except ImportError:  # Python < 3.11
    try:
        import tomli as tomllib
    except ImportError:
        sys.exit("tomllib/tomli is required to read benchmark_config.toml "
                  "(Python < 3.11: python -m pip install tomli)")

try:
    from openpyxl import Workbook, load_workbook
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter
except ImportError:
    sys.exit("openpyxl is required: python -m pip install openpyxl")

# Output parsing is decoupled into its own module (see benchmark_parse.py):
# this runner only orchestrates + writes xlsx; it does not know the table format.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_parse import parse_output, parse_interactive_output

# ---------------------------------------------------------------------------
# Config loading -- benchmark_config.toml is the single source of truth for
# machine paths and the model list (see that file for the documented schema).
# Code below only supplies fallbacks for [run] tuning knobs the config is
# allowed to omit.
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(SCRIPT_DIR, "benchmark_config.toml")

FALLBACK_ENGINES = ["CUDA", "OPTIX", "EEVEE"]
FALLBACK_TEST_TIMEOUT_SEC = 600
FALLBACK_FULL_TIMEOUT_SEC = 10800
FALLBACK_LOG_DIR = "logs"
FALLBACK_OUTPUT_XLSX = os.path.join("results", "results.xlsx")
FALLBACK_MODE = "test"
FALLBACK_RENDER_DIR = os.path.join("results", "renders")
FALLBACK_SAVE_RENDERS = False


def _resolve(base_dir, path):
    """Relative config paths resolve against the config file's directory, not
    the current working directory, so behavior doesn't depend on invoke-from-cwd."""
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(base_dir, path))


def load_config(config_path):
    if not os.path.isfile(config_path):
        sys.exit(f"Config file not found: {config_path}\n"
                  f"Pass --config to point at one, or create benchmark_config.toml "
                  f"next to {os.path.basename(__file__)}.")
    try:
        with open(config_path, "rb") as f:
            cfg = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        sys.exit(f"Could not parse {config_path}: {e}")

    errors = []
    blender = cfg.get("blender", {})
    # Accept EITHER a single 'exe' (legacy) OR a [blender.versions] table.
    has_exe = isinstance(blender.get("exe"), str) and blender["exe"]
    versions = blender.get("versions", {})
    has_versions = isinstance(versions, dict) and len(versions) > 0
    if not has_exe and not has_versions:
        errors.append("[blender] needs either 'exe = \"...\"' or a non-empty "
                      "[blender.versions] table")
    if isinstance(versions, dict):
        for k, v in versions.items():
            if not isinstance(v, dict):
                errors.append(f'[blender.versions."{k}"] must be a table with exe = "..."')
                continue
            builds = v.get("builds")
            if builds is not None:
                if (not isinstance(builds, list) or not builds
                        or any(not isinstance(b, str) or not b.startswith(f"{k}.") for b in builds)):
                    errors.append(f'[blender.versions."{k}"] builds must be a non-empty list of '
                                  f'"{k}.<patch>" strings')
                elif v.get("default_build", builds[-1]) not in builds:
                    errors.append(f'[blender.versions."{k}"] default_build must be one of its builds')
            elif not isinstance(v.get("exe"), str) or not v["exe"]:
                errors.append(f'[blender.versions."{k}"] needs builds = [...] or exe = "..."')
            for lk in ("models", "cycles_models", "eevee_models", "interactive_models",
                       "archviz_models"):
                if lk in v and (not isinstance(v[lk], list)
                                or any(not isinstance(m, str) for m in v[lk])):
                    errors.append(f'[blender.versions."{k}"] {lk} must be a list of model keys')
            for sk in ("compositor_cycles", "compositor_eevee"):
                if sk in v and not isinstance(v[sk], str):
                    errors.append(f'[blender.versions."{k}"] {sk} must be a model key (string)')
        dv = blender.get("default_version")
        if dv is not None and dv not in versions:
            errors.append(f"[blender] default_version '{dv}' is not a key in "
                          f"[blender.versions] ({', '.join(versions) or 'empty'})")
    elif versions:
        errors.append("[blender.versions] must be a table of versions")
    if not isinstance(blender.get("script"), str) or not blender["script"]:
        errors.append("[blender] script must be a non-empty string")

    models = cfg.get("models", {})
    # `root` is a shared path prefix, not a model -- pull it out before validating.
    model_root = models.pop("root", None) if isinstance(models, dict) else None
    if model_root is not None and not isinstance(model_root, str):
        errors.append("[models].root must be a string path")
    if not models or not isinstance(models, dict):
        errors.append("[models] must contain at least one key = \"path.blend\"")
    else:
        for key, val in models.items():
            if not isinstance(val, str) or not val:
                errors.append(f"[models] '{key}' must be a non-empty string path")
    # every model key a version references must exist in the catalog
    if isinstance(versions, dict) and isinstance(models, dict):
        for k, v in versions.items():
            if not isinstance(v, dict):
                continue
            refs = (list(v.get("models", [])) + list(v.get("cycles_models", []))
                    + list(v.get("eevee_models", [])) + list(v.get("interactive_models", []))
                    + list(v.get("archviz_models", [])))
            for sk in ("compositor_cycles", "compositor_eevee"):
                if isinstance(v.get(sk), str):
                    refs.append(v[sk])
            for mk in refs:
                if mk not in models:
                    errors.append(f'[blender.versions."{k}"] references unknown model key \'{mk}\'')

    run = cfg.get("run", {})
    if "engines" in run and (not isinstance(run["engines"], list) or not run["engines"]):
        errors.append("[run] engines must be a non-empty list")
    for key in ("timeout_test_sec", "timeout_full_sec", "warm"):
        if key in run and not isinstance(run[key], int):
            errors.append(f"[run] {key} must be an integer")
    if "mode" in run and run["mode"] not in ("test", "full"):
        errors.append("[run] mode must be \"test\" or \"full\"")
    if "save_renders" in run and not isinstance(run["save_renders"], bool):
        errors.append("[run] save_renders must be true or false")

    if errors:
        details = "\n  - ".join(errors)
        sys.exit(f"Invalid config {config_path}:\n  - {details}")

    base_dir = os.path.dirname(os.path.abspath(config_path))
    cfg["_base_dir"] = base_dir
    if blender.get("exe"):
        blender["exe"] = _resolve(base_dir, blender["exe"])
    # Each exact build lives in <builds_dir>/<x.y.z>/ and inherits its line's
    # model lists; the line's own exe is its default_build (else the last build).
    builds_dir = _resolve(base_dir, blender.get("builds_dir", "blender"))
    build_index = {}
    if isinstance(blender.get("versions"), dict):
        for _k, _v in blender["versions"].items():
            if not isinstance(_v, dict):
                continue
            for _b in _v.get("builds", []):
                build_index[_b] = {**_v, "exe": os.path.join(builds_dir, _b, "blender.exe"),
                                   "line": _k}
            if _v.get("builds"):
                _v["default_build"] = _v.get("default_build", _v["builds"][-1])
                _v["exe"] = build_index[_v["default_build"]]["exe"]
            elif _v.get("exe"):
                _v["exe"] = _resolve(base_dir, _v["exe"])
    blender["build_index"] = build_index
    blender["script"] = _resolve(base_dir, blender["script"])
    run["log_dir"] = _resolve(base_dir, run.get("log_dir", FALLBACK_LOG_DIR))
    run["output_xlsx"] = _resolve(base_dir, run.get("output_xlsx", FALLBACK_OUTPUT_XLSX))
    run["render_dir"] = _resolve(base_dir, run.get("render_dir", FALLBACK_RENDER_DIR))
    run.setdefault("mode", FALLBACK_MODE)
    run.setdefault("engines", list(FALLBACK_ENGINES))
    run.setdefault("timeout_test_sec", FALLBACK_TEST_TIMEOUT_SEC)
    run.setdefault("timeout_full_sec", FALLBACK_FULL_TIMEOUT_SEC)
    run.setdefault("save_renders", FALLBACK_SAVE_RENDERS)
    cfg["run"] = run
    cfg["blender"] = blender
    # Model paths resolve against [models].root (falling back to the config dir).
    root_abs = _resolve(base_dir, model_root) if model_root else base_dir
    cfg["models_root"] = root_abs
    cfg["models"] = {
        k: (v if os.path.isabs(v) else os.path.normpath(os.path.join(root_abs, v)))
        for k, v in models.items()
    }
    return cfg


def select_blender(blender_cfg, explicit_path=None, version_key=None):
    """Pick (exe_path, version_models, label) for this run. Exe precedence:
      1. --blender "<path>"        raw override, wins  (version_models = None)
      2. --blender-version <key>   that version's exe + its `models` list
      3. [blender].default_version
      4. legacy single [blender].exe
    version_models = the list of catalog keys the selected version should run, or
    None (raw path / legacy exe / version with no `models`) -> caller uses all."""
    versions = blender_cfg.get("versions") or {}
    builds = blender_cfg.get("build_index") or {}
    if explicit_path:
        return explicit_path, None, "--blender path"
    if versions:
        key = version_key or blender_cfg.get("default_version")
        if not key:
            sys.exit("No Blender version selected: set [blender].default_version or "
                     f"pass --blender-version <{'|'.join(sorted(versions))}>.")
        if key in builds:
            return builds[key]["exe"], builds[key].get("models"), f"Blender {key}"
        if key not in versions:
            sys.exit(f"--blender-version '{key}' is not a line or build in [blender.versions]. "
                     f"Lines: {', '.join(sorted(versions))}; builds: "
                     f"{', '.join(sorted(builds)) or '-'}")
        v = versions[key]
        note = "" if version_key else " (default)"
        build = f" = {v['default_build']}" if v.get("default_build") else ""
        return v["exe"], v.get("models"), f"Blender {key}{build}{note}"
    if blender_cfg.get("exe"):
        return blender_cfg["exe"], None, "[blender].exe"
    sys.exit("No Blender executable configured ([blender].exe or [blender.versions]).")


def version_cfg_for(blender_cfg, key):
    """The [blender.versions] table governing a line or build key (a build carries
    its line's model lists), or None for a raw --blender path."""
    if not key:
        return None
    builds = blender_cfg.get("build_index") or {}
    return builds.get(key) or (blender_cfg.get("versions") or {}).get(key)


def expand_selection(blender_cfg, key, all_builds):
    """[(version_key, exe, label)] for one --blender-version key. The version key
    is the exact build when the line lists builds, so every xlsx row and log name
    says which patch ran; --all-builds expands a line into all of its builds."""
    v = (blender_cfg.get("versions") or {}).get(key)
    if v and v.get("builds"):
        keys = v["builds"] if all_builds else [v["default_build"]]
        return [(b, *[select_blender(blender_cfg, None, b)[i] for i in (0, 2)]) for b in keys]
    exe, _vm, label = select_blender(blender_cfg, None, key)
    return [(key, exe, label)]


def print_versions(blender_cfg):
    versions = blender_cfg.get("versions") or {}
    builds = blender_cfg.get("build_index") or {}
    default = blender_cfg.get("default_version")
    if not versions:
        print(f"No [blender.versions] configured. Single exe: {blender_cfg.get('exe')}")
        return
    print("Configured Blender versions ([blender.versions]):")
    for k in sorted(versions):
        v = versions[k]
        star = "  (default)" if k == default else ""
        print(f"  {k}{star}")
        print(f"       exe   : {v.get('exe')}")
        if v.get("builds"):
            listed = [f"{b}{'*' if b == v['default_build'] else ''}"
                      f"{'' if os.path.isfile(builds[b]['exe']) else ' (missing)'}"
                      for b in v["builds"]]
            print(f"       builds: {', '.join(listed)}   (* = default)")
        print(f"       cycles: {', '.join(v.get('cycles_models', v.get('models', []))) or '-'}")
        print(f"       eevee : {', '.join(v.get('eevee_models', [])) or '-'}")
        print(f"       archviz: {', '.join(v.get('archviz_models', [])) or '-'}")

# ---------------------------------------------------------------------------
# Execution  (output parsing lives in benchmark_parse.parse_output)
# ---------------------------------------------------------------------------

# Windows NTSTATUS codes commonly seen when a Blender subprocess crashes.
_NTSTATUS = {
    0xC0000005: "ACCESS_VIOLATION (segfault)",
    0xC0000409: "STACK_BUFFER_OVERRUN / __fastfail (heap/stack corruption; often a GPU-driver fault)",
    0xC00000FD: "STACK_OVERFLOW",
    0xC0000135: "DLL_NOT_FOUND",
    0xC0000142: "DLL_INIT_FAILED",
    0xC000001D: "ILLEGAL_INSTRUCTION",
    0xC0000094: "INTEGER_DIVIDE_BY_ZERO",
}


def decode_process_status(run_status):
    """Human-readable description of run_one()'s run_status
    ('OK' | 'EXIT_<n>' | 'TIMEOUT' | 'LAUNCH_FAILED' | 'DRY_RUN')."""
    if run_status in ("OK", "DRY_RUN"):
        return "clean exit"
    if run_status == "TIMEOUT":
        return "TIMEOUT (killed at the per-run ceiling)"
    if run_status == "LAUNCH_FAILED":
        return "LAUNCH FAILED (blender.exe not found / not runnable)"
    if run_status.startswith("EXIT_"):
        try:
            code = int(run_status[5:]) & 0xFFFFFFFF
        except ValueError:
            return f"nonzero exit ({run_status})"
        name = _NTSTATUS.get(code)
        return f"exit 0x{code:08X}" + (f" {name}" if name else " (unrecognised crash code)")
    return run_status


def failure_status_label(run_status):
    """Short, definitive Status-cell label for a non-clean exit."""
    if run_status.startswith("EXIT_"):
        return "CRASH"
    if run_status == "TIMEOUT":
        return "TIMEOUT"
    if run_status == "LAUNCH_FAILED":
        return "LAUNCH_FAIL"
    return "ERROR"
def build_cmd(blender_exe, benchmark_script, model_path, engine, mode, warm,
              save_renders, render_dir, extra_args, render_tag=""):
    # --enable-autoexec: allow the .blend's own Python (expression drivers, e.g. the
    # camera drivers in the version splash scenes) to run. Off by default in -b/-P
    # for security; the GUI has it via user prefs, so without it those scenes render
    # wrong ("restricted access disallows name 'scene'"). Must precede the .blend.
    cmd = [blender_exe, "--enable-autoexec", "-b", model_path,
           "-P", benchmark_script, "--", "--engine", engine]
    if mode == "test":
        cmd.append("--test")
    if warm is not None:
        cmd += ["--warm", str(warm)]
    if save_renders:
        cmd += ["--save-renders", "--render-dir", render_dir]
        if render_tag:
            cmd += ["--render-tag", render_tag]   # "<version>_<model>" -> named, comparable renders
    cmd += extra_args
    return cmd


def _exec(cmd, log_path, log_dir, timeout_sec):
    """Run one Blender subprocess, capture combined stdout+stderr to log_path, and
    classify the exit. Returns (text, run_status, elapsed_sec). Shared by the render
    and interactive runners so process handling/crash decoding stays identical."""
    os.makedirs(log_dir, exist_ok=True)
    start = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=timeout_sec)
        text = (proc.stdout or "") + "\n" + (proc.stderr or "")
        run_status = "OK" if proc.returncode == 0 else f"EXIT_{proc.returncode}"
    except subprocess.TimeoutExpired as e:
        text = (e.stdout or "") + "\n" + (e.stderr or "")
        run_status = "TIMEOUT"
    except FileNotFoundError as e:
        text = str(e)
        run_status = "LAUNCH_FAILED"
    elapsed = time.time() - start
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text, run_status, elapsed


def run_one(blender_exe, benchmark_script, model_key, model_path, engine, mode,
            warm, save_renders, render_dir, extra_args, log_dir, timeout_sec, dry_run,
            tag="", version=None):
    cmd = build_cmd(blender_exe, benchmark_script, model_path, engine, mode, warm,
                     save_renders, render_dir, extra_args, render_tag=f"{tag}{model_key}")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{tag}{model_key}_{engine}_{mode}_{ts}.log")

    print(f"[run] {tag}{model_key} / {engine} ({mode})  -> {os.path.basename(log_path)}")
    if dry_run:
        print("       " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
        return {"model_key": model_key, "model_path": model_path, "engine_arg": engine,
                "version": version, "mode": mode, "log_path": log_path,
                "run_status": "DRY_RUN", "meta": {}, "rows": []}

    text, run_status, elapsed = _exec(cmd, log_path, log_dir, timeout_sec)

    meta, rows = parse_output(text)
    # Blender exits 1 (not a crash code) when a render reported an error; the
    # harness already caught it and printed the full table with a FAIL row.
    table_survived = run_status == "EXIT_1" and any(r["status"] == "FAIL" for r in rows)
    if run_status != "OK" and not table_survived:
        # Non-clean exit -> DEFINITIVE failure row. A crashed process may have
        # half-printed a table; that partial scrape is not trustworthy, so we
        # replace it with one explicit failure row rather than let it read as data.
        label = failure_status_label(run_status)
        rows = [{"engine_label": engine, "status": label, "cold_s": None,
                 "warm_raw_med_s": None, "warm_den_med_s": None, "denoise_plus_s": None,
                 "warm_raw_min_s": None, "warm_raw_max_s": None, "devices": None,
                 "cold_render_path": None,
                 "reason": f"{decode_process_status(run_status)} after {elapsed:.0f}s -- see log"}]

    n_ok = sum(1 for r in rows if r["status"] == "OK")
    if run_status != "OK" and not table_survived:
        verdict, detail = failure_status_label(run_status), decode_process_status(run_status)
    elif n_ok:
        verdict, detail = "OK", f"{n_ok}/{len(rows)} engine-row(s) OK"
    elif any(r["status"] == "FAIL" for r in rows):
        failed = next(r for r in rows if r["status"] == "FAIL")
        verdict, detail = "FAILED", f"render error: {failed['reason']}"
    else:
        verdict, detail = ("SKIPPED" if rows else "NO DATA"), f"0/{len(rows)} OK"
    flag = "" if verdict == "OK" else "   <<< SEE LOG"
    print(f"       {verdict:8} {elapsed:6.1f}s  {detail}{flag}")
    return {"model_key": model_key, "model_path": model_path, "engine_arg": engine,
            "version": version, "mode": mode, "log_path": log_path,
            "run_status": run_status, "meta": meta, "rows": rows}


def build_interactive_cmd(blender_exe, script, model_path, mode, warm, extra_args):
    """Interactive harness invocation: no --engine (depsgraph-eval cost is engine-
    independent, so it runs once per model). --enable-autoexec for the same driver
    reason as the render harness."""
    cmd = [blender_exe, "--enable-autoexec", "-b", model_path, "-P", script, "--"]
    if mode == "test":
        cmd.append("--test")
    if warm is not None:
        cmd += ["--warm", str(warm)]
    cmd += extra_args
    return cmd


def run_one_interactive(blender_exe, script, model_key, model_path, mode, warm,
                        extra_args, log_dir, timeout_sec, dry_run, tag="", version=None):
    """Run interactive_benchmark.py on one model; parse its metric table. Mirrors
    run_one but with no engine dimension and the interactive parser/row shape."""
    cmd = build_interactive_cmd(blender_exe, script, model_path, mode, warm, extra_args)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{tag}{model_key}_interactive_{mode}_{ts}.log")

    print(f"[run] {tag}{model_key} / interactive ({mode})  -> {os.path.basename(log_path)}")
    if dry_run:
        print("       " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
        return {"model_key": model_key, "model_path": model_path, "version": version,
                "mode": mode, "log_path": log_path, "run_status": "DRY_RUN",
                "meta": {}, "rows": []}

    text, run_status, elapsed = _exec(cmd, log_path, log_dir, timeout_sec)
    meta, rows = parse_interactive_output(text)
    if run_status != "OK":
        # Non-clean exit -> one DEFINITIVE failure row (a crashed process may have
        # half-printed the table; that partial scrape is not trustworthy).
        label = failure_status_label(run_status)
        rows = [{"metric": "(all)", "status": label, "median_ms": None, "min_ms": None,
                 "max_ms": None, "derived": None,
                 "reason": f"{decode_process_status(run_status)} after {elapsed:.0f}s -- see log"}]

    n_ok = sum(1 for r in rows if r["status"] == "OK")
    if run_status != "OK":
        verdict, detail = failure_status_label(run_status), decode_process_status(run_status)
    elif n_ok:
        verdict, detail = "OK", f"{n_ok}/{len(rows)} metric-row(s) OK"
    else:
        verdict, detail = ("SKIPPED" if rows else "NO DATA"), f"0/{len(rows)} OK"
    flag = "" if verdict == "OK" else "   <<< SEE LOG"
    print(f"       {verdict:8} {elapsed:6.1f}s  {detail}{flag}")
    return {"model_key": model_key, "model_path": model_path, "version": version,
            "mode": mode, "log_path": log_path, "run_status": run_status,
            "meta": meta, "rows": rows}


# ---------------------------------------------------------------------------
# Spreadsheet output
# ---------------------------------------------------------------------------
COLUMNS = [
    "Run Started", "Label", "Version", "Model", "Mode", "Engine", "Status", "Cold (s)",
    "Warm Raw Med (s)", "Warm Raw Min (s)", "Warm Raw Max (s)",
    "Warm Denoised Med (s)", "Denoise+ (s)", "Cold Render Path", "Devices",
    "Reported GPU", "Blender Build", "Resolution", "Cycles Samples",
    "Invariants Source", "Compositor", "Reason", "Process Status", "Log File",
]
# "Label" = free-text run label from --label (e.g. the test build/host/GPU, or
#           "native") -- the config/environment the Cycles log can't record itself.
# "Version"      = the build we launched (e.g. 4.5.0; the line key for a line with no
#                  builds) -- ALWAYS set, even on a crash, so every row is attributable.
# "Blender Build"= the exact build string the run reported (e.g. 4.5.3) -- parsed
#                  from output, so it's blank on a crashed/never-started run.


# Interactive suite writes its own sheet (different row shape from the render race).
INTERACTIVE_COLUMNS = [
    "Run Started", "Label", "Version", "Model", "Mode", "Metric", "Status",
    "Median (ms)", "Min (ms)", "Max (ms)", "Derived", "Frames",
    "Reported GPU", "Blender Build", "Invariants Source", "Reason",
    "Process Status", "Log File",
]


def _set_width(ws, columns, name, width):
    """Set a column's width by header NAME (robust to inserts like Label)."""
    if name in columns:
        ws.column_dimensions[get_column_letter(columns.index(name) + 1)].width = width


def _init_header(ws, columns):
    # openpyxl's create_sheet() leaves a phantom empty row 1 (max_row==1, A1 None),
    # so a plain append() would push the header to row 2 and leave row 1 blank.
    # Write the header straight into row 1 instead; data appends then start at row 2.
    if ws.max_row == 1 and all(c.value is None for c in ws[1]):
        for col, val in enumerate(columns, start=1):
            ws.cell(row=1, column=col, value=val)
    else:
        ws.append(columns)
    for cell in ws[1]:
        cell.font = Font(bold=True)
    ws.freeze_panes = "A2"


def _open_sheet(out_path, title, columns):
    """Open (or create) the named worksheet in out_path, preserving any other
    sheets already in the file. Lets the render and interactive suites write into
    the SAME workbook on separate tabs within one invocation."""
    if os.path.exists(out_path):
        wb = load_workbook(out_path)
        ws = wb[title] if title in wb.sheetnames else wb.create_sheet(title)
        if ws.max_row == 0 or ws["A1"].value is None:
            _init_header(ws, columns)
    else:
        wb = Workbook()
        ws = wb.active
        ws.title = title
        _init_header(ws, columns)
    return wb, ws


def write_xlsx(out_path, run_started, records, label=""):
    wb, ws = _open_sheet(out_path, "Results", COLUMNS)

    for rec in records:
        meta = rec["meta"]
        for row in rec["rows"]:
            ws.append([
                run_started, label, rec.get("version"), rec["model_key"], rec["mode"],
                row["engine_label"], row["status"], row["cold_s"], row["warm_raw_med_s"],
                row["warm_raw_min_s"], row["warm_raw_max_s"], row["warm_den_med_s"],
                row["denoise_plus_s"], row.get("cold_render_path"), row["devices"],
                meta.get("gpu"), meta.get("version"), meta.get("resolution"),
                meta.get("cycles_samples"), meta.get("invariants_source"),
                meta.get("compositor"), row["reason"],
                rec["run_status"], os.path.relpath(rec["log_path"], SCRIPT_DIR),
            ])

    for i, _ in enumerate(COLUMNS, start=1):
        ws.column_dimensions[get_column_letter(i)].width = 18
    _set_width(ws, COLUMNS, "Label", 34)
    _set_width(ws, COLUMNS, "Cold Render Path", 50)
    _set_width(ws, COLUMNS, "Devices", 30)
    _set_width(ws, COLUMNS, "Reason", 30)
    _set_width(ws, COLUMNS, "Log File", 40)

    wb.save(out_path)


def write_interactive_xlsx(out_path, run_started, records, label=""):
    """Append interactive-suite results to the 'Interactive' sheet (one row per
    metric per model), preserving the render 'Results' sheet in the same file."""
    wb, ws = _open_sheet(out_path, "Interactive", INTERACTIVE_COLUMNS)
    for rec in records:
        meta = rec["meta"]
        for row in rec["rows"]:
            ws.append([
                run_started, label, rec.get("version"), rec["model_key"], rec["mode"],
                row["metric"], row["status"], row.get("median_ms"), row.get("min_ms"),
                row.get("max_ms"), row.get("derived"), meta.get("frames"),
                meta.get("gpu"), meta.get("version"), meta.get("invariants_source"),
                row.get("reason"), rec["run_status"],
                os.path.relpath(rec["log_path"], SCRIPT_DIR),
            ])
    for i, _ in enumerate(INTERACTIVE_COLUMNS, start=1):
        ws.column_dimensions[get_column_letter(i)].width = 18
    _set_width(ws, INTERACTIVE_COLUMNS, "Label", 34)
    _set_width(ws, INTERACTIVE_COLUMNS, "Reason", 30)
    _set_width(ws, INTERACTIVE_COLUMNS, "Log File", 40)
    wb.save(out_path)


# ---------------------------------------------------------------------------
# CLI -- every flag here overrides the matching benchmark_config.toml value;
# leaving a flag unset means "use whatever the config says."
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=DEFAULT_CONFIG_PATH,
                   help=f"path to config TOML (default: {DEFAULT_CONFIG_PATH})")
    p.add_argument("--mode", choices=["test", "full"], default=None,
                   help="test = --test (fast, non-representative); full = real benchmark run "
                        "(default: [run].mode in config)")
    p.add_argument("--suite", choices=["cycles", "eevee", "archviz", "interactive", "all"],
                   default=None,
                   help="cycles = CUDA+OptiX; eevee = EEVEE; archviz = all three engines; "
                        "interactive = headless depsgraph evaluation; all = cycles+eevee")
    p.add_argument("--models", default=None,
                   help="comma-separated model keys from config, or full .blend paths, or 'all'. "
                        "Default: the models listed for the selected --blender-version.")
    p.add_argument("--engines", default=None,
                   help="comma-separated engines; default: [blender].<suite>_engines")
    p.add_argument("--warm", type=int, default=None, help="override warm-run count passed to engine_benchmark.py")
    p.add_argument("--timeout", type=int, default=None,
                   help="per (model, engine) subprocess timeout in seconds (default: mode-dependent, from config)")
    p.add_argument("--blender", default=None,
                   help="full path to a blender.exe -- raw override, wins over --blender-version")
    p.add_argument("--blender-version", default=None,
                   help="select Blender by line or exact build, comma-separated (e.g. 4.5, "
                        "4.5.0, or 3.3,4.5.0). A line runs its default build; a build runs "
                        "with its line's configured models.")
    p.add_argument("--all-versions", action="store_true",
                   help="run EVERY version in [blender.versions] (each with its own models), "
                        "all into one results .xlsx for cross-version comparison")
    p.add_argument("--all-builds", action="store_true",
                   help="expand each selected line into ALL of its builds (e.g. 4.5 -> "
                        "4.5.0, 4.5.9), to compare patch releases of the same line")
    p.add_argument("--list-versions", action="store_true",
                   help="print the configured [blender.versions] (exe + models) and exit")
    p.add_argument("--script", default=None, help="default: [blender].script in config")
    p.add_argument("--log-dir", default=None, help="default: [run].log_dir in config")
    p.add_argument("--out", default=None, help="default: [run].output_xlsx in config")
    p.add_argument("--label", default=None,
                   help="free-text label recorded in the Label column (e.g. the test "
                        "build/host/GPU, or 'native') -- the config the logs can't capture")
    p.add_argument("--save-renders", action="store_true", default=None,
                   help="write each cold frame to disk for visual comparison later "
                        "(default: [run].save_renders in config, normally off)")
    p.add_argument("--render-dir", default=None, help="default: [run].render_dir in config")
    p.add_argument("--dry-run", action="store_true", help="print commands, execute nothing")
    p.add_argument("--extra-arg", action="append", default=[],
                   help="extra pass-through arg to engine_benchmark.py, repeatable "
                        "(e.g. --extra-arg --compositor --extra-arg GPU)")
    return p.parse_args()


def resolve_models(spec, config_models):
    if spec == "all":
        return dict(config_models)
    out = {}
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if token in config_models:
            out[token] = config_models[token]
        else:
            key = os.path.splitext(os.path.basename(token))[0]
            out[key] = token
    return out


def suites_requested(args, run_cfg):
    """Which test suite(s) to run. 'all' = the render race (cycles+eevee); the
    Tier-1 'interactive' suite is opt-in (different measurement + its own sheet)."""
    s = (args.suite or run_cfg.get("suite") or "cycles").lower()
    if s == "all":
        return ["cycles", "eevee"]
    if s not in ("cycles", "eevee", "archviz", "interactive"):
        sys.exit(f"--suite must be cycles|eevee|archviz|interactive|all "
                 f"(got '{s}')")
    return [s]


SUITE_DEFAULT_ENGINES = {
    "cycles": ["CUDA", "OPTIX"],
    "eevee": ["EEVEE"],
    "archviz": ["CUDA", "OPTIX", "EEVEE"],
}


def engines_for_suite(suite, blender_cfg, args):
    """Engines for a suite: --engines overrides; else [blender].<suite>_engines
    ('-' in a suite name is '_' in its config key)."""
    if args.engines:
        return [e.strip().upper() for e in args.engines.split(",") if e.strip()]
    key = f"{suite.replace('-', '_')}_engines"
    return [e.upper() for e in (blender_cfg.get(key) or SUITE_DEFAULT_ENGINES[suite])]


def models_for_suite(version_cfg, suite):
    """Catalog keys a version runs for a suite (cycles_models / eevee_models /
    interactive_models). None => raw/legacy exe with no per-suite lists (caller
    falls back to catalog).

    The interactive suite measures depsgraph RE-EVAL, which is engine-independent
    -- so with no explicit interactive_models it runs on the UNION of this version's
    cycles_models + eevee_models (both scene sets, deduped), not just the eevee ones.
    Single-frame scenes simply skip the frame-scrub metric; transform/subsurf still run."""
    if not isinstance(version_cfg, dict):
        return None
    if suite == "interactive" and "interactive_models" not in version_cfg:
        merged = []
        for k in (list(version_cfg.get("cycles_models", []))
                  + list(version_cfg.get("eevee_models", []))):
            if k not in merged:
                merged.append(k)
        return merged or version_cfg.get("models", [])
    return version_cfg.get(f"{suite}_models", version_cfg.get("models", []))


def main():
    args = parse_args()
    cfg = load_config(args.config)
    run_cfg, blender_cfg = cfg["run"], cfg["blender"]

    if args.list_versions:
        print_versions(blender_cfg)
        return

    mode = args.mode or run_cfg["mode"]
    label = args.label or ""
    requested_suites = suites_requested(args, run_cfg)
    warm = args.warm if args.warm is not None else run_cfg.get("warm")
    benchmark_script = args.script or blender_cfg["script"]
    # Interactive harness: [blender].interactive_script if set, else sits next to
    # the render script (both ship in the repo root together).
    if blender_cfg.get("interactive_script"):
        interactive_script = _resolve(cfg["_base_dir"], blender_cfg["interactive_script"])
    else:
        interactive_script = os.path.join(os.path.dirname(benchmark_script),
                                          "interactive_benchmark.py")
    log_dir = args.log_dir or run_cfg["log_dir"]
    save_renders = args.save_renders if args.save_renders is not None else run_cfg["save_renders"]
    render_dir = args.render_dir or run_cfg["render_dir"]
    timeout_sec = args.timeout or (run_cfg["timeout_test_sec"] if mode == "test" else run_cfg["timeout_full_sec"])

    # Which Blender(s) to run -> list of (log_tag, exe, version_models, label).
    #   --blender <path> : one raw exe   |  --all-versions : every configured version
    #   --blender-version a,b : those keys  |  else : the config default_version
    # The tag prefixes log files so the same model across versions doesn't collide.
    all_versions = blender_cfg.get("versions") or {}
    if args.blender:
        exe, _vm, lbl = select_blender(blender_cfg, args.blender, None)
        selections = [(None, exe, lbl)]
    elif not all_versions:
        exe, _vm, lbl = select_blender(blender_cfg, None, None)   # legacy single exe
        selections = [(None, exe, lbl)]
    else:
        if args.all_versions:
            keys = sorted(all_versions)
        elif args.blender_version:
            keys = [k.strip() for k in args.blender_version.split(",") if k.strip()]
        else:
            select_blender(blender_cfg, None, None)   # exits if no default_version
            keys = [blender_cfg["default_version"]]
        selections = [s for k in keys for s in expand_selection(blender_cfg, k, args.all_builds)]

    if not os.path.isfile(benchmark_script):
        sys.exit(f"engine_benchmark.py not found: {benchmark_script}")
    if "interactive" in requested_suites and not args.dry_run \
            and not os.path.isfile(interactive_script):
        sys.exit(f"interactive_benchmark.py not found: {interactive_script}")

    now = datetime.datetime.now()
    run_started = now.strftime("%Y-%m-%d %H:%M:%S")
    # One timestamped .xlsx per invocation; all versions share it, so the
    # "Blender Version" column lets you compare across them. --out appends to a fixed file.
    if args.out:
        out_xlsx = args.out
    else:
        root, ext = os.path.splitext(run_cfg["output_xlsx"])
        out_xlsx = f"{root}_{now.strftime('%Y%m%d_%H%M%S')}{ext or '.xlsx'}"
    os.makedirs(os.path.dirname(os.path.abspath(out_xlsx)), exist_ok=True)
    print(f"=== run_benchmark_suite: {len(selections)} version(s) x suites={requested_suites}  "
          f"mode={mode} save_renders={save_renders} -> {os.path.basename(out_xlsx)} ===")

    # Write after EVERY (version, suite, model, engine) result -- a crash partway
    # (legacy Eevee, VRAM-heavy scenes) never loses what is already measured.
    total_rows = 0
    for vkey, blender_exe, blabel in selections:
        if not args.dry_run and not os.path.isfile(blender_exe):
            print(f"[warn] {blabel}: blender.exe not found, skipping version: {blender_exe}")
            continue
        version_cfg = version_cfg_for(blender_cfg, vkey)
        for suite in requested_suites:
            # per-suite models: explicit --models wins; else the version's
            # {suite}_models list; else (raw/legacy exe) the whole catalog.
            if args.models:
                models = resolve_models(args.models, cfg["models"])
            else:
                mkeys = models_for_suite(version_cfg, suite)
                models = (dict(cfg["models"]) if mkeys is None
                          else {k: cfg["models"][k] for k in mkeys if k in cfg["models"]})
            for key, path in list(models.items()):
                if not args.dry_run and not os.path.isfile(path):
                    print(f"[warn] {blabel}/{suite}: model '{key}' not found, skipping: {path}")
            models = {k: v for k, v in models.items() if args.dry_run or os.path.isfile(v)}
            if not models:
                print(f"    {blabel} / {suite}: no models to run")
                continue
            tag = f"{vkey}_{suite}_" if vkey else f"{suite}_"

            # --- Tier-1 interactive suite: no engine dimension, own sheet ---
            if suite == "interactive":
                print(f"\n--- {blabel} / interactive   models={list(models)} ---")
                for model_key, model_path in models.items():
                    rec = run_one_interactive(blender_exe, interactive_script, model_key,
                                              model_path, mode, warm, args.extra_arg,
                                              log_dir, timeout_sec, args.dry_run,
                                              tag=tag, version=vkey)
                    if not args.dry_run:
                        write_interactive_xlsx(out_xlsx, run_started, [rec], label=label)
                        total_rows += len(rec["rows"])
                        print(f"       -> saved to {out_xlsx} ({total_rows} row(s) so far)")
                continue

            # --- render race (cycles / eevee) ---
            engines = engines_for_suite(suite, blender_cfg, args)
            print(f"\n--- {blabel} / {suite}   engines={engines}   models={list(models)} ---")
            for model_key, model_path in models.items():
                for engine in engines:
                    rec = run_one(blender_exe, benchmark_script, model_key, model_path, engine,
                                  mode, warm, save_renders, render_dir, args.extra_arg,
                                  log_dir, timeout_sec, args.dry_run, tag=tag, version=vkey)
                    if not args.dry_run:
                        write_xlsx(out_xlsx, run_started, [rec], label=label)
                        total_rows += len(rec["rows"])
                        print(f"       -> saved to {out_xlsx} ({total_rows} row(s) so far)")

    if args.dry_run:
        print("\n[dry-run] no results written.")
        return
    print(f"\n=== done: wrote {total_rows} row(s) to {out_xlsx} ===")


if __name__ == "__main__":
    main()
