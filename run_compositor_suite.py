"""
run_compositor_suite.py
=======================
Orchestrates the COMPOSITOR CPU-vs-GPU A/B (engine_benchmark.py --compositor-ab)
across the compositor-bearing models and collects the CPU / GPU / delta figures
into one timestamped .xlsx.

This is the compositor counterpart to run_benchmark_suite.py (the engine-race
runner). It exists as a separate script because the A/B prints a DIFFERENT table
(CPU comp med | GPU comp med | GPU-CPU) that the engine-race parser doesn't
understand. It shares benchmark_config.toml (via load_config) so machine paths,
the model list, engines, and timeouts stay single-sourced.

WHAT THE A/B MEASURES (recap)
-----------------------------
For each (model, engine) with a real compositor tree, engine_benchmark.py times
the full warm render+compositor with scene.render.compositor_device on CPU, then
on GPU, and reports delta = GPU - CPU. delta > 0 => GPU compositing is the slower
option. Models with no meaningful compositor tree are SKIPPED by engine_benchmark.py.

WHY one Blender process per (model, engine)
-------------------------------------------
Same rationale as run_benchmark_suite.py: crash isolation (a failure voids one
row, not the whole run) and no thermal/cache carryover. Each process writes its
own timestamped log into logs/ (or whatever [run].log_dir is set to) alongside the engine-race logs.
USAGE
-----
  Validate (fast, the 3 compositor models, non-representative numbers):
    python run_compositor_suite.py --mode test

  Full A/B across the compositor models:
    python run_compositor_suite.py --mode full

  A subset / a specific engine:
    python run_compositor_suite.py --models npr,archviz --engines OPTIX
    python run_compositor_suite.py --models all      # runs all; non-compositor skip

  Dry run (print commands, execute nothing):
    python run_compositor_suite.py --mode test --dry-run

Each invocation writes its own timestamped compositor_results_<ts>.xlsx (pass
--out to append into a fixed file). Rows are written incrementally after every
(model, engine) so a crash partway loses nothing already measured.
"""

import argparse
import datetime
import os
import subprocess
import sys
import time

try:
    from openpyxl import Workbook, load_workbook
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter
except ImportError:
    sys.exit("openpyxl is required: python -m pip install openpyxl")

# Config loading is shared with the engine-race runner; output parsing is shared
# via benchmark_parse. Both are decoupled from this orchestrator.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_benchmark_suite import (load_config, SCRIPT_DIR, DEFAULT_CONFIG_PATH,
                                 decode_process_status, failure_status_label,
                                 select_blender, print_versions, engines_for_suite,
                                 expand_selection, version_cfg_for)


def comp_targets(version_cfg, blender_cfg, args):
    """(model_key, engines) pairs the compositor A/B runs for a version:
    compositor_cycles under the cycles engines, compositor_eevee under EEVEE.
    --models overrides to the given keys under the cycles engines."""
    if args.models:
        eng = engines_for_suite("cycles", blender_cfg, args)
        return [(k.strip(), eng) for k in args.models.split(",") if k.strip()]
    out = []
    if isinstance(version_cfg, dict):
        if version_cfg.get("compositor_cycles"):
            out.append((version_cfg["compositor_cycles"], engines_for_suite("cycles", blender_cfg, args)))
        if version_cfg.get("compositor_eevee"):
            out.append((version_cfg["compositor_eevee"], engines_for_suite("eevee", blender_cfg, args)))
    return out
from benchmark_parse import parse_compositor_output

# Models known (verified from run logs) to ship a meaningful compositor tree.
# engine_benchmark.py --compositor-ab SKIPS any model without one, so this is just
# the sensible default set; --models all runs everything and lets the others skip.
DEFAULT_COMPOSITOR_MODELS = ["npr", "vram8k", "archviz"]
DEFAULT_OUT_BASENAME = "compositor_results.xlsx"

# ---------------------------------------------------------------------------
# Execution  (output parsing lives in benchmark_parse.parse_compositor_output)
# ---------------------------------------------------------------------------
def build_cmd(blender_exe, benchmark_script, model_path, engine, mode, warm, extra_args):
    # --enable-autoexec: let the .blend's own Python (expression drivers) run, as the
    # GUI does -- off by default in -b/-P. Must precede the .blend.
    cmd = [blender_exe, "--enable-autoexec", "-b", model_path, "-P", benchmark_script, "--",
           "--compositor-ab", "--engine", engine]
    if mode == "test":
        cmd.append("--test")
    if warm is not None:
        cmd += ["--warm", str(warm)]
    cmd += extra_args
    return cmd


def run_one(blender_exe, benchmark_script, model_key, model_path, engine, mode,
            warm, extra_args, log_dir, timeout_sec, dry_run, tag=""):
    cmd = build_cmd(blender_exe, benchmark_script, model_path, engine, mode, warm, extra_args)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # 'compAB' marker distinguishes these from the engine-race logs in the same dir.
    log_path = os.path.join(log_dir, f"{tag}{model_key}_{engine}_compAB_{mode}_{ts}.log")

    print(f"[comp] {tag}{model_key} / {engine} ({mode})  -> {os.path.basename(log_path)}")
    if dry_run:
        print("       " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
        return {"model_key": model_key, "model_path": model_path, "engine_arg": engine,
                "mode": mode, "log_path": log_path, "run_status": "DRY_RUN",
                "meta": {}, "rows": []}

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

    meta, rows = parse_compositor_output(text)
    if run_status != "OK":
        # Non-clean exit -> DEFINITIVE failure row (a crashed process's partial
        # table is not trustworthy; don't let it read as data).
        label = failure_status_label(run_status)
        rows = [{"engine_label": engine, "status": label, "cpu_comp_med": None,
                 "gpu_comp_med": None, "delta": None, "gpu_error": None,
                 "devices": None,
                 "reason": f"{decode_process_status(run_status)} after {elapsed:.0f}s -- see log"}]

    n_ok = sum(1 for r in rows if r["status"] == "OK")
    if run_status != "OK":
        verdict, detail = failure_status_label(run_status), decode_process_status(run_status)
    elif n_ok:
        verdict, detail = "OK", f"{n_ok}/{len(rows)} engine-row(s) OK"
    else:
        verdict, detail = ("SKIPPED" if rows else "NO DATA"), f"0/{len(rows)} OK"
    flag = "" if verdict == "OK" else "   <<< SEE LOG"
    print(f"       {verdict:8} {elapsed:6.1f}s  {detail}{flag}")
    return {"model_key": model_key, "model_path": model_path, "engine_arg": engine,
            "mode": mode, "log_path": log_path, "run_status": run_status,
            "meta": meta, "rows": rows}


# ---------------------------------------------------------------------------
# Spreadsheet output
# ---------------------------------------------------------------------------
COLUMNS = [
    "Run Started", "Model", "Mode", "Engine", "Status", "CPU Comp Med (s)",
    "GPU Comp Med (s)", "GPU-CPU Delta (s)", "GPU Error", "Devices",
    "Reported GPU", "Blender Version", "Resolution", "Reason",
    "Process Status", "Log File",
]


def write_xlsx(out_path, run_started, records):
    if os.path.exists(out_path):
        wb = load_workbook(out_path)
        ws = wb.active
    else:
        wb = Workbook()
        ws = wb.active
        ws.title = "Compositor AB"
        ws.append(COLUMNS)
        for cell in ws[1]:
            cell.font = Font(bold=True)
        ws.freeze_panes = "A2"

    for rec in records:
        meta = rec["meta"]
        for row in rec["rows"]:
            ws.append([
                run_started, rec["model_key"], rec["mode"], row["engine_label"],
                row["status"], row["cpu_comp_med"], row["gpu_comp_med"], row["delta"],
                row["gpu_error"], row["devices"], meta.get("gpu"), meta.get("version"),
                meta.get("resolution"), row["reason"], rec["run_status"],
                os.path.relpath(rec["log_path"], SCRIPT_DIR),
            ])

    for i, _ in enumerate(COLUMNS, start=1):
        ws.column_dimensions[get_column_letter(i)].width = 18
    ws.column_dimensions["I"].width = 26   # GPU Error
    ws.column_dimensions["J"].width = 40   # Devices
    ws.column_dimensions["K"].width = 30   # Reported GPU
    ws.column_dimensions["P"].width = 44   # Log File
    wb.save(out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=DEFAULT_CONFIG_PATH,
                   help=f"path to config TOML (default: {DEFAULT_CONFIG_PATH})")
    p.add_argument("--mode", choices=["test", "full"], default=None,
                   help="test = --test (fast, non-representative); full = real (default: [run].mode)")
    p.add_argument("--models", default=None,
                   help="comma-separated model keys, full .blend paths, or 'all'. Default: the "
                        "selected --blender-version's models, else the known compositor-bearing "
                        f"models ({', '.join(DEFAULT_COMPOSITOR_MODELS)}).")
    p.add_argument("--engines", default=None,
                   help="comma-separated engines (default: [run].engines in config)")
    p.add_argument("--warm", type=int, default=None, help="override warm-run count")
    p.add_argument("--timeout", type=int, default=None,
                   help="per (model, engine) subprocess timeout in seconds (default: mode-dependent)")
    p.add_argument("--blender", default=None,
                   help="full path to a blender.exe -- raw override, wins over --blender-version")
    p.add_argument("--blender-version", default=None,
                   help="select Blender by line or exact build, comma-separated "
                        "(e.g. 4.5 or 4.5.0)")
    p.add_argument("--all-versions", action="store_true",
                   help="run the compositor A/B for every configured version")
    p.add_argument("--all-builds", action="store_true",
                   help="expand each selected line into ALL of its builds")
    p.add_argument("--list-versions", action="store_true",
                   help="print the configured [blender.versions] and exit")
    p.add_argument("--script", default=None, help="default: [blender].script in config")
    p.add_argument("--log-dir", default=None, help="default: [run].log_dir in config")
    p.add_argument("--out", default=None,
                   help="fixed output xlsx to append into (default: a timestamped "
                        f"{DEFAULT_OUT_BASENAME} next to the config)")
    p.add_argument("--dry-run", action="store_true", help="print commands, execute nothing")
    p.add_argument("--extra-arg", action="append", default=[],
                   help="extra pass-through arg to engine_benchmark.py, repeatable")
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


def main():
    args = parse_args()
    cfg = load_config(args.config)
    run_cfg, blender_cfg = cfg["run"], cfg["blender"]

    if args.list_versions:
        print_versions(blender_cfg)
        return

    mode = args.mode or run_cfg["mode"]
    warm = args.warm if args.warm is not None else run_cfg.get("warm")
    benchmark_script = args.script or blender_cfg["script"]
    log_dir = args.log_dir or run_cfg["log_dir"]
    timeout_sec = args.timeout or (run_cfg["timeout_test_sec"] if mode == "test"
                                   else run_cfg["timeout_full_sec"])

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

    now = datetime.datetime.now()
    run_started = now.strftime("%Y-%m-%d %H:%M:%S")
    if args.out:
        out_xlsx = args.out
    else:
        results_dir = os.path.dirname(run_cfg["output_xlsx"])
        base = os.path.join(results_dir, DEFAULT_OUT_BASENAME)
        root, ext = os.path.splitext(base)
        out_xlsx = f"{root}_{now.strftime('%Y%m%d_%H%M%S')}{ext}"
    os.makedirs(os.path.dirname(os.path.abspath(out_xlsx)), exist_ok=True)

    print(f"=== run_compositor_suite: {len(selections)} version(s)  mode={mode} "
          f"-> {os.path.basename(out_xlsx)} ===")

    total_rows = 0
    for vkey, blender_exe, blabel in selections:
        if not args.dry_run and not os.path.isfile(blender_exe):
            print(f"[warn] {blabel}: blender.exe not found, skipping: {blender_exe}")
            continue
        version_cfg = version_cfg_for(blender_cfg, vkey)
        targets = comp_targets(version_cfg, blender_cfg, args)   # [(model_key, engines), ...]
        if not targets:
            print(f"    {blabel}: no compositor models configured (compositor_cycles/_eevee)")
            continue
        tag = f"{vkey}_" if vkey else ""
        print(f"\n--- {blabel}   compositor A/B on {[t[0] for t in targets]} ---")
        for model_key, engines in targets:
            model_path = cfg["models"].get(model_key, model_key)
            if not args.dry_run and not os.path.isfile(model_path):
                print(f"[warn] {blabel}: compositor model '{model_key}' not found: {model_path}")
                continue
            for engine in engines:
                rec = run_one(blender_exe, benchmark_script, model_key, model_path, engine,
                              mode, warm, args.extra_arg, log_dir, timeout_sec, args.dry_run, tag=tag)
                if not args.dry_run:
                    write_xlsx(out_xlsx, run_started, [rec])
                    total_rows += len(rec["rows"])
                    print(f"       -> saved to {out_xlsx} ({total_rows} row(s) so far)")

    if args.dry_run:
        print("\n[dry-run] no results written.")
        return
    print(f"\n=== done: wrote {total_rows} row(s) to {out_xlsx} ===")


if __name__ == "__main__":
    main()
