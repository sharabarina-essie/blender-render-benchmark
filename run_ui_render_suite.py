"""
run_ui_render_suite.py
======================
Drive ui_render_test.py (F12 render in a full-screen Blender UI) across Blender
builds and render-setting permutations: Cycles on OptiX / CUDA / CPU x denoiser
(none, OIDN on CPU, OIDN on GPU, OptiX), plus EEVEE and Workbench.

One Blender session per build runs all its perms. If Blender crashes, or a perm
makes no progress for --timeout seconds, that perm is recorded as CRASH / TIMEOUT
and a fresh session resumes with the remaining perms -- one bad combination never
costs the rest of the matrix. A window "Not Responding" for 2 minutes while
Blender uses (almost) no CPU is a HANG.

Broken scenarios are never repeated:
  * every crash / hang / timeout is recorded in results/ui_render_broken.json
    (build, model, perm, status, log) and skipped by later runs (--retry-broken
    to re-test, --list-broken to print the list for a bug report);
  * after a Cycles denoise mode breaks, the build's remaining perms with that
    denoiser are UNTESTED rather than run into the same bug;
  * two sessions in a row dying before any render finishes = the build is broken.

Usage (system Python 3.11+, from the repo root):
  python run_ui_render_suite.py --test                          # default build, all perms, smoke
  python run_ui_render_suite.py --test --all-versions           # every line's default build
  python run_ui_render_suite.py --test --all-versions --all-builds   # every installed build
  python run_ui_render_suite.py --blender-version 4.5.9 --perms gpu,raster --runs 2 --label native
  python run_ui_render_suite.py --list-perms
  python run_ui_render_suite.py --golden --all-versions --label native   # baseline, nothing skipped

At the end the session's images are validated (validate_ui_renders.py: sanity,
display vs render, device and denoiser agreement, golden references; --save-reference
to bless this run, --no-validate to skip).

Rows land in results/ui_render.csv (one per render, tagged with this run's session id);
per-session Blender logs in logs/ui_render_<session>_<build>_<n>.log. A summary
table (build x perm) prints at the end.
"""

import argparse
import csv
import ctypes
import json
import os
import re
import subprocess
import sys
import time
from ctypes import wintypes

from run_benchmark_suite import (DEFAULT_CONFIG_PATH, expand_selection, load_config,
                                 select_blender, version_cfg_for)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UI_SCRIPT = os.path.join(SCRIPT_DIR, "ui_render_test.py")
CSV_PATH = os.path.join(SCRIPT_DIR, "results", "ui_render.csv")
BROKEN_PATH = os.path.join(SCRIPT_DIR, "results", "ui_render_broken.json")
DEFAULT_MODEL = "archviz_classroom"
# A deadlock = window "Not Responding" for HANG_SEC while Blender is idle. Legacy EEVEE
# compiles shaders on the main thread (window also Not Responding) but keeps a core busy.
HANG_SEC = 120
HANG_MAX_CPU_SEC = 10.0

# Keep in sync with ui_render_test.PERMS / GROUPS (that module imports bpy).
COMPUTES = ("optix", "cuda", "cpu")
DENOISES = ("none", "oidn-cpu", "oidn-gpu", "optix")
PERMS = ([f"cycles-{c}-{d}" for c in COMPUTES if c != "cpu" for d in DENOISES]
         + ["eevee", "workbench"] + [f"cycles-cpu-{d}" for d in DENOISES])
GROUPS = {
    "all": PERMS,
    "cycles": [p for p in PERMS if p.startswith("cycles-")],
    "gpu": [p for p in PERMS if p.startswith(("cycles-optix-", "cycles-cuda-"))],
    "cpu": [p for p in PERMS if p.startswith("cycles-cpu-")],
    "raster": ["eevee", "workbench"],
}
RESULT_RE = re.compile(r"\[ui-render\] RESULT (\S+) run(\d+) (\S+) (\S+)")
DONE_RE = re.compile(r"\[ui-render\] PERM_DONE (\S+)")


def expand_perms(spec):
    out = []
    for tok in (t.strip().lower() for t in spec.split(",") if t.strip()):
        for p in GROUPS.get(tok, [tok]):
            if p not in PERMS:
                sys.exit(f"unknown perm '{tok}'. Perms: {', '.join(PERMS)}; "
                         f"groups: {', '.join(GROUPS)}")
            if p not in out:
                out.append(p)
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    p.add_argument("--blender-version", default=None,
                   help="comma-separated lines or exact builds (default: config default_version)")
    p.add_argument("--all-versions", action="store_true", help="every configured line")
    p.add_argument("--all-builds", action="store_true",
                   help="expand each line into all of its builds (else its default build)")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"model key (default {DEFAULT_MODEL})")
    p.add_argument("--perms", default="all", help="perms / groups (default all)")
    p.add_argument("--device", default="OPTIX", choices=["OPTIX", "CUDA"],
                   help="GPU backend in Preferences for CPU-compute perms (GPU denoise)")
    p.add_argument("--gpu", default=None, help="only GPUs whose name contains this")
    p.add_argument("--runs", type=int, default=1, help="renders per perm (1 = cold only)")
    p.add_argument("--samples", type=int, default=None, help="override Cycles samples")
    p.add_argument("--test", action="store_true",
                   help="smoke: 480x270, 16 Cycles / 8 EEVEE samples -- quick, not representative")
    p.add_argument("--label", default="", help="free text recorded per row")
    p.add_argument("--timeout", type=int, default=None,
                   help="seconds a perm may run without finishing (default 600 test / 5400 full)")
    p.add_argument("--retry-broken", action="store_true",
                   help="rerun scenarios listed in results/ui_render_broken.json")
    p.add_argument("--list-broken", action="store_true",
                   help="print the known-broken scenarios (for the dev report) and exit")
    p.add_argument("--no-validate", action="store_true",
                   help="skip the image validation (validate_ui_renders.py) at the end")
    p.add_argument("--save-reference", action="store_true",
                   help="store this run's sane renders as golden references for later runs")
    p.add_argument("--force-reference", action="store_true",
                   help="let --save-reference/--golden replace references saved under another label")
    p.add_argument("--list-perms", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="print commands, run nothing")
    p.add_argument("--golden", action="store_true",
                   help="baseline run: nothing is skipped (ignores the broken list, no "
                        "same-denoiser or whole-build skips) and renders become the references")
    args = p.parse_args()
    if args.golden:
        args.retry_broken = args.save_reference = True
    return args


def build_cmd(exe, model_path, perms, args, session):
    cmd = [exe, "-W", "--enable-autoexec", model_path, "--python", UI_SCRIPT, "--",
           "--perms", ",".join(perms), "--device", args.device, "--runs", str(args.runs),
           "--session", session]
    if args.test:
        cmd.append("--test")
    if args.label:
        cmd += ["--label", args.label]
    if args.gpu:
        cmd += ["--gpu", args.gpu]
    if args.samples:
        cmd += ["--samples", str(args.samples)]
    return cmd


def _kill_tree(proc):
    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                   capture_output=True, check=False)


def _scan(log_path, results, done):
    """Pick up RESULT / PERM_DONE lines Blender has written so far."""
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return
    for perm, run, status, secs in RESULT_RE.findall(text):
        results.setdefault(perm, {})[int(run)] = (status, secs)
    for perm in DONE_RE.findall(text):
        if perm not in done:
            done.append(perm)


def _append_failure_row(session, label, build, model, perm, status, log_path, test):
    """CSV row for a perm the Blender process never reported (crash / hang)."""
    if os.path.isfile(CSV_PATH):
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), [])
    else:
        return
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=header, extrasaction="ignore").writerow({
            "session": session, "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "label": label, "blender": build, "scene": model, "perm": perm,
            "test_mode": test, "status": f"{status}: see {os.path.relpath(log_path, SCRIPT_DIR)}",
        })


def load_broken():
    try:
        with open(BROKEN_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return []


def find_broken(broken, build, model, perm):
    """Registry entry covering this scenario (the perm itself or the whole build)."""
    for e in broken:
        if e["build"] == build and e["model"] == model and e["perm"] in (perm, "*"):
            return e
    return None


def mark_broken(broken, build, model, perm, status, log_path, session, note=""):
    if find_broken(broken, build, model, perm):
        return
    broken.append({"build": build, "model": model, "perm": perm, "status": status,
                   "log": os.path.relpath(log_path, SCRIPT_DIR), "session": session,
                   "first_seen": time.strftime("%Y-%m-%d %H:%M:%S"), "note": note})
    os.makedirs(os.path.dirname(BROKEN_PATH), exist_ok=True)
    with open(BROKEN_PATH, "w", encoding="utf-8") as f:
        json.dump(broken, f, indent=2)


def _denoise_of(perm):
    return perm.split("-", 2)[2] if perm.startswith("cycles-") else None


def _window_hung(pid):
    """True if every visible top-level window of pid is 'Not Responding'."""
    user32 = ctypes.windll.user32
    hwnds = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def _cb(hwnd, _):
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid and user32.IsWindowVisible(hwnd):
            hwnds.append(hwnd)
        return True

    user32.EnumWindows(_cb, 0)
    return bool(hwnds) and all(user32.IsHungAppWindow(h) for h in hwnds)


def _cpu_seconds(proc):
    """User + kernel CPU time Blender has used so far."""
    times = [wintypes.FILETIME() for _ in range(4)]
    ctypes.windll.kernel32.GetProcessTimes(int(proc._handle), *map(ctypes.byref, times))
    return sum((t.dwHighDateTime << 32 | t.dwLowDateTime) for t in times[2:]) / 1e7


def run_build(vkey, exe, model_key, model_path, perms, args, session, log_dir, timeout,
              broken):
    """Run all perms for one build, resuming after crashes/hangs. Returns {perm: cell}.
    Crashes/hangs go to the broken registry; known-broken scenarios are not rerun."""
    remaining, cells = [], {}
    for perm in perms:
        e = None if args.retry_broken else find_broken(broken, vkey, model_key, perm)
        if e:
            cells[perm] = f"BROKEN ({e['status'].split(':')[0]})"
            print(f"    {perm:<24} {cells[perm]} -- known, skipped (see {e['log']})", flush=True)
        else:
            remaining.append(perm)
    results, done = {}, []
    attempt = idle_sessions = 0
    while remaining:
        attempt += 1
        log_path = os.path.join(log_dir, f"ui_render_{session}_{vkey}_{attempt}.log")
        cmd = build_cmd(exe, model_path, remaining, args, session)
        print(f"  [{vkey}] session {attempt}: {len(remaining)} perm(s) -> "
              f"{os.path.relpath(log_path, SCRIPT_DIR)}", flush=True)
        if args.dry_run:
            print("    " + subprocess.list2cmdline(cmd))
            return {p: "DRY_RUN" for p in perms}
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=SCRIPT_DIR)
            last_progress, seen = time.time(), 0
            hung_since = hung_cpu = None
            status = None
            while proc.poll() is None:
                time.sleep(2)
                _scan(log_path, results, done)
                progress = len(done) + sum(len(v) for v in results.values())
                if progress != seen:
                    seen, last_progress = progress, time.time()
                    for perm in done:
                        if perm in remaining:
                            remaining.remove(perm)
                            cells[perm] = _cell(results.get(perm, {}))
                            print(f"    {perm:<24} {cells[perm]}", flush=True)
                if _window_hung(proc.pid):
                    if hung_since is None:
                        hung_since, hung_cpu = time.time(), _cpu_seconds(proc)
                    elif time.time() - hung_since > HANG_SEC:
                        busy = _cpu_seconds(proc) - hung_cpu
                        if busy < HANG_MAX_CPU_SEC:
                            _kill_tree(proc)
                            status = (f"HANG (window not responding {HANG_SEC}s, "
                                      f"{busy:.1f}s CPU)")
                            break
                        hung_since = None     # busy (e.g. shader compile): look again
                else:
                    hung_since = None
                if time.time() - last_progress > timeout:
                    _kill_tree(proc)
                    status = "TIMEOUT"
                    break
            proc.wait()
        _scan(log_path, results, done)
        for perm in done:
            if perm in remaining:
                remaining.remove(perm)
                cells[perm] = _cell(results.get(perm, {}))
                print(f"    {perm:<24} {cells[perm]}", flush=True)
        if not remaining:
            break
        culprit = remaining.pop(0)
        status = status or ("CRASH" if proc.returncode else "NO_RESULT")
        if proc.returncode and status == "CRASH":
            status = f"CRASH 0x{proc.returncode & 0xFFFFFFFF:08X}"
        cells[culprit] = f"BROKEN ({status})"
        _append_failure_row(session, args.label, vkey, model_key, culprit, status,
                            log_path, args.test)
        idle_sessions = idle_sessions + 1 if seen == 0 else 0
        if idle_sessions >= 2 and not args.golden:
            print(f"    {culprit:<24} {status} -- second session in a row died before "
                  f"finishing any render; build marked broken", flush=True)
            mark_broken(broken, vkey, model_key, "*", status, log_path, session,
                        f"every UI render crashes (first seen on {culprit})")
            for perm in remaining:
                cells[perm] = "BROKEN (build)"
            break
        mark_broken(broken, vkey, model_key, culprit, status, log_path, session)
        print(f"    {culprit:<24} {status} -- marked broken", flush=True)
        # Same Cycles denoise mode later in this build: almost certainly the same bug.
        dn = _denoise_of(culprit)
        if dn and dn != "none" and not args.golden:
            for perm in [p for p in remaining if _denoise_of(p) == dn]:
                remaining.remove(perm)
                cells[perm] = f"UNTESTED ({dn} broke in {culprit})"
                print(f"    {perm:<24} {cells[perm]}", flush=True)
    return cells


def _cell(runs):
    """Summary cell: 'OK 12.3s' (cold) or 'OK 41.0s / 9.8s' (cold / last warm)."""
    if not runs:
        return "NO_RESULT"
    first = runs[min(runs)]
    if first[0] == "SKIPPED":
        return "SKIPPED"
    if first[0] != "OK":
        return first[0]
    cold = f"{float(first[1]):.1f}s"
    if len(runs) > 1:
        last = runs[max(runs)]
        return f"OK {cold} / {float(last[1]):.1f}s" if last[0] == "OK" else f"OK {cold} / {last[0]}"
    return f"OK {cold}"


def print_summary(table, perms):
    builds = list(table)
    widths = {b: 2 + max(len(b), *(len(table[b].get(p, "-")) for p in perms)) for b in builds}
    print("\n=== UI render matrix " + "=" * 50)
    print(f"{'perm':<24}" + "".join(f"{b:<{widths[b]}}" for b in builds))
    for perm in perms:
        print(f"{perm:<24}" + "".join(f"{table[b].get(perm, '-'):<{widths[b]}}" for b in builds))


def print_broken(broken):
    print(f"\n=== Known-broken scenarios ({os.path.relpath(BROKEN_PATH, SCRIPT_DIR)}) ===")
    if not broken:
        print("  none")
    for e in broken:
        perm = "ALL perms" if e["perm"] == "*" else e["perm"]
        note = f" -- {e['note']}" if e.get("note") else ""
        print(f"  {e['build']:<8} {e['model']:<18} {perm:<24} {e['status']}{note}\n"
              f"           first seen {e['first_seen']}, log {e['log']}")


def main():
    args = parse_args()
    if args.list_broken:
        print_broken(load_broken())
        return
    if args.list_perms:
        print("perms : " + ", ".join(PERMS))
        for g, ps in GROUPS.items():
            print(f"{g:<7}: {', '.join(ps)}")
        return
    perms = expand_perms(args.perms)
    timeout = args.timeout or (600 if args.test else 5400)

    cfg = load_config(args.config)
    blender_cfg = cfg["blender"]
    versions = blender_cfg.get("versions") or {}
    if args.all_versions:
        keys = sorted(versions, key=lambda k: [int(x) for x in k.split(".")])
    elif args.blender_version:
        keys = [k.strip() for k in args.blender_version.split(",") if k.strip()]
    else:
        select_blender(blender_cfg, None, None)
        keys = [blender_cfg["default_version"]]
    selections = [s for k in keys for s in expand_selection(blender_cfg, k, args.all_builds)]

    model_path = cfg["models"].get(args.model)
    if not model_path:
        sys.exit(f"unknown model key '{args.model}'")
    log_dir = cfg["run"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    session = time.strftime("%Y%m%d_%H%M%S")
    print(f"=== run_ui_render_suite {session}: {len(selections)} build(s) x {len(perms)} perm(s) "
          f"| model {args.model} | {'TEST' if args.test else 'FULL'} | runs {args.runs} "
          f"{'| GOLDEN (nothing skipped) ' if args.golden else ''}"
          f"| per-perm timeout {timeout}s ===", flush=True)

    table = {}
    broken = load_broken()
    for vkey, exe, label in selections:
        vcfg = version_cfg_for(blender_cfg, vkey) or {}
        allowed = (set(vcfg.get("archviz_models", [])) | set(vcfg.get("heavy_models", []))
                   | set(vcfg.get("models", [])))
        print(f"\n{label}", flush=True)
        if not args.dry_run and not os.path.isfile(exe):
            print(f"  [skip] blender.exe not found: {exe}")
            table[vkey] = {p: "MISSING" for p in perms}
            continue
        if allowed and args.model not in allowed:
            print(f"  [skip] {args.model} is not listed for this line (can't open it?)")
            table[vkey] = {p: "NO_MODEL" for p in perms}
            continue
        table[vkey] = run_build(vkey, exe, args.model, model_path, perms, args,
                                session, log_dir, timeout, broken)

    print_summary(table, perms)
    print_broken(broken)
    print(f"\nrows: {CSV_PATH} (session {session})")
    if not args.dry_run and not args.no_validate:
        from validate_ui_renders import validate
        validate(session, save_reference=args.save_reference, force=args.force_reference)


if __name__ == "__main__":
    main()
