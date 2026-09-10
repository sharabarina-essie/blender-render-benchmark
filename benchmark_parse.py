"""
benchmark_parse.py
==================
The ONE place that understands engine_benchmark.py's printed output. Both suite
runners (run_benchmark_suite.py, run_compositor_suite.py) import from here rather
than embedding their own regexes, so:

  * parsing is decoupled from ORCHESTRATION -- the suites only drive Blender and
    hand the captured stdout to these pure functions (text in, dict/rows out); and
  * parsing is decoupled from the TEST -- engine_benchmark.py never imports this;
    it just prints. The format contract lives here, in one module, not copied
    across suites.

COUPLING THAT REMAINS (by design, and isolated to this file): these parsers mirror
engine_benchmark.py's human-readable tables (print_table / print_compositor_table)
by regex -- that printed format IS the contract. Change those print_* functions and
you update the regexes here, and ONLY here. The stricter decoupling is to have
engine_benchmark.py also emit a machine-readable JSON block that these functions
read instead of scraping the table; if/when we do that, it lands in this module.
"""

import re


def _num(token):
    """Parse one table cell to float, or None for blanks / 'n/a'."""
    token = token.strip()
    if not token or "n/a" in token.lower():
        return None
    token = token.lstrip("+")  # Denoise+ column always prepends '+', even when negative
    try:
        return float(token)
    except ValueError:
        return None


# ===========================================================================
# Engine-race table  (mirrors print_table() in engine_benchmark.py)
# ===========================================================================
HEADER_RE = re.compile(
    r"Blender\s+(?P<version>\S+)\s+BENCHMARK\s+GPU=(?P<gpu>.+?)\s+"
    r"profile=(?P<profile>\S+)\s+(?P<res>\d+x\d+)\s+"
    r"cycles_samples=(?P<samples>\d+)\s*\(adaptive OFF\)(?P<tag>.*)"
)
INVARIANTS_RE = re.compile(r"invariants source:\s*(?P<inv>.+)")
COMPOSITOR_RE = re.compile(r"compositor:\s*(?P<comp>.+)")
SPREAD_RE = re.compile(
    r"^\s{4}(?P<name>.+?):\s+warm raw spread\s+(?P<min>[\d.]+)-(?P<max>[\d.]+)s"
    r"(?:, devices \[(?P<devices>.*)\])?\s*$",
    re.MULTILINE,
)
REASON_RE = re.compile(r"\((?P<reason>[^()]*)\)\s*$")
RENDER_SAVED_RE = re.compile(r"^\s*Saved cold render -> (?P<path>.+)$", re.MULTILINE)


def parse_output(text):
    """Extract header metadata + one row per engine from a raw Blender log.
    Returns (meta_dict, [row_dict, ...]). Tolerant of a truncated/crashed log:
    only the final summary table and spread footnote are required per row, and
    engines never reached simply don't produce a row."""
    meta = {"version": None, "gpu": None, "profile": None, "resolution": None,
            "cycles_samples": None, "test_tag": None, "invariants_source": None,
            "compositor": None}

    m = HEADER_RE.search(text)
    if m:
        meta.update(
            version=m.group("version"), gpu=m.group("gpu").strip(),
            profile=m.group("profile"), resolution=m.group("res"),
            cycles_samples=m.group("samples"),
            test_tag=m.group("tag").strip() or None,
        )
    m = INVARIANTS_RE.search(text)
    if m:
        meta["invariants_source"] = m.group("inv").strip()
    m = COMPOSITOR_RE.search(text)
    if m:
        meta["compositor"] = m.group("comp").strip()

    spreads = {}
    for m in SPREAD_RE.finditer(text):
        spreads[m.group("name").strip()] = {
            "min": _num(m.group("min")), "max": _num(m.group("max")),
            "devices": m.group("devices") or None,
        }

    rows = []
    in_table = False
    for line in text.splitlines():
        if line.startswith("Engine") and "Status" in line and "|" in line:
            in_table = True
            continue
        if not in_table:
            continue
        if line.startswith("-" * 10) or line.startswith("=" * 10):
            if rows:  # second divider = end of table
                break
            continue
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 6:
            continue
        name, status = parts[0], parts[1]
        row = {"engine_label": name, "status": status}
        if status == "OK":
            row["cold_s"] = _num(parts[2])
            row["warm_raw_med_s"] = _num(parts[3])
            row["warm_den_med_s"] = _num(parts[4])
            row["denoise_plus_s"] = _num(parts[5])
            sp = spreads.get(name, {})
            row["warm_raw_min_s"] = sp.get("min")
            row["warm_raw_max_s"] = sp.get("max")
            row["devices"] = sp.get("devices")
            row["reason"] = None
        else:  # SKIP
            rm = REASON_RE.search(parts[5])
            row.update(cold_s=None, warm_raw_med_s=None, warm_den_med_s=None,
                       denoise_plus_s=None, warm_raw_min_s=None, warm_raw_max_s=None,
                       devices=None, reason=(rm.group("reason") if rm else None))
        rows.append(row)

    # One engine per Blender process, so at most one "Saved cold render ->" line
    # per log -- attach it to the (single) OK row.
    render_match = RENDER_SAVED_RE.search(text)
    cold_render_path = render_match.group("path").strip() if render_match else None
    for row in rows:
        row["cold_render_path"] = cold_render_path if row["status"] == "OK" else None

    return meta, rows


# ===========================================================================
# Compositor A/B table  (mirrors print_compositor_table() in engine_benchmark.py)
# ===========================================================================
COMP_HEADER_RE = re.compile(
    r"Blender\s+(?P<version>\S+)\s+COMPOSITOR A/B\s+GPU=(?P<gpu>.+?)\s+"
    r"profile=(?P<profile>\S+)\s+(?P<res>\d+x\d+)(?P<tag>.*)"
)
DEV_FOOT_RE = re.compile(
    r"^\s{4}(?P<name>.+?): devices \[(?P<devices>.*)\]\s*$", re.MULTILINE)
PARENS_RE = re.compile(r"\((?P<msg>[^()]*)\)\s*$")


def parse_compositor_output(text):
    """Extract header metadata + one row per engine from a --compositor-ab log.
    Tolerant of a truncated/crashed log and of the GPU-compositor-failed case
    (engine_benchmark prints gpu='ERR' + a parenthesised error in the delta cell)."""
    meta = {"version": None, "gpu": None, "profile": None, "resolution": None,
            "test_tag": None}
    m = COMP_HEADER_RE.search(text)
    if m:
        meta.update(version=m.group("version"), gpu=m.group("gpu").strip(),
                    profile=m.group("profile"), resolution=m.group("res"),
                    test_tag=m.group("tag").strip() or None)

    devices_by_name = {}
    for m in DEV_FOOT_RE.finditer(text):
        devices_by_name[m.group("name").strip()] = m.group("devices") or None

    rows = []
    in_table = False
    for line in text.splitlines():
        if "CPU comp med" in line and "|" in line:
            in_table = True
            continue
        if not in_table:
            continue
        if line.startswith("-" * 10) or line.startswith("=" * 10):
            if rows:            # second divider ends the table
                break
            continue
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 5:
            continue
        name, status = parts[0], parts[1]
        row = {"engine_label": name, "status": status, "cpu_comp_med": None,
               "gpu_comp_med": None, "delta": None, "gpu_error": None,
               "devices": None, "reason": None}
        if status == "OK":
            row["cpu_comp_med"] = _num(parts[2])
            gpu_val = _num(parts[3])
            if gpu_val is None:                      # GPU compositor failed -> 'ERR'
                row["gpu_comp_med"] = None
                em = PARENS_RE.search(parts[4])
                row["gpu_error"] = em.group("msg") if em else parts[4] or "unknown"
            else:
                row["gpu_comp_med"] = gpu_val
                row["delta"] = _num(parts[4])
            row["devices"] = devices_by_name.get(name)
        else:                                         # SKIP
            rm = PARENS_RE.search(parts[4] if len(parts) > 4 else "")
            row["reason"] = rm.group("msg") if rm else None
        rows.append(row)
    return meta, rows
