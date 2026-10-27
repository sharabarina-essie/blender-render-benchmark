"""Sequential, isolated full-GUI orbit measurements on every installed Blender build."""
import argparse
import csv
from datetime import datetime
import html
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time

from run_benchmark_suite import load_config
from run_viewport_performance import digest
from viewport_performance import MODEL_KEYS, STANDARD_PROFILE, aggregate_runs, expand_cases
from keep_viewport_awake import awake_request

ROOT = Path(__file__).resolve().parent


def read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def blender_executable(build):
    direct = build / 'blender.exe'
    if direct.is_file():
        return direct
    nested = sorted(build.glob('*/blender.exe'))
    if len(nested) > 1:
        raise ValueError(f'Multiple nested Blender executables in {build}; normalize the installation first')
    return nested[0] if nested else None


def installed_builds(builds_dir):
    if not builds_dir.is_dir():
        return []
    return sorted((path for path in builds_dir.iterdir()
                   if path.is_dir() and re.fullmatch(r'\d+\.\d+\.\d+', path.name)
                   and blender_executable(path)),
                  key=lambda path: tuple(map(int, path.name.split('.'))))


def save_matrix(output, matrix):
    cells = matrix['cells']
    matrix['counts'] = {status: sum(cell['status'] == status for cell in cells)
                        for status in ('PENDING', 'RUNNING', 'PASS', 'FAIL')}
    matrix['updated_at'] = datetime.now().isoformat(timespec='seconds')
    temporary = output / 'matrix.tmp'
    temporary.write_text(json.dumps(matrix, indent=2), encoding='utf-8')
    temporary.replace(output / 'matrix.json')
    fields = ['build', 'case', 'run', 'cell_status', 'status', 'viewport_draw_fps',
              'draws', 'duration_s', 'p95_frame_ms', 'p99_frame_ms', 'one_percent_low_fps',
              'relative_stutters', 'gaps_over_500_ms', 'longest_without_new_draw_ms', 'report']
    with (output / 'trials.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for cell in cells:
            for row in cell.get('results', []):
                values = {key: row.get(key) for key in fields}
                values.update(build=cell['build'], case=cell['case'],
                              cell_status=cell['status'], report=cell['directory'] + '/report.html')
                writer.writerow(values)
    esc = lambda value: html.escape(str(value))
    table_rows = []
    for build in matrix['builds']:
        columns = [f'<th>{esc(build)}</th>']
        for case in matrix['cases']:
            cell = next(c for c in cells if c['build'] == build and c['case'] == case)
            summary = aggregate_runs(cell.get('results', [])).get(case, {})
            fps = summary.get('median_draw_fps')
            measured = '' if fps is None else f'<br>{fps:.2f} draw FPS'
            stutters = sum(row.get('relative_stutters', 0) for row in cell.get('results', []))
            count = len(cell.get('results', []))
            details = f'<br>{count}/{matrix["runs"]} trials; {stutters} stutters' if count else ''
            target = cell['directory'] + ('/report.html' if (output / cell['directory'] / 'report.html').is_file() else '/report.json')
            text = f'{esc(cell["status"])}{measured}{details}'
            if cell['status'] != 'PENDING':
                text = f'<a href="{esc(target)}">{text}</a>'
                if (output / cell['directory'] / 'quality.html').is_file():
                    text += f'<br><a href="{esc(cell["directory"] + "/quality.html")}">image quality</a>'
            columns.append(f'<td class="{cell["status"].lower()}">{text}</td>')
        table_rows.append('<tr>' + ''.join(columns) + '</tr>')
    errors = ''.join(f'<li>{esc(cell["build"])} / {esc(cell["case"])}: {esc("; ".join(cell.get("errors", [])))}</li>'
                     for cell in cells if cell['status'] == 'FAIL')
    count_text = ', '.join(f'{value} {status.lower()}' for status, value in matrix['counts'].items())
    document = f'''<!doctype html><meta charset="utf-8"><title>Blender orbit matrix</title>
<style>body{{font:16px system-ui;margin:30px;background:#fafafa;color:#222}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:12px;min-width:125px;text-align:left}}a{{color:inherit}}
.pass{{background:#e4f4e8}}.fail{{background:#ffe2e2}}.running{{background:#fff2c6}}
.note{{padding:15px;background:#fff2c6;max-width:1100px}}code{{overflow-wrap:anywhere}}</style>
<h1>Blender orbit matrix — {esc(matrix['status'])}</h1>
<p>{esc(matrix['label'])} · {esc(matrix['updated_at'])} · {esc(count_text)}</p>
<p>Scene: <code>{esc(matrix['scene'])}</code><br>
{matrix['runs']} × {matrix['duration_s']:g}s per combination; warm-up at least {matrix['warmup_s']:g}s.
OpenGL · actual 1920×1080 viewport · 64 configured samples · one Blender process at a time.</p>
<p class="note">Values are median changed-view viewport draw callbacks per second, not GPU completion or client presentation FPS.
Cycles may display a partially refined image. Compare builds within the same renderer generation and settings.
Stutters: between-draw gaps &gt; max(100ms, 3 × each trial's median interval).
PASS confirms complete measurements, size/backend and image sanity, not image equivalence or a minimum FPS.</p>
<div style="overflow-x:auto"><table><tr><th>Blender</th>{''.join(f'<th>{esc(c)}</th>' for c in matrix['cases'])}</tr>
{''.join(table_rows)}</table></div><ul>{errors}</ul>
<p><a href="matrix.json">Matrix JSON</a> · <a href="trials.csv">All trial metrics CSV</a></p>'''
    temporary = output / 'report.tmp'
    temporary.write_text(document, encoding='utf-8')
    temporary.replace(output / 'report.html')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'benchmark_config.toml')
    parser.add_argument('--blender-dir', type=Path, help='Existing portable build directory, otherwise use config builds_dir')
    parser.add_argument('--builds', help='Optional comma-separated exact build names')
    parser.add_argument('--cases', default='all')
    parser.add_argument('--runs', type=int)
    parser.add_argument('--duration', type=float)
    parser.add_argument('--warmup', type=float)
    parser.add_argument('--timeout', type=float, default=360, help='Per-combination Blender ceiling in seconds')
    parser.add_argument('--gpu', help='Optional device-name substring; also checked against the OpenGL renderer')
    parser.add_argument('--label', default='unlabeled')
    parser.add_argument('--smoke', action='store_true', help='Classroom, one 30s trial; full defaults to Medieval Hut, 3 x 30s')
    parser.add_argument('--model', help='Override the scene catalog key or .blend path')
    parser.add_argument('--quality-capture', action='store_true')
    parser.add_argument('--capture-method', choices=['desktop', 'blender'],
                        default='desktop' if sys.platform == 'win32' else 'blender')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    args.runs = args.runs if args.runs is not None else (1 if args.smoke else 3)
    args.duration = args.duration if args.duration is not None else 30
    if args.duration < 30:
        parser.error('Orbit trials must last at least 30 seconds')
    args.warmup = args.warmup if args.warmup is not None else (12 if args.smoke else 15)
    if not all(math.isfinite(v) and v > 0 for v in (args.runs, args.duration, args.warmup, args.timeout)):
        parser.error('Counts and durations must be positive')
    cfg = load_config(str(args.config))
    cases = expand_cases(args.cases)
    builds_dir = (args.blender_dir or Path(cfg['_base_dir']) / cfg['blender'].get('builds_dir', 'blender')).resolve()
    builds = installed_builds(builds_dir)
    if args.builds:
        requested = args.builds.split(',')
        if set(requested) - {path.name for path in builds}:
            parser.error('A requested build has no installed executable')
        builds = [path for path in builds if path.name in requested]
    if not builds:
        parser.error('No installed Blender executables found')
    output = args.output.resolve() if args.output else ROOT / 'results' / ('viewport_matrix_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True, exist_ok=False)
    key = args.model or MODEL_KEYS['smoke' if args.smoke else 'full']
    scene = Path(cfg['models'].get(key, key)).resolve()
    if not scene.is_file():
        parser.error(f'Scene not found: {scene}. Run setup_viewport_test.py first.')
    matrix = {'status': 'RUNNING', 'started_at': datetime.now().isoformat(timespec='seconds'),
              'capture_method': args.capture_method,
              'label': args.label, 'quality_capture': args.quality_capture,
              'scene': str(scene), 'scene_sha256': digest(scene),
              'builds': [path.name for path in builds], 'cases': cases,
              'runs': args.runs, 'duration_s': args.duration, 'warmup_s': args.warmup,
              'timeout_s': args.timeout, 'gpu_filter': args.gpu, 'quality_profile': dict(STANDARD_PROFILE),
              'harness_sha256': {name: digest(ROOT / name) for name in (
                  'run_viewport_matrix.py', 'run_viewport_performance.py',
                  'blender_viewport_performance.py', 'viewport_performance.py', 'engine_benchmark.py',
                  'keep_viewport_awake.py', 'viewport_quality.py', 'viewport_capture.py')},
              'cells': [{'build': build.name, 'case': case, 'status': 'PENDING',
                         'directory': f'{build.name}/{case}', 'results': []}
                        for build in builds for case in cases]}
    save_matrix(output, matrix)
    print(f'Matrix: {len(builds)} builds × {len(cases)} cases × {args.runs} trials -> {output}', flush=True)
    for index, cell in enumerate(matrix['cells'], 1):
        destination = output / cell['directory']
        destination.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-u', str(ROOT / 'run_viewport_performance.py'),
                   '--config', str(args.config), '--blender', str(blender_executable(builds_dir / cell['build'])),
                   '--model', str(scene), '--cases', cell['case'], '--runs', str(args.runs),
                   '--duration', str(args.duration), '--warmup', str(args.warmup),
                   '--timeout', str(args.timeout), '--label', args.label,
                   '--output', str(destination), '--capture-method', args.capture_method]
        if args.gpu:
            command += ['--gpu', args.gpu]
        if args.quality_capture:
            command.append('--quality-capture')
        cell.update(status='RUNNING', command=command, started_at=datetime.now().isoformat(timespec='seconds'))
        save_matrix(output, matrix)
        print(f'[{index}/{len(matrix["cells"])}] {cell["build"]} / {cell["case"]}: starting', flush=True)
        with (destination.parent / (cell['case'] + '.runner.log')).open('w', encoding='utf-8') as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            last_count = 0
            last_progress = time.monotonic()
            while process.poll() is None:
                report = read_json(destination / 'report.json')
                rows = report.get('results', [])
                if len(rows) != last_count:
                    last_count = len(rows)
                    cell['results'] = rows
                    save_matrix(output, matrix)
                if time.monotonic() - last_progress >= 30:
                    last_progress = time.monotonic()
                    print(f'  {cell["build"]} / {cell["case"]}: {len(rows)}/{args.runs} trials recorded', flush=True)
                time.sleep(2)
        summary = read_json(destination / 'summary.json')
        report = summary.get('report', read_json(destination / 'report.json'))
        cell.update(status=summary.get('status', 'FAIL'), results=report.get('results', []),
                    errors=summary.get('errors', []), runner_returncode=process.returncode,
                    graphics_backend=report.get('graphics_backend'), renderer=report.get('renderer'),
                    finished_at=datetime.now().isoformat(timespec='seconds'))
        if process.returncode != 0 or cell['status'] != 'PASS':
            cell['status'] = 'FAIL'
            if not cell['errors']:
                cell['errors'] = [f'Runner exited {process.returncode}; see the combination runner log']
        if args.gpu and args.gpu.lower() not in (cell.get('renderer') or '').lower():
            cell['status'] = 'FAIL'
            cell['errors'].append('OpenGL renderer did not match the requested GPU filter')
        aggregate = aggregate_runs(cell['results']).get(cell['case'], {})
        print(f'  {cell["status"]}: {aggregate.get("median_draw_fps", 0):.2f} median draw FPS; '
              f'{sum(row.get("relative_stutters", 0) for row in cell["results"])} relative stutters', flush=True)
        for error in cell['errors']:
            print('  ' + error, flush=True)
        save_matrix(output, matrix)
    matrix['scene_unchanged'] = digest(scene) == matrix['scene_sha256']
    matrix['harness_unchanged'] = all(digest(ROOT / name) == value for name, value in matrix['harness_sha256'].items())
    matrix['status'] = 'PASS' if all(cell['status'] == 'PASS' for cell in matrix['cells']) and matrix['scene_unchanged'] and matrix['harness_unchanged'] else 'FAIL'
    matrix['finished_at'] = datetime.now().isoformat(timespec='seconds')
    save_matrix(output, matrix)
    print(f'{matrix["status"]}: {output / "report.html"}', flush=True)
    return int(matrix['status'] != 'PASS')


if __name__ == '__main__':
    with awake_request():
        sys.exit(main())
