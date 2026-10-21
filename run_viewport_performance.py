"""Measure orbit draw FPS in Blender's actual UI; never changes Boost sessions.

Smoke defaults to Classroom (one 5s run); full defaults to Medieval Fisher Hut
(three 30s runs). Both start with Solid and Material Preview at Pixel Size 1x.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import html
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from run_benchmark_suite import load_config, select_blender
from viewport_performance import DEFAULT_CASES, MODEL_KEYS, STANDARD_PROFILE, DrawRecorder, aggregate_runs, expand_cases

ROOT = Path(__file__).resolve().parent


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def assess(report, cases, runs, code, backend='opengl', gpu_filter=None):
    errors = []
    if code != 0:
        errors.append(f'Blender process exit: {code}')
    if not report.get('complete'):
        errors.append('Blender report is incomplete: ' + report.get('error', 'see blender.log'))
    rows = report.get('results', [])
    expected = [(case, run) for case in cases for run in range(1, runs + 1)]
    if [(r.get('case'), r.get('run')) for r in rows] != expected:
        errors.append('Missing, duplicate, or out-of-order case/run results')
    for row in rows:
        if row.get('status') != 'PASS' or row.get('viewport_draw_fps', 0) <= 0 or row.get('draws', 0) < 3:
            errors.append(f"{row.get('case')} run {row.get('run')}: invalid draw measurement {row.get('errors', [])}")
        if row.get('viewport_px') != STANDARD_PROFILE['viewport_px']:
            errors.append(f"{row.get('case')} run {row.get('run')}: viewport is not 1920x1080")
    actual = report.get('graphics_backend', '').lower()
    if not actual.startswith(backend):
        errors.append(f'Graphics backend mismatch: requested {backend}, actual {actual}')
    if gpu_filter and gpu_filter.casefold() not in report.get('renderer', '').casefold():
        errors.append('OpenGL renderer did not match the requested GPU filter: ' + str(report.get('renderer')))
    return errors


def check_artifacts(report, output):
    """Catch absent evidence and blank/black/magenta views outside the timed loop."""
    from PIL import Image, ImageStat
    errors = []
    checked = {}
    for row in report.get('results', []):
        if not (output / row.get('frames_csv', '__missing__')).is_file():
            errors.append(f"Missing frame CSV for {row.get('case')} run {row.get('run')}")
        shot = row.get('screenshot', '')
        if shot not in checked:
            try:
                with Image.open(output / shot) as source:
                    image = source.convert('RGB')
                    w, h = image.size
                    # Exclude the HUD, editor borders, and window chrome.
                    crop = image.crop((w // 4, h // 4, 3 * w // 4, 3 * h // 4))
                    stat = ImageStat.Stat(crop)
                    mean = stat.mean
                    flat = max(stat.stddev) < 1.0
                    black = max(mean) < 2
                    magenta = mean[0] > 180 and mean[1] < 40 and mean[2] > 180
                    checked[shot] = {'status': 'FAIL' if flat or black or magenta else 'PASS',
                                     'center_mean_rgb': mean, 'center_stddev_rgb': stat.stddev}
            except (OSError, ValueError) as exc:
                checked[shot] = {'status': 'FAIL', 'error': str(exc)}
        row['image_sanity'] = checked[shot]
        if checked[shot]['status'] != 'PASS':
            row['status'] = 'FAIL'
            errors.append(f"{row.get('case')}: screenshot missing, blank, black, or magenta")
    return list(dict.fromkeys(errors))


def frame_chart(output, row):
    """Offline SVG; no per-frame screenshots or readbacks affect measurement."""
    try:
        with (output / row['frames_csv']).open(newline='', encoding='utf-8') as stream:
            frames = list(csv.DictReader(stream))
        samples = [(float(f['time_s']), float(f['interval_ms'])) for f in frames[1:]]
    except (OSError, KeyError, ValueError):
        return ''
    if not samples:
        return '<p>Insufficient completed intervals for a frame-time plot.</p>'
    duration = row['duration_s']
    threshold = row.get('stutter_threshold_ms', 100)
    ceiling = max(100, threshold * 1.15, max(dt for _, dt in samples) * 1.05)
    x = lambda t: 55 + 900 * t / duration
    y = lambda dt: 190 - 165 * dt / ceiling
    points = ' '.join(f'{x(t):.2f},{y(dt):.2f}' for t, dt in samples)
    markers = []
    for event in row.get('gap_events', []):
        if event['relative_stutter']:
            title = html.escape(f"{event['start_s']:.3f}–{event['end_s']:.3f}s: {event['duration_ms']:.1f}ms gap")
            markers.append(f'<circle cx="{x(event["end_s"]):.2f}" cy="{y(event["duration_ms"]):.2f}" r="4" fill="#c22"><title>{title}</title></circle>')
    label = html.escape(f"{row['case']} run {row['run']}")
    first = row.get('time_to_first_draw_ms') or 0
    tail = row.get('tail_without_new_draw_ms') or 0
    return f'''<figure><figcaption><b>{label}</b> — between-draw intervals; higher spikes mean longer waits.
First draw: {first:.1f} ms; end gap: {tail:.1f} ms.</figcaption>
<svg viewBox="0 0 1000 230" role="img" aria-label="Frame-time plot for {label}">
<rect x="55" y="25" width="900" height="165" fill="#fff" stroke="#aaa"/>
<line x1="55" x2="955" y1="{y(threshold):.2f}" y2="{y(threshold):.2f}" stroke="#c22" stroke-dasharray="5 4"/>
<polyline points="{points}" fill="none" stroke="#1766aa" stroke-width="1.2"/>{''.join(markers)}
<text x="5" y="30">{ceiling:.0f}ms</text><text x="15" y="192">0ms</text>
<text x="55" y="214">0s</text><text x="915" y="214">{duration:.1f}s</text>
</svg></figure>'''


def write_html(output, summary):
    report = summary['report']
    esc = lambda value: html.escape(str(value))
    rows = []
    for row in report.get('results', []):
        def number(name):
            value = row.get(name)
            return f'{value:.2f}' if isinstance(value, (int, float)) else '—'
        rows.append('<tr>' + ''.join(f'<td>{esc(v)}</td>' for v in (
            row['case'], row['run'], row['status'], number('viewport_draw_fps'),
            number('p95_frame_ms'), number('p99_frame_ms'), number('one_percent_low_fps'),
            row.get('relative_stutters', '—'), row.get('gaps_over_500_ms', '—'), number('longest_without_new_draw_ms'),
            row['draws'], '×'.join(map(str, row['viewport_px'])))) +
            f'<td><a href="{esc(row["frames_csv"])}">frames CSV</a> · '
            f'<a href="{esc(row.get("gaps_csv", ""))}">gap events</a> · '
            f'<a href="{esc(row.get("screenshot", ""))}">image</a></td></tr>')
    medians = []
    aggregates = aggregate_runs(report.get('results', []))
    for case, values in aggregates.items():
        medians.append(f"{case}: median {values['median_draw_fps']:.2f} draw FPS; "
                       f"range {values['min_draw_fps']:.2f}–{values['max_draw_fps']:.2f} ({values['runs']} runs)")
    warnings = [f'{case}: {values["warning"]}' for case, values in aggregates.items() if values['warning']]
    document = f'''<!doctype html><meta charset="utf-8"><title>Viewport orbit performance</title>
<style>body{{font:16px system-ui;max-width:1350px;margin:35px auto;padding:0 20px;background:#fafafa;color:#222}}
table{{border-collapse:collapse;width:100%}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #ccc}}
code{{overflow-wrap:anywhere}}.note{{padding:15px;background:#fff3cd}}img{{max-width:800px;width:100%}}
table{{display:block;overflow-x:auto;white-space:nowrap}}svg{{width:100%;max-height:300px}}figure{{margin:25px 0}}</style>
<h1>Viewport orbit performance — {esc(summary['status'])}</h1>
<p>{esc(summary['label'])} · {esc(summary['profile'])} · Blender {esc(report.get('blender','unknown'))}<br>
Scene: <code>{esc(summary['scene'])}</code><br>Graphics: {esc(report.get('graphics_backend','unknown'))} · {esc(report.get('renderer','unknown'))}</p>
<p>Quality profile: {esc(report.get('quality_profile', {}).get('id', 'historical / unspecified'))} ·
requested viewport {esc(summary.get('requested_window_px'))} · {esc(summary.get('samples'))} samples.</p>
{'<p><a href="quality.html">Motion and stationary image-quality captures</a></p>' if report.get('quality_cases') else ''}
<p class="note">Measures unique Blender viewport draw callbacks after view changes. This is not a measurement of GPU completion,
rendered-image freshness, input-to-photon latency, or Boost client presentation FPS. Smoke results validate the harness;
compare performance only with the same scene, build, settings, dimensions, and machine conditions.</p>
<p>{esc('; '.join(medians))}</p>
{''.join(f'<p class="note">{esc(warning)}</p>' for warning in warnings)}
<table><tr><th>Case</th><th>Run</th><th>Status</th><th>Draw FPS</th><th>p95 ms</th><th>p99 ms</th>
<th>1% low FPS</th><th>Relative stutters</th><th>Gaps &gt;500ms</th><th>Longest gap ms</th><th>Draws</th><th>Viewport px</th><th>Evidence</th></tr>{''.join(rows)}</table>
<p>Stutter rule: a between-draw interval above max(100ms, 3 × that run's median interval).
This distinguishes irregular hitches from consistently low FPS. Initial and end delays are separate events.
Red dashed lines show the threshold; red points mark relative stutters. These identify gaps, not their cause.</p>
{''.join(frame_chart(output, row) for row in report.get('results', []))}
<p>{'<br>'.join(esc(e) for e in summary['errors'])}</p>
<p><a href="summary.json">Summary JSON</a> · <a href="report.json">Blender report</a> · <a href="blender.log">Log</a></p>'''
    (output / 'report.html').write_text(document, encoding='utf-8')


def reanalyze(output):
    """Add gap events/plots from saved timestamps, without launching Blender."""
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    report = summary['report']
    for row in report.get('results', []):
        recorder = DrawRecorder(0)
        with (output / row['frames_csv']).open(newline='', encoding='utf-8') as stream:
            recorder.samples = [{'time_s': float(f['time_s']), 'revision': int(f['revision']),
                                 'angle_deg': float(f['angle_deg']), 'interval_ms': float(f['interval_ms'])}
                                for f in csv.DictReader(stream)]
        row.update(recorder.summarize(row['duration_s']))
        row['gap_events'] = recorder.gap_events(row['duration_s'])
        row['gaps_csv'] = f"{row['case']}_run{row['run']}_gaps.csv"
        with (output / row['gaps_csv']).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['kind', 'start_s', 'end_s', 'duration_ms',
                'start_angle_deg', 'end_angle_deg', 'relative_stutter', 'over_500_ms'])
            writer.writeheader()
            writer.writerows(row['gap_events'])
    summary['analysis_version'] = 2
    summary['case_summaries'] = aggregate_runs(report.get('results', []))
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    write_html(output, summary)
    print('Updated gap analysis: ' + str(output / 'report.html'))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--blender-version', default='4.5')
    selection.add_argument('--blender', type=Path)
    parser.add_argument('--config', type=Path, default=ROOT / 'benchmark_config.toml')
    parser.add_argument('--analyze', type=Path, help='Reanalyze an existing results directory without launching Blender')
    parser.add_argument('--output', type=Path, help='New results directory (existing nonempty directories are rejected)')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--model', help='Override the configured model key or provide a .blend path')
    parser.add_argument('--cases', default=','.join(DEFAULT_CASES))
    parser.add_argument('--duration', type=float)
    parser.add_argument('--warmup', type=float)
    parser.add_argument('--runs', type=int)
    parser.add_argument('--samples', type=int, choices=[STANDARD_PROFILE['samples']], default=STANDARD_PROFILE['samples'])
    parser.add_argument('--gpu', help='Cycles GPU device name substring')
    parser.add_argument('--graphics-backend', choices=['opengl'], default='opengl')
    parser.add_argument('--label', default='unlabeled')
    parser.add_argument('--timeout', type=float, default=900)
    parser.add_argument('--quality-capture', action='store_true',
                        help='After FPS trials, capture a separate orbit and 15s stop/refine sequence')
    parser.add_argument('--window-size', nargs=2, type=int, default=list(STANDARD_PROFILE['viewport_px']), metavar=('WIDTH', 'HEIGHT'), help='The standard profile requires 1920 1080')
    args = parser.parse_args()
    if args.analyze:
        return reanalyze(args.analyze.resolve())
    profile = 'smoke' if args.smoke else 'full'
    duration = args.duration if args.duration is not None else (5 if args.smoke else 30)
    warmup = args.warmup if args.warmup is not None else (12 if args.smoke else 15)
    runs = args.runs if args.runs is not None else (1 if args.smoke else 3)
    if not all(math.isfinite(value) and value > 0 for value in (duration, warmup, runs, args.samples, args.timeout, *args.window_size)):
        parser.error('Durations, counts, samples, timeout, and window dimensions must be positive')
    if args.window_size != STANDARD_PROFILE['viewport_px']:
        parser.error('The standard viewport profile requires --window-size 1920 1080')
    try:
        cases = expand_cases(args.cases)
    except ValueError as exc:
        parser.error(str(exc))
    cfg = load_config(str(args.config))
    key = args.model or MODEL_KEYS[profile]
    scene = Path(cfg['models'].get(key, key))
    if not scene.is_absolute():
        scene = ROOT / scene
    scene = scene.resolve()
    if not scene.is_file():
        parser.error(f'Scene not found: {scene}')
    exe = str(args.blender.resolve()) if args.blender else select_blender(cfg['blender'], version_key=args.blender_version)[0]
    output = args.output.resolve() if args.output else ROOT / 'results' / ('viewport_performance_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    if output.exists() and any(output.iterdir()):
        parser.error(f'Results directory is not empty: {output}')
    output.mkdir(parents=True, exist_ok=True)
    summary = {'status': 'RUNNING', 'profile': profile, 'label': args.label, 'scene': str(scene),
               'scene_sha256': digest(scene), 'executable': exe, 'cases': cases, 'runs': runs,
               'requested_duration_s': duration, 'warmup_s': warmup, 'samples': args.samples,
               'requested_window_px': args.window_size, 'graphics_backend_requested': args.graphics_backend}
    summary['quality_profile'] = dict(STANDARD_PROFILE)
    summary['quality_capture_requested'] = args.quality_capture
    summary['harness_sha256'] = {name: digest(ROOT / name) for name in (
        'run_viewport_performance.py', 'blender_viewport_performance.py',
        'viewport_performance.py', 'engine_benchmark.py', 'viewport_quality.py', 'keep_viewport_awake.py')}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    cmd = [exe, '--factory-startup', '-p', '40', '40', *map(str, args.window_size),
           '--python-exit-code', '2', '--python', str(ROOT / 'blender_viewport_performance.py'), '--',
           '--scene', str(scene), '--output', str(output), '--cases', ','.join(cases),
           '--duration', str(duration), '--warmup', str(warmup), '--runs', str(runs),
           '--samples', str(args.samples), '--label', args.label, '--graphics-backend', args.graphics_backend]
    # Older releases have only OpenGL and do not accept --gpu-backend. Read the
    # executable's CLI capabilities rather than assuming a major/minor boundary.
    help_text = subprocess.check_output([exe, '--help'], text=True, errors='replace', timeout=30)
    if '--gpu-backend' in help_text:
        cmd[1:1] = ['--gpu-backend', args.graphics_backend]
    if args.gpu:
        cmd += ['--gpu', args.gpu]
    if args.quality_capture:
        cmd += ['--quality-capture']
    summary['command'] = cmd
    summary['launch_mode'] = 'Full Blender GUI, factory startup, load scene, Layout workspace, real 3D Viewport'
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'Viewport {profile}: {scene.name}, {cases}, {runs} x {duration:g}s -> {output}', flush=True)
    started = time.monotonic()
    from keep_viewport_awake import awake_request
    with awake_request(), (output / 'blender.log').open('w', encoding='utf-8') as stream:
        try:
            process = subprocess.Popen(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            try:
                code = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                code = 'TIMEOUT'
        except OSError as exc:
            stream.write(str(exc))
            code = f'LAUNCH_ERROR: {exc}'
    try:
        report = json.loads((output / 'report.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        report = {}
    errors = assess(report, cases, runs, code, args.graphics_backend, args.gpu)
    errors += check_artifacts(report, output)
    if args.quality_capture:
        from viewport_quality import analyze_quality
        if [row.get('case') for row in report.get('quality_cases', [])] != cases:
            errors.append('Quality capture case set mismatch / missing evidence')
        errors += analyze_quality(output, report)
    summary.update({'status': 'FAIL' if errors else 'PASS', 'returncode': code, 'errors': errors,
                    'elapsed_s': time.monotonic() - started, 'report': report,
                    'case_summaries': aggregate_runs(report.get('results', []))})
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    write_html(output, summary)
    for row in report.get('results', []):
        print(f"  {row['case']} run {row['run']}: {row['viewport_draw_fps']:.2f} viewport draw FPS ({row['status']})", flush=True)
    for error in errors:
        print('  ' + error, flush=True)
    print(f"{summary['status']}: {output / 'report.html'}", flush=True)
    return int(bool(errors))


if __name__ == '__main__':
    sys.exit(main())
