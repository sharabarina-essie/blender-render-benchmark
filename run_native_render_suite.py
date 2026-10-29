"""Portable stock Cycles/EEVEE GUI renders for native workstation baselines."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from PIL import Image
import numpy as np
from native_render_support import CASES, final_result_errors

ROOT = Path(__file__).resolve().parent


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
    for _ in range(100):
        try: temporary.replace(path); return
        except PermissionError: time.sleep(.02)
    raise RuntimeError('Cannot publish ' + str(path))


def build_command(exe, source, case, directory, args):
    command = [str(exe), '--factory-startup', '--enable-autoexec' if args.enable_autoexec else '--disable-autoexec',
               '--gpu-backend', 'opengl', str(source), '--python', str(ROOT / 'blender_native_render.py'), '--',
               '--output', str(directory), '--source', str(source), '--case', case, '--gpu', args.gpu,
               '--gpu-index', str(args.gpu_index), '--width', str(args.width), '--height', str(args.height),
               '--samples', str(args.samples), '--eevee-samples', str(args.eevee_samples), '--profile', args.profile]
    if args.all_matching_gpus: command.append('--all-matching-gpus')
    return command


def publish_report(out, summary):
    summary['counts'] = dict(Counter(r['status'] for r in summary['rows']))
    write_json(out / 'summary.json', summary)
    page = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Native render baseline</title>',
            '' if summary['complete'] else '<meta http-equiv="refresh" content="10">',
            '<style>body{font:16px system-ui;background:#101820;color:#eee;margin:30px}a{color:#7dd3fc}table{border-collapse:collapse;width:100%}td,th{border:1px solid #567;padding:12px;vertical-align:top}img{max-width:480px;width:100%}pre{white-space:pre-wrap;max-width:650px}</style>',
            '<h1>Native workstation render baseline</h1>',
            '<p>' + ('Complete' if summary['complete'] else 'Running') + ' · ' + html.escape(str(summary['counts'])) + '</p>',
            '<p>Full Blender GUI/F12, OpenGL, one fresh process per scene/case. PNGs are saved from Render Result. '
            'PASS verifies rendering, device/settings evidence, source preservation and PNG decoding; it is not a visual parity threshold.</p>',
            '<p>Profile: ' + html.escape(summary['profile']) + ' · output ' + str(summary['resolution']) +
            ' · Cycles samples ' + str(summary['samples']) + ' fixed · EEVEE samples ' + str(summary['eevee_samples']) + '</p>',
            '<p>EEVEE uses the OpenGL adapter selected by Windows; the Cycles GPU index does not control that adapter. '
            'Device names and IDs, renderer, authored camera/world/color and effective settings are in the evidence.</p>',
            '<p><a href="summary.json">All evidence</a></p>',
            '<table><tr><th>Scene</th><th>Case</th><th>Status</th><th>Render</th><th>Evidence / error</th></tr>']
    for row in summary['rows']:
        image = ''
        if row.get('image') and Path(row['image']).is_file():
            rel = Path(row['image']).relative_to(out).as_posix()
            image = '<a href="' + html.escape(rel) + '"><img alt="Saved render" src="' + html.escape(rel) + '"></a>'
        evidence = Path(row['directory']).relative_to(out).as_posix() + '/result.json'
        error = row.get('host_errors') or row.get('error') or row.get('audit_errors') or ''
        page.append('<tr>' + ''.join('<td>' + html.escape(str(v)) + '</td>' for v in
                    [row['model'], row['case'], row['status']]) + '<td>' + image + '</td><td><a href="' +
                    html.escape(evidence) + '">Result</a><pre>' + html.escape(str(error)) + '</pre></td></tr>')
    page.append('</table></html>')
    (out / 'report.html').write_text('\n'.join(page), encoding='utf-8')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    blender = p.add_mutually_exclusive_group()
    blender.add_argument('--blender', type=Path, help='Path to an existing Blender4.2+ executable')
    blender.add_argument('--blender-version', help='Build under blender/, default4.5.15')
    p.add_argument('--model', type=Path, required=True, action='append', help='Scene path; repeat for multiple scenes')
    p.add_argument('--cases', default=','.join(CASES))
    p.add_argument('--gpu', default='RTX 6000', help='GPU name substring; CPU fallback is rejected')
    p.add_argument('--gpu-index', type=int, default=0, help='Zero-based matching Cycles GPU, sorted by device ID')
    p.add_argument('--all-matching-gpus', action='store_true', help='Explicit multi-GPU run, not a single-GPU comparison')
    p.add_argument('--width', type=int, default=480)
    p.add_argument('--height', type=int, default=270)
    p.add_argument('--samples', type=int, default=16)
    p.add_argument('--eevee-samples', type=int, default=64)
    p.add_argument('--profile', choices=['authored-look', 'smoke'], default='authored-look')
    p.add_argument('--enable-autoexec', action='store_true', help='Allow scene scripts/drivers after reviewing the source')
    p.add_argument('--timeout', type=float, default=600, help='Seconds per process, including scene loading')
    p.add_argument('--output', type=Path)
    p.add_argument('--dry-run', action='store_true', help='Validate inputs and print commands, without launching Blender')
    args = p.parse_args(argv)
    cases = list(dict.fromkeys(args.cases.split(',')))
    if any(c not in CASES for c in cases): p.error('Cases: ' + ','.join(CASES))
    if min(args.width, args.height, args.samples, args.eevee_samples) < 1 or args.gpu_index < 0 or args.timeout <= 0:
        p.error('Dimensions/samples/timeout must be positive; GPU index must be nonnegative')
    if max(args.width, args.height) > 16384 or args.width * args.height > 40000000:
        p.error('Output limited to40 million pixels and16384 pixels per dimension')
    if not args.gpu.strip(): p.error('GPU substring must not be empty')
    exe = (args.blender or ROOT / 'blender' / (args.blender_version or '4.5.15') / 'blender.exe').resolve()
    models = list(dict.fromkeys(path.resolve() for path in args.model))
    if not exe.is_file(): p.error('Blender not found: ' + str(exe))
    for source in models:
        if not source.is_file() or source.suffix.lower() != '.blend': p.error('Missing .blend source: ' + str(source))
    out = (args.output or ROOT / 'results' / ('native_render_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    jobs = []
    for source in models:
        sha = hashlib.sha256(source.read_bytes()).hexdigest()
        name = re.sub(r'[^A-Za-z0-9_-]', '_', source.stem)[:50] + '_' + sha[:8]
        for case in cases:
            directory = out / name / case
            jobs.append(dict(model=source.name, source=str(source), source_sha256=sha, case=case,
                             directory=str(directory), command=build_command(exe, source, case, directory, args)))
    if args.dry_run:
        print(json.dumps(dict(output=str(out), jobs=jobs), indent=2)); return 0
    if out.exists(): p.error('Output directory already exists; choose a new path: ' + str(out))
    out.mkdir(parents=True, exist_ok=False)
    summary = dict(started=datetime.now().isoformat(), complete=False, profile=args.profile,
                   resolution=[args.width, args.height], samples=args.samples, eevee_samples=args.eevee_samples,
                   gpu=args.gpu, gpu_index=args.gpu_index, all_matching_gpus=args.all_matching_gpus,
                   requested_cases=len(jobs), rows=[])
    publish_report(out, summary)
    print(out, flush=True)
    for job in jobs:
        directory = Path(job['directory']); directory.mkdir(parents=True)
        row = dict(job, status='RUNNING'); summary['rows'].append(row)
        write_json(directory / 'result.json', row); publish_report(out, summary)
        print(row['model'], row['case'], flush=True)
        stopped = None
        with (directory / 'blender.log').open('w', encoding='utf-8') as log:
            proc = subprocess.Popen(job['command'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            started = time.monotonic()
            try:
                while proc.poll() is None:
                    if (out / 'STOP').exists() or time.monotonic() - started > args.timeout:
                        stopped = 'INTERRUPTED' if (out / 'STOP').exists() else 'TIMEOUT'
                        proc.kill(); proc.wait(); break
                    time.sleep(.25)
            except KeyboardInterrupt:
                stopped = 'INTERRUPTED'
                if proc.poll() is None: proc.kill()
                proc.wait()
        worker = directory / 'worker.json'
        if worker.exists():
            try: row.update(json.loads(worker.read_text(encoding='utf-8')))
            except ValueError: row.update(status='INVALID EVIDENCE', error='Cannot parse worker result')
        row['process_exit'] = proc.returncode; row['process_exit_hex'] = f'0x{proc.returncode & 0xffffffff:08X}'
        if stopped:
            row.update(status=stopped, error='STOP requested' if stopped == 'INTERRUPTED' else f'Process exceeded{args.timeout}s')
        else:
            issues = final_result_errors(row, proc.returncode, [args.width, args.height])
            if row['status'] == 'PASS':
                try:
                    with Image.open(row['image']) as image:
                        image.load()
                        if image.size != (args.width, args.height): issues.append('Decoded PNG has wrong dimensions')
                        rgba = np.asarray(image.convert('RGBA'))
                        row['alpha_nonopaque_pixels'] = int(np.count_nonzero(rgba[:, :, 3] < 255))
                        row['alpha_fully_transparent_pixels'] = int(np.count_nonzero(rgba[:, :, 3] == 0))
                        row['image_sha256'] = hashlib.sha256(Path(row['image']).read_bytes()).hexdigest()
                except (OSError, KeyError) as exc: issues.append(str(exc))
            if issues:
                row['host_errors'] = issues
                if row['status'] in ['PASS', 'RUNNING', 'CONFIGURING']: row['status'] = 'VALIDATION OR PROCESS FAILED'
        # The host also checks preservation when a worker crashes before its final audit.
        row['source_unchanged_host'] = hashlib.sha256(Path(job['source']).read_bytes()).hexdigest() == job['source_sha256']
        if not row['source_unchanged_host']: row.update(status='SOURCE CHANGED', error='Source hash changed')
        write_json(directory / 'result.json', row); publish_report(out, summary)
        print(' ', row['status'], flush=True)
        if stopped == 'INTERRUPTED':
            summary['interrupted'] = True; publish_report(out, summary); return 1
    summary['complete'] = True; summary['completed'] = datetime.now().isoformat()
    publish_report(out, summary)
    return 0 if all(r['status'] == 'PASS' for r in summary['rows']) else 1


if __name__ == '__main__': raise SystemExit(main())
