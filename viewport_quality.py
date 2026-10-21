"""Offline image evidence for viewport motion and stationary refinement.

The final stationary capture is an observed reference, not a full-resolution
ground truth. No image metric in this module identifies a codec or infers the
renderer buffer dimensions. Moving views are never compared against a different
pose. NumPy/Pillow are host dependencies, not Blender dependencies.
"""
import html
import hashlib
import json
from pathlib import Path


def image_metrics(candidate, reference):
    """RGB MAE and 8x8-window luminance SSIM on identically aligned RGB arrays."""
    import numpy as np
    a = np.asarray(candidate, dtype=np.float32)
    b = np.asarray(reference, dtype=np.float32)
    if a.shape != b.shape or a.ndim != 3 or a.shape[2] != 3:
        raise ValueError('Comparison requires equally sized RGB images at the same pose')
    if min(a.shape[:2]) < 8:
        raise ValueError('Comparison needs at least an 8x8 image')
    weights = np.array([.2126, .7152, .0722], dtype=np.float32)
    x, y = a @ weights, b @ weights
    h, w = x.shape
    h, w = h // 8 * 8, w // 8 * 8
    blocks = lambda v: v[:h, :w].reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3)
    x, y = blocks(x), blocks(y)
    mx, my = x.mean((2, 3)), y.mean((2, 3))
    vx = ((x - mx[..., None, None]) ** 2).mean((2, 3))
    vy = ((y - my[..., None, None]) ** 2).mean((2, 3))
    cov = ((x - mx[..., None, None]) * (y - my[..., None, None])).mean((2, 3))
    c1, c2 = (255 * .01) ** 2, (255 * .03) ** 2
    ssim = ((2 * mx * my + c1) * (2 * cov + c2) /
            ((mx * mx + my * my + c1) * (vx + vy + c2))).mean()
    return {'ssim_luma_8x8': float(ssim), 'rgb_mae_255': float(np.abs(a - b).mean())}


def detail_energy(image):
    """RMS adjacent-pixel luma differences; noise can also increase this value."""
    import numpy as np
    gray = np.asarray(image, dtype=np.float32) @ np.array([.2126, .7152, .0722], dtype=np.float32)
    dx, dy = np.diff(gray, axis=1), np.diff(gray, axis=0)
    return float(np.sqrt((np.mean(dx ** 2) + np.mean(dy ** 2)) / 2))


def analyze_quality(output, report):
    """Validate evidence, compare stationary frames, and save JSON/HTML reports."""
    import numpy as np
    from PIL import Image
    output = Path(output)
    cases, errors = [], []
    for source in report.get('quality_cases', []):
        case = dict(source)
        case['frames'] = [dict(frame) for frame in source['frames']]
        cases.append(case)
        try:
            stopped = [f for f in case['frames'] if f['phase'] == 'stopped']
            moving = [f for f in case['frames'] if f['phase'] == 'moving']
            if not case.get('complete') or len(stopped) != 8 or not moving:
                raise ValueError('Incomplete moving / stationary capture sequence')
            pose = lambda f: (f['view_rotation'], f['view_location'], f['view_distance'])
            if any(pose(f) != pose(stopped[-1]) for f in stopped):
                raise ValueError('Stationary captures have different requested poses')
            if any(b['actual_time_s'] < a['actual_time_s'] for a, b in zip(stopped, stopped[1:])):
                raise ValueError('Non-monotonic stationary capture timestamps')

            def read(frame):
                with Image.open(output / frame['file']) as image:
                    if image.size != (1920, 1080):
                        raise ValueError(f'Expected 1920x1080 PNG, got {image.size}')
                    # Exclude the HUD/logo area and editor borders in every image.
                    return np.array(image.convert('RGB'))[64:-8, 8:-8]

            reference = read(stopped[-1])
            if reference.std(axis=(0, 1)).max() < 1 or reference.mean() < 2:
                raise ValueError('Reference image is flat or black')
            reference_detail = detail_energy(reference)
            for frame in case['frames']:
                array = read(frame)
                frame['detail_rms_luma_255'] = detail_energy(array)
                if frame['phase'] == 'stopped':
                    frame.update(image_metrics(array, reference))
                    frame['detail_ratio_to_reference'] = (frame['detail_rms_luma_255'] / reference_detail
                                                         if reference_detail > 0 else None)
            close = lambda f: f['ssim_luma_8x8'] >= .99 and f['rgb_mae_255'] <= 1
            matching = [f for i, f in enumerate(stopped) if all(close(g) for g in stopped[i:])]
            # The final image always equals itself, so only report a bound when
            # the preceding capture is also close; otherwise refinement is unknown.
            case['reference_stable_in_last_two_captures'] = close(stopped[-2])
            case['observed_reference_match_by_s'] = (matching[0]['actual_time_s']
                if case['reference_stable_in_last_two_captures'] else None)
            case['reference_file'] = stopped[-1]['file']
            case['status'] = 'COMPLETE'
        except (OSError, ValueError, KeyError) as exc:
            case.update(status='INVALID', error=str(exc))
            errors.append(f"{case['case']}: quality evidence invalid: {exc}")
    analysis = {'schema_version': 1, 'cases': cases,
                'analysis_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'roi': 'x=8..1911, y=64..1071, no resampling',
                'reference': 'Last stationary PNG at/after 15s; not ground truth or proof of 64 completed samples',
                'similarity_rule': '8x8 luma SSIM >=0.99 and RGB MAE <=1/255, retained at later checkpoints',
                'measurement_scope': 'Separate capture workload after FPS trials; readbacks perturb capture cadence',
                'pose_scope': 'Stored pose is the requested Blender view; displayed Cycles content may still lag it',
                'errors': errors}
    (output / 'quality.json').write_text(json.dumps(analysis, indent=2), encoding='utf-8')
    write_html(output, analysis, report)
    return errors


def write_html(output, analysis, report):
    esc = lambda v: html.escape(str(v), quote=True)
    sections = []
    for case in analysis['cases']:
        frames = case['frames']
        stopped = [f for f in frames if f['phase'] == 'stopped']
        rows = []
        for f in frames:
            n = lambda key: f'{f[key]:.3f}' if isinstance(f.get(key), (int, float)) else '—'
            rows.append(f'<tr><td>{esc(f["phase"])}</td><td>{n("requested_time_s")}</td>'
                        f'<td>{n("actual_time_s")}</td><td>{n("readback_duration_s")}</td>'
                        f'<td>{n("ssim_luma_8x8")}</td><td>{n("rgb_mae_255")}</td>'
                        f'<td>{n("detail_ratio_to_reference")}</td>'
                        f'<td><a href="{esc(f["file"])}">lossless PNG</a></td></tr>')
        gallery = ''.join(f'<figure><a href="{esc(f["file"])}"><img loading="lazy" src="{esc(f["file"])}"></a>'
                          f'<figcaption>{esc(f["phase"])} · {f["actual_time_s"]:.3f}s</figcaption></figure>'
                          for f in frames)
        comparison = ''
        if stopped:
            index = len(sections)
            files = json.dumps([f['file'] for f in stopped])
            labels = json.dumps([f'{f["actual_time_s"]:.3f}s after stop' for f in stopped])
            comparison = f'''<p>Same requested pose, after orbit stops. Move the slider or open a PNG at 100%.</p>
<input style="width:100%" aria-label="Refinement checkpoint" type="range" min="0" max="{len(stopped)-1}" value="0"
oninput='const f={esc(files)},l={esc(labels)};document.getElementById("shot{index}").src=f[this.value];document.getElementById("label{index}").textContent=l[this.value]'>
<p id="label{index}">{stopped[0]['actual_time_s']:.3f}s after stop</p>
<img class="compare" id="shot{index}" src="{esc(stopped[0]['file'])}">'''
        sections.append(f'''<section><h2>{esc(case['case'])} — {esc(case['status'])}</h2>
<p>Last two captures similar: {esc(case.get('reference_stable_in_last_two_captures', 'unknown'))};
observed reference match by: {esc(case.get('observed_reference_match_by_s'))} seconds.</p>
<p>{esc(case.get('error', ''))}</p>{comparison}
<table><tr><th>Phase</th><th>Requested s</th><th>Actual s</th><th>Readback s</th><th>SSIM</th><th>RGB MAE /255</th><th>Detail ratio</th><th>Evidence</th></tr>{''.join(rows)}</table>
<details><summary>All captures, including moving views</summary><div class="gallery">{gallery}</div></details></section>''')
    document = f'''<!doctype html><meta charset="utf-8"><title>Viewport image quality</title>
<style>body{{font:16px system-ui;margin:30px auto;max-width:1400px;padding:0 24px;background:#fafafa;color:#222}}img{{max-width:100%}}.compare{{width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid #ccc;text-align:left}}.gallery{{display:grid;grid-template-columns:1fr 1fr}}figure{{margin:10px}}.note{{background:#fff0c2;padding:16px}}section{{margin-bottom:50px}}</style>
<h1>Viewport image quality — motion and refinement</h1>
<p>Blender {esc(report.get('blender'))} · {esc(report.get('renderer'))} · OpenGL · 1920×1080 · 64 samples configured</p>
<p class="note">Captured in a separate workload after FPS trials. Screenshots introduce readback delays;
actual timestamps and capture costs are recorded. The final stationary image is an observed reference,
not a native baseline or proof of full internal resolution / completed samples. Similarity measures image change,
not codec compression. Moving images have different poses and are not scored against the stationary reference.
PNG pixel dimensions do not establish Cycles' internal rendering resolution.</p>
<p>The frozen pose is the requested Blender view. A delayed rendered image can still show an older pose;
the comparison captures that lag as well as detail/noise changes.</p>
<p>Metric ROI excludes the top 64 pixels and 8-pixel borders. Detail ratio can also change with noise.
Similarity rule: SSIM ≥0.99 and RGB MAE ≤1/255 at this and all later checkpoints.
<a href="quality.json">Raw metadata and metrics</a> · <a href="report.html">FPS report</a></p>{''.join(sections)}'''
    (output / 'quality.html').write_text(document, encoding='utf-8')
