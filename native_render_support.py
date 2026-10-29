"""Portable validation shared by the native render runner and Blender worker."""
import struct
from pathlib import Path

CASES = ('cycles-cuda-oidn-gpu', 'cycles-optix-optix', 'eevee')
STANDARD_BOUNCES = [16, 4, 4, 12, 0, 16]
BOUNCE_FIELDS = ('max_bounces', 'diffuse_bounces', 'glossy_bounces',
                 'transmission_bounces', 'volume_bounces', 'transparent_max_bounces')
COLOR_FIELDS = ('view_transform', 'look', 'exposure', 'gamma')


def select_gpu_ids(devices, backend, gpu_name, index=0, all_matching=False):
    """Choose one matching physical GPU by default, never a CPU or another API."""
    matches = sorted((d for d in devices if d['type'] == backend and
                      gpu_name.casefold() in d['name'].casefold()), key=lambda d: d['id'])
    if not matches:
        raise ValueError(f'No {backend} GPU matching {gpu_name!r}')
    if index < 0 or index >= len(matches):
        raise ValueError(f'GPU index {index} unavailable; {len(matches)} matching {backend} device(s)')
    return [d['id'] for d in (matches if all_matching else [matches[index]])]


def gpu_errors(state, case, gpu_name, selected_ids=()):
    if case not in CASES: raise ValueError('Unsupported case: ' + case)
    issues = []
    if state.get('graphics_backend') != 'OPENGL': issues.append('OpenGL not active')
    if case == 'eevee':
        if state.get('engine') not in ('BLENDER_EEVEE_NEXT', 'BLENDER_EEVEE'):
            issues.append('EEVEE engine not active')
        if gpu_name.casefold() not in state.get('renderer', '').casefold():
            issues.append('OpenGL renderer does not match requested GPU')
        return issues
    backend = 'CUDA' if case == CASES[0] else 'OPTIX'
    denoiser = 'OPENIMAGEDENOISE' if backend == 'CUDA' else 'OPTIX'
    if state.get('engine') != 'CYCLES' or state.get('scene_device') != 'GPU' or state.get('backend') != backend:
        issues.append('Cycles GPU Compute/backend mismatch')
    enabled = [d for d in state.get('devices', []) if d.get('use')]
    if not enabled or any(d['type'] != backend or gpu_name.casefold() not in d['name'].casefold() for d in enabled):
        issues.append('GPU-only selection missing, includes CPU, or selects the wrong device')
    if set(d['id'] for d in enabled) != set(selected_ids):
        issues.append('Enabled devices differ from explicitly selected GPU IDs')
    if state.get('active_gpu') is not True:
        issues.append('Cycles reports no active GPU')
    if not state.get('denoising') or state.get('denoiser') != denoiser or state.get('denoising_use_gpu') is not True:
        issues.append('Requested GPU denoiser is not enabled')
    if backend == 'CUDA' and state.get('oidn_gpu_supported') is not True:
        issues.append('Selected CUDA GPU does not advertise GPU OIDN support')
    return issues


def png_dimensions(path):
    with Path(path).open('rb') as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != b'\x89PNG\r\n\x1a\n' or header[12:16] != b'IHDR':
        raise ValueError('Invalid saved PNG header')
    return list(struct.unpack('>II', header[16:24]))


def final_result_errors(row, exit_code, expected_size):
    errors = []
    if exit_code != 0:
        errors.append(f'Blender process exited {exit_code} (0x{exit_code & 0xffffffff:08X})')
    if not row.get('render_complete'):
        errors.append('No completed GUI render job')
    if not row.get('source_unchanged'):
        errors.append('Source file unchanged check missing or failed')
    if row.get('saved_image_dimensions') != list(expected_size):
        errors.append('Saved PNG dimensions differ from requested output')
    if not row.get('image') or not Path(row['image']).is_file():
        errors.append('Saved render is missing')
    errors.extend(row.get('audit_errors', []))
    return errors
