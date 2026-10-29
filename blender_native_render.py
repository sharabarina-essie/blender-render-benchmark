"""Visible F12 worker for run_native_render_suite.py; no source/preferences saves."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import bpy
import gpu
from native_render_support import (CASES, STANDARD_BOUNCES, BOUNCE_FIELDS, COLOR_FIELDS,
                                   select_gpu_ids, gpu_errors, png_dimensions)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--case', choices=CASES, required=True)
    p.add_argument('--gpu', required=True)
    p.add_argument('--gpu-index', type=int, default=0)
    p.add_argument('--all-matching-gpus', action='store_true')
    p.add_argument('--width', type=int, default=480)
    p.add_argument('--height', type=int, default=270)
    p.add_argument('--samples', type=int, default=16)
    p.add_argument('--eevee-samples', type=int, default=64)
    p.add_argument('--profile', choices=['authored-look', 'smoke'], default='authored-look')
    args = p.parse_args(sys.argv[sys.argv.index('--') + 1:])
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    source = args.source.resolve()
    scene = bpy.context.scene; window = bpy.context.window
    bpy.context.preferences.use_preferences_save = False
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    authored_color = {k: getattr(scene.view_settings, k) for k in COLOR_FIELDS}
    authored_bounces = [getattr(scene.cycles, k) for k in BOUNCE_FIELDS]
    selected = []
    state = dict(status='CONFIGURING', case=args.case, source=str(source),
                 source_sha256=digest, blender=bpy.app.version_string,
                 build_hash=bpy.app.build_hash.decode(), profile=args.profile,
                 worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 authored_color=authored_color, authored_bounces=authored_bounces,
                 render_complete=False)

    def publish():
        temporary = out / 'worker.tmp'
        temporary.write_text(json.dumps(state, indent=2), encoding='utf-8')
        for _ in range(100):
            try: temporary.replace(out / 'worker.json'); return
            except PermissionError: time.sleep(.02)
        raise RuntimeError('Cannot publish worker evidence')

    def devices(prefs):
        return [dict(name=d.name, id=d.id, type=d.type, use=bool(d.use)) for d in prefs.devices]

    def snapshot():
        result = dict(engine=scene.render.engine, graphics_backend=gpu.platform.backend_type_get(),
                      renderer=gpu.platform.renderer_get(), vendor=gpu.platform.vendor_get())
        if args.case != 'eevee':
            prefs = bpy.context.preferences.addons['cycles'].preferences
            cy = scene.cycles
            result.update(scene_device=cy.device, backend=prefs.compute_device_type, devices=devices(prefs),
                          active_gpu=bool(prefs.has_active_device()), denoising=cy.use_denoising,
                          denoiser=cy.denoiser, denoising_use_gpu=getattr(cy, 'denoising_use_gpu', None),
                          oidn_gpu_supported=bool(prefs.has_oidn_gpu_devices()) if hasattr(prefs, 'has_oidn_gpu_devices') else None)
        return result

    def settings():
        r = scene.render; cy = scene.cycles; ev = scene.eevee
        return dict(engine=r.engine, resolution=[r.resolution_x, r.resolution_y, r.resolution_percentage],
                    scene=scene.name, frame=scene.frame_current, camera=scene.camera.name,
                    camera_dof=scene.camera.data.dof.use_dof, world=scene.world.name if scene.world else None,
                    color={k: getattr(scene.view_settings, k) for k in COLOR_FIELDS},
                    bounces=[getattr(cy, k) for k in BOUNCE_FIELDS], samples=cy.samples,
                    adaptive_sampling=cy.use_adaptive_sampling, seed=cy.seed, animated_seed=cy.use_animated_seed,
                    time_limit=cy.time_limit, denoising=cy.use_denoising, denoiser=cy.denoiser,
                    denoising_use_gpu=getattr(cy, 'denoising_use_gpu', None),
                    compositor=r.use_compositing, compositor_device=getattr(r, 'compositor_device', 'CPU'),
                    persistent_data=r.use_persistent_data, motion_blur=r.use_motion_blur, simplify=r.use_simplify,
                    film_transparent=r.film_transparent, eevee_samples=ev.taa_render_samples,
                    raytracing=ev.use_raytracing, raytracing_resolution=ev.ray_tracing_options.resolution_scale,
                    shadow_rays=ev.shadow_ray_count, shadow_steps=ev.shadow_step_count,
                    volume_shadows=ev.use_volumetric_shadows)

    def checks():
        errors = gpu_errors(snapshot(), args.case, args.gpu, selected)
        now = settings()
        # Compare full snapshots to detect a setting/device change during the render.
        if now != state['settings']: errors.append('Render settings changed during the job')
        return errors

    def pre(*_):
        try:
            state['pre_audit'] = snapshot(); state['pre_settings'] = settings()
            state['pre_errors'] = checks()
        except Exception: state['pre_errors'] = [traceback.format_exc()]
        publish()

    def complete(*_): state.update(render_complete=True, ended=time.perf_counter())
    def cancel(*_): state.update(cancelled=True, ended=time.perf_counter())

    def poll():
        if 'ended' not in state or bpy.app.is_job_running('RENDER'): return .25
        try:
            if state.get('cancelled') or not state['render_complete']: raise RuntimeError('Render cancelled')
            image = bpy.data.images.get('Render Result')
            if image is None: raise RuntimeError('Missing Render Result')
            path = out / 'render.png'; image.save_render(str(path), scene=scene)
            state['image'] = str(path); state['saved_image_dimensions'] = png_dimensions(path)
            state['post_audit'] = snapshot()
            errors = checks() + state.get('pre_errors', [])
            if not state.get('pre_audit'): errors.append('No render_pre GPU audit')
            if state['saved_image_dimensions'] != [args.width, args.height]: errors.append('Wrong saved PNG dimensions')
            state['source_unchanged'] = hashlib.sha256(source.read_bytes()).hexdigest() == digest
            if not state['source_unchanged']: errors.append('Source file changed')
            state.update(status='PASS' if not errors else 'AUDIT FAILED', audit_errors=errors,
                         render_s=state['ended'] - state['started'])
        except Exception: state.update(status='FAILED', error=traceback.format_exc())
        publish(); print('NATIVE_RENDER_DONE', args.case, state['status'], flush=True)
        bpy.ops.wm.quit_blender(); return None

    def start():
        nonlocal selected
        try:
            if window is None or bpy.app.background: raise RuntimeError('Requires a visible Blender GUI')
            if Path(bpy.data.filepath).resolve() != source: raise RuntimeError('Requested source did not load')
            if getattr(bpy.app, 'autoexec_fail', False): raise RuntimeError('Scene requires auto-execution; review it and rerun with --enable-autoexec')
            if not scene.camera: raise RuntimeError('No active camera')
            if scene.render.use_multiview: raise RuntimeError('This runner saves mono views; multiview needs a separate fixture test')
            missing = []
            for image in bpy.data.images:
                if not image.users or image.source not in ('FILE', 'TILED'): continue
                if image.packed_file or len(image.packed_files): continue
                path = Path(bpy.path.abspath(image.filepath, library=image.library))
                exists = path.is_file()
                if image.source == 'TILED' and not exists:
                    exists = any(path.parent.glob(path.name.replace('<UDIM>', '*').replace('<UVTILE>', '*')))
                if not exists: missing.append(str(path))
            for library in bpy.data.libraries:
                if not library.packed_file and not Path(bpy.path.abspath(library.filepath)).is_file():
                    missing.append(library.filepath)
            state['missing_used_dependencies'] = missing
            if missing: raise RuntimeError('Missing used image/library dependencies: ' + '; '.join(missing))
            r = scene.render; cy = scene.cycles
            if args.case == 'eevee':
                r.engine = 'BLENDER_EEVEE' if bpy.app.version >= (5, 0, 0) else 'BLENDER_EEVEE_NEXT'
                ev = scene.eevee; ev.taa_render_samples = args.eevee_samples
                ev.use_raytracing = True; ev.ray_tracing_method = 'SCREEN'
                ev.ray_tracing_options.resolution_scale = '1'
                ev.shadow_ray_count = 4; ev.shadow_step_count = 12
                ev.use_volumetric_shadows = True; ev.volumetric_shadow_samples = 16
            else:
                r.engine = 'CYCLES'
                backend = 'CUDA' if args.case == CASES[0] else 'OPTIX'
                prefs = bpy.context.preferences.addons['cycles'].preferences
                prefs.compute_device_type = backend
                prefs.refresh_devices()
                selected = select_gpu_ids(devices(prefs), backend, args.gpu, args.gpu_index, args.all_matching_gpus)
                for device in prefs.devices: device.use = device.id in selected
                cy.device = 'GPU'; cy.samples = args.samples; cy.use_adaptive_sampling = False
                cy.time_limit = 0; cy.seed = 0; cy.use_animated_seed = False; cy.use_denoising = True
                cy.denoiser = 'OPENIMAGEDENOISE' if backend == 'CUDA' else 'OPTIX'
                cy.denoising_use_gpu = True; cy.denoising_prefilter = 'ACCURATE'
                cy.denoising_input_passes = 'RGB_ALBEDO_NORMAL'
                if hasattr(cy, 'denoising_quality'): cy.denoising_quality = 'HIGH'
                target_bounces = authored_bounces if args.profile == 'authored-look' else STANDARD_BOUNCES
                for key, value in zip(BOUNCE_FIELDS, target_bounces): setattr(cy, key, value)
                for layer in scene.view_layers: layer.cycles.use_denoising = True
            if args.profile == 'smoke':
                scene.view_settings.view_transform = 'AgX'; scene.view_settings.look = 'None'
                scene.view_settings.exposure = 0; scene.view_settings.gamma = 1
            # Match the existing visual smoke convention:1920x1080 at25%=480x270.
            r.resolution_x = args.width * 4; r.resolution_y = args.height * 4; r.resolution_percentage = 25
            r.use_border = False; r.use_crop_to_border = False
            r.use_motion_blur = False; r.use_simplify = False; r.use_sequencer = False
            r.use_stamp = False; r.dither_intensity = 0; r.use_persistent_data = True
            if hasattr(r, 'compositor_device'): r.compositor_device = 'CPU'
            for layer in scene.view_layers: layer.samples = 0
            fmt = r.image_settings
            if hasattr(fmt, 'media_type'): fmt.media_type = 'IMAGE'
            fmt.file_format = 'PNG'; fmt.color_mode = 'RGBA'; fmt.color_depth = '8'
            bpy.context.preferences.view.render_display_type = 'SCREEN'
            state['selected_gpu_ids'] = selected; state['audit'] = snapshot(); state['settings'] = settings()
            errors = checks()
            if errors: raise RuntimeError('; '.join(errors))
            bpy.app.handlers.render_pre.append(pre); bpy.app.handlers.render_complete.append(complete)
            bpy.app.handlers.render_cancel.append(cancel)
            state.update(status='RUNNING', started=time.perf_counter()); publish()
            print('NATIVE_RENDER_START', args.case, flush=True)
            with bpy.context.temp_override(window=window): result = bpy.ops.render.render('INVOKE_DEFAULT', write_still=False)
            if 'RUNNING_MODAL' not in result: raise RuntimeError('GUI/F12 job did not start')
            bpy.app.timers.register(poll, first_interval=.25)
        except Exception:
            state.update(status='SETUP FAILED', error=traceback.format_exc()); publish()
            print(state['error'], flush=True); bpy.ops.wm.quit_blender()
        return None

    publish(); bpy.app.timers.register(start, first_interval=3)


if __name__ == '__main__': main()
