"""Run inside Blender's UI via run_viewport_performance.py.

The modal timer requests wall-clock-driven view changes. Only POST_PIXEL callbacks
for a new view revision in the target region count toward viewport draw FPS.
The timeline stays stationary. No forced GPU finish/readback occurs in the timed
loop. Draw callbacks do not establish GPU completion or remote presentation.
"""
import argparse
from collections import deque
import csv
import json
import math
from pathlib import Path
import sys
import time
import traceback

import blf
import bpy
import gpu
from mathutils import Quaternion, Vector

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from viewport_performance import CASES, STANDARD_PROFILE, DrawRecorder, expand_cases


def log(message):
    print('[viewport-performance] ' + message, flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', default='solid,material')
    parser.add_argument('--duration', type=float, required=True)
    parser.add_argument('--warmup', type=float, required=True)
    parser.add_argument('--runs', type=int, required=True)
    parser.add_argument('--samples', type=int, choices=[STANDARD_PROFILE['samples']], default=STANDARD_PROFILE['samples'])
    parser.add_argument('--gpu')
    parser.add_argument('--label', default='')
    parser.add_argument('--graphics-backend', choices=['opengl'], default='opengl')
    parser.add_argument('--keep-open', action='store_true')
    parser.add_argument('--quality-capture', action='store_true',
                        help='Separate orbit and stop/refine screenshots after the FPS trials')
    return parser.parse_args(sys.argv[sys.argv.index('--') + 1:])


ARGS = None
REPORT = None
BENCH = None


def scalar_settings(group):
    """Read back effective settings, including defaults supplied by this build."""
    result = {}
    for prop in group.bl_rna.properties:
        if prop.identifier == 'rna_type' or prop.type not in ('BOOLEAN', 'INT', 'FLOAT', 'ENUM', 'STRING'):
            continue
        value = getattr(group, prop.identifier)
        if getattr(prop, 'is_array', False):
            value = list(value)
        if isinstance(value, set):
            value = sorted(value)
        result[prop.identifier] = value
    return result


def reset_scalars(group):
    for prop in group.bl_rna.properties:
        if not prop.is_readonly and prop.type in ('BOOLEAN', 'INT', 'FLOAT', 'ENUM', 'STRING'):
            group.property_unset(prop.identifier)


def apply_standard_profile(scene, space):
    import engine_benchmark as eb
    profile = STANDARD_PROFILE
    # Discard model-authored renderer settings, retaining geometry/materials/lights.
    reset_scalars(scene.eevee)
    reset_scalars(scene.cycles)
    if hasattr(scene.eevee, 'ray_tracing_options'):
        reset_scalars(scene.eevee.ray_tracing_options)
    eb.apply_invariants(scene)
    r = scene.render
    r.resolution_x, r.resolution_y = profile['viewport_px']
    r.resolution_percentage = 100
    r.preview_pixel_size = '1'
    r.use_simplify = False
    r.use_border = r.use_crop_to_border = False
    r.use_motion_blur = profile['motion_blur']
    r.use_persistent_data = True
    r.dither_intensity = 0
    if scene.camera:
        scene.camera.data.dof.use_dof = profile['depth_of_field']
    scene.display.viewport_aa = '8'
    if hasattr(bpy.context.preferences.system, 'viewport_aa'):
        bpy.context.preferences.system.viewport_aa = '8'
    if hasattr(space.shading, 'use_compositor'):
        space.shading.use_compositor = profile['viewport_compositor']
    ev, cy = scene.eevee, scene.cycles
    ev.taa_samples = ev.taa_render_samples = profile['samples']
    ev.use_taa_reprojection = profile['eevee_temporal_reprojection']
    legacy = not hasattr(ev, 'use_raytracing')
    compatibility_notes = []
    if not legacy:
        ev.use_raytracing = profile['eevee_raytracing']
        ev.ray_tracing_method = profile['eevee_raytracing_method']
        if hasattr(ev, 'use_fast_gi'):
            ev.use_fast_gi = profile['eevee_fast_gi']
        else:
            compatibility_notes.append('This EEVEE build has no separate Fast GI switch; trace_max_roughness=1.0 disables the roughness fallback to Fast GI.')
        ev.use_shadow_jitter_viewport = False
        ev.use_shadows = True
        ev.shadow_ray_count = 1
        ev.shadow_step_count = 6
        ev.shadow_resolution_scale = 1.0
        ev.use_volumetric_shadows = False
        rt = ev.ray_tracing_options
        rt.resolution_scale = profile['eevee_raytracing_resolution']
        rt.screen_trace_quality = .5
        rt.screen_trace_thickness = .2
        rt.trace_max_roughness = 1.0  # Fast GI stays disabled for all surfaces.
        rt.use_denoise = profile['eevee_raytracing_denoising']
        rt.denoise_spatial = rt.denoise_temporal = rt.denoise_bilateral = True
    else:
        compatibility_notes.append('Legacy EEVEE uses SSR; Next raytracing denoising is unavailable. Compare within the same Blender build.')
        # EEVEE 3.x has SSR instead of EEVEE Next's ray-tracing/denoiser module.
        ev.use_ssr = True
        ev.use_ssr_halfres = True
        ev.ssr_quality = .5
        ev.ssr_thickness = .2
        ev.ssr_max_roughness = 1.0
        ev.use_gtao = False
        ev.use_bloom = False
    cy.preview_samples = cy.samples = profile['samples']
    cy.use_preview_adaptive_sampling = cy.use_adaptive_sampling = False
    cy.use_preview_denoising = cy.use_denoising = profile['cycles_preview_denoising']
    cy.preview_denoiser = cy.denoiser = profile['cycles_preview_denoiser']
    cy.preview_denoising_start_sample = profile['cycles_denoising_start_sample']
    cy.preview_denoising_input_passes = 'RGB_ALBEDO_NORMAL'
    if hasattr(cy, 'preview_denoising_use_gpu'):
        cy.preview_denoising_use_gpu = True
    cy.preview_pause = False
    for name, value in {'max_bounces': 16, 'diffuse_bounces': 4, 'glossy_bounces': 4,
                        'transmission_bounces': 12, 'volume_bounces': 0,
                        'transparent_max_bounces': 16, 'sample_clamp_direct': 0.0,
                        'sample_clamp_indirect': 10.0, 'seed': 0, 'use_animated_seed': False}.items():
        setattr(cy, name, value)
    return {'id': profile['id'], 'requested': dict(profile),
            'eevee_generation': 'legacy SSR' if legacy else 'EEVEE Next raytracing',
            'compatibility_notes': compatibility_notes,
            'eevee': scalar_settings(ev),
            'eevee_raytracing_options': scalar_settings(ev.ray_tracing_options) if not legacy else None,
            'cycles': scalar_settings(cy),
            'color_management': scalar_settings(scene.view_settings),
            'workbench_viewport_aa': scene.display.viewport_aa,
            'compositor': profile['viewport_compositor'], 'motion_blur': r.use_motion_blur,
            'depth_of_field': scene.camera.data.dof.use_dof if scene.camera else False}


def resize_own_window(delta_width, delta_height):
    """Adjust only this Blender process's window, never monitor resolution."""
    if sys.platform != 'win32':
        raise RuntimeError('Viewport size differs from the standard; automatic window calibration requires Windows')
    import ctypes
    from ctypes import wintypes
    import os
    user32 = ctypes.windll.user32
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]
    user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
    windows = []
    def collect(hwnd, _):
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        rect = wintypes.RECT()
        if pid.value == os.getpid() and user32.GetClientRect(hwnd, ctypes.byref(rect)):
            windows.append(((rect.right - rect.left) * (rect.bottom - rect.top), hwnd))
        return True
    user32.EnumWindows(callback_type(collect), 0)
    if not windows:
        raise RuntimeError('Cannot find this Blender process window for viewport calibration')
    hwnd = max(windows)[1]
    rect = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    width, height = rect.right - rect.left + delta_width, rect.bottom - rect.top + delta_height
    if not user32.SetWindowPos(hwnd, None, 0, 0, width, height, 0x0002 | 0x0004 | 0x0010):
        raise RuntimeError('Failed to resize the benchmark window')


def save_report():
    temp = ARGS.output / 'report.tmp'
    temp.write_text(json.dumps(REPORT, indent=2), encoding='utf-8')
    temp.replace(ARGS.output / 'report.json')


def quit_later():
    if not ARGS.keep_open:
        def quit_app():
            bpy.ops.wm.quit_blender()
        bpy.app.timers.register(quit_app, first_interval=.5)


class VIEWPORT_OT_orbit_performance(bpy.types.Operator):
    bl_idname = 'wm.orbit_performance'
    bl_label = 'Viewport Orbit Performance'

    def execute(self, context):
        try:
            return self.setup(context)
        except Exception:
            self.finish(traceback.format_exc())
            return {'CANCELLED'}

    def setup(self, context):
        global BENCH
        BENCH = self
        self.window = context.window
        self.area = max((a for a in self.window.screen.areas if a.type == 'VIEW_3D'),
                        key=lambda a: a.width * a.height)
        self.region = next(r for r in self.area.regions if r.type == 'WINDOW')
        self.space = self.area.spaces.active
        self.rv3d = self.space.region_3d
        self.scene = context.scene
        self.original_rotation = self.rv3d.view_rotation.copy()
        self.original_location = self.rv3d.view_location.copy()
        self.original_distance = self.rv3d.view_distance
        self.original_perspective = self.rv3d.view_perspective
        self.base_rotation = self.original_rotation.copy()
        self.set_camera_view()
        self.space.overlay.show_overlays = False
        self.space.show_gizmo = False
        self.space.show_region_ui = False
        self.space.show_region_toolbar = False
        REPORT['quality_profile'] = apply_standard_profile(self.scene, self.space)
        self.cases = expand_cases(ARGS.cases)
        self.case_index = -1
        self.recorder = None
        self.callback_error = None
        self.revision = 0
        self.drawn_revision = 0
        self.request_phase = None
        self.angle = 0.0
        self.phase = 'setup'
        self.hud = 'Preparing viewport orbit benchmark'
        self.handle = bpy.types.SpaceView3D.draw_handler_add(self.on_draw, (), 'WINDOW', 'POST_PIXEL')
        self.timer = context.window_manager.event_timer_add(.001, window=self.window)
        context.window_manager.modal_handler_add(self)
        REPORT['viewport'] = {
            'region_px': [self.region.width, self.region.height], 'pixel_size': 1,
            'workspace': self.window.workspace.name, 'overlays': False,
            'hdr_environment': 'forest.exr', 'samples': ARGS.samples,
            'cycles_preview_denoising': STANDARD_PROFILE['cycles_preview_denoising'],
            'cycles_preview_adaptive_sampling': False,
            'eevee_temporal_reprojection': getattr(self.scene.eevee, 'use_taa_reprojection', None),
            'timer_requested_interval_s': .001, 'timeline_frame': self.scene.frame_current,
            'view_requests': 'One outstanding view change; advance after its draw callback',
            'base_rotation': list(self.base_rotation), 'pivot': list(self.rv3d.view_location),
            'distance': self.rv3d.view_distance,
        }
        self.next_case()
        return {'RUNNING_MODAL'}

    def set_camera_view(self):
        """Start at the authored camera, with an orbit pivot along its view ray."""
        camera = self.scene.camera
        if camera:
            rotation = camera.matrix_world.to_quaternion()
            forward = rotation @ Vector((0, 0, -1))
            # Median object positions avoid a single huge ground plane setting the pivot.
            import statistics
            points = [o.matrix_world.translation for o in self.scene.objects
                      if o.type == 'MESH' and o.visible_get()]
            center = Vector(tuple(statistics.median(p[i] for p in points) for i in range(3))) if points else Vector((0, 0, 0))
            distance = max(.5, (center - camera.matrix_world.translation).dot(forward))
            self.rv3d.view_location = camera.matrix_world.translation + forward * distance
            self.rv3d.view_distance = distance
            self.space.lens = camera.data.lens
            self.space.clip_start = camera.data.clip_start
            self.space.clip_end = camera.data.clip_end
            self.base_rotation = rotation
        self.rv3d.view_perspective = 'PERSP'
        self.rv3d.view_rotation = self.base_rotation

    def next_case(self):
        self.case_index += 1
        if self.case_index >= len(self.cases):
            self.finish()
            return
        self.case = self.cases[self.case_index]
        shading, compute = CASES[self.case]
        self.space.shading.type = 'SOLID'
        import engine_benchmark as eb
        if compute in ('CUDA', 'OPTIX'):
            eb.GPU_SUBSTR = ARGS.gpu
            ok, devices, reason = eb.configure_backend(compute)
            if not ok:
                raise RuntimeError(f'{compute} device selection failed: {reason}')
            self.devices = devices
        else:
            self.scene.render.engine = eb._resolve_engine_id('BLENDER_EEVEE_NEXT', self.scene)
            self.devices = []
        self.space.shading.use_scene_lights = self.space.shading.use_scene_world = self.case == 'material-world'
        self.space.shading.use_scene_lights_render = self.space.shading.use_scene_world_render = True
        self.space.shading.light = 'STUDIO'
        self.space.shading.studiolight_rotate_z = STANDARD_PROFILE['hdr_rotation']
        self.space.shading.studiolight_intensity = STANDARD_PROFILE['hdr_intensity']
        self.space.shading.studiolight_background_alpha = 0
        self.space.shading.type = shading
        if shading == 'MATERIAL':
            self.space.shading.studio_light = STANDARD_PROFILE['hdr_environment']
        elif shading == 'SOLID':
            self.space.shading.studio_light = 'paint.sl'
            self.space.shading.color_type = 'MATERIAL'
            self.space.shading.show_shadows = True
            self.space.shading.show_cavity = False
        self.case_shading_settings = scalar_settings(self.space.shading)
        self.case_engine_settings = {'cycles_device': self.scene.cycles.device,
                                     'compute_backend': compute,
                                     'cycles': scalar_settings(self.scene.cycles)} if compute in ('CUDA', 'OPTIX') else None
        self.run = 0
        self.case_start = self.phase_start = time.perf_counter()
        self.phase = 'warmup'
        self.warm_draws = 0
        self.warm_last_revision = -1
        self.last_progress = self.phase_start
        self.hud = f'{self.case}: warm-up'
        REPORT['active_case'] = self.case
        save_report()
        log(f'CASE {self.case}: warming for at least {ARGS.warmup:g}s')

    def request_view(self, now):
        # Do not invalidate an undrawn view on every timer tick. That can starve
        # expensive engines and would benchmark the request generator instead.
        if self.revision > self.drawn_revision:
            return
        duration = ARGS.warmup if self.phase == 'warmup' else ARGS.duration
        fraction = (now - self.phase_start) / max(duration, .001)
        self.angle = fraction * 360
        self.rv3d.view_rotation = Quaternion((0, 0, 1), math.radians(self.angle)) @ self.base_rotation
        self.revision += 1
        self.request_phase = self.phase
        self.area.tag_redraw()

    def on_draw(self):
        try:
            if bpy.context.area != self.area or bpy.context.region != self.region:
                return
            if self.phase == 'measure' and (self.region.width, self.region.height) != self.start_size:
                self.resolution_changed = True
            now = time.perf_counter()
            changed = self.revision > self.drawn_revision
            self.drawn_revision = self.revision
            if changed and self.request_phase == self.phase and self.phase == 'measure':
                if self.recorder.record(now, self.revision, self.angle):
                    self.recent_draws.append(now)
                    while self.recent_draws and self.recent_draws[0] < now - 1:
                        self.recent_draws.popleft()
                    window = min(1.0, now - self.recorder.started)
                    fps = len(self.recent_draws) / window if window > 0 else 0
                    average = len(self.recorder.samples) / (now - self.recorder.started)
                    self.hud = (f'{self.case} | run {self.run}/{ARGS.runs} | '
                                f'{fps:.1f} draw FPS (avg {average:.1f}) | {now - self.phase_start:.1f}/{ARGS.duration:g}s')
            elif changed and self.request_phase == self.phase and self.phase == 'warmup':
                self.warm_draws += 1
                self.warm_last_revision = self.revision
            # Keep text out of quality comparisons. Quality capture runs only
            # after the timed trials, and never adds readbacks to their loop.
            if self.phase.startswith('quality-'):
                return
            try:
                blf.size(0, 18)
            except TypeError:
                blf.size(0, 18, 72)
            blf.color(0, 1, 1, 0.25, 1)
            blf.position(0, 18, self.region.height - 32, 0)
            blf.draw(0, self.hud)
        except Exception:
            self.callback_error = traceback.format_exc()

    def modal(self, context, event):
        if event.type == 'ESC':
            self.finish('Cancelled by user')
            return {'CANCELLED'}
        if self.phase == 'finished':
            return {'FINISHED'}
        if event.type != 'TIMER':
            return {'RUNNING_MODAL'}
        try:
            if self.callback_error:
                raise RuntimeError(self.callback_error)
            now = time.perf_counter()
            if self.phase == 'warmup':
                compiling = hasattr(bpy.app, 'is_job_running') and bpy.app.is_job_running('SHADER_COMPILATION')
                if now - self.phase_start >= ARGS.warmup and self.warm_draws >= 3 and not compiling:
                    self.start_run(now)
            elif self.phase == 'measure' and now - self.phase_start >= ARGS.duration:
                self.end_run(now)
                if self.run >= ARGS.runs:
                    self.capture()
                    if ARGS.quality_capture:
                        self.start_quality()
                    else:
                        self.next_case()
                else:
                    self.start_run(time.perf_counter())
            elif self.phase.startswith('quality-'):
                self.advance_quality(now)
            if self.phase in ('warmup', 'measure'):
                self.request_view(time.perf_counter())
            if now - self.last_progress > 5:
                self.last_progress = now
                log(self.hud)
        except Exception:
            self.finish(traceback.format_exc())
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def start_run(self, now):
        if [self.region.width, self.region.height] != STANDARD_PROFILE['viewport_px']:
            raise RuntimeError(f'Viewport must be exactly 1920x1080, actual {self.region.width}x{self.region.height}')
        if self.run == 0:
            self.warmup_elapsed = now - self.case_start
        self.run += 1
        self.phase = 'measure'
        self.phase_start = now
        self.recorder = DrawRecorder(now)
        self.recent_draws = deque()
        self.start_size = (self.region.width, self.region.height)
        self.resolution_changed = False
        self.start_frame = self.scene.frame_current
        log(f'{self.case} run {self.run}: measuring {ARGS.duration:g}s')

    def end_run(self, now):
        self.phase = 'between-runs'
        metrics = self.recorder.summarize(now)
        errors = []
        if metrics['draws'] < 3:
            errors.append('Fewer than three distinct views were drawn')
        if self.resolution_changed or (self.region.width, self.region.height) != self.start_size:
            errors.append('Viewport resized during measurement')
        if self.scene.frame_current != self.start_frame:
            errors.append('Timeline advanced during the orbit')
        data_file = f'{self.case}_run{self.run}_frames.csv'
        with (ARGS.output / data_file).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['time_s', 'revision', 'angle_deg', 'interval_ms'])
            writer.writeheader()
            writer.writerows(self.recorder.samples)
        gaps_file = f'{self.case}_run{self.run}_gaps.csv'
        events = self.recorder.gap_events(now)
        with (ARGS.output / gaps_file).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['kind', 'start_s', 'end_s', 'duration_ms',
                'start_angle_deg', 'end_angle_deg', 'relative_stutter', 'over_500_ms'])
            writer.writeheader()
            writer.writerows(events)
        row = {'case': self.case, 'run': self.run, 'status': 'FAIL' if errors else 'PASS',
               'errors': errors, 'frames_csv': data_file, 'viewport_px': list(self.start_size),
               'gaps_csv': gaps_file, 'gap_events': events,
               'shading': self.space.shading.type, 'render_engine': self.scene.render.engine,
               'viewport_engine': ('WORKBENCH' if self.space.shading.type == 'SOLID' else
                                   'EEVEE' if self.space.shading.type == 'MATERIAL' else self.scene.render.engine),
               'viewport_shading_settings': self.case_shading_settings,
               'case_engine_settings': self.case_engine_settings,
               'compute_devices': self.devices, 'warmup_elapsed_s': self.warmup_elapsed,
               **metrics}
        REPORT['results'].append(row)
        save_report()
        log(f"RESULT {self.case} run {self.run}: {row['status']} {metrics['viewport_draw_fps']:.2f} draw FPS, "
            f"p95 {metrics['p95_frame_ms']} ms, {metrics['draws']} draws")

    def capture(self):
        path = ARGS.output / f'{self.case}.png'
        with bpy.context.temp_override(window=self.window, area=self.area, region=self.region):
            bpy.ops.screen.screenshot_area(filepath=str(path))
        for row in REPORT['results']:
            if row['case'] == self.case:
                row['screenshot'] = path.name
        save_report()

    def start_quality(self):
        self.phase = 'quality-motion'
        self.phase_start = time.perf_counter()
        self.quality_next_motion = ARGS.duration / 4
        self.quality_stop = None
        self.quality_checkpoints = deque([0, .25, .5, 1, 2, 5, 10, 15])
        self.quality = {'case': self.case, 'frames': [], 'complete': False,
                        'orbit_duration_requested_s': ARGS.duration,
                        'settle_duration_requested_s': 15,
                        'internal_render_resolution_px': None,
                        'completed_samples': None,
                        'settings': self.case_engine_settings,
                        'capture_scope': 'Blender screenshot_area framebuffer readback; '
                            'not an agent pre-encoder capture or independent client presentation measurement'}
        REPORT.setdefault('quality_cases', []).append(self.quality)
        self.hud = f'{self.case}: separate image-quality pass'
        # Remove the previous trial HUD before the first evidence frame.
        self.area.tag_redraw()
        log(f'{self.case}: separate orbit captures, then 15s stationary refinement')

    def quality_frame(self, phase, requested):
        frame = {'phase': phase, 'requested_time_s': requested,
                 'view_rotation': list(self.rv3d.view_rotation),
                 'view_location': list(self.rv3d.view_location),
                 'view_distance': self.rv3d.view_distance,
                 'requested_revision': self.revision, 'drawn_revision': self.drawn_revision,
                 'viewport_px': [self.region.width, self.region.height],
                 'preview_pixel_size': self.scene.render.preview_pixel_size}
        origin = self.phase_start if phase == 'moving' else self.quality_stop
        before = time.perf_counter()
        frame['actual_time_s'] = before - origin
        path = ARGS.output / 'quality' / self.case / f'{len(self.quality["frames"]):02d}_{phase}.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        with bpy.context.temp_override(window=self.window, area=self.area, region=self.region):
            bpy.ops.screen.screenshot_area(filepath=str(path))
        frame['readback_duration_s'] = time.perf_counter() - before
        frame['file'] = path.relative_to(ARGS.output).as_posix()
        self.quality['frames'].append(frame)
        save_report()

    def advance_quality(self, now):
        if self.phase == 'quality-motion':
            elapsed = now - self.phase_start
            if elapsed >= ARGS.duration:
                # Freeze exactly the last requested pose; do not jump to a new
                # pose between the moving image and the stationary reference.
                self.phase = 'quality-drain'
                self.quality_stop = time.perf_counter()
            elif elapsed >= self.quality_next_motion and self.revision == self.drawn_revision:
                self.quality_frame('moving', self.quality_next_motion)
                self.quality_next_motion += ARGS.duration / 4
            if self.phase == 'quality-motion':
                self.request_view(time.perf_counter())
        elif self.phase == 'quality-drain' and self.revision == self.drawn_revision:
            self.phase = 'quality-settle'
            self.quality_frame('stopped', self.quality_checkpoints.popleft())
        elif self.phase == 'quality-settle':
            if self.quality_checkpoints and now - self.quality_stop >= self.quality_checkpoints[0]:
                self.quality_frame('stopped', self.quality_checkpoints.popleft())
            if not self.quality_checkpoints:
                self.quality['complete'] = True
                save_report()
                self.next_case()

    def finish(self, error=None):
        self.phase = 'finished'
        if hasattr(self, 'timer'):
            bpy.context.window_manager.event_timer_remove(self.timer)
        if hasattr(self, 'handle'):
            bpy.types.SpaceView3D.draw_handler_remove(self.handle, 'WINDOW')
        if hasattr(self, 'original_rotation'):
            self.rv3d.view_rotation = self.original_rotation
            self.rv3d.view_location = self.original_location
            self.rv3d.view_distance = self.original_distance
            self.rv3d.view_perspective = self.original_perspective
        REPORT['complete'] = error is None
        if error:
            REPORT['error'] = error
            log('ERROR ' + error)
        REPORT['active_case'] = None
        save_report()
        log('Finished: ' + str(ARGS.output / 'report.json'))
        quit_later()


def start():
    try:
        window = bpy.context.window
        workspace = bpy.data.workspaces.get('Layout')
        if workspace:
            window.workspace = workspace
        area = max((a for a in window.screen.areas if a.type == 'VIEW_3D'), key=lambda a: a.width * a.height)
        region = next(r for r in area.regions if r.type == 'WINDOW')
        with bpy.context.temp_override(window=window, area=area, region=region):
            bpy.ops.screen.screen_full_area(use_hide_panels=True)
        bpy.app.timers.register(invoke, first_interval=.5)
    except Exception:
        REPORT['error'] = traceback.format_exc()
        save_report()
        log(REPORT['error'])
        quit_later()


def invoke(attempt=0):
    try:
        window = bpy.context.window
        area = max((a for a in window.screen.areas if a.type == 'VIEW_3D'), key=lambda a: a.width * a.height)
        region = next(r for r in area.regions if r.type == 'WINDOW')
        target_width, target_height = STANDARD_PROFILE['viewport_px']
        if (region.width, region.height) != (target_width, target_height):
            if attempt >= 3:
                raise RuntimeError(f'Cannot obtain 1920x1080 viewport; actual {region.width}x{region.height}. Check monitor size/scaling.')
            log(f'Calibrating viewport {region.width}x{region.height} -> 1920x1080')
            resize_own_window(target_width - region.width, target_height - region.height)
            bpy.app.timers.register(lambda: invoke(attempt + 1), first_interval=1)
            return
        bpy.utils.register_class(VIEWPORT_OT_orbit_performance)
        with bpy.context.temp_override(window=window, area=area, region=region):
            bpy.ops.wm.orbit_performance()
    except Exception:
        REPORT['error'] = traceback.format_exc()
        save_report()
        log(REPORT['error'])
        quit_later()


def main():
    global ARGS, REPORT
    ARGS = parse_args()
    ARGS.output.mkdir(parents=True, exist_ok=True)
    REPORT = {'schema_version': 1, 'complete': False, 'results': [], 'label': ARGS.label,
              'scene': str(ARGS.scene), 'requested_cases': expand_cases(ARGS.cases),
              'requested_duration_s': ARGS.duration, 'requested_warmup_s': ARGS.warmup,
              'requested_runs': ARGS.runs,
              'measurement_scope': 'Unique target-region POST_PIXEL callbacks after a requested view change; '
                  'not GPU completion, rendered-image freshness, input-to-photon latency, or client presentation FPS.'}
    save_report()
    try:
        if bpy.app.background:
            raise RuntimeError('This benchmark requires a real Blender window')
        bpy.context.preferences.view.show_splash = False
        bpy.ops.wm.open_mainfile(filepath=str(ARGS.scene), load_ui=False, use_scripts=False)
        REPORT.update({'blender': bpy.app.version_string,
                       'build_hash': bpy.app.build_hash.decode(),
                       'graphics_backend': gpu.platform.backend_type_get() if hasattr(gpu.platform, 'backend_type_get') else 'OPENGL (legacy Windows inference)',
                       'renderer': gpu.platform.renderer_get(), 'vendor': gpu.platform.vendor_get(),
                       'graphics_version': gpu.platform.version_get(),
                       'requested_graphics_backend': ARGS.graphics_backend})
        save_report()
        bpy.app.timers.register(start, first_interval=1)
    except Exception:
        REPORT['error'] = traceback.format_exc()
        save_report()
        log(REPORT['error'])
        quit_later()


if __name__ == '__main__':
    main()
