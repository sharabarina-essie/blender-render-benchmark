"""Host-independent orbit configuration and draw-cadence statistics.

These are Blender POST_PIXEL draw callbacks for changed views, not GPU timestamps
or proof that a remote frame has reached the client's display.
"""
import math
import statistics


STANDARD_PROFILE = {
    'id': 'opengl-1080p-v1',
    'graphics_backend': 'OPENGL',
    'viewport_px': [1920, 1080],
    'pixel_size': 1,
    'samples': 64,
    'cycles_preview_adaptive_sampling': False,
    'cycles_preview_denoising': True,
    'cycles_preview_denoiser': 'OPTIX',
    'cycles_denoising_start_sample': 1,
    'eevee_raytracing': True,
    'eevee_raytracing_method': 'SCREEN',
    'eevee_raytracing_resolution': '2',
    'eevee_raytracing_denoising': True,
    'eevee_fast_gi': False,
    'eevee_temporal_reprojection': False,
    'motion_blur': False,
    'depth_of_field': False,
    'viewport_compositor': 'DISABLED',
    'seed': 0,
    'hdr_environment': 'forest.exr',
    'hdr_intensity': 1.0,
    'hdr_rotation': 0.0,
}

MODEL_KEYS = {'smoke': 'archviz_classroom', 'full': 'heavy_fisher_hut'}
DEFAULT_CASES = ('solid', 'material')
CASES = {
    'solid': ('SOLID', None),
    'material': ('MATERIAL', None),
    'material-world': ('MATERIAL', None),
    'rendered-eevee': ('RENDERED', 'EEVEE'),
    'rendered-cycles-optix': ('RENDERED', 'OPTIX'),
    'rendered-cycles-cuda': ('RENDERED', 'CUDA'),
}


def expand_cases(spec):
    names = list(CASES) if spec == 'all' else list(dict.fromkeys(spec.split(',')))
    if not names or any(name not in CASES for name in names):
        raise ValueError('Cases must be all or a comma-separated selection of ' + ','.join(CASES))
    return names


def percentile(values, fraction):
    """Linearly interpolated percentile, including both endpoints."""
    ordered = sorted(values)
    if not ordered:
        return None
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def aggregate_runs(rows):
    """Expose run-to-run variation instead of hiding it behind a median."""
    result = {}
    for row in rows:
        if row.get('status') == 'PASS':
            result.setdefault(row['case'], []).append(row['viewport_draw_fps'])
    for case, values in result.items():
        median = statistics.median(values)
        variation = (max(values) - min(values)) / median * 100 if median else 0
        result[case] = {'runs': len(values), 'median_draw_fps': median,
                        'min_draw_fps': min(values), 'max_draw_fps': max(values),
                        'run_variation_pct': variation,
                        'warning': 'Run range exceeds 25% of median; inspect individual trials before comparing environments.'
                        if len(values) > 1 and variation > 25 else None}
    return result


class DrawRecorder:
    """Deduplicate redraws of the same requested view; clocks are monotonic."""

    def __init__(self, started):
        self.started = started
        self.last_revision = -1
        self.samples = []

    def record(self, now, revision, angle_deg):
        if revision <= self.last_revision or now < self.started:
            return False
        if self.samples and now <= self.samples[-1]['time_s'] + self.started:
            return False
        self.last_revision = revision
        elapsed = now - self.started
        previous = self.samples[-1]['time_s'] if self.samples else 0.0
        self.samples.append({'time_s': elapsed, 'revision': revision,
                             'angle_deg': angle_deg, 'interval_ms': (elapsed - previous) * 1000})
        return True

    def summarize(self, ended):
        elapsed = ended - self.started
        if elapsed <= 0 or any(s['time_s'] > elapsed for s in self.samples):
            raise ValueError('Measurement end must follow its start and all draws')
        intervals = [s['interval_ms'] for s in self.samples[1:]]
        first = self.samples[0]['time_s'] * 1000 if self.samples else None
        tail = (elapsed - self.samples[-1]['time_s']) * 1000 if self.samples else elapsed * 1000
        silences = intervals + [tail] + ([first] if first is not None else [])
        worst_count = max(1, math.ceil(len(intervals) * .01))
        worst = sorted(intervals, reverse=True)[:worst_count]
        events = self.gap_events(ended)
        relative = [event for event in events if event['relative_stutter']]
        threshold = max(100.0, 3 * statistics.median(intervals)) if intervals else 100.0
        return {
            'metric': 'unique_viewport_draw_callbacks_per_second',
            'duration_s': elapsed, 'draws': len(self.samples),
            'viewport_draw_fps': len(self.samples) / elapsed,
            'median_frame_ms': statistics.median(intervals) if intervals else None,
            'p95_frame_ms': percentile(intervals, .95),
            'p99_frame_ms': percentile(intervals, .99),
            'one_percent_low_fps': 1000 / statistics.mean(worst) if worst else None,
            'time_to_first_draw_ms': first, 'tail_without_new_draw_ms': tail,
            'longest_without_new_draw_ms': max(silences),
            'gaps_over_100_ms': sum(value > 100 for value in silences),
            'gaps_over_250_ms': sum(value > 250 for value in silences),
            'gaps_over_500_ms': sum(value > 500 for value in silences),
            'stutter_threshold_ms': threshold,
            'relative_stutters': len(relative),
            'relative_stutter_time_ms': sum(event['duration_ms'] for event in relative),
            'stutter_policy': 'Between-draw gap > max(100ms, 3 x median between-draw interval). Initial/tail delays are separate.',
            'client_presentation_fps': None,
        }

    def gap_events(self, ended):
        """Timestamp gaps without labelling every frame of a slow run a stutter."""
        elapsed = ended - self.started
        if elapsed <= 0 or any(s['time_s'] > elapsed for s in self.samples):
            raise ValueError('Invalid measurement bounds')
        intervals = [s['interval_ms'] for s in self.samples[1:]]
        threshold = max(100.0, 3 * statistics.median(intervals)) if intervals else 100.0
        events = []

        def append(kind, start, end, start_angle, end_angle):
            duration = (end - start) * 1000
            if duration > 100:
                events.append({'kind': kind, 'start_s': start, 'end_s': end,
                               'duration_ms': duration, 'start_angle_deg': start_angle,
                               'end_angle_deg': end_angle,
                               'relative_stutter': kind == 'between_draws' and duration > threshold,
                               'over_500_ms': duration > 500})

        if not self.samples:
            append('no_draws', 0, elapsed, None, None)
            return events
        append('start_to_first_draw', 0, self.samples[0]['time_s'], None, self.samples[0]['angle_deg'])
        for previous, current in zip(self.samples, self.samples[1:]):
            append('between_draws', previous['time_s'], current['time_s'],
                   previous['angle_deg'], current['angle_deg'])
        append('last_draw_to_end', self.samples[-1]['time_s'], elapsed, self.samples[-1]['angle_deg'], None)
        return events
