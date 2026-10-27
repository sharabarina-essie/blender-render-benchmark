import unittest

from run_viewport_performance import assess
from viewport_performance import DrawRecorder, MODEL_KEYS, STANDARD_PROFILE, aggregate_runs, expand_cases


class ViewportPerformanceTests(unittest.TestCase):
    def test_standard_profile_is_opengl_full_hd_with_explicit_quality(self):
        self.assertEqual(STANDARD_PROFILE['graphics_backend'], 'OPENGL')
        self.assertEqual(STANDARD_PROFILE['viewport_px'], [1920, 1080])
        self.assertEqual(STANDARD_PROFILE['samples'], 64)
        self.assertTrue(STANDARD_PROFILE['eevee_raytracing'])
        self.assertTrue(STANDARD_PROFILE['eevee_raytracing_denoising'])
        self.assertTrue(STANDARD_PROFILE['cycles_preview_denoising'])
        self.assertEqual(STANDARD_PROFILE['cycles_preview_denoiser'], 'OPTIX')
    def test_large_run_variation_is_visible_and_failed_runs_are_excluded(self):
        rows = [{'case': 'material', 'status': 'PASS', 'viewport_draw_fps': fps} for fps in (9, 21, 24)]
        rows.append({'case': 'material', 'status': 'FAIL', 'viewport_draw_fps': 999})
        result = aggregate_runs(rows)['material']
        self.assertEqual(result['median_draw_fps'], 21)
        self.assertEqual(result['max_draw_fps'], 24)
        self.assertTrue(result['warning'])

    def test_smoke_and_full_select_requested_production_models(self):
        self.assertEqual(MODEL_KEYS['smoke'], 'archviz_classroom')
        self.assertEqual(MODEL_KEYS['full'], 'heavy_fisher_hut')

    def test_timer_ticks_and_repeated_draws_are_not_frames(self):
        recorder = DrawRecorder(10)
        self.assertTrue(recorder.record(10.1, 1, 5))
        self.assertFalse(recorder.record(10.2, 1, 5))
        self.assertTrue(recorder.record(10.3, 4, 15))
        stats = recorder.summarize(11)
        self.assertEqual(stats['draws'], 2)
        self.assertEqual(stats['viewport_draw_fps'], 2)
        self.assertAlmostEqual(stats['p95_frame_ms'], 200)
        self.assertAlmostEqual(stats['tail_without_new_draw_ms'], 700)
        self.assertIsNone(stats['client_presentation_fps'])

    def test_stalls_and_idle_tail_reduce_full_window_fps(self):
        recorder = DrawRecorder(0)
        for revision, now in enumerate([.01, .02, .03, .53, .54]):
            recorder.record(now, revision, revision)
        stats = recorder.summarize(1)
        self.assertEqual(stats['viewport_draw_fps'], 5)
        self.assertAlmostEqual(stats['one_percent_low_fps'], 2)
        self.assertAlmostEqual(stats['longest_without_new_draw_ms'], 500)
        self.assertEqual(stats['gaps_over_100_ms'], 2)

    def test_no_draws_produces_zero_fps_not_timer_rate(self):
        stats = DrawRecorder(5).summarize(7)
        self.assertEqual(stats['viewport_draw_fps'], 0)
        self.assertEqual(stats['longest_without_new_draw_ms'], 2000)
        self.assertIsNone(stats['p95_frame_ms'])

    def test_consistently_slow_frames_are_long_gaps_not_relative_stutters(self):
        recorder = DrawRecorder(0)
        for revision, now in enumerate([.2, .4, .6, .8, 1.0]):
            recorder.record(now, revision, revision * 10)
        stats = recorder.summarize(1.05)
        self.assertEqual(stats['relative_stutters'], 0)
        self.assertAlmostEqual(stats['stutter_threshold_ms'], 600)
        self.assertGreater(stats['gaps_over_100_ms'], 0)

    def test_stutter_events_preserve_times_and_orbit_angles(self):
        recorder = DrawRecorder(0)
        for revision, now in enumerate([.01, .02, .03, .54, .55]):
            recorder.record(now, revision, revision * 10)
        events = recorder.gap_events(.6)
        stutters = [event for event in events if event['relative_stutter']]
        self.assertEqual(len(stutters), 1)
        self.assertAlmostEqual(stutters[0]['start_s'], .03)
        self.assertAlmostEqual(stutters[0]['end_s'], .54)
        self.assertAlmostEqual(stutters[0]['duration_ms'], 510)
        self.assertEqual(stutters[0]['start_angle_deg'], 20)
        self.assertEqual(stutters[0]['end_angle_deg'], 30)
        self.assertTrue(stutters[0]['over_500_ms'])

    def test_initial_and_terminal_waits_are_not_completed_frame_stutters(self):
        recorder = DrawRecorder(0)
        for revision, now in enumerate([1, 1.01, 1.02]):
            recorder.record(now, revision, revision)
        events = recorder.gap_events(2)
        self.assertEqual([event['kind'] for event in events], ['start_to_first_draw', 'last_draw_to_end'])
        self.assertFalse(any(event['relative_stutter'] for event in events))
        self.assertEqual(recorder.summarize(2)['gaps_over_500_ms'], 2)

    def test_backwards_draws_or_clock_are_rejected(self):
        recorder = DrawRecorder(10)
        self.assertFalse(recorder.record(9, 1, 1))
        recorder.record(11, 3, 3)
        self.assertFalse(recorder.record(12, 2, 2))
        self.assertFalse(recorder.record(10.5, 4, 4))
        with self.assertRaises(ValueError):
            recorder.summarize(10.5)

    def test_exit_crash_and_missing_trials_fail_even_with_valid_measurements(self):
        report = {'complete': True, 'graphics_backend': 'OPENGL', 'results': [
            {'case': 'material', 'run': 1, 'status': 'PASS', 'viewport_draw_fps': 10, 'draws': 50,
             'viewport_px': [1920, 1080]}]}
        self.assertEqual(assess(report, ['material'], 1, 0, 'opengl'), [])
        self.assertTrue(assess(report, ['material'], 1, 1))
        self.assertTrue(assess(report, ['material'], 2, 0))
        report['graphics_backend'] = 'VULKAN'
        self.assertTrue(assess(report, ['material'], 1, 0))

    def test_wrong_viewport_resolution_is_not_a_passing_standard_run(self):
        report = {'complete': True, 'graphics_backend': 'OPENGL', 'results': [
            {'case': 'material', 'run': 1, 'status': 'PASS', 'viewport_draw_fps': 10, 'draws': 50,
             'viewport_px': [1280, 800]}]}
        self.assertTrue(any('1920x1080' in error for error in assess(report, ['material'], 1, 0)))

    def test_case_selection_is_explicit_and_order_preserving(self):
        self.assertEqual(expand_cases('material,solid,material'), ['material', 'solid'])
        self.assertIn('rendered-cycles-optix', expand_cases('all'))
        with self.assertRaises(ValueError):
            expand_cases('not-a-case')


if __name__ == '__main__':
    unittest.main()
