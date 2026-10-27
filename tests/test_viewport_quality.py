import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageFilter

from viewport_quality import analyze_quality, detail_energy, image_metrics


class ViewportQualityTests(unittest.TestCase):
    def setUp(self):
        y, x = np.indices((128, 128))
        self.reference = np.repeat((((x // 3 + y // 3) % 2) * 255)[..., None], 3, axis=2).astype('uint8')

    def test_identical_image_has_perfect_similarity(self):
        metrics = image_metrics(self.reference, self.reference)
        self.assertAlmostEqual(metrics['ssim_luma_8x8'], 1)
        self.assertEqual(metrics['rgb_mae_255'], 0)

    def test_coarse_upscaled_and_blurred_images_do_not_pass_similarity(self):
        image = Image.fromarray(self.reference)
        coarse = image.resize((16, 16), Image.Resampling.BOX).resize(image.size, Image.Resampling.NEAREST)
        blurred = image.filter(ImageFilter.GaussianBlur(2))
        for degraded in (coarse, blurred):
            with self.subTest(image=degraded):
                metrics = image_metrics(np.array(degraded), self.reference)
                self.assertLess(metrics['ssim_luma_8x8'], .8)
                self.assertGreater(metrics['rgb_mae_255'], 10)
                self.assertLess(detail_energy(degraded), detail_energy(self.reference) / 2)

    def test_different_dimensions_are_rejected_instead_of_resampled(self):
        with self.assertRaises(ValueError):
            image_metrics(self.reference[:64], self.reference)

    def test_five_second_checkpoint_reports_refinement_and_actual_timing(self):
        reference = Image.fromarray(self.reference).resize((1920, 1080), Image.Resampling.NEAREST)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            reference.save(output / 'final.png')
            reference.filter(ImageFilter.GaussianBlur(4)).save(output / 'initial.png')
            frames = [{'phase': 'moving', 'file': 'initial.png', 'actual_time_s': 1,
                       'requested_time_s': 1, 'readback_duration_s': .01}]
            for t in (0, .25, .5, 1, 2, 5, 10, 15):
                frames.append({'phase': 'stopped', 'file': 'final.png' if t >= 5 else 'initial.png',
                               'actual_time_s': t+.05, 'requested_time_s': t, 'readback_duration_s': .01,
                               'view_rotation': [1, 0, 0, 0], 'view_location': [0, 0, 0], 'view_distance': 10})
            for index in (1, 2, 3):
                for t in (0, .25, .5, 1, 2, 4, 5):
                    frames.append({'phase': 'pause', 'pose_index': index, 'orbit_angle_deg': 90*index,
                                   'file': 'final.png' if t >= 4 else 'initial.png',
                                   'actual_time_s': t+.05, 'requested_time_s': t, 'readback_duration_s': .01,
                                   'view_rotation': [1, 0, 0, 0], 'view_location': [index, 0, 0], 'view_distance': 10})
            self.assertFalse(analyze_quality(output, {'quality_cases': [
                {'case': 'rendered-cycles-optix', 'complete': True, 'frames': frames,
                 'pause_angles_deg': [90, 180, 270],
                 'pauses': [{'pose_index': i, 'actual_hold_s': 5.4} for i in (1, 2, 3)]}]}))
            case = json.loads((output / 'quality.json').read_text())['cases'][0]
            five = case['five_second_refinement']
            self.assertTrue(five['closer_to_reference_than_initial'])
            self.assertTrue(five['close_to_final_reference'])
            self.assertTrue(five['timing_within_half_second'])
            self.assertAlmostEqual(five['actual_time_s'], 5.05)
            self.assertEqual(len(case['pause_comparisons']), 3)
            for pause in case['pause_comparisons']:
                self.assertTrue(pause['image_changed_during_hold'])
                self.assertTrue(pause['four_and_five_second_images_similar'])
            frames[-1]['view_distance'] = 11
            errors = analyze_quality(output, {'quality_cases': [
                {'case': 'rendered-cycles-optix', 'complete': True, 'frames': frames}]})
            self.assertIn('Intermediate pause captures have different requested poses', errors[0])

    def test_different_stationary_poses_invalidate_the_comparison(self):
        frames = [{'phase': 'moving'}] + [
            {'phase': 'stopped', 'view_rotation': [1, 0, 0, 0],
             'view_location': [0, 0, 0], 'view_distance': 10} for _ in range(8)]
        frames[-1]['view_distance'] = 11
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            # Invalid evidence should still produce a readable diagnostic report.
            for i, f in enumerate(frames):
                f.update(file=f'{i}.png', actual_time_s=float(i), requested_time_s=float(i),
                         readback_duration_s=0)
            errors = analyze_quality(output, {'quality_cases': [
                {'case': 'rendered-cycles-optix', 'complete': True, 'frames': frames}]})
            self.assertIn('different requested poses', errors[0])
            data = json.loads((output / 'quality.json').read_text())
            self.assertEqual(data['cases'][0]['status'], 'INVALID')
            self.assertTrue((output / 'quality.html').is_file())


if __name__ == '__main__':
    unittest.main()
