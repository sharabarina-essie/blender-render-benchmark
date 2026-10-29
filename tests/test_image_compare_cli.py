import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageCms

from compare_images import compare_arrays, read_image


SCRIPT = Path(__file__).resolve().parents[1] / "compare_images.py"


class ImageCompareCliTests(unittest.TestCase):
    def test_identical_and_known_constant_error(self):
        a = np.full((9, 9, 4), [50, 100, 150, 255], dtype=np.uint8)
        same, _ = compare_arrays(a, a)
        self.assertEqual(same["rgb_mae_255"], 0)
        self.assertAlmostEqual(same["ssim_luma_8x8"], 1, places=12)
        self.assertIsNone(same["rgb_psnr_db"])
        self.assertTrue(same["rgb_identical"])
        b = np.full_like(a, [60, 110, 160, 255])
        m, diff = compare_arrays(a, b, tolerance=10, gain=4, make_difference=True)
        self.assertEqual(m["rgb_mae_255"], 10)
        self.assertEqual(m["rgb_rmse_255"], 10)
        self.assertAlmostEqual(m["rgb_psnr_db"], 20 * math.log10(255 / 10))
        self.assertEqual(m["rgb_changed_pixels"], 0)
        self.assertTrue((diff[:, :, :3] == 40).all())
        self.assertEqual(compare_arrays(a, b, tolerance=9)[0]["rgb_changed_pixels"], 81)

    def test_partial_edge_blocks_and_non_aligned_roi(self):
        a = np.full((9, 9, 4), [0, 0, 0, 255], dtype=np.uint8)
        b = a.copy()
        b[-1, -1, :3] = 255
        m, _ = compare_arrays(a, b)
        self.assertEqual(m["pixel_count"], 81)
        self.assertEqual(m["rgb_changed_pixels"], 1)
        self.assertAlmostEqual(m["ssim_luma_8x8"], (80 + 2.55**2 / (255**2 + 2.55**2)) / 81)
        m, d = compare_arrays(a, b, margins=dict(top=1, right=1, bottom=1, left=2), make_difference=True)
        self.assertEqual(m["pixel_count"], 42)
        self.assertEqual(m["rgb_mae_255"], 0)
        self.assertEqual(m["ssim_luma_8x8"], 1)
        self.assertTrue((d[:, :, :3] == 0).all())
        self.assertTrue((d[:, :, 3] == 255).all())

    def test_alpha_is_independent_and_rounding_matches_javascript(self):
        a = np.array([[[255, 0, 0, 128], [255, 255, 255, 0]]], dtype=np.uint8)
        b = np.array([[[128, 0, 0, 255], [0, 0, 0, 0]]], dtype=np.uint8)
        m, _ = compare_arrays(a, b)
        self.assertEqual(m["rgb_mae_255"], 0)
        self.assertEqual(m["alpha_mae_255"], 63.5)
        self.assertEqual(m["alpha_changed_pixels"], 1)
        # Python round(0.5) differs from JS Math.round; 128*128/255 must round up.
        a = np.array([[[128, 0, 0, 128]]], dtype=np.uint8)
        b = np.array([[[64, 0, 0, 255]]], dtype=np.uint8)
        self.assertEqual(compare_arrays(a, b)[0]["rgb_mae_255"], 0)

    def test_invalid_inputs_do_not_get_resized_or_scored(self):
        a = np.zeros((9, 9, 4), dtype=np.uint8)
        for options in [dict(tolerance=-1), dict(tolerance=1.5), dict(gain=0),
                        dict(margins=dict(top=9, right=0, bottom=0, left=0))]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                compare_arrays(a, a, **options)
        with self.assertRaises(ValueError):
            compare_arrays(a, a[:8])

    def test_cli_writes_both_outputs_and_protects_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b = root / "A image.png", root / "B image.png"
            x = np.full((17, 19, 4), [10, 20, 30, 255], dtype=np.uint8)
            y = x.copy()
            y[4, 5, :3] += 2
            Image.fromarray(x).save(a)
            Image.fromarray(y).save(b)
            report, diff = root / "out" / "metrics.json", root / "out" / "diff.png"
            run = subprocess.run([sys.executable, str(SCRIPT), str(a), str(b),
                                  "--top", "1", "--left", "2", "--tolerance", "1",
                                  "--gain", "4", "--json", str(report), "--diff", str(diff)],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            data = json.loads(report.read_text())
            self.assertEqual(data["metrics"]["rgb_changed_pixels"], 1)
            self.assertEqual(data["metrics"]["pixel_count"], 16 * 17)
            self.assertEqual(data["reference"]["name"], "A image.png")
            d = np.array(Image.open(diff))
            self.assertEqual(d.shape, x.shape)
            np.testing.assert_array_equal(d[4, 5], [8, 8, 8, 255])
            self.assertTrue((d[0, :, :3] == 0).all())
            before = a.read_bytes()
            invalid = subprocess.run([sys.executable, str(SCRIPT), str(a), str(b),
                                      "--diff", str(a), "--force"], capture_output=True, text=True)
            self.assertEqual(invalid.returncode, 2)
            self.assertIn("source image", invalid.stderr)
            self.assertEqual(a.read_bytes(), before)
            pure_json = subprocess.run([sys.executable, str(SCRIPT), str(a), str(a), "--json", "-"],
                                       capture_output=True, text=True)
            self.assertEqual(pure_json.returncode, 0, pure_json.stderr)
            identical = json.loads(pure_json.stdout)
            self.assertIsNone(identical["metrics"]["rgb_psnr_db"])
            self.assertTrue(identical["metrics"]["rgb_identical"])

    def test_decoder_rejects_unsupported_and_animated_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            im = Image.new("RGB", (8, 8))
            tiff = root / "unsupported.tif"
            im.save(tiff)
            with self.assertRaisesRegex(ValueError, "PNG, JPEG or WebP"):
                read_image(tiff)
            frames = root / "animated.png"
            im.save(frames, save_all=True, append_images=[Image.new("RGB", (8, 8), "red")], duration=100)
            with self.assertRaisesRegex(ValueError, "animated"):
                read_image(frames)

    def test_decoder_honors_srgb_profile_and_exif_orientation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            im = Image.new("RGBA", (13, 9), (50, 100, 150, 128))
            tagged = root / "tagged.png"
            profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
            im.save(tagged, icc_profile=profile)
            decoded, info = read_image(tagged)
            np.testing.assert_array_equal(decoded, np.array(im))
            self.assertTrue(info["icc_to_srgb"])
            oriented = root / "oriented.png"
            exif = Image.Exif()
            exif[274] = 6
            im.save(oriented, exif=exif)
            decoded, info = read_image(oriented)
            self.assertEqual(decoded.shape, (13, 9, 4))
            self.assertEqual((info["width"], info["height"]), (9, 13))


if __name__ == "__main__":
    unittest.main()
