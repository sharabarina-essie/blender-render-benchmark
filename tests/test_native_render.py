import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from native_render_support import select_gpu_ids, gpu_errors, png_dimensions, final_result_errors

ROOT = Path(__file__).resolve().parents[1]


class NativeRenderTests(unittest.TestCase):
    def devices(self):
        return [dict(name='RTX 6000 Ada', id='cuda-b', type='CUDA', use=True),
                dict(name='CPU', id='cpu', type='CPU', use=True),
                dict(name='RTX 6000 Ada', id='cuda-a', type='CUDA', use=True),
                dict(name='RTX 6000 Ada', id='optix-a', type='OPTIX', use=True)]

    def state(self):
        return dict(engine='CYCLES', scene_device='GPU', backend='CUDA', graphics_backend='OPENGL',
                    devices=[dict(name='RTX 6000 Ada', id='cuda-a', type='CUDA', use=True)],
                    active_gpu=True, denoising=True, denoiser='OPENIMAGEDENOISE',
                    denoising_use_gpu=True, oidn_gpu_supported=True)

    def test_two_gpu_host_defaults_to_one_sorted_device(self):
        self.assertEqual(select_gpu_ids(self.devices(), 'CUDA', 'RTX 6000'), ['cuda-a'])
        self.assertEqual(select_gpu_ids(self.devices(), 'CUDA', 'RTX 6000', 1), ['cuda-b'])
        self.assertEqual(select_gpu_ids(self.devices(), 'CUDA', 'RTX 6000', all_matching=True), ['cuda-a', 'cuda-b'])
        for backend, index in [('CUDA', 2), ('CUDA', -1), ('HIP', 0)]:
            with self.assertRaises(ValueError): select_gpu_ids(self.devices(), backend, 'RTX 6000', index)

    def test_audit_rejects_cpu_second_gpu_wrong_api_and_oidn_fallback(self):
        good = self.state()
        self.assertEqual(gpu_errors(good, 'cycles-cuda-oidn-gpu', 'RTX 6000', ['cuda-a']), [])
        for field, value in [('active_gpu', False), ('oidn_gpu_supported', False), ('backend', 'OPTIX'),
                             ('scene_device', 'CPU'), ('denoising_use_gpu', False), ('graphics_backend', 'VULKAN')]:
            self.assertTrue(gpu_errors(dict(good, **{field: value}), 'cycles-cuda-oidn-gpu', 'RTX 6000', ['cuda-a']))
        for extra in [dict(name='CPU', id='cpu', type='CPU', use=True),
                      dict(name='RTX 6000 Ada', id='cuda-b', type='CUDA', use=True)]:
            state = dict(good, devices=good['devices'] + [extra])
            self.assertTrue(gpu_errors(state, 'cycles-cuda-oidn-gpu', 'RTX 6000', ['cuda-a']))

    def test_eevee_audit_requires_matching_opengl_adapter(self):
        good = dict(engine='BLENDER_EEVEE_NEXT', graphics_backend='OPENGL', renderer='NVIDIA RTX 6000 Ada/PCIe/SSE2')
        self.assertEqual(gpu_errors(good, 'eevee', 'RTX 6000'), [])
        self.assertTrue(gpu_errors(dict(good, renderer='Integrated Graphics'), 'eevee', 'RTX 6000'))
        with self.assertRaises(ValueError): gpu_errors(good, 'unknown', 'RTX 6000')

    def test_saved_png_header_not_render_result_rna_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'render.png'
            path.write_bytes(b'\x89PNG\r\n\x1a\n' + struct.pack('>I', 13) + b'IHDR' + struct.pack('>II', 480, 270))
            self.assertEqual(png_dimensions(path), [480, 270])
            path.write_bytes(b'not a PNG')
            with self.assertRaises(ValueError): png_dimensions(path)

    def test_stale_image_or_crash_cannot_be_a_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'old.png'; path.write_bytes(b'old')
            row = dict(image=str(path), render_complete=False, source_unchanged=True, saved_image_dimensions=[480,270])
            self.assertIn('No completed GUI render job', final_result_errors(row, 0, [480,270]))
            row['render_complete'] = True
            self.assertTrue(final_result_errors(row, -1073740791, [480,270]))
            row['source_unchanged'] = False
            self.assertTrue(final_result_errors(row, 0, [480,270]))

    def test_clean_checkout_cli_dry_run_arbitrary_scene_path(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'scene with spaces.blend'; source.write_bytes(b'fixture')
            output = Path(directory) / 'output'
            run = subprocess.run([sys.executable, str(ROOT/'run_native_render_suite.py'), '--model', str(source),
                                  '--blender', sys.executable, '--output', str(output), '--dry-run'],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            plan = json.loads(run.stdout)
            self.assertEqual(len(plan['jobs']), 3)
            self.assertEqual({j['case'] for j in plan['jobs']}, {'cycles-cuda-oidn-gpu','cycles-optix-optix','eevee'})
            self.assertTrue(all(str(source) in j['command'] for j in plan['jobs']))
            self.assertTrue(all('--background' not in j['command'] and '-b' not in j['command'] for j in plan['jobs']))
            self.assertTrue(all('0' == j['command'][j['command'].index('--gpu-index')+1] for j in plan['jobs']))
            self.assertFalse(output.exists())
            bad = subprocess.run([sys.executable, str(ROOT/'run_native_render_suite.py'), '--model', str(source),
                                  '--blender', sys.executable, '--cases', 'cpu', '--dry-run'], capture_output=True, text=True)
            self.assertNotEqual(bad.returncode, 0)

    def test_process_failure_is_reported_without_worker_or_image(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'fixture.blend'; source.write_bytes(b'fixture')
            output = Path(directory) / 'results'
            # Python rejects Blender flags immediately: exercise the real host failure path without a GPU job.
            run = subprocess.run([sys.executable, str(ROOT/'run_native_render_suite.py'), '--model', str(source),
                                  '--blender', sys.executable, '--cases', 'eevee', '--output', str(output)],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 1, run.stderr)
            summary = json.loads((output/'summary.json').read_text())
            self.assertTrue(summary['complete'])
            self.assertNotEqual(summary['rows'][0]['status'], 'PASS')
            self.assertTrue(summary['rows'][0]['host_errors'])
            self.assertTrue(summary['rows'][0]['source_unchanged_host'])
            report = (output/'report.html').read_text(encoding='utf-8')
            self.assertIn('VALIDATION OR PROCESS FAILED', report)


if __name__ == '__main__': unittest.main()
