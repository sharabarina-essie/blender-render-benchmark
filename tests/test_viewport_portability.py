import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import run_viewport_matrix as matrix
import run_viewport_performance as runner
from run_viewport_performance import assess
from setup_viewport_test import extract_archive, prepare_scene, safe_member, verify_scene


class ViewportPortabilityTests(unittest.TestCase):
    def test_short_orbits_are_rejected_by_both_runners(self):
        for module in (matrix, runner):
            with self.subTest(module=module.__name__), patch('sys.argv', ['runner', '--duration', '5']), \
                    patch('sys.stderr', io.StringIO()), self.assertRaises(SystemExit) as raised:
                module.main()
            self.assertEqual(raised.exception.code, 2)

    def test_requested_nvidia_gpu_rejects_integrated_opengl(self):
        report = {'complete': True, 'graphics_backend': 'OPENGL',
                  'renderer': 'AMD Radeon integrated', 'results': [
                      {'case': 'solid', 'run': 1, 'status': 'PASS',
                       'viewport_draw_fps': 30, 'draws': 30, 'viewport_px': [1920,1080]}]}
        errors = assess(report, ['solid'], 1, 0, gpu_filter='RTX 600')
        self.assertTrue(any('GPU filter' in e for e in errors))
        report['renderer'] = 'NVIDIA RTX 600 Ada Generation'
        self.assertFalse(assess(report, ['solid'], 1, 0, gpu_filter='RTX 600'))

    def test_empty_and_nested_blender_installations(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(matrix.installed_builds(root/'missing'), [])
            nested = root/'4.5.14'/'blender-4.5.14-windows-x64'
            nested.mkdir(parents=True)
            (nested/'blender.exe').touch()
            direct = root/'4.5.9'
            direct.mkdir()
            (direct/'blender.exe').touch()
            self.assertEqual([p.name for p in matrix.installed_builds(root)], ['4.5.9','4.5.14'])

    def test_archive_paths_cannot_escape_staging_on_windows(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('../escape', '/escape', 'C:/escape', '..\\escape', 'folder\\escape'):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    safe_member(root, name)
            self.assertEqual(safe_member(root, 'folder/scene.blend'), root/'folder'/'scene.blend')

    def test_malicious_archive_does_not_write_outside_staging(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root/'archive.zip'
            with zipfile.ZipFile(archive, 'w') as source:
                source.writestr('../outside.blend', b'bad')
            with self.assertRaises(ValueError):
                extract_archive(archive, root/'staging')
            self.assertFalse((root/'outside.blend').exists())

    def test_manual_scene_import_checks_hash_and_preserves_source(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary)
            source=root/'download.blend'
            source.write_bytes(b'original scene fixture')
            scene={'path':'models/medieval_fisher_hut.blend',
                   'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
            prepare_scene(root, 'full', scene, source)
            self.assertEqual((root/scene['path']).read_bytes(), source.read_bytes())
            with self.assertRaises(ValueError):
                prepare_scene(root, 'full', {**scene,'sha256':'wrong'}, source)
            self.assertEqual(source.read_bytes(), b'original scene fixture')

    def test_missing_manual_scene_never_attempts_a_download(self):
        with TemporaryDirectory() as temporary, patch('setup_viewport_test.download') as download:
            scene={'path':'models/medieval_fisher_hut.blend','source_page':'https://blendswap.com/blend/30099'}
            with self.assertRaisesRegex(ValueError,'one-time copy'):
                prepare_scene(Path(temporary), 'full', scene)
            download.assert_not_called()

    def test_missing_linked_classroom_assets_are_detected(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary)
            scene=root/'main.blend'
            scene.write_bytes(b'classroom')
            with self.assertRaisesRegex(ValueError,'linked Classroom'):
                verify_scene(root, {'path':'main.blend',
                    'sha256':hashlib.sha256(scene.read_bytes()).hexdigest(),
                    'required_links':['assets/chairs/chairs.blend']})

    def test_matrix_launches_only_runner_and_forwards_quality(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary)
            build=root/'blender'/'5.2.1'
            build.mkdir(parents=True)
            (build/'blender.exe').touch()
            scene=root/'classroom.blend'
            scene.touch()
            output=root/'matrix'
            config={'_base_dir':str(root),'blender':{'builds_dir':'blender'},
                    'models':{'archviz_classroom':str(scene)}}
            commands=[]
            def launch(command, **kwargs):
                commands.append(command)
                target=Path(command[command.index('--output')+1])
                target.mkdir(parents=True)
                report={'renderer':'NVIDIA RTX 600 Ada Generation','graphics_backend':'OPENGL',
                        'results':[{'case':'solid','run':1,'status':'PASS','viewport_draw_fps':30}]}
                (target/'summary.json').write_text(json.dumps({'status':'PASS','report':report}))
                return SimpleNamespace(returncode=0,poll=lambda:0)
            args=['run_viewport_matrix.py','--smoke','--cases','solid','--quality-capture',
                  '--gpu','RTX 600','--output',str(output)]
            with patch('sys.argv',args), patch.object(matrix,'ROOT',root), \
                 patch.object(matrix,'load_config',return_value=config), \
                 patch.object(matrix,'digest',return_value='fixture-sha'), \
                 patch.object(matrix.subprocess,'run') as external_commands, \
                 patch.object(matrix.subprocess,'Popen',side_effect=launch):
                self.assertEqual(matrix.main(),0)
                external_commands.assert_not_called()
            self.assertIn('--quality-capture',commands[0])
            self.assertEqual(commands[0][commands[0].index('--model')+1],str(scene))
            saved=json.loads((output/'matrix.json').read_text())
            self.assertEqual(saved['runs'],1)
            self.assertEqual(saved['duration_s'],30)
            self.assertIn('--capture-method',commands[0])
            self.assertEqual(saved['label'],'unlabeled')


if __name__ == '__main__':
    unittest.main()
