import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from viewport_capture import DesktopCaptureService, atomic_json


class DesktopCaptureTests(unittest.TestCase):
    def test_atomic_publish_retries_a_windows_reader_sharing_violation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'response.json'
            replace = Path.replace
            attempts = []
            def briefly_locked(source, target):
                attempts.append(1)
                if len(attempts) == 1:
                    raise PermissionError('reader holds handle without delete sharing')
                return replace(source, target)
            with patch.object(Path, 'replace', briefly_locked):
                atomic_json(path, {'id': 123})
            self.assertEqual(json.loads(path.read_text()), {'id': 123})
            self.assertEqual(len(attempts), 2)

    def test_capture_is_serviced_once_and_preserves_external_timestamps(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            calls = []
            def capture(request, path):
                calls.append(path)
                return {'captured_at': 123.5, 'completed_at': 123.6}
            atomic_json(output / 'capture_request.json', {'id': 42, 'pid': 7, 'file': 'quality/0.png'})
            service = DesktopCaptureService(output, 7, capture)
            service.poll()
            service.poll()
            response = json.loads((output / 'capture_response.json').read_text())
            self.assertEqual(response['captured_at'], 123.5)
            self.assertEqual(len(calls), 1)

    def test_wrong_process_and_path_escape_are_rejected_before_capture(self):
        for pid, file in ((8, '0.png'), (7, '../escape.png')):
            with self.subTest(pid=pid, file=file), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)
                atomic_json(output / 'capture_request.json', {'id': 42, 'pid': pid, 'file': file})
                def forbidden(*_):
                    self.fail('Invalid request reached capture')
                DesktopCaptureService(output, 7, forbidden).poll()
                self.assertIn('error', json.loads((output / 'capture_response.json').read_text()))

    def test_capture_failure_is_returned_to_blender(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            atomic_json(output / 'capture_request.json', {'id': 42, 'pid': 7, 'file': '0.png'})
            def obscured(*_):
                raise ValueError('Viewport obscured')
            DesktopCaptureService(output, 7, obscured).poll()
            self.assertEqual(json.loads((output / 'capture_response.json').read_text())['error'], 'Viewport obscured')
