"""Host-side Windows desktop capture, independent of Blender's GL readback."""
import json
from pathlib import Path
import time


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    # Windows readers may briefly hold a handle without FILE_SHARE_DELETE.
    # Retry publication rather than abandoning a successfully captured image.
    deadline = time.monotonic() + 2
    while True:
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(.01)


def capture_desktop(request, destination):
    """Capture the visible physical-pixel viewport; reject occluded/offscreen views."""
    import ctypes
    from ctypes import wintypes
    from PIL import ImageGrab
    user = ctypes.windll.user32
    user.SetThreadDpiAwarenessContext.argtypes = [ctypes.c_void_p]
    user.SetThreadDpiAwarenessContext.restype = ctypes.c_void_p
    previous = user.SetThreadDpiAwarenessContext(ctypes.c_void_p(-4))
    try:
        callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        user.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        user.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        user.IsWindowVisible.argtypes = [wintypes.HWND]
        user.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]
        user.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
        user.WindowFromPoint.argtypes = [wintypes.POINT]
        user.WindowFromPoint.restype = wintypes.HWND
        user.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
        user.GetAncestor.restype = wintypes.HWND
        user.BringWindowToTop.argtypes = [wintypes.HWND]
        user.SetForegroundWindow.argtypes = [wintypes.HWND]
        windows = []
        def collect(hwnd, _):
            pid = wintypes.DWORD()
            user.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            rect = wintypes.RECT()
            if pid.value == request['pid'] and user.IsWindowVisible(hwnd) and user.GetClientRect(hwnd, ctypes.byref(rect)):
                windows.append(((rect.right * rect.bottom), hwnd, rect.right, rect.bottom))
            return True
        user.EnumWindows(callback_type(collect), 0)
        if not windows:
            raise ValueError('No visible Blender window for desktop capture')
        _, hwnd, width, height = max(windows)
        user.BringWindowToTop(hwnd)
        user.SetForegroundWindow(hwnd)
        time.sleep(.05)  # Let the desktop compositor expose this test window.
        if [width, height] != request['window_px']:
            raise ValueError('Desktop/Blender client pixel sizes differ; check DPI scaling')
        x, y, w, h = request['region']
        if min(x, y) < 0 or w <= 0 or h <= 0 or x+w > width or y+h > height:
            raise ValueError('Viewport bounds outside the Blender client')
        origin = wintypes.POINT(0, 0)
        if not user.ClientToScreen(hwnd, ctypes.byref(origin)):
            raise OSError('Cannot locate Blender client on desktop')
        left, top = origin.x+x, origin.y+height-y-h
        bbox = [left, top, left+w, top+h]
        for px, py in ((left+2, top+2), (left+w-3, top+2), (left+2, top+h-3),
                       (left+w-3, top+h-3), (left+w//2, top+h//2)):
            hit = user.WindowFromPoint(wintypes.POINT(px, py))
            if user.GetAncestor(hit, 2) != hwnd:
                owner = wintypes.DWORD()
                user.GetWindowThreadProcessId(hit, ctypes.byref(owner))
                raise ValueError(f'Viewport obscured/offscreen at {px},{py} by process {owner.value}; keep Blender visible')
        captured_at = time.perf_counter()
        image = ImageGrab.grab(bbox=tuple(bbox), all_screens=True).convert('RGB')
        if image.size != (w, h):
            raise ValueError('Desktop capture pixel dimensions differ from viewport')
        grabbed_at = time.perf_counter()
        image.save(destination)
        return {'captured_at': captured_at, 'capture_completed_at': grabbed_at,
                'completed_at': time.perf_counter(), 'desktop_bbox': bbox,
                'capture_method': 'Windows desktop ImageGrab (GDI)'}
    finally:
        if previous:
            user.SetThreadDpiAwarenessContext(previous)


class DesktopCaptureService:
    def __init__(self, output, pid, capture=capture_desktop):
        self.output, self.pid, self.capture = Path(output), pid, capture
        self.last_id = None

    def poll(self):
        try:
            request = json.loads((self.output / 'capture_request.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return
        if request.get('id') == self.last_id:
            return
        self.last_id = request.get('id')
        response = {'id': self.last_id}
        try:
            if request['pid'] != self.pid:
                raise ValueError('Capture request does not belong to the launched Blender process')
            path = (self.output / request['file']).resolve()
            if not path.is_relative_to(self.output.resolve()):
                raise ValueError('Capture output escapes result directory')
            path.parent.mkdir(parents=True, exist_ok=True)
            response.update(self.capture(request, path))
        except Exception as exc:
            response['error'] = str(exc)
        atomic_json(self.output / 'capture_response.json', response)
