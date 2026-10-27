"""Temporary Windows display/system execution request for a full-GUI test run."""
import argparse
import ctypes
from contextlib import contextmanager
from pathlib import Path
import sys
import time


@contextmanager
def awake_request():
    if sys.platform != 'win32':
        yield
        return
    set_state = ctypes.windll.kernel32.SetThreadExecutionState
    set_state.argtypes = [ctypes.c_uint]
    set_state.restype = ctypes.c_uint
    if not set_state(0x80000003):
        raise OSError('Cannot request an awake system and display')
    print('Temporary system/display awake request active', flush=True)
    try:
        yield
    finally:
        set_state(0x80000000)
        print('Temporary awake request released', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stop-file', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=7200)
    args = parser.parse_args()
    with awake_request():
        end = time.monotonic() + args.seconds
        while time.monotonic() < end and not args.stop_file.exists():
            time.sleep(1)


if __name__ == '__main__':
    main()
