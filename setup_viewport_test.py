"""Prepare only the portable Blender builds and scenes needed for viewport tests.

Python 3.10+, standard library only. Existing scenes must match pinned hashes.
Large binaries stay in gitignored blender/ and models/ folders.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import subprocess
import tarfile
import tempfile
from urllib.request import urlopen
import zipfile

ROOT = Path(__file__).resolve().parent
ALL_BUILDS = ('3.3.0', '3.3.9', '3.6.0', '3.6.9', '4.2.0', '4.2.9', '4.2.23',
              '4.5.0', '4.5.9', '4.5.14', '5.2.0', '5.2.1', '5.2.2')


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def safe_member(destination, name):
    """Reject archive traversal and Windows absolute paths before extraction."""
    pure = PurePosixPath(name)
    if pure.is_absolute() or '..' in pure.parts or chr(92) in name or ':' in name:
        raise ValueError(f'Unsafe archive member: {name}')
    path = destination.joinpath(*pure.parts).resolve()
    if not path.is_relative_to(destination.resolve()):
        raise ValueError(f'Archive member escaped destination: {name}')
    return path


def extract_archive(archive, destination):
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                target = safe_member(destination, member.filename)
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('Symbolic links are not supported in setup archives')
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source.open(member) as inp, target.open('wb') as out:
                        shutil.copyfileobj(inp, out)
    else:
        with tarfile.open(archive, 'r:*') as source:
            for member in source:
                target = safe_member(destination, member.name)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source.extractfile(member) as inp, target.open('wb') as out:
                        shutil.copyfileobj(inp, out)
                else:
                    raise ValueError('Links and special files are not supported in setup archives')


def download(url, destination):
    print('Downloading ' + url, flush=True)
    # Windows curl uses the system certificate/proxy configuration and matches
    # the downloader already used by bootstrap.ps1.
    if sys.platform == 'win32' and shutil.which('curl.exe'):
        subprocess.run(['curl.exe', '--location', '--fail', '--retry', '2',
                        '--silent', '--show-error', '--connect-timeout', '30',
                        '--output', str(destination), url], check=True)
    else:
        with urlopen(url, timeout=60) as response, destination.open('wb') as stream:
            shutil.copyfileobj(response, stream, length=1024 * 1024)


def installed_executable(directory):
    direct = directory / 'blender.exe'
    if direct.is_file():
        return direct
    matches = list(directory.glob('*/blender.exe'))
    return matches[0] if len(matches) == 1 else None


def prepare_build(root, version, verify_only=False):
    if not re.fullmatch(r'\d+\.\d+\.\d+', version):
        raise ValueError(f'Expected an exact Blender x.y.z build, got {version!r}')
    builds_root = (root / 'blender').resolve()
    destination = builds_root / version
    exe = installed_executable(destination)
    if exe:
        print('Blender ready: ' + str(exe), flush=True)
        return
    if verify_only:
        raise ValueError(f'Blender {version} is missing')
    if destination.exists():
        raise ValueError(f'{destination} exists without a unique blender.exe; inspect it before retrying')
    builds_root.mkdir(parents=True, exist_ok=True)
    line = '.'.join(version.split('.')[:2])
    url = f'https://download.blender.org/release/Blender{line}/blender-{version}-windows-x64.zip'
    with tempfile.TemporaryDirectory(prefix='viewport-setup-', dir=builds_root) as temporary:
        staging = Path(temporary).resolve()
        assert staging.is_relative_to(builds_root)
        archive = staging / 'blender.zip'
        download(url, archive)
        unpacked = staging / 'unpacked'
        extract_archive(archive, unpacked)
        matches = list(unpacked.glob('*/blender.exe'))
        if len(matches) != 1:
            raise ValueError('Official archive did not contain one portable Blender installation')
        assert destination.resolve().is_relative_to(builds_root)
        matches[0].parent.rename(destination)
    print('Installed Blender ' + version, flush=True)


def verify_scene(root, scene):
    path = root / scene['path']
    if not path.is_file():
        raise ValueError(f'Missing {path}')
    if file_hash(path) != scene['sha256']:
        raise ValueError(f'Scene SHA-256 mismatch: {path}. Keep the original file; do not resave it.')
    for relative in scene.get('required_links', []):
        if not (path.parent / relative).is_file():
            raise ValueError(f'Missing linked Classroom asset: {relative}; copy/extract the complete scene directory')
    print('Scene verified: ' + str(path), flush=True)


def prepare_scene(root, name, scene, fisher_source=None, verify_only=False):
    destination = root / scene['path']
    if destination.is_file() or verify_only:
        verify_scene(root, scene)
        return
    models_root = (root / 'models').resolve()
    models_root.mkdir(parents=True, exist_ok=True)
    if name == 'full':
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not fisher_source:
            raise ValueError('Medieval Fisher Hut requires a one-time copy of the original 1 GB file. '
                'Use --fisher-hut-source PATH, or download it after sign-in from ' + scene['source_page'])
        if not fisher_source.is_file() or file_hash(fisher_source) != scene['sha256']:
            raise ValueError('The supplied Medieval Fisher Hut file does not match the baseline SHA-256')
        with tempfile.TemporaryDirectory(prefix='viewport-setup-', dir=models_root) as temporary:
            staging = Path(temporary).resolve()
            assert staging.is_relative_to(models_root)
            part = staging / 'hut.blend'
            shutil.copyfile(fisher_source, part)
            verify_scene(staging, {**scene, 'path': 'hut.blend'})
            part.rename(destination)
    else:
        target_folder = root / 'models/archviz/classroom'
        if target_folder.exists():
            raise ValueError(f'{target_folder} is incomplete; inspect or rename it before retrying')
        with tempfile.TemporaryDirectory(prefix='viewport-setup-', dir=models_root) as temporary:
            staging = Path(temporary).resolve()
            assert staging.is_relative_to(models_root)
            archive = staging / 'classroom.tar.bz2'
            download(scene['url'], archive)
            unpacked = staging / 'unpacked'
            extract_archive(archive, unpacked)
            relative = destination.relative_to(target_folder)
            verify_scene(unpacked, {**scene, 'path': relative.as_posix()})
            target_folder.parent.mkdir(parents=True, exist_ok=True)
            assert target_folder.resolve().is_relative_to(models_root)
            unpacked.rename(target_folder)
    verify_scene(root, scene)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--builds', default='', help='Comma-separated exact builds, or all; omitted skips Blender downloads')
    parser.add_argument('--scene', choices=['smoke', 'full', 'all'], default='all')
    parser.add_argument('--fisher-hut-source', type=Path, help='Existing original .blend to copy into models/')
    parser.add_argument('--verify-only', action='store_true', help='Check existing files without network or writes')
    args = parser.parse_args()
    builds = ALL_BUILDS if args.builds == 'all' else list(dict.fromkeys(filter(None, args.builds.split(','))))
    scenes = json.loads((ROOT / 'viewport_assets.json').read_text(encoding='utf-8'))['scenes']
    try:
        for build in builds:
            prepare_build(ROOT, build, args.verify_only)
        for name in (['smoke', 'full'] if args.scene == 'all' else [args.scene]):
            prepare_scene(ROOT, name, scenes[name], args.fisher_hut_source, args.verify_only)
    except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile, subprocess.CalledProcessError) as exc:
        print('SETUP FAILED: ' + str(exc), file=sys.stderr)
        return 1
    print('Viewport test setup ready. Source .blend files were not modified.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
