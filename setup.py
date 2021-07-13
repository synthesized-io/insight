import pathlib
import re
import subprocess
import os
from typing import List, Union

from setuptools import setup
from setuptools.extension import Extension

if os.environ.get('CYTHONIZE', "True") == "True":
    print("Compiling source with Cython.")
    USE_CYTHON = True
    from Cython.Build import cythonize
    from Cython.Distutils.build_ext import new_build_ext
else:
    USE_CYTHON = False

setup_kwargs = {}


def subprocess_command(cmd: List[str]):
    out: Union[str, bytes] = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    if not isinstance(out, str):
        out = out.decode('utf-8')
    ver = out.strip()
    return ver


def get_version_from_git_describe():
    """Determine a version according to git.

    This calls `git describe`, if git is available.

    Returns:
        version from `git describe --tags --always --dirty` if git is
        available; otherwise, None
    """
    ver = subprocess_command(["git", "describe", "--first-parent"])
    match = re.match(r'v(\d)\.(\d)(-rc\d)?(-\d+)?(-g[0-9a-f]+)?', ver)
    if match is None:
        return None

    major, minor, pre, n, git = match.groups()
    if n is None:
        version = f'{major}.{minor}{pre or ""}'
    elif pre is None:
        version = f'{major}.{int(minor)+1}-dev{n[1:]}'
    else:
        ver2 = subprocess_command(['git', 'describe', "--first-parent", "--match", f"v{major}.{int(minor)-1}"])
        match = re.match(r'v(\d)\.(\d)(-\d+)(-g[0-9a-f]+)', ver2)
        major, minor, n, git = match.groups()
        version = f'{major}.{int(minor)+1}-dev{n[1:]}'
    return version


def get_git_revision(base_path):
    git_dir = pathlib.Path(base_path) / '.git'
    if not git_dir.exists():
        return ''
    with (git_dir / 'HEAD').open('r') as head:
        ref = head.readline().split(' ')[-1].strip()
    try:
        with (git_dir / ref).open('r') as git_hash:
            return git_hash.readline().strip()[:7]
    except FileNotFoundError:
        return ref[:7]


def source_files(base_dir):
    result = []
    for f in os.listdir(base_dir):
        p = os.path.join(base_dir, f)
        if os.path.isdir(p):
            result.extend(source_files(p))
        elif p.endswith('.py') and not p.endswith('__init__.py'):
            result.append(p)
    return result


here = os.path.abspath(os.path.dirname(__file__))
__version__ = get_version_from_git_describe()

git_revision = get_git_revision(here)
if git_revision:
    __version__ += '+' + git_revision


if USE_CYTHON:
    modules = [
        Extension(f.replace('/', '.').replace('\\', '.')[:-3], [f]) for f in source_files('synthesized')
        if re.match(r"synthesized(/|\\)config\.py", f) is None  # windows uses '\\' instead of '/'
    ]
    ext_modules = cythonize(
        modules, build_dir="build", language_level=3, compiler_directives={'always_allow_keywords': True}
    )
    setup_kwargs['ext_modules'] = ext_modules
    setup_kwargs['cmdclass'] = {'build_ext': new_build_ext}


setup(version=__version__,
      **setup_kwargs
      )
