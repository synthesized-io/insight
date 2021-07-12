import pathlib
import re
import subprocess
from codecs import open
from os import listdir, path
from typing import List, Union

from Cython.Build import cythonize
from Cython.Distutils import build_ext
from setuptools import find_packages, setup
from setuptools.extension import Extension


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


here = path.abspath(path.dirname(__file__))
__version__ = get_version_from_git_describe()

git_revision = get_git_revision(here)
if git_revision:
    __version__ += '+' + git_revision

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# define the package dependencies
install_requires = [
    "dataclasses ~= 0.6;python_version<'3.7'",
    "numpy >= 1.18.4, < 1.20.0",
    "tensorflow ~= 2.4",
    "tensorflow_probability ~= 0.12",
    "tensorflow_privacy ~= 0.5",
    "scipy ~= 1.5",
    "scikit_learn ~= 0.23",
    "pandas ~= 1.1",
    "matplotlib ~= 3.3",
    "seaborn ~= 0.11",
    "faker ~= 5.0",
    "simplejson ~= 3.17",
    "pyyaml ~= 5.3",
    "rstr ~= 2.2",
    "rsa ~= 4.7"
]

packages = find_packages(exclude=['tests*', 'web*'])


def source_files(base_dir):
    result = []
    for f in listdir(base_dir):
        p = path.join(base_dir, f)
        if path.isdir(p):
            result.extend(source_files(p))
        elif p.endswith('.py') and not p.endswith('__init__.py'):
            result.append(p)
    return result


ext_modules = [
    Extension(f.replace('/', '.').replace('\\', '.')[:-3], [f]) for f in source_files('synthesized')
    if re.match(r"synthesized(/|\\)config\.py", f) is None  # windows uses '\\' instead of '/'
]

setup(
    name='synthesized',
    version=__version__,
    description='synthesized.io',
    long_description=long_description,
    url='https://github.com/synthesized-io/synthesized',
    author='Synthesized Ltd.',
    author_email='team@synthesized.io',
    license='Proprietary',
    packages=packages,
    package_data={"synthesized": [".pubkey", "fonts/inter-v3-latin-regular.ttf"]},
    install_requires=install_requires,
    python_requires='>=3.6,<3.9',
    ext_modules=cythonize(
        ext_modules, build_dir="build", language_level=3, compiler_directives={'always_allow_keywords': True}
    ),
    cmdclass={'build_ext': build_ext}
)
