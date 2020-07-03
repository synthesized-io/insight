from codecs import open
from os import path, listdir
import pathlib

from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

__version__ = ''
exec(open('synthesized/version.py').read())

here = path.abspath(path.dirname(__file__))


def get_git_revision(base_path):
    git_dir = pathlib.Path(base_path) / '.git'
    if not git_dir.exists():
        return ''
    with (git_dir / 'HEAD').open('r') as head:
        ref = head.readline().split(' ')[-1].strip()
    with (git_dir / ref).open('r') as git_hash:
        return git_hash.readline().strip()[:7]


git_revision = get_git_revision(here)

if git_revision:
    __version__ += '.rev' + git_revision

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# get the dependencies and installs
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    install_requires = f.read().split('\n')

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
    if f.replace('/', '.').replace('\\', '.')[:-3] != 'synthesized.config'
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
    install_requires=install_requires,
    ext_modules=cythonize(ext_modules, build_dir="build", language_level=3, compiler_directives={'always_allow_keywords': True}),
    cmdclass={'build_ext': build_ext}
)
