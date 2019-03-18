from codecs import open
from os import path

from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

__version__ = ''
exec(open('synthesized/version.py').read())

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# get the dependencies and installs
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    install_requires = f.read().split('\n')

packages = find_packages(exclude=['tests*', 'web'])
ext_modules = [Extension(p + '.*', [p.replace('.', '/') + '/*.py']) for p in packages]

setup(
    name='synthesized',
    version=__version__,
    description='synthesized.io',
    long_description=long_description,
    url='https://github.com/synthesized-io/synthesized',
    author='Synthesized Ltd.',
    author_email='team@synthesized.io',
    license='Proprietary',
    packages=[],
    install_requires=install_requires,
    ext_modules=cythonize(ext_modules, build_dir="build", language_level=3, compiler_directives={'always_allow_keywords': True}),
    cmdclass={'build_ext': build_ext}
)
