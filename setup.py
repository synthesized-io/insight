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

packages = find_packages(exclude=['tests*'])

# TODO: compile all modules
# ext_modules = [Extension(p + '.*', [p.replace('.', '/') + '/*.py'], include_dirs=['.']) for p in packages]

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
    ext_modules=cythonize(
        [
            Extension("synthesized.core.basic_synthesizer", ["synthesized/core/basic_synthesizer.py"]),
            Extension("synthesized.core.id_synthesizer", ["synthesized/core/id_synthesizer.py"]),
            Extension("synthesized.testing.linkage_attack", ["synthesized/testing/linkage_attack.py"]),
         ],
        build_dir="build",
    ),
    cmdclass={'build_ext': build_ext}
)

# run this to delete original files
# zip -d dist/synthesized-1.0.0-cp36-cp36m-macosx_10_11_x86_64.whl "synthesized/core/basic_synthesizer.py" "synthesized/core/id_synthesizer.py" "synthesized/testing/linkage_attack.py"