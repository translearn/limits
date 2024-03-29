from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
import os
import re
import numpy as np


def get_version():
    with open(os.path.join(os.path.abspath(os.path.dirname(__file__)),
                           'klimits', '__init__.py'), encoding='utf-8') as f:
        init_file = f.read()
        version = re.search(r"__version__\W*=\W*'([^']+)'", init_file)
        return version.group(1) if version is not None else '0.0.0'


with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.md'), encoding='utf-8') as f:
    readme_file = f.read()

if os.name == 'nt':
    os_extra_compile_args = ['-DSIZEOF_VOID_P=8', '-DMS_WIN64']
    os_extra_link_args = ['-static-libgcc', '-static-libstdc++', '-Wl,-Bstatic,--whole-archive', '-lwinpthread',
                          '-Wl,--no-whole-archive']
else:
    os_extra_compile_args = []
    os_extra_link_args = []


ext_mods = [Extension(
    '_klimits', ['klimits/_klimits/_klimits_module.pyx', 'klimits/_klimits/_klimits_code.c'],
    include_dirs=[np.get_include()],
    library_dirs=[],
    libraries=[],
    extra_compile_args=['-O3', '-std=c99', '-fopenmp'] + os_extra_compile_args,
    extra_link_args=['-fopenmp'] + os_extra_link_args
)]
setup(name='klimits',
      version=get_version(),
      packages=['klimits'],
      author='Jonas C. Kiemel',
      author_email='jonas.kiemel@kit.edu',
      url='https://github.com/translearn/limits',
      description='An action space representation for learning robot trajectories without exceeding limits on the '
                  'position, velocity, acceleration and jerk of each robot joint.',
      long_description=readme_file,
      long_description_content_type='text/markdown',
      license='MIT',
      classifiers=["License :: OSI Approved :: MIT License", "Intended Audience :: Developers"],
      install_requires=[
          'numpy',
          'matplotlib',
      ],
      ext_modules=cythonize(ext_mods))
