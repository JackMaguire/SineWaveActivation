from __future__ import absolute_import
from __future__ import print_function

import io
import re
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup

setup(
    name='sinact',
    version='0.0.1',
    description='Use learned sine waves to approximate any activaiton function',
    author='Jack Maguire',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    python_requires='>=3.6.0',
    install_requires=[
        'tensorflow>=2.1.0',
    ],
)
