from os.path import join, dirname, realpath
from setuptools import setup
import sys

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, \
    "The Spinning Up repo is designed to work with Python 3.6 and greater." \
    + "Please install it before proceeding."

with open(join("iape", "version.py")) as version_file:
    exec(version_file.read())

setup(
    name='spinup',
    py_modules=['spinup'],
    version=0.0,#'0.1',
    description="Minimal implementation of Instance Agnostic Policy Ensembles (IAPE) based on spinning up code.",
    author="Martin Bertran",
    install_requires=[
            'cloudpickle>=1.2.1',
            # 'gym[atari,box2d,classic_control]~=0.15.3',
            'gym>=0.15.3',
            'ipython',
            'joblib',
            'matplotlib>=3.1.1',
            'mpi4py',
            'numpy',
            'pandas',
            'pytest',
            'psutil',
            'scipy',
            'seaborn>=0.8.1',
            'torch>=1.4',
            'tqdm'
        ],
)
