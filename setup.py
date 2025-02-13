#!/opt/conda/bin/python
from setuptools import find_packages
from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=find_packages()
)
print("Executing map_tr setup.py")
setup(**setup_args)
