#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='adelson_motion_model',
    version='0.1',
    description='Adelson Motion Model',
    author='Josue Ortega Caro',
    author_email='josueortc@gmail.com',
    packages=find_packages(exclude=[]),
    install_requires=['numpy', 'tqdm', 'torch'],
)
