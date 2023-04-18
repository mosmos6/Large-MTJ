#!/usr/bin/env python3

from setuptools import setup, find_packages
import versioneer

setup(
    name='mesh_transformer',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(include=['mesh_transformer', 'mesh_transformer.*'])
)
