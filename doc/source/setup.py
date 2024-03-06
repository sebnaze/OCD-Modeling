#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='OCD_modeling',
      version='1.0',
      description='Modeling frontostriatal mechanisms and simulating intervention targets restoring \
      ventro-dorsal dynamics in obsessive-compulsive disorder.',
      author='Sebastien Naze',
      author_email='sebastien.naze@gmail.com',
      url='https://github.com/sebnaze/OCD-Modeling',
      packages=find_packages(),
      install_requires=['dill',
                        'neurolib',
                        'nilearn', 
                        'numpy', 
                        'pandas',
                        'pingouin',
                        'pickle',
                        'pyabc',
                        'PyDSTool',
                        'pymeshfix',
                        'pyvista',
                        'scipy', 
                        'seaborn',
                        'sklearn',
                        'sympy'
                        ]
     )
