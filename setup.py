#!/usr/bin/env python
from setuptools import setup, find_packages

setup(name='dh_segment_torch',
      version='0.1.0',
      license='GPL',
      url='https://github.com/dhlab-epfl/dhSegment-torch',
      description='Generic framework for historical document processing',
      packages=find_packages(),
      project_urls={
          'Paper': 'https://arxiv.org/abs/1804.10371',
          'Source Code': 'https://github.com/dhlab-epfl/dhSegment-torch'
      },
      install_requires=[
          'sacred==0.8',
      ],
      extras_require={
          'doc': [
              'sphinx',
              'sphinx-autodoc-typehints',
              'sphinx-rtd-theme',
              'sphinxcontrib-bibtex',
              'sphinxcontrib-websupport'
          ],
      },
      zip_safe=False)
