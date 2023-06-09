#!/usr/bin/env python
from setuptools import setup, find_packages
import sys

if sys.platform == 'win32':
    jsonnet = "jsonnetbin==0.17.0"
else:
    jsonnet = "jsonnet==0.17.0"

setup(
    name="dh_segment_torch",
    version="0.1.0",
    license="GPL",
    url="https://github.com/dhlab-epfl/dhSegment-torch",
    description="Generic framework for historical document processing",
    packages=find_packages(),
    project_urls={
        "Paper": "https://arxiv.org/abs/1804.10371",
        "Source Code": "https://github.com/dhlab-epfl/dhSegment-torch",
    },
    install_requires=[
        "sacred==0.8.1",
        "torch==1.9.0",
        "torchvision==0.10.1",
        "tensorboard==2.4.1",
        "scikit-image==0.18.1",
        "scikit-learn==0.24.2",
        "pandas==1.3.2",
        "numpy==1.21.0",
        "scipy==1.7.1",
        "networkx==2.5.1",
        "lxml==4.6.3",
        "pretrainedmodels==0.7.4",
        "opencv-python-headless==4.5.3.56",
        "PyYaml==5.4.1",
        "frozendict==2.0.6",
        "albumentations==1.0.3",
        "shapely==1.7.1",
        "tqdm==4.61.2",
        "requests==2.31.0",
        "jsonnet==0.17.0"
    ],
    extras_require={
        "doc": [
            "sphinx",
            "sphinx-autodoc-typehints",
            "sphinx-rtd-theme",
            "sphinxcontrib-bibtex",
            "sphinxcontrib-websupport",
        ],
        "w&b": ["wandb"],
    },
    test_require=["pytest"],
    zip_safe=False,
)
