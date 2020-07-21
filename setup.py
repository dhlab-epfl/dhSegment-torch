#!/usr/bin/env python
from setuptools import setup, find_packages

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
        "sacred==0.8",
        "torch==1.5.1",
        "torchvision==0.6.1",
        "tensorboard>=2.0.0",
        "scikit-image>=0.15.0,<0.16.0",
        "scikit-learn>=0.22.1,<0.23.0",
        "pandas>=1.0.3",
        "numpy>=1.18.1,<1.19.0",
        "scipy>=1.3.0,<1.4.0",
        "networkx>=2.4",
        "sacred>=0.8.0",
        "pretrainedmodels>=0.7.4",
        "opencv-python-headless>=4.3.0",
        "PyYaml>=5.3.0",
        "frozendict>=1.2",
        "albumentations>=0.4.5",
        "shapely>=1.7.0",
        "tqdm>=4.41.1",
        "requests>=2.23.0",
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
