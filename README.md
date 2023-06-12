[![PyPI - Python Version](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/downloads/release/python-31011/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-green.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![open issues](https://isitmaintained.com/badge/open/dhlab-epfl/dhSegment-torch.svg)](https://github.com/dhlab-epfl/dhSegment-torch/issues)

<a href="https://colab.research.google.com/github/dhlab-epfl/dhSegment-torch/blob/master/demo/dhSegment_demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# dhSegment

**dhSegment** is a tool for Historical Document Processing. Its generic approach allows to segment regions and
extract content from different type of documents. 

This respository contains the PyTorch version of **dhSegment** which the one currently in development.

The no longer developed Tensorflow version can still be found at this [address](https://dhsegment.readthedocs.io).

The original version was created by [Benoit Seguin](https://twitter.com/Seguin_Be) and Sofia Ares Oliveira at DHLAB, EPFL.

The complete porting to Pytorch was done by Sofia Ares Oliveira, Raphäel Barman, and Rémi Petitpierre.

## Installation

dhSegment will not work properly if the dependencies are not respected. In particular, inaccurate dependencies may result in an inability to converge, even if no error is displayed. Therefore, we highly recommend to create a dedicated environment as following :

```
conda env create --name dhs --file environment.yml
source activate dhs
python setup.py install
```

## Usage

### Training the demo model locally

Demo train script and dataset are provided. This requires only 6 GB GPU RAM and circa 20 min to train. 
```
python3 demo/train_demo.py
```

### Running the demo notebook on Colab

The compatibility with Google Colab notebooks is guaranteed, as of June 2023.

## Citation
If you are using this code for your research, you can cite the corresponding paper as :
```
@inproceedings{oliveiraseguinkaplan2018dhsegment,
  title={dhSegment: A generic deep-learning approach for document segmentation},
  author={Ares Oliveira, Sofia and Seguin, Benoit and Kaplan, Frederic},
  booktitle={Frontiers in Handwriting Recognition (ICFHR), 2018 16th International Conference on},
  pages={7--12},
  year={2018},
  organization={IEEE}
}
```
