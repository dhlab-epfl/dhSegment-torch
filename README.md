# dhSegment <a href="https://colab.research.google.com/github/dhlab-epfl/dhSegment-torch/blob/master/demo/dhSegment_demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

**dhSegment** is a tool for Historical Document Processing. Its generic approach allows to segment regions and
extract content from different type of documents. 

This respository contains the PyTorch version of **dhSegment** which the one currently in development.

The no longer developed Tensorflow version can still be found at this [address](https://dhsegment.readthedocs.io).

The original version was created by [Benoit Seguin](https://twitter.com/Seguin_Be) and Sofia Ares Oliveira at DHLAB, EPFL.

The complete rewrite in Pytorch was done by Sofia Ares Oliveira and Raphäel Barman.

## Installation

dhSegment will not work properly if the dependencies are not respected. In particular, inaccurate dependencies may result in an inability to converge, even if no error is displayed. Therefore, we highly recommend to create a dedicated environment as following :

```
conda env create --name dhs --file environment.yml
source activate dhs
python setup.py install
```

## Usage

Documentation is in progress and will be available as soon as possible.

### Training the demo model

Demo train script and dataset are provided. This requires only 6 GB GPU RAM and circa 30 min to train. 
```
python3 demo/train_demo.py
```

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
