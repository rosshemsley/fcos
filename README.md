# 🔎  FCOS Pytorch

_A model implementing 2D object detection in images, trained on Cityscapes_

This is a pure Python 3.8 implementation of the fully convolutional one-stage anchor free FCOS algorithm.
It has been slightly adapted from the original paper [FCOS: Fully Convolutional One-Stage Object Detection](https://arxiv.org/pdf/1904.01355.pdf).


⚠️  _Note_ This is still just a toy implementation. You probably shouldn't try and use it for anything yet.


## Installing with pip
You can **pip install this package directly** if your pip is recent enough (20.1 is known to work).

```
$ pip install git+https://github.com/rosshemsley/fcos
```

## Local development
It is recommended that you use `pyenv` to set up a local version of Python. From the project root, run

```
$ pyenv local 3.8
```

Poetry can be used to install the package and dependencies
```
$ poetry install
```

To run the unit tests, you can use
```
$ poetry run pytest tests
```

## Training
Once you have installed the package, you can use the bundled CLI to train and test the network. Training status is logged so that tensorboard can read it.


```
$ poetry run train \
    --cityscapes-dir <path/to/Cityscapes> \
    --verbose
```

To track status using tensorboard, you can run
```
$ poetry run tensorboard --logdir runs
```

Models are written to the same directory as the tensorboard logs for now, the default is at `runs/`.

### Evaluating on the test set
A CLI is provided for testing inference on the Cityscapes test set.

```
$ poetry run test
    --cityscapes-dir <path/to/Cityscapes> \
    --model-checkpoint <path/to/checkpoint.chkpt> \
    --output ./output 
```

This will write predicted bounding boxes to all of the images in the test set into the output directory.
