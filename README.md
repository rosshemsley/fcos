# 🔎  FCOS Pytorch

_A model implementing 2D object detection in images, trained on Cityscapes_

This is a pure python implementation of the fully convolutional one-stage anchor free FCOS algorithm.
It has been slightly adapted from the original paper [FCOS: Fully Convolutional One-Stage Object Detection](https://arxiv.org/pdf/1904.01355.pdf).

⚠️  _Note_ This is still just a toy implementation. You probably shouldn't try and use it for anything yet.

## Local development
This project uses python 3.8. It is recommended that you use `pyenv` to set up a local version of Python.
You will need to install pyenv and python 3.8 first.

From the project root, run
```
$ pyenv local 3.8
```

You can now use poetry to set up the package. You will need to install poetry first.
```
$ poetry install
```

To create a pure python wheel, which can be published and installed elsewhere
```
$ poetry build
```

## Training
Once you have built the project, you can use the following to train the network
```
$ poetry run train
```
