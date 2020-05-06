# 🔎  FCOS Pytorch

A pure Pytorch implementation of FCOS, adapted from the paper [https://arxiv.org/pdf/1904.01355.pdf](FCOS: Fully Convolutional One-Stage Object Detection).

⚠️  _Note_ This is still just a toy implementation. You probably shouldn't try and use it for anything yet.


## Local development
This project uses Python 3.8. It is recommended that you use `pyenv` to set up a local version of Python.
You will need to install pyenv and Python 3.8 first.

From the project root, run
```
$ pyenv local 3.8
```

You can now use Poetry to set up the package
```
$ poetry install
```

To create a pure Python wheel, which can be published and installed elsewhere
```
$ poetry build
```

## Training
Once you have built the project, you can use
```
$ poetry run train
```
