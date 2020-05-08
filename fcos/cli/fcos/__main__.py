import click

from .train import train
from .test import test


@click.group()
def fcos():
    """
    A CLI to train and test the fcos 2D object detection model, taken from the paper
    'FCOS: Fully Convolutional One-Stage Object Detection'.
    """
    pass


fcos.add_command(train)
fcos.add_command(test)


if __name__ == "__main__":
    fcos()
