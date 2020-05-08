import click

from .train import train
from .test import test


@click.group()
def fcos():
    pass


fcos.add_command(train)
fcos.add_command(test)


if __name__ == "__main__":
    fcos()
