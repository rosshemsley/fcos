import click

from fcos.datasets import CityscapesData
from fcos.training import train

@click.command()
def main():
    dataset = CityscapesData()
    train(dataset)


if __name__ == "__main__":
    main()
