from typing import Optional
import logging
import pathlib

import click

from torch.utils.tensorboard import SummaryWriter
from fcos.datasets import CityscapesData
from fcos.training import train

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DEBUG_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@click.command()
@click.option(
    "--verbose", "-v", type=bool, is_flag=True,
)
@click.option(
    "--debug", type=bool, is_flag=True,
)
@click.option(
    "--log-dir", type=click.Path(file_okay=False, dir_okay=True),
)
@click.option(
    "--cityscapes-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    required=True,
)
def train(
    cityscapes_dir: pathlib.Path, log_dir: Optional[pathlib.Path], verbose: bool, debug: bool,
):
    if debug:
        logging.basicConfig(level=logging.INFO, format=DEBUG_LOG_FORMAT)
    elif verbose:
        logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    else:
        logging.basicConfig(level=logging.WARNING, format=LOG_FORMAT)

    logger = logging.getLogger(__name__)

    with SummaryWriter(log_dir=log_dir) as writer:
        logger.info(f"Logging tensorboard logs to '{writer.log_dir}'")
        train(cityscapes_dir, writer)
