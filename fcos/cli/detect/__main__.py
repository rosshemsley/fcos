import click
import torch

from fcos.model import FCOS
from fcos.inference import compute_detections


@click.command()
@click.argument("image", required=True, help="path to images", type=click.Path(exists=True))
@click.argument("model", required=True, help="path to images", type=click.Path(exists=True))
def main(image):

    if torch.cuda.is_available():
        print("using cuda")
        device = torch.device("cuda")
    else:
        print("using cpu")
        device = torch.device("cpu")

    model = FCOS()
    model.to(device)


    model = FCOS()
    result = compute_detections(model, img)

    print(result)


if __name__ == "__main__":
    main()
