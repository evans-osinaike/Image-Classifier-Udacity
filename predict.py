import argparse
from pathlib import Path
from PIL import Image
import torch

from cli_utils import json_file_type
from model_utils import rebuild_model_from_checkpoint
from data_utils import process_image
from device_utils import get_device

# Note to run: python predict.py (flower file path) (checkpoint)

def get_cli_arguments():
    """
    Retrieves and parses the command line arguments provided by the user when
    they run the program from a terminal window. If the user fails to provide
    some or all arguments, then the default values are used for the missing
    arguments.
    Returns:
        parse_args (Dict): data structure that stores the command line
        arguments object
    """

    parser = argparse.ArgumentParser(
        description="A script that takes an image and a checkpoint of a model, \
        then returns the top k most likely classes along with the \
        probabilities",
        epilog="\
        Examples:\n\
        * Predict flower name from an image:\n\
        \tpython predict.py ./image_06743.jpg ./checkpoint.pth\n\n\
        * Return top 3 most likely classes:\n\
        \tpython predict.py ./image_06743.jpg ./checkpoint.pth --top_k 3\n\n\
        * Use a mapping of categories to real names:\n\
        \tpython predict.py ./image_06743.jpg ./checkpoint.pth \
        --category_names cat_to_name.json \n\n\
        * Use GPU for inference:\n\
        \tpython predict.py ./image_06743.jpg --gpu\
        ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "input",
        type=Path,
        help="path to the image to classify",
    )

    parser.add_argument(
        "checkpoint",
        type=Path,
        help="path to the checkpoint of the model to use for inference",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="the top k most probable classes",
    )

    parser.add_argument(
        "--category_names",
        type=json_file_type,
        default= 'cat_to_name.json',
        help="the path to the file containing the mapping of categories to \
        real names",
    )

    parser.add_argument(
        "--gpu",
        default= True,
        action="store_true",
        help="Use the gpu for inference",
    )

    return parser.parse_args()


def predict(model, image, top_k, device):
    """
    Predict the class (or classes) of an image using a trained deep learning
    model.
    Args:
        model (nn.Module): a convolutional neural network.
        image (torch.FloatTensor): a tensor representing the image.
        top_k (int): the number of top most likely classes to show.
        device (torch.device): the device to use for inference.
    return:
        top_p (List): the probabilities of of the top k classes.
        top_idx (List): the list of top classes indices.
    """
    model = model.to(device)
    image = image.to(device)

    model.eval()
    with torch.no_grad():
        logps = model.forward(image)
        ps = torch.exp(logps)
        top_p, top_idx = ps.topk(top_k)

        return top_p, top_idx


def idx_to_flower_name(top_idx, class_to_idx, cat_to_name):
    """
    Map a list of predicted indices to their corresponding class names.
    Args:
        top_idx (List): a list of predicted classes indices.
        class_to_idx (Dict): a dictionary containing the mapping of each
        class number to it's corresponding indice in the dataset.
        cat_to_name (Dict): a dictionary containing the mapping of each
        class number to it's corresponding label.
    return:
        top_class (List): the list of top classes labels.
    """
    # Dict for reverse lookup
    idx_to_cat = {v: k for k, v in class_to_idx.items()}
    top_cat = [idx_to_cat[idx] for idx in top_idx]
    top_class = [cat_to_name[cat] for cat in top_cat]
    return top_class


def main():
    cli_args = get_cli_arguments()
    model = rebuild_model_from_checkpoint(cli_args.checkpoint)
    class_to_idx = model.class_to_idx
    device = get_device(cli_args.gpu)

    with Image.open(cli_args.input) as im:
        image = process_image(train=False)(im)
        image = image.unsqueeze(0)
        probs, classes = predict(model, image, cli_args.top_k, device)
        probs, classes = probs[0].tolist(), classes[0].tolist()

        if cli_args.category_names:
            classes = idx_to_flower_name(
                classes, class_to_idx, cli_args.category_names
            )

        print("-" * 43)
        print(f"| {'class name':25s} | {'probability':10} |")
        print("-" * 43)
        for label, prob in zip(classes, probs):
            print(f"| {str(label):25s} | {prob:11.2%} |")
        print("-" * 43)


if __name__ == "__main__":
    main()