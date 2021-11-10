import argparse
from pathlib import Path
import json
import torch

# data_dir = 'flowers'
def datasets_folder_type(data_dir):
    """
    Check if a directory is a valid datasets folder for training a neural
    network. To be valid, the directory must contain at least two subfolders
    named /train and /valid containing respectively training data and cross
    validation data. If the directory fails to match this structure an error
    is raised.
    This function is use by argparse to check if the specified command line
    argument data_dir is a valid data folder and raises an error
    Args:
        data_dir (string): path to the data directory.
    Returns:
        data_dir_dict (Dict): a dictionnary containing the paths to the
        training and validation folders.
    """

    train_dir = Path(data_dir, "train")
    valid_dir = Path(data_dir, "valid")

    if (
        # If /train doesn't exist
        not train_dir.is_dir()
        # If /valid doesn't exist
        or not valid_dir.is_dir()
        # If /train is empty
        or not any(True for _ in train_dir.iterdir())
        # If /valid is empty
        or not any(True for _ in valid_dir.iterdir())
    ):
        raise argparse.ArgumentTypeError(
            f"{data_dir} does not contain the required subfolders or \
            data for training (/train) and cross validation (/valid)"
        )
    else:
        data_dir_dict = {
            "train": train_dir,
            "valid": valid_dir,
        }

        return data_dir_dict


def probability_type(p):
    """
    Check if a value is a valid probability. To be valid the value must be
    a float between 0 and 1. This function raises an error if the test fails.
    This function is use by argparse to check if probability of droupout
    specified by the user as a cli argument is a valid value.
    Args:
        p (Float): the probability of dropout.
    Returns:
        p (Float): the probability of dropout.
    """
    p = float(p)
    if p < 0 or p > 1:
        raise ValueError(
            f"dropout probability has to be between 0 and 1, but got {p}"
        )
    else:
        return p


def json_file_type(filepath):
    """
    Check if a file is a valid json file.
    This function is used by argparse to load the category to name dictionnary.
    Args:
        filepath (Path): the path to the json file.
    Returns:
        cat_to_name (dict): a dictionnary containing the mapping of each
        category in the dataset to its corresponding label.
    """

    filepath = Path(filepath)
    with open(filepath, "r") as f:
        try:
            cat_to_name = json.load(f)
            return cat_to_name
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Can't load {filepath} as a json object. "
                f"The decoding has failed."
            )


class GPUAction(argparse.Action):
    """
    Argparse action that check if cuda is available on the machine when the gpu
    flag is specified. If cuda is not available, it raises an error warning the
    user that he can't use the gpu for training a model or performing
    inference on this device.
    """

    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if not torch.cuda.is_available():
            raise ValueError(
                "cuda is not available on this machine. Cannot use the GPU"
            )
        else:
            setattr(namespace, self.dest, True)