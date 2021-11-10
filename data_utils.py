import torch
from torchvision import datasets, transforms


def process_image(
    train=False,
    thumbnail_size=256,
    image_size=224,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    rotation_angle=30
):
    """
    Return the function to process a dataset of images for training or
    inference.
    Args:
        train (Bool): whether or not the processing will be applied to a
        training dataset.
        thumbnail_size (Int) - default 256: the size of the images shortest
        side before center-cropping to image_size.
        image_size (Int) - default 224: the size of image expected by the
        model that will perform the inference. The default is 224 because most
        of torch.models modules expect this size.
        mean (List) - default [0.485, 0.456, 0.406]: value of the mean of the
        dataset. The default is the mean of ImageNet.
        std (List) - default [0.229, 0.224, 0.225]: value of the standard
        deviation of the dataset. The default is the standard deviation of
        ImageNet.
        rotation_angle (Int) - default 30: maximum range of degrees to randomly
        rotate the training dataset to. It is used to perform the training
        data augmentation.
    Returns:
        transform (torchvision.transforms): a function containing the
        transformations to perform on the images.
    """

    if train:
        # Preprocess and augment data
        return transforms.Compose(
            [
                transforms.RandomRotation(rotation_angle),
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize(thumbnail_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )


def get_datasets(data_dir_dict):
    """
    Fetch the images from the training and the cross validation folder
    with their correct labels and preprocess them.
    Args:
        data_dir_dict (Dict): a dictionnary containing the path to the training
        and the validation images folder.
    Returns:
        datasets (Dict): a dictionnary containing the training and the
        validation datasets preprocessed along with their correct labels.
    """

    dataset = {
        "train": datasets.ImageFolder(
            data_dir_dict["train"],
            transform=process_image(train=True),
        ),
        "valid": datasets.ImageFolder(
            data_dir_dict["valid"],
            transform=process_image(train=False),
        ),
    }

    return dataset


def get_dataloaders(datasets, batch_size):
    """
    Load data from the dataset for training or inference.
    Args:
        datasets (Dict): a dictionnary containing the training and the
        validation datasets.
        batch_size (Int): the quantity of data to load at each
        iteration.
    Returns:
        dataloaders (Dict): a dictionnary containing the dataloaders.
    """
    dataloaders = {
        "train": torch.utils.data.DataLoader(
            datasets["train"], batch_size=batch_size, shuffle=True
        ),
        "valid": torch.utils.data.DataLoader(
            datasets["valid"], batch_size=batch_size, shuffle=True
        ),
    }

    return dataloaders