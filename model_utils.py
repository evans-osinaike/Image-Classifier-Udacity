import torch
from torch import nn
from torchvision import models


def freeze_model_params_(model):
    """
    Freeze all the parameters of a model.
    Args:
        model (nn.Module): a convolutional neural network
    Returns:
        None
    """
    for param in model.parameters():
        param.requires_grad = False


def get_model_trainable_params(model):
    """
    Get the unfreezed params in a model.
    Args:
        model (nn.Module): a convolutional neural network
    Returns:
        trainable_params (List): list of parameters to learn during the
        training.
    """
    return filter(lambda param: param.requires_grad, model.parameters())


def build_classifier(num_features, num_hidden_units, num_classes, dropout):
    """
    Build a classifier.
    Args:
        num_features (Int): the number of units in the input layer
        num_hidden_units (Int): the number of units in the hidden layer
        num_classes (Int): the number of units in the output layer
    Returns:
        classifier (nn.Module): a model capable of classification
    """
    classifier = nn.Sequential(
        nn.Linear(num_features, num_hidden_units),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(num_hidden_units, num_classes),
        nn.LogSoftmax(dim=1),
    )

    return classifier


#  Inspired by pytorch documentation.
# SEE:
# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
def build_model(arch, num_hidden_units, num_classes, dropout):
    """
    Download the appropriate pretrained model according to the specified
    architecture and adjust the classifier.
    Args:
        arch (String): the architecture of the pretrained models.
        num_hidden_units (Unt): the number of units in the hidden layer of the
        classifier.
        num_classes (Int): the number of classes in the dataset.
    Returns:
        model (nn.Module): a convolutional neural network with a classifier
        layer ready to be trained.
    """
    model = None

    if arch == "resnet50":
        model = models.resnet50(pretrained=True)
        freeze_model_params_(model)
        num_features = model.fc.in_features
        model.fc = build_classifier(
            num_features, num_hidden_units, num_classes, dropout
        )

    elif arch == "alexnet":
        model = models.alexnet(pretrained=True)
        freeze_model_params_(model)
        num_features = model.classifier[0].in_features
        model.classifier = build_classifier(
            num_features, num_hidden_units, num_classes, dropout
        )

    elif arch == "vgg13":
        model = models.vgg13(pretrained=True)
        freeze_model_params_(model)
        num_features = model.classifier[0].in_features
        model.classifier = build_classifier(
            num_features, num_hidden_units, num_classes, dropout
        )

    elif arch == "vgg16":
        model = models.vgg16(pretrained=True)
        freeze_model_params_(model)
        num_features = model.classifier[0].in_features
        model.classifier = build_classifier(
            num_features, num_hidden_units, num_classes, dropout
        )

    elif arch == "densenet161":
        model = models.densenet161(pretrained=True)
        freeze_model_params_(model)
        num_features = model.classifier.in_features
        model.classifier = build_classifier(
            num_features, num_hidden_units, num_classes, dropout
        )

    else:
        print("Invalid model name, exiting...")
        exit()

    return model


def rebuild_model_from_checkpoint(checkpoint_path):
    """
    Rebuild a model from it's checkpoint.
    Args:
        checkpoint_path (Path): a path to the model's checkpoint.
    Returns:
        model (nn.Module): a convolutional neural network with a classifier
        layer ready to be retrained of inferred.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = build_model(
        checkpoint["arch"],
        checkpoint["num_hidden_units"],
        checkpoint["num_classes"],
        checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.class_to_idx = checkpoint["class_to_idx"]

    return model