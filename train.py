import argparse
from pathlib import Path
from time import time

import torch
from torch import nn, optim
import numpy as np
from sklearn.metrics import f1_score

from cli_utils import datasets_folder_type, probability_type
from data_utils import get_datasets, get_dataloaders
from model_utils import build_model, get_model_trainable_params
from device_utils import get_device

# Note to run: python train.py (directory of data)

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
        description="A script that train a new network on a dataset and save the \
        model as a checkpoint",
        epilog="\
        Examples:\n\
        * Train the model of a flower dataset:\n\
        \tpython train.py ./flower\n\n\
        * Set the directory to save the model checkpoints:\n\
        \tpython train.py ./flower --save_dir ./saved_models\n\n\
        * Choose the architecture:\n\
        \tpython train.py ./flower --arch 'vgg13'\n\n\
        * Set the hyperparameters:\n\
        \tpython train.py ./flowers --learning_rate 0.01 --hidden_units 512 \
        --dropout 0.5 --epochs 20\n\n\
        * Use the GPU for training:\n\
        \tpython train.py ./flower --gpu\
        ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "data_dir",
        type=datasets_folder_type,
        help="path to the folder containing the training and the cross \
        validation data. It must contain at least two subfolders namely: \
        train (which contains the training images) and valid (which contains \
        the cross validation images). Each of these subfolders should follow \
        the torchvision.datasets.ImageFolder default arrangement",
    )

    parser.add_argument(
        "--save_dir",
        type=Path,
        default=".",
        help="the folder where to save the model checkpoints",
    )

    parser.add_argument(
        "--arch",
        type=str,
        default="vgg16",
        choices=[
            "resnet50",
            "alexnet",
            "vgg13",
            "vgg16",
            "densenet161",
        ],
        help="the CNN model architecture",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="the quantity of data at each training iteration",
    )

    parser.add_argument(
        "--hidden_units",
        type=int,
        default=512,
        help="the number of units in the classifier hidden layer",
    )

    parser.add_argument(
        "--dropout",
        type=probability_type,
        default=0.5,
        help="the probability of dropout of a unit",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.05,
        help="the learning rate of the optimizer",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="the number of learning epochs",
    )

    parser.add_argument(
        "--gpu",
        default=True,
        action="store_true",
        help="perform the training on GPU",
    )

    return parser.parse_args()


def train_model_(
    model,
    dataloaders,
    criterion,
    optimizer,
    device,
    epochs,
    checkpoint,
    save_dir,
):
    """
    train a new network on a dataset.
    Args:
        model (nn.Module): a convolutional neural network.
        dataloaders (Dict): a dictionnary containing the dataloaders.
        criterion (nn.Module): the loss function.
        optimizer (nn.Optimizer): the optimizer of weights.
        device (torch.device): the device to use for inference.
        epochs (Int): the number of epochs for the training.
        checkpoint (dict): the checkpoint to update an save.
        save_dir (Path): the directory where to save the checkpoint.
    Return:
        None
    """
    model.to(device)

    print("-" * 62)
    print(
        "| {:8s} | {:10s} | {:10s} | {:10s} | {:8s} |".format(
            "epoch", "Train loss", "Val loss", "Accuracy", "f1 score"
        )
    )
    print("-" * 62)
    
    start_time = time()
    
    train_loss = 0
    best_accuracy = 0
    print_every = 10
    steps = 0
    
    for epoch in range(epochs):        
        for inputs, labels in dataloaders["train"]:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            logps = model(inputs)
            loss = criterion(logps, labels)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

            # Validation
            if steps % print_every == 0:
                model.eval()
                targets = []
                outputs = []
                valid_loss = 0
                accuracy = 0
                with torch.no_grad():
                    for inputs, labels in dataloaders["valid"]:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)

                        # validation loss
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        _, top_class = ps.topk(1, dim=1)

                        # Calculate accuracy
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                        targets.extend(labels)
                        outputs.extend(top_class)
                    else:
                        train_loss /= print_every
                        valid_loss /= len(dataloaders["valid"])
                        accuracy /= len(dataloaders["valid"])
                        f1 = f1_score(
                            targets,
                            outputs,
                            average="weighted",
                            labels=np.unique(outputs)
                        )

                        if accuracy > best_accuracy:
                            save_dir = Path(save_dir)
                            if not save_dir.is_dir():
                                save_dir.mkdir()

                            checkpoint["model_state_dict"] = model.state_dict()
                            checkpoint_path = save_dir / (checkpoint["arch"] + ".pth")
                            torch.save(checkpoint, checkpoint_path)
            
                        print(
                            f"| {(str(epoch+1) + '/' + str(epochs)):8s} "
                            f"| {train_loss:10.3f} "
                            f"| {valid_loss:10.3f} "
                            f"| {accuracy:10.2%} "
                            f"| {f1:8.3f} |"
                        )
            
                    model.train()
                
    else:
        tot_time = time() - start_time
        print("-" * 62)
        print()
        print("Training complete in {:.0f}h {:.0f}m {:.0f}s".format(
            tot_time / 3600, (tot_time % 3600) / 60, (tot_time % 3600) % 60)
             )


def main():
    cli_args = get_cli_arguments()
    datasets = get_datasets(cli_args.data_dir)
    dataloaders = get_dataloaders(datasets, cli_args.batch_size)
    # The number of classes in the datasets
    num_classes = len(datasets["train"].class_to_idx)

    model = build_model(
        cli_args.arch, cli_args.hidden_units, num_classes, cli_args.dropout
    )

    checkpoint = {
        # Important to rebuild the model later
        "arch": cli_args.arch,
        "num_hidden_units": cli_args.hidden_units,
        "num_classes": num_classes,
        "dropout": cli_args.dropout,
        # Mapping of classes to indices of the datasets
        "class_to_idx": datasets["train"].class_to_idx,
        "model_state_dict": model.state_dict(),
    }

    # negative log likelihood loss (multiclass classification)
    criterion = nn.NLLLoss()
    params_to_learn = get_model_trainable_params(model)
    optimizer = optim.SGD(params_to_learn, lr=cli_args.learning_rate)
    device = get_device(cli_args.gpu)
    train_model_(
        model,
        dataloaders,
        criterion,
        optimizer,
        device,
        cli_args.epochs,
        checkpoint,
        cli_args.save_dir,
    )


if __name__ == "__main__":
    main()