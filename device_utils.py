import torch


def get_device(gpu=False):
    """
    Get the device of which to perform the training or the inference.
    Args:
        gpu (Bool): whether one should try to use the GPU or not.
    Returns:
        dataloaders (Device): the device on which to perform the training or
        the inference.
    """
    if (gpu and not torch.cuda.is_available()):
        raise ValueError(
            "cuda is not available on this machine. Cannot use the GPU. "
            "Reexecute the program without the --gpu option or try on another "
            "machine"
        )
    else:
        device = torch.device("cuda" if gpu else "cpu")
        return device