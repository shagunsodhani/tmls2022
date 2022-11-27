import torch


from torch.utils.data.distributed import DistributedSampler
from config import Config

import torchvision


def get_mnist_dataloaders_and_samplers(config: Config, rank: int, world_size: int):
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    datasets = {
        mode: torchvision.datasets.MNIST(
            "../data",
            train=mode == "train",
            download=mode == "train",
            transform=transform,
        )
        for mode in ["train", "test"]
    }

    if world_size > 1:
        samplers = {
            mode: DistributedSampler(
                dataset,
                rank=rank,
                num_replicas=world_size,
                shuffle=mode == "train",
            )
            for mode, dataset in datasets.items()
        }
    else:
        samplers = {mode: None for mode in datasets}

    kwargs = {
        mode: {
            "batch_size": config.batch_size[mode],
            "sampler": sampler,
            "shuffle": world_size == 1 and mode == "train",
        }
        for mode, sampler in samplers.items()
    }

    cuda_kwargs = {"num_workers": 2, "pin_memory": True}

    for mode in kwargs:
        kwargs[mode].update(cuda_kwargs)

    dataloaders = {
        mode: torch.utils.data.DataLoader(dataset, **kwargs[mode])
        for mode, dataset in datasets.items()
    }

    return dataloaders, samplers


def get_dataloaders_and_samplers(name: str, config: Config, rank: int, world_size: int):
    if name == "mnist":
        return get_mnist_dataloaders_and_samplers(
            config=config, rank=rank, world_size=world_size
        )
    else:
        raise NotImplementedError(
            f"name = {name} is not supported for building dataloaders and samplers."
        )
