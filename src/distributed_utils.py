import os

import torch.distributed as dist
import torch
from config import Config

import functools
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from model import build_model
from config import Config
from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def build_model_for_distributed_training(config: Config, rank: int, world_size: int):
    model = build_model(name=config.dataset).to(rank)
    if world_size > 1:
        if config.strategy == "fsdp":
            auto_wrap_policy = functools.partial(
                size_based_auto_wrap_policy, min_num_params=20000
            )
            cpu_offload = CPUOffload(offload_params=True)
            model = FSDP(
                model,
                # auto_wrap_policy=auto_wrap_policy,
                # cpu_offload=cpu_offload,
            )
        elif config.strategy == "ddp":
            model = DDP(model)
        else:
            raise NotImplementedError(
                f"strategy = {config.strategy} is not supported for building distributed models."
            )
    return model
