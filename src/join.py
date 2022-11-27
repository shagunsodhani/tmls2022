# modified from https://pytorch.org/tutorials/advanced/generic_join.html
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.algorithms.join import Join
from torch.nn.parallel import DistributedDataParallel as DDP
import datetime

BACKEND = "nccl"
WORLD_SIZE = 2
NUM_INPUTS = 50


def worker(rank):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(
        BACKEND,
        rank=rank,
        world_size=WORLD_SIZE,
        timeout=datetime.timedelta(seconds=2),
    )

    model = DDP(torch.nn.Linear(1, 1).to(rank), device_ids=[rank])
    # Rank 1 gets one more input than rank 0
    inputs = [torch.tensor([1]).float() for _ in range(NUM_INPUTS + rank * 100)]

    num_inputs = 0
    with Join([model], enable=False, throw_on_early_termination=False):
        for input in inputs:
            num_inputs += 1
            loss = model(input).sum()
            loss.backward()

    print(f"Rank {rank} has exhausted all {num_inputs} of its inputs!")


def main():
    mp.spawn(worker, nprocs=WORLD_SIZE, join=True)


if __name__ == "__main__":
    main()
