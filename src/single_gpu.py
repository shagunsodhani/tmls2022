import torch
import torch.optim as optim


from torch.optim.lr_scheduler import StepLR

from trainer import train, test
from config import Config
from data import get_dataloaders_and_samplers
from model import build_model


def main(config):
    rank = 0
    world_size = 1
    is_rank_0 = True

    dataloaders, samplers = get_dataloaders_and_samplers(
        name=config.dataset, config=config, rank=rank, world_size=world_size
    )

    model = build_model(name=config.dataset).to(rank)

    optimizer = optim.Adadelta(model.parameters(), lr=config.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=config.gamma)
    if is_rank_0 and config.should_record_time:
        init_start_event = torch.cuda.Event(enable_timing=True)
        init_end_event = torch.cuda.Event(enable_timing=True)
        init_start_event.record()

    for epoch in range(1, config.epochs + 1):
        train(
            model=model,
            rank=rank,
            world_size=world_size,
            train_loader=dataloaders["train"],
            optimizer=optimizer,
            epoch=epoch,
            sampler=samplers["train"],
        )
        test(
            model=model,
            rank=rank,
            world_size=world_size,
            test_loader=dataloaders["test"],
        )
        scheduler.step()

    if is_rank_0:
        if config.should_record_time:
            init_end_event.record()
            print(
                f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec"
            )
        print(f"{model}")

    if config.save_model:
        state = model.state_dict()
        if rank == 0:
            torch.save(state, "mnist_cnn.pt")


if __name__ == "__main__":
    config = Config.build()
    torch.manual_seed(config.seed)
    main(config)
