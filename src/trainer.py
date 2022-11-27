# Based on: https://github.com/pytorch/examples/blob/master/mnist/main.py
import torch
import torch.nn.functional as F


import torch.distributed as dist


def train(model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
    model.train()
    ddp_loss = torch.zeros(2).to(rank)
    if sampler:
        sampler.set_epoch(epoch)
    for data, target in train_loader:
        data, target = data.to(rank), target.to(rank)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target, reduction="sum")
        loss.backward()
        optimizer.step()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(data)

    if world_size > 1:
        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        print("Train Epoch: {} \tLoss: {:.6f}".format(epoch, ddp_loss[0] / ddp_loss[1]))


def test(model, rank, world_size, test_loader):
    model.eval()
    ddp_loss = torch.zeros(3).to(rank)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(rank), target.to(rank)
            output = model(data)
            ddp_loss[0] += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            ddp_loss[1] += pred.eq(target.view_as(pred)).sum().item()
            ddp_loss[2] += len(data)

    if world_size > 1:
        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    if rank == 0:
        test_loss = ddp_loss[0] / ddp_loss[2]
        print(
            "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
                test_loss,
                int(ddp_loss[1]),
                int(ddp_loss[2]),
                100.0 * ddp_loss[1] / ddp_loss[2],
            )
        )
