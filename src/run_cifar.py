import sys
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def main():

    torch.backends.cudnn.enabled = True

    print(f"Using pytorch version: {torch.__version__}")

    on_gpu = torch.cuda.is_available()
    if not on_gpu:
        sys.exit("Can't find GPU")

    device = torch.device("cuda:0")
    print(f"Using: {torch.cuda.get_device_name(device)}")

    trainloader = get_data()
    net = torchvision.models.resnet50(pretrained=True)
    net = swap_classifier(net)
    net.to(device)

    comp_conf = Conf(
        trainloader=trainloader,
        net=net,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optim.SGD(net.parameters(), lr=1e-2, momentum=0.9),
        device=device,
    )
    train(comp_conf)


@dataclass
class Conf:
    trainloader: DataLoader
    net: nn.Module
    criterion: nn.Module
    optimizer: Any
    device: torch.device


def get_data():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = DataLoader(trainset, batch_size=512, shuffle=True, num_workers=6)

    return trainloader


def swap_classifier(model):
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        OrderedDict(
            [
                ("cls_lin1", nn.Linear(num_ftrs, 512)),
                ("cls_relu", nn.ReLU()),
                ("cls_bn", nn.BatchNorm1d(512)),
                ("cls_lin2", nn.Linear(512, 10)),
            ]
        )
    )
    return model


def train(conf: Conf, n_epochs: int = 10):
    now = datetime.now()
    print(now.strftime("%d-%m-%Y-%H-%M"))

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(conf.trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(conf.device), labels.to(conf.device)

            # zero the parameter gradients
            conf.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = conf.net(inputs)
            loss = conf.criterion(outputs, labels)
            loss.backward()
            conf.optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 20 == 19:
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0

    print("Finished Training")
    now = datetime.now()
    print(now.strftime("%d-%m-%Y-%H-%M"))


if __name__ == "__main__":
    main()
