import os
import torch
import torch.nn as nn
from model import Model
import torch.optim as optim
from utils import load, config


def load_dataloader():
    processed_path = config()["path"]["processed_path"]

    if os.path.exists(processed_path):
        train_dataloader = load(
            filename=os.path.join(processed_path, "train_dataloader.pkl")
        )

        test_dataloader = load(
            filename=os.path.join(processed_path, "test_dataloader.pkl")
        )

        return {
            "train_dataloader": train_dataloader,
            "test_dataloader": test_dataloader,
        }


def helpers(**kwargs):
    lr = kwargs["lr"]
    adam = kwargs["adam"]
    SGD = kwargs["SGD"]
    beta1 = kwargs["beta1"]
    beta2 = kwargs["beta2"]
    momentum = kwargs["momentum"]

    netBreastCancer = Model()

    if adam:
        optimizer = optim.Adam(
            params=netBreastCancer.parameters(), lr=lr, betas=(beta1, beta2)
        )

    elif SGD:
        optimizer = optim.SGD(
            params=netBreastCancer.parameters(), lr=lr, momentum=momentum
        )

    criterion = nn.BCELoss(reduction="mean")

    dataloader = load_dataloader()

    return {
        "model": netBreastCancer,
        "optimizer": optimizer,
        "criterion": criterion,
        "train_dataloader": dataloader["train_dataloader"],
        "test_dataloader": dataloader["test_dataloader"],
    }


if __name__ == "__main__":
    init = helpers(
        lr=0.01,
        adam=True,
        SGD=False,
        beta1=0.9,
        beta2=0.5,
        momentum=0.9,
    )

    print(init["model"])
    print(init["train_dataloader"])
    print(init["test_dataloader"])

    assert init["optimizer"].__class__.__name__ == "Adam"
    assert init["criterion"].__class__.__name__ == "BCELoss"
