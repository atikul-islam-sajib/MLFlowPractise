import os
import argparse
from test import TestModel
from trainer import Trainer
from dataloader import Loader
from utils import config


def cli():
    parser = argparse.ArgumentParser(description="CLI for the project".capitalize())
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.path.join(config()["path"]["raw_path"], "breast-cancer.csv"),
        help="Path to the dataset".capitalize(),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for the dataloader".capitalize(),
    )
    parser.add_argument(
        "--split_size",
        default=0.20,
        type=float,
        help="Split size for the dataloader".capitalize(),
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs".capitalize()
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="Learning rate".capitalize()
    )
    parser.add_argument("--beta1", type=float, default=0.5, help="Beta1".capitalize())
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2".capitalize())
    parser.add_argument("--adam", type=bool, default=True, help="Adam".capitalize())
    parser.add_argument("--SGD", type=bool, default=False, help="SGD".capitalize())
    parser.add_argument("--device", type=str, default="cpu", help="Device".capitalize())
    parser.add_argument(
        "--display", type=bool, default=True, help="Display".capitalize()
    )
    parser.add_argument("--train", action="store_true", help="Train Model".capitalize())
    parser.add_argument("--test", action="store_true", help="Test Model".capitalize())

    args = parser.parse_args()

    if args.train:
        loader = Loader(
            dataset=args.dataset,
            batch_size=args.batch_size,
            split_size=args.split_size,
        )

        loader.create_dataloader()

        trainer = Trainer(
            epochs=args.epochs,
            lr=args.lr,
            beta1=args.beta1,
            beta2=args.beta2,
            adam=args.adam,
            SGD=args.SGD,
            device=args.device,
            is_display=args.display,
        )

        trainer.train()

        trainer.plot_history()

    elif args.test:
        test = TestModel()

        test.test()


if __name__ == "__main__":
    cli()
