import os
import torch
import argparse
import torch.nn as nn
from utils import config
from torchsummary import summary
from torchview import draw_graph


class Model(nn.Module):
    def __init__(self, in_features=30):
        super(Model, self).__init__()

        self.in_features = in_features
        self.out_features = 128

        self.layers = []

        for _ in range(4):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(
                        in_features=self.in_features, out_features=self.out_features
                    ),
                    nn.ReLU(),
                    nn.BatchNorm1d(num_features=self.out_features),
                )
            )

            self.in_features = self.out_features
            self.out_features = self.out_features // 2

        self.model = nn.Sequential(*self.layers)

        self.out = nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=1), nn.Sigmoid()
        )

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = self.model(x)
            return self.out(x)

        else:
            raise ValueError("Input must be a torch.Tensor".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model for breast Cancer".title())
    parser.add_argument(
        "--model",
        action="store_true",
        default=None,
        help="Model for Breast Cancer".capitalize(),
    )

    args = parser.parse_args()

    if args.model:

        model = Model()

        summary(model=model, input_size=(30,))

        draw_graph(model=model, input_data=torch.randn(32, 30)).visual_graph.render(
            filename=os.path.join(config()["path"]["files_path"], "Model"),
            format="jpeg",
        )
