import os
import numpy as np
import traceback
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class Loader:
    def __init__(self, dataset=None, batch_size=32, split_size=0.30):
        self.dataset = dataset
        self.batch_size = batch_size
        self.split_size = split_size

    def transform(self, data):
        if isinstance(data, np.ndarray):
            standard_scaler = StandardScaler()
            return standard_scaler.fit_transform(data)

        else:
            raise ValueError("Data must be a numpy array".capitalize())

    def data_preprocessing(self):
        df = pd.read_csv(self.dataset)

        if "id" in df.columns:
            df.drop(["id"], inplace=True, axis=1)

        df.loc[:, "diagnosis"] = df.loc[:, "diagnosis"].map({"B": 1, "M": 0})

        X = df.iloc[:, 1:].values
        y = df.iloc[:, 0].values
        y = y.astype(int)

        try:
            X = self.transform(data=X)

        except ValueError as e:
            traceback.print_exc()
            raise

        except Exception as e:
            traceback.print_exc()
            raise

        X = torch.tensor(data=X, dtype=torch.float)
        y = torch.tensor(data=y, dtype=torch.long)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.split_size, random_state=42
        )

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }

    def create_dataloader(self):
        data = self.data_preprocessing()

        train_dataloader = DataLoader(
            dataset=list(zip(data["X_train"], data["y_train"])),
            batch_size=self.batch_size,
            shuffle=True,
        )

        test_dataloader = DataLoader(
            dataset=list(zip(data["X_test"], data["y_test"])),
            batch_size=self.batch_size,
            shuffle=True,
        )


if __name__ == "__main__":
    loader = Loader(
        dataset="/Users/shahmuhammadraditrahman/Desktop/MLFlowPractise/data/raw/breast-cancer.csv"
    )
    loader.create_dataloader()
