import os
import torch
import argparse
from utils import load, config
from model import Model
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)


class TestModel:
    def __init__(self):

        self.model = Model()

    def load_dataloader(self):
        return load(
            filename=os.path.join(
                config()["path"]["processed_path"], "test_dataloader.pkl"
            )
        )

    def select_best_model(self):
        if os.path.exists(config()["path"]["best_model_path"]):
            best_model_path = config()["path"]["best_model_path"]

            best_model = torch.load(os.path.join(best_model_path, "best_model.pth"))

            self.model.load_state_dict(best_model["model"])

    def test(self):
        dataloader = self.load_dataloader()

        print(dataloader)

        self.select_best_model()

        self.predicted = []
        self.actual = []

        for X, y in dataloader:
            predicted = self.model(X)
            predicted = predicted.view(-1)
            predicted = torch.where(predicted > 0.5, 1, 0)
            predicted = predicted.detach().flatten()

            self.actual.extend(y.detach().flatten())
            self.predicted.extend(predicted)

        accuracy = accuracy_score(self.predicted, self.actual)
        precision = precision_score(self.predicted, self.actual)
        recall = recall_score(self.predicted, self.actual)
        f1 = f1_score(self.predicted, self.actual)

        confusion = confusion_matrix(self.predicted, self.actual)
        classification = classification_report(self.predicted, self.actual)

        print(f"Test Accuracy: {accuracy}")
        print(f"Test Precision: {precision}")
        print(f"Test Recall: {recall}")
        print(f"Test F1 Score: {f1}")

        print("\n")

        print(f"Confusion Matrix: \n {confusion}")
        print("\n")

        print(f"Classification Report: \n {classification}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Model for Breast Cancer".title())
    parser.add_argument("--test", action="store_true", help="Test Model".capitalize())

    args = parser.parse_args()

    if args.test:
        test = TestModel()

        test.test()
