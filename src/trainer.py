import os
import torch
import numpy as np
from tqdm import tqdm
from helper import helpers
from utils import config
from sklearn.metrics import accuracy_score


class Trainer:
    def __init__(
        self,
        epochs=100,
        lr=0.01,
        beta1=0.5,
        beta2=0.999,
        momentum=0.9,
        adam=True,
        SGD=False,
        device="cpu",
        is_display=True,
    ):
        self.epochs = epochs
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.adam = adam
        self.SGD = SGD
        self.momentum = momentum
        self.device = device
        self.is_display = is_display

        self.init = helpers(
            lr=self.lr,
            beta1=self.beta1,
            beta2=self.beta2,
            momentum=self.momentum,
            adam=self.adam,
            SGD=self.SGD,
        )

        self.train_dataloader = self.init["train_dataloader"]
        self.valid_dataloader = self.init["test_dataloader"]

        self.model = self.init["model"]
        self.criterion = self.init["criterion"]
        self.optimizer = self.init["optimizer"]

        self.loss = float("inf")
        self.config = config()

        self.train_models_path = self.config["path"]["train_models_path"]
        self.best_model_path = self.config["path"]["best_model_path"]

    def update_model(self, X, y):
        if isinstance(X, torch.Tensor) and isinstance(y, torch.Tensor):

            self.optimizer.zero_grad()

            predicted = self.model(X)
            predicted_loss = self.criterion(predicted, y)

            predicted_loss.backward()
            self.optimizer.step()

            return predicted, predicted_loss.item()

        else:
            raise TypeError("X and y must be torch.Tensor")

    def display_progress(
        self,
        epoch,
        train_loss,
        test_loss,
        train_pred,
        train_actual,
        valid_pred,
        valid_actual,
    ):

        train_pred_flat = np.concatenate(train_pred)
        train_actual_flat = np.concatenate(train_actual)
        valid_pred_flat = np.concatenate(valid_pred)
        valid_actual_flat = np.concatenate(valid_actual)

        if self.is_display:

            print(
                "Epochs - [{}/{}] - Train Loss: {:.4f} - Test Loss: {:.4f} - train_accuracy: {:.4f} - test_accuracy: {:.4f}".format(
                    epoch + 1,
                    self.epochs,
                    np.mean(train_loss),
                    np.mean(test_loss),
                    accuracy_score(train_actual_flat, train_pred_flat),
                    accuracy_score(valid_actual_flat, valid_pred_flat),
                )
            )

        else:
            print("Epochs - [{}/{}] is completed".format(epoch + 1, self.epochs))

    def saved_models(self, epoch=None, train_loss=None):
        torch.save(
            self.model.state_dict(),
            os.path.join(self.train_models_path, "model{}.pth".format(epoch)),
        )

        if train_loss is not None:

            if self.loss > train_loss:
                self.loss = train_loss
                torch.save(
                    {
                        "model": self.model.state_dict(),
                        "loss": train_loss,
                        "epoch": epoch,
                    },
                    os.path.join(self.best_model_path, "best_model.pth"),
                )

        else:
            raise ValueError("Train loss cannot be None".capitalize())

    def train(self):
        for epoch in tqdm(range(self.epochs)):
            train_predcit = []
            valid_predict = []
            train_actual = []
            valid_actual = []
            train_loss = []
            test_loss = []

            for index, (X, y) in enumerate(self.train_dataloader):
                X, y = X.to(self.device), y.to(self.device)
                predicted, loss = self.update_model(X=X, y=y.unsqueeze(1).float())

                train_predcit.append(
                    torch.where(predicted > 0.5, 1, 0).detach().cpu().numpy()
                )
                train_actual.append(torch.where(y > 0.5, 1, 0).detach().cpu().numpy())

                train_loss.append(loss)

            for _, (X, y) in enumerate(self.valid_dataloader):
                X, y = X.to(self.device), y.to(self.device)
                predicted = self.model(X)
                loss = self.criterion(predicted, y.unsqueeze(1).float())

                valid_predict.append(
                    torch.where(predicted > 0.5, 1, 0).detach().cpu().numpy()
                )
                valid_actual.append(torch.where(y > 0.5, 1, 0).detach().cpu().numpy())

                test_loss.append(loss.item())

            self.display_progress(
                epoch,
                train_loss,
                test_loss,
                train_predcit,
                train_actual,
                valid_predict,
                valid_actual,
            )

            self.saved_models(epoch=epoch + 1, train_loss=np.mean(train_loss))


if __name__ == "__main__":
    trainer = Trainer(
        epochs=10, lr=0.01, beta1=0.5, beta2=0.999, adam=True, SGD=False, device="cpu"
    )
    trainer.train()
