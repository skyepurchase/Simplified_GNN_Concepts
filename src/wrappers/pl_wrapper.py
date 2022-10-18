import torch
import pytorch_lightning as pl

# type annotations
from torch.nn import Module
from typing import TypedDict
from collections.abc import Callable
from pytorch_lightning.loggers import TensorBoardLogger


class BatchDict(TypedDict):
    loss: torch.Tensor 
    acc: float
    log: dict
    correct: int
    total: int


class PLModel(pl.LightningModule):
    def __init__(self,
                 model: Module,
                 learning_rate: int) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.model = model
        self.criterion :  Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = torch.nn.CrossEntropyLoss()

    def forward(self, x) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {'optimizer': optimizer}


    def training_step(self, batch, batch_idx: int) -> BatchDict:
        z = self.model(batch)[batch.train_mask]
        y = batch.y[batch.train_mask]
       
        correct = z.argmax(dim=1).eq(y).sum().item()
        total = len(y)

        train_loss = self.criterion(z, y)
        train_acc = z.argmax(dim=1).eq(y).sum()/len(y)

        self.log("train_acc", train_acc*100, prog_bar=True)

        logs = {"train_loss": train_loss,
                "train_acc": train_acc}

        batch_dictionary : BatchDict = {
            'loss': train_loss,
            'acc': train_acc,
            'log': logs,
            'correct': correct,
            'total': total
        }
       
        return batch_dictionary

    def train_epoch_end(self, outputs: list[BatchDict]) -> None:
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        correct = sum([x['correct'] for x in outputs])
        total = sum([x['total'] for x in outputs])

        if isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_scalar("Loss/Train",
                                              avg_loss,
                                              self.current_epoch)

            self.logger.experiment.add_scalar("Accuracy/Train",
                                              correct * 100/total,
                                              self.current_epoch)

    def validation_step(self, batch, batch_idx: int) -> BatchDict:
        z = self.model(batch)[batch.val_mask]
        y = batch.y[batch.val_mask]
       
        correct = z.argmax(dim=1).eq(y).sum().item()
        total = len(y)

        train_loss = self.criterion(z, y)
        train_acc = z.argmax(dim=1).eq(y).sum()/len(y)

        self.log("train_acc", train_acc*100, prog_bar=True)

        logs = {"train_loss": train_loss,
                "train_acc": train_acc}

        batch_dictionary : BatchDict = {
            "loss": train_loss,
            "acc": train_acc,
            "log": logs,
            "correct": correct,
            "total": total
        }
       
        return batch_dictionary

    def validation_epoch_end(self, outputs: list[BatchDict]) -> None:
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        correct = sum([x['correct'] for x in outputs])
        total = sum([x['total'] for x in outputs])

        if isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_scalar("Loss/Train",
                                              avg_loss,
                                              self.current_epoch)

            self.logger.experiment.add_scalar("Accuracy/Train",
                                              correct * 100/total,
                                              self.current_epoch)

