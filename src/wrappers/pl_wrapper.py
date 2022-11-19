from os import WCOREDUMP
import torch
import pytorch_lightning as pl

# type annotations
from torch.nn import Module
from typing import TypedDict
from collections.abc import Callable
from pytorch_lightning.loggers import TensorBoardLogger
from torch.types import Number
from torch.functional import Tensor


class BatchDict(TypedDict):
    loss: Tensor 
    acc: Tensor 
    log: dict
    correct: Number 
    total: int


class GraphWrapper(pl.LightningModule):
    def __init__(self,
                 model: Module,
                 learning_rate: float,
                 weight_decay: float) -> None:
        super().__init__()
        self.lr: float = learning_rate
        self.model: Module = model
        self.criterion: Callable[[Tensor, Tensor], Tensor] = torch.nn.CrossEntropyLoss()
        self.weight_decay: float = weight_decay

    def forward(self, batch) -> torch.Tensor:
        return self.model(batch.x, batch.edge_index)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return {'optimizer': optimizer}

    def training_step(self, batch, batch_idx: int) -> BatchDict:
        z: Tensor = self.model(batch.x, batch.edge_index)[batch.train_mask]
        y: Tensor = batch.y[batch.train_mask]
       
        correct: Number = z.argmax(dim=1).eq(y).sum().item()
        total: int = len(y)

        train_loss: Tensor = self.criterion(z, y)
        train_acc: Tensor = z.argmax(dim=1).eq(y).sum()/len(y)

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
        avg_loss: Tensor = torch.stack([x['loss'] for x in outputs]).mean()

        correct: Number = sum([x['correct'] for x in outputs])
        total: int = sum([x['total'] for x in outputs])

        if isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_scalar("Loss/Train",
                                              avg_loss,
                                              self.current_epoch)

            self.logger.experiment.add_scalar("Accuracy/Train",
                                              correct * 100/total,
                                              self.current_epoch)

    def validation_step(self, batch, batch_idx: int) -> BatchDict:
        z: Tensor = self.model(batch.x, batch.edge_index)[batch.val_mask]
        y: Tensor = batch.y[batch.val_mask]
       
        correct: Number = z.argmax(dim=1).eq(y).sum().item()
        total: int = len(y)

        val_loss: Tensor = self.criterion(z, y)
        val_acc: Tensor = z.argmax(dim=1).eq(y).sum()/len(y)

        self.log("val_acc", val_acc*100, prog_bar=True)

        logs = {"val_loss": val_loss,
                "val_acc": val_acc}

        batch_dictionary : BatchDict = {
            'loss': val_loss,
            'acc': val_acc,
            'log': logs,
            'correct': correct,
            'total': total
        }
       
        return batch_dictionary

    def validation_epoch_end(self, outputs: list[BatchDict]) -> None:
        avg_loss: Tensor = torch.stack([x['loss'] for x in outputs]).mean()

        correct: Number = sum([x['correct'] for x in outputs])
        total: int = sum([x['total'] for x in outputs])

        if isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_scalar("Loss/Validation",
                                              avg_loss,
                                              self.current_epoch)

            self.logger.experiment.add_scalar("Accuracy/Validation",
                                              correct * 100/total,
                                              self.current_epoch)


class LinearWrapper(pl.LightningModule):
    def __init__(self,
                 model: Module,
                 learning_rate: float,
                 weight_decay: float = 0.0) -> None:
        super().__init__()
        self.lr: float = learning_rate
        self.model: Module = model
        self.criterion: Callable[[Tensor, Tensor], Tensor] = torch.nn.CrossEntropyLoss()
        self.weight_decay: float = weight_decay

    def forward(self, batch) -> torch.Tensor:
        return self.model(batch.x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return {'optimizer': optimizer}

    def training_step(self, batch, batch_idx: int) -> BatchDict:
        z: Tensor = self.model(batch.x)[batch.train_mask]
        y: Tensor = batch.y[batch.train_mask]
       
        correct: Number = z.argmax(dim=1).eq(y).sum().item()
        total: int = len(y)

        train_loss: Tensor = self.criterion(z, y)
        train_acc: Tensor = z.argmax(dim=1).eq(y).sum()/len(y)

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
        avg_loss: Tensor = torch.stack([x['loss'] for x in outputs]).mean()

        correct: Number = sum([x['correct'] for x in outputs])
        total: int = sum([x['total'] for x in outputs])

        if isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_scalar("Loss/Train",
                                              avg_loss,
                                              self.current_epoch)

            self.logger.experiment.add_scalar("Accuracy/Train",
                                              correct * 100/total,
                                              self.current_epoch)

    def validation_step(self, batch, batch_idx: int) -> BatchDict:
        z: Tensor = self.model(batch.x)[batch.val_mask]
        y: Tensor = batch.y[batch.val_mask]
       
        correct: Number = z.argmax(dim=1).eq(y).sum().item()
        total: int = len(y)

        val_loss: Tensor = self.criterion(z, y)
        val_acc: Tensor = z.argmax(dim=1).eq(y).sum()/len(y)

        self.log("val_acc", val_acc*100, prog_bar=True)

        logs = {"val_loss": val_loss,
                "val_acc": val_acc}

        batch_dictionary : BatchDict = {
            'loss': val_loss,
            'acc': val_acc,
            'log': logs,
            'correct': correct,
            'total': total
        }
       
        return batch_dictionary

    def validation_epoch_end(self, outputs: list[BatchDict]) -> None:
        avg_loss: Tensor = torch.stack([x['loss'] for x in outputs]).mean()

        correct: Number = sum([x['correct'] for x in outputs])
        total: int = sum([x['total'] for x in outputs])

        if isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_scalar("Loss/Validation",
                                              avg_loss,
                                              self.current_epoch)

            self.logger.experiment.add_scalar("Accuracy/Validation",
                                              correct * 100/total,
                                              self.current_epoch)

