import pytorch_lightning as pl
import torch
from torch import nn

from zerospot.models.model import BcResNetModel


class LightningEngine(pl.LightningModule):
    def __init__(self, lr_rate):
        super(LightningEngine, self).__init__()

        self.model = BcResNetModel()
        self.lr_rate = lr_rate
        self.criterion = nn.CrossEntropyLoss()

    def training_step(self, train_batch, batch_idx):
        spg, label = train_batch
        spg = spg.unsqueeze(1)
        label_predict = self.model(spg)
        loss = self.criterion(label_predict, label)
        self.log('crossentropy', loss.item())
        return loss

    def validation_step(self, val_batch, batch_idx):
        spg, label = val_batch
        spg = spg.unsqueeze(1)
        label_predict = self.model(spg)
        loss = self.criterion(label_predict, label)
        self.log('val_loss', loss.item())
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr_rate)
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=0.95
            ),
            "name": "expo_lr",
        }
        return [optimizer], [lr_scheduler]
