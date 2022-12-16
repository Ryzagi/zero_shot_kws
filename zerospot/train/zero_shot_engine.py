import pytorch_lightning as pl
import torch
from torch import nn

from zerospot.models.zero_shot_model import ZeroShotModel


class LightningEngine(pl.LightningModule):
    def __init__(self, lr_rate):
        super(LightningEngine, self).__init__()

        self.model = ZeroShotModel(30, 32, 1)
        self.lr_rate = lr_rate
        self.criterion = nn.BCEWithLogitsLoss()

    def training_step(self, train_batch, batch_idx):
        spg, tokens_ids_tensor, tokens_lentghts_tensor, label = train_batch
        spg = spg.unsqueeze(1)
        label_predict = self.model(tokens_ids_tensor, tokens_lentghts_tensor, spg)
        loss = self.criterion(label_predict, label.float())
        self.log('crossentropy', loss.item())
        return loss

    def validation_step(self, val_batch, batch_idx):
        spg, tokens_ids_tensor, tokens_lentghts_tensor, label = val_batch
        spg = spg.unsqueeze(1)
        label_predict = self.model(tokens_ids_tensor,tokens_lentghts_tensor, spg)
        loss = self.criterion(label_predict, label.float())
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
