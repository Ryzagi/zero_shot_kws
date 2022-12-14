from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from zerospot.data.data import WavDataClass, collate_fn
from zerospot.train.engine import LightningEngine

PATH_TO_TRAIN = Path(__file__).parent.parent / "data" / "train.csv"
PATH_TO_TEST = Path(__file__).parent.parent / "data" / "test.csv"
MODEL_PATH = Path(__file__).parent / "checkpoints"


def main():
    train_ds = WavDataClass(PATH_TO_TRAIN)
    test_ds = WavDataClass(PATH_TO_TEST)
    train_dataloader = DataLoader(
        train_ds,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )
    test_dataloader = DataLoader(
        test_ds,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    checkpoint_callback = ModelCheckpoint(dirpath=MODEL_PATH,
                                          filename='BcResNetModel_{epoch}-{val_loss:.2f}',
                                          monitor='val_loss', mode='min', save_top_k=3)

    logger = TensorBoardLogger(MODEL_PATH, name="logs")

    trainer = pl.Trainer(callbacks=[checkpoint_callback], logger=logger, gpus=1)

    model = LightningEngine(lr_rate=1e-3)

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=test_dataloader,
    )


if __name__ == "__main__":
    main()
