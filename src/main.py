import argparse

import lightning as L
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichProgressBar,
    StochasticWeightAveraging,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.tuner.tuning import Tuner

from x3d import X3D, FireDataModule

URL = "./data/01.원천데이터/"
torch.set_float32_matmul_precision("high")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--stage",
        type=str,
        default="train",
        help="train | valid | test | predict",
    )
    parser.add_argument(
        "-d",
        "--data_path",
        type=str,
        default=URL,
        nargs="+",
        help="data path",
    )
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("-n", "--num_workers", type=int, default=8, help="num workers")
    parser.add_argument(
        "--devices",
        type=str,
        default="auto",
        help="number of devices. ex) 1, 2 or [0, 2]",
    )
    parser.add_argument(
        "-p",
        "--profile",
        default=False,
        action="store_true",
        help="profile model",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    x3d: L.LightningModule = X3D()
    dm: L.LightningDataModule = FireDataModule(
        root=args.data_path,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )

    if args.profile:
        print("[INFO] Profiling Model...")
        L.Trainer(
            profiler="simple",
            max_epochs=2,
            precision="16-mixed",
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        ).fit(x3d, dm)
        exit()

    trainer: L.Trainer = L.Trainer(
        logger=WandbLogger(project="x3d"),
        devices=args.devices,
        accelerator="auto",
        precision="16-mixed",
        callbacks=[
            RichProgressBar(),
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=5,
                verbose=True,
            ),
            ModelCheckpoint(
                dirpath="./checkpoints/",
                filename="x3d-{epoch:02d}-{val_acc:.2f}",
                save_top_k=5,
                verbose=True,
                monitor="val_loss",
                mode="min",
            ),
            LearningRateMonitor(logging_interval="step"),
            StochasticWeightAveraging(swa_lrs=1e-2),
        ],
    )
    tuner: Tuner = Tuner(trainer)
    tuner.lr_find(x3d, dm)

    if args.stage == "train":
        trainer.fit(x3d, dm)
    elif args.stage == "valid":
        trainer.validate(x3d, dm)
    elif args.stage == "test":
        trainer.test(x3d, dm)
    elif args.stage == "predict":
        trainer.predict(x3d, dm)
