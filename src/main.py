import lightning as L
import torch
from lightning.pytorch.callbacks import (
    RichProgressBar,
)
from x3d import X3D, FireDataModule

URL = "./data/01.원천데이터/"
torch.set_float32_matmul_precision("highest")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x3d: L.LightningModule = X3D()
    dm: L.LightningDataModule = FireDataModule(
        root=URL,
        num_workers=8,
        batch_size=8,
    )
    trainer: L.Trainer = L.Trainer(
        precision="16-mixed",
        callbacks=[RichProgressBar()],
    )
    trainer.fit(x3d, dm)
