import lightning as L
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from pytorchvideo.models.hub import x3d_l
from torch import nn
from torchmetrics import Accuracy, F1Score


class X3D(L.LightningModule):
    def __init__(
        self,
        num_class: int = 3,
        pretrained: bool = True,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.lr = lr

        self.x3d = x3d_l(
            model_num_class=400,
            pretrained=pretrained,
        )
        self.x3d.blocks[-1].proj = nn.Linear(2048, num_class)

        self.loss = nn.CrossEntropyLoss()
        self.f1 = F1Score(task="multiclass", num_classes=num_class, average="macro")
        self.acc = Accuracy(task="multiclass", num_classes=num_class, average="macro")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.x3d(x)

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        x, label = batch
        logits = self.x3d(x)
        loss = self.loss(logits, label)

        label_idx = torch.argmax(label, dim=1)

        acc = self.acc(logits, label_idx)
        f1 = self.f1(logits, label_idx)

        self.__log__(stage="train", loss=loss, acc=acc, f1=f1)
        return loss

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        x, label = batch
        logits = self.x3d(x)
        loss = self.loss(logits, label)

        label_idx = torch.argmax(label, dim=1)

        acc = self.acc(logits, label_idx)
        f1 = self.f1(logits, label_idx)

        self.__log__(stage="val", loss=loss, acc=acc, f1=f1)
        return loss

    def configure_optimizers(
        self,
    ) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=1e-5,
            max_lr=1e-3,
            gamma=0.85,
            mode="exp_range",
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
            },
        }

    def __log__(self, stage: str, **kwargs: dict[str, torch.Tensor]) -> None:
        self.log_dict(
            {f"{stage}_{k}": v for k, v in kwargs.items()},
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )


if __name__ == "__main__":
    x3d = X3D(num_class=10)
    print(x3d)
    dummy = torch.randn(4, 3, 16, 312, 312)
    print(x3d(dummy).shape)
