import lightning as L
import torch
from pytorchvideo.models.hub import x3d_l
from torch import nn
from torchmetrics import F1Score, Accuracy


class X3D(L.LightningModule):
    def __init__(
        self,
        num_class: int = 3,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        self.x3d = x3d_l(
            model_num_class=400,
            pretrained=pretrained,
        )
        self.x3d.blocks[-1].proj = nn.Linear(2048, num_class)

        self.loss = nn.BCEWithLogitsLoss()
        self.f1 = F1Score(task="multilabel", num_labels=num_class, average="macro")
        self.acc = Accuracy(task="multilabel", num_labels=num_class, average="macro")

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

        pred = (logits > 0.5).float()
        acc = self.acc(pred, label)
        f1 = self.f1(pred, label)

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

        pred = (logits > 0.5).float()
        acc = self.acc(pred, label)
        f1 = self.f1(pred, label)

        self.__log__(stage="val", loss=loss, acc=acc, f1=f1)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=1e-3)

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
