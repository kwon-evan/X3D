import lightning as L
import torch
from pytorchvideo.models.hub import x3d_l
from torch import nn


class X3D(L.LightningModule):
    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.x3d = x3d_l(model_num_class=num_classes)
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.x3d(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, label = batch
        logits = self.x3d(x)
        loss = self.criterion(logits, label)
        acc = (logits.argmax(1) == label.argmax(1)).float().mean()

        self.__log__(stage="train", loss=loss, acc=acc)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, label = batch
        logits = self.x3d(x)
        loss = self.criterion(logits, label)
        acc = (logits.argmax(1) == label.argmax(1)).float().mean()

        self.__log__(stage="val", loss=loss, acc=acc)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def __log__(self, stage: str, **kwargs: dict[str, torch.Tensor]) -> None:
        self.log_dict(
            {f"{stage}_{k}": v for k, v in kwargs.items()},
            on_epoch=True,
            prog_bar=True,
        )


if __name__ == "__main__":
    x3d = X3D()
    print(x3d)
    dummy = torch.randn(4, 3, 16, 312, 312)
    print(x3d(dummy).shape)
