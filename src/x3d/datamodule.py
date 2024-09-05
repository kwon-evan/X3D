from glob import glob

import albumentations as A
import cv2
import lightning as L
import numpy as np
import numpy.typing as npt
import torch
from albumentations.pytorch import ToTensorV2
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split


class FireDataset(Dataset):
    def __init__(
        self,
        root: str,
        len_clip: int = 16,
        img_size: int = 312,
        num_classes: int = 3,
    ) -> None:
        super().__init__()

        self.root: str = root
        self.len_clip: int = len_clip
        self.img_size: int = img_size
        self.num_classes: int = num_classes

        imgs: npt.NDArray[np.object_] = np.array(
            sorted(glob(f"{self.root}/**/화재현상/**/*.jpg", recursive=True))
        )  # (n)
        self.clips: npt.NDArray[np.object_] = self.imgs2clips(
            imgs
        )  # (n // len_clip, len_clip)
        self.labels: npt.NDArray[np.int_] = self.get_labels()  # (n // len_clip)

        self.transform = A.Compose(
            [
                A.Resize(img_size, img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        def path2tensor(path: str) -> Tensor:
            try:
                img = cv2.imread(path)
                if img is None:
                    raise FileNotFoundError(f"Could not read image file: {path}")
                img = self.transform(image=img)["image"]
                return img
            except Exception as e:
                print(f"Error processing image {path}: {str(e)}")
                # Return a blank image tensor as a fallback
                return torch.zeros((3, 312, 312), dtype=torch.float32)

        clip: npt.NDArray[np.object_] = self.clips[idx]
        x = torch.stack([path2tensor(p) for p in clip])  # (len_clip, 3, 312, 312)
        x = x.permute(1, 0, 2, 3)  # (3, len_clip, 312, 312)

        label = torch.zeros(3, dtype=torch.float)  # (3,)
        label[self.labels[idx]] = 1.0

        return x, label  # (3, len_clip, 312, 312), (3,)

    def imgs2clips(
        self,
        imgs: npt.NDArray[np.object_],  # (n)
    ) -> npt.NDArray[np.object_]:  # (n // len_clip, len_clip)
        num_full_clips = len(imgs) // self.len_clip
        selected_imgs = imgs[: num_full_clips * self.len_clip]
        clips = selected_imgs.reshape(-1, self.len_clip)
        return clips

    def get_labels(self) -> npt.NDArray[np.int_]:
        def get_label(clip: str) -> int:
            try:
                filename = clip[0].split("/")[-1]
                filename, _ = filename.split(".")
                _, label, _, _ = filename.split("_")
                if label == "NONE":
                    return 0
                elif label == "FL":
                    return 1
                elif label == "SM":
                    return 2
                else:
                    raise ValueError(f"Unexpected label: {label}")
            except Exception as e:
                print(f"Error processing label for clip {clip}: {str(e)}")
                return -1  # Return a sentinel value for error cases

        return np.array([get_label(clip) for clip in self.clips])


class FireDataModule(L.LightningDataModule):
    def __init__(
        self,
        root: str,
        len_clip: int = 16,
        img_size: int = 312,
        num_workers: int = 8,
        batch_size: int = 8,
    ):
        super().__init__()

        self.dataset: Dataset = FireDataset(
            root=root,
            len_clip=len_clip,
            img_size=img_size,
        )
        self.num_workers: int = num_workers
        self.batch_size: int = batch_size

    def setup(self, stage):
        if stage == "fit":
            self.train, self.val = random_split(
                dataset=self.dataset,
                lengths=[0.8, 0.2],
            )
        if stage == "test":
            self.test = self.dataset

    def train_dataloader(self):
        return DataLoader(
            self.train,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
        )


if __name__ == "__main__":
    URL = "./data/01.원천데이터/"
    dataset = FireDataset(root=URL)  # [3, len_clip, 312, 312]
    print("Dataset length:", len(dataset))
    # for data in dataset:
    #     x, label = data
    #     print(x.shape, label)

    datamodule = FireDataModule(root=URL)
    datamodule.setup(stage="fit")
