import os
from glob import glob

import albumentations as A
import cv2
import lightning as L
import numpy as np
import numpy.typing as npt
import torch
from albumentations.pytorch import ToTensorV2
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class FireDataset(Dataset):
    def __init__(
        self,
        roots: str | list[str],
        stage: str,
        len_clip: int = 16,
        img_size: int = 312,
        num_classes: int = 3,
    ) -> None:
        super().__init__()

        if isinstance(roots, str):
            roots = [roots]

        self.roots: list[str] = roots
        self.videos: list[str] = []
        self.len_clip: int = len_clip
        self.img_size: int = img_size
        self.num_classes: int = num_classes

        for root in self.roots:
            print("Getting videos from", root)
            videos = glob(
                f"{root}/**/{stage}/**/JPG/",
                recursive=True,
            )
            print("Got", len(videos), "Videos")
            self.videos.extend(videos)
        self.clips = self.get_clips()
        self.labels = self.get_labels()

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
        clip: npt.NDArray[np.object_] = self.clips[idx]

        x = torch.stack([self.path2tensor(p) for p in clip])  # (len_clip, 3, 312, 312)
        x = x.permute(1, 0, 2, 3)  # (3, len_clip, 312, 312)

        label = torch.zeros(3, dtype=torch.float)  # (3,)
        label[self.labels[idx]] = 1.0

        return x, label  # (3, len_clip, 312, 312), (3,)

    def get_clips(self) -> npt.NDArray[np.object_]:
        ret = []
        video_dict = {
            k: sorted(os.listdir(k)) for k in self.videos
        }  # video_path: [frame1, frame2, ...]

        for k, v in video_dict.items():
            # make len(frames) to be multiple of len_clip
            video_frames = v[: len(v) // self.len_clip * self.len_clip]
            # make filename to full path
            video_frames = [os.path.join(k, frame) for frame in video_frames]
            # take clips of len_clip from each video
            clips = [
                video_frames[i : i + self.len_clip]
                for i in range(0, len(video_frames), self.len_clip)
            ]
            ret.extend(clips)

        return np.array(ret)

    def get_labels(self) -> npt.NDArray[np.int_]:
        def get_label(clip: str) -> int:
            try:
                filename = clip[0].split("/")[-1]
                filename, _ = filename.split(".")
                _, label, _, _ = filename.split("_")
                if label == "NONE":
                    return 0
                elif label == "SM":
                    return 1
                elif label == "FL":
                    return 2
                else:
                    raise ValueError(f"Unexpected label: {label}")
            except Exception as e:
                print(f"Error processing label for clip {clip}: {str(e)}")
                return -1  # Return a sentinel value for error cases

        return np.array([get_label(clip) for clip in self.clips])

    def path2tensor(self, path: str) -> Tensor:
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

    def imgs2clips(
        self,
        imgs: npt.NDArray[np.object_],  # (n)
    ) -> npt.NDArray[np.object_]:  # (n // len_clip, len_clip)
        num_full_clips = len(imgs) // self.len_clip
        selected_imgs = imgs[: num_full_clips * self.len_clip]
        clips = selected_imgs.reshape(-1, self.len_clip)
        return clips


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

        self.root: str = root
        self.len_clip: int = len_clip
        self.img_size: int = img_size
        self.num_workers: int = num_workers
        self.batch_size: int = batch_size

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train: Dataset = FireDataset(
                roots=self.root,
                stage="Training",
                len_clip=self.len_clip,
                img_size=self.img_size,
            )
            self.val: Dataset = FireDataset(
                roots=self.root,
                stage="Validation",
                len_clip=self.len_clip,
                img_size=self.img_size,
            )
            print("Train dataset length:", len(self.train))
            print("Val dataset length:", len(self.val))

        if stage == "test":
            self.test: Dataset = FireDataset(
                roots=self.root,
                stage="Validation",
                len_clip=self.len_clip,
                img_size=self.img_size,
            )
            print("Test dataset length:", len(self.test))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
        )


if __name__ == "__main__":
    from rich import print

    URL = "./data/"
    dataset = FireDataset(
        roots=URL, stage="Training", len_clip=16
    )  # [3, len_clip, 312, 312]
    print("Dataset length:", len(dataset))

    datamodule = FireDataModule(root=URL, batch_size=2)
    datamodule.setup(stage="fit")

    for batch in datamodule.train_dataloader():
        data, lable = batch
        print(data.shape, lable)
        break
