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

        self.videos = glob(f"{self.root}/**/화재현상/**/JPG/", recursive=True)
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

        self.dataset: Dataset = FireDataset(
            root=root,
            len_clip=len_clip,
            img_size=img_size,
        )
        self.num_workers: int = num_workers
        self.batch_size: int = batch_size

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train, self.val = random_split(
                dataset=self.dataset,
                lengths=[0.8, 0.2],
            )
        if stage == "test":
            self.test = self.dataset

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

    URL = "./data/01.원천데이터/"
    dataset = FireDataset(root=URL)  # [3, len_clip, 312, 312]
    print("Dataset length:", len(dataset))
    for data in dataset:
        print(data[0].shape, data[1].argmax())  # (3, len_clip, 312, 312), (3,)

    datamodule = FireDataModule(root=URL)
    datamodule.setup(stage="fit")
    print("Train dataset length:", len(datamodule.train))
    print("Val dataset length:", len(datamodule.val))

    # code for checking images
    import os

    os.makedirs("images", exist_ok=True)
    for i, (x, label) in enumerate(dataset):
        x = x.permute(1, 0, 2, 3)  # (len_clip, 3, 312, 312)
        for j, tensor in enumerate(x):
            img = tensor.permute(1, 2, 0).numpy() * 255
            cv2.imwrite(f"images/{i}_{j}.jpg", img)
            print(f"img {i}_{j} saved")
