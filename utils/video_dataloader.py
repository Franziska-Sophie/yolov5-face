import os
import cv2
import random
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np


class VideoYOLODataset(Dataset):
    """
    Custom dataset that loads video frames dynamically and reads YOLO-format annotations.

    Expected structure:
        cabin-pre-annotations/
        ├── cabin_footage/
        │   ├── video1.mp4
        │   └── video2.mp4
        └── annotations/
            ├── classes.txt
            ├── yolo_train/
            │   ├── video1/
            │   │   ├── video1_frame_00001.txt
            │   │   ├── video1_frame_00002.txt
            │   │   └── ...
            │   └── video2/
            │       ├── video2_frame_00001.txt
            │       └── ...
            └── yolo_test/
                ├── ...
    """

    def __init__(
        self,
        video_root,
        label_root,
        img_size=640,
        frame_skip=1,
        sample_frames=None,
        transform=None,
    ):
        self.video_root = Path(video_root)
        self.label_root = Path(label_root)
        self.img_size = img_size
        self.frame_skip = frame_skip
        self.sample_frames = sample_frames
        self.transform = transform

        self.index = self._build_index()

        if self.sample_frames:
            self.index = random.sample(
                self.index, min(self.sample_frames, len(self.index))
            )

        self.labels = []
        self.shapes = []
        for _, _, label_file in self.index:
            if os.path.exists(label_file):
                with open(label_file, "r") as f:
                    lines = f.readlines()
                frame_labels = []
                for line in lines:
                    parts = list(map(float, line.strip().split()))

                    # If only bbox (5 values: class, x, y, w, h), pad with 10 dummy landmark coords
                    if len(parts) == 5:
                        parts += [0.0] * 10  # add x1 y1 ... x5 y5 = zeros

                    frame_labels.append(parts)

                frame_labels = np.array(frame_labels, dtype=np.float32)
                self.labels.append(frame_labels)
            else:
                self.labels.append(np.zeros((0, 15), dtype=np.float32))  # no labels
            self.shapes.append(
                (self.img_size, self.img_size)
            )  # fake shape, scaled to img_size

        self.shapes = np.array(self.shapes, dtype=np.float32)

        print(f"✅ Found {len(self.index)} labeled frames in {self.video_root}")

    def _build_index(self):
        """Build (video_path, frame_id, label_path) triplets."""
        index = []
        video_files = sorted(
            [
                f
                for f in self.video_root.glob("*.*")
                if f.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]
            ]
        )

        for video_path in video_files:
            video_name = video_path.stem
            label_folder = self.label_root / "labels" / video_name
            if not label_folder.exists():
                print(f"⚠️ Warning: no label folder found at {label_folder}")
                continue

            frame_labels = sorted(
                label_folder.glob(
                    f"{video_name}_{video_path.suffix.lstrip('.')}_frame_*.txt"
                )
            )
            for label_file in frame_labels:
                frame_str = label_file.stem.split("_frame_")[-1]
                if not frame_str.isdigit():
                    continue
                frame_id = int(frame_str)
                if frame_id % self.frame_skip:
                    index.append((video_path, frame_id, label_file))
        return index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        video_path, frame_id, label_path = self.index[idx]
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError(f"❌ Could not read frame {frame_id} from {video_path}")

        # Resize and normalize
        frame = cv2.resize(frame, (self.img_size, self.img_size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.transpose(2, 0, 1) / 255.0
        frame = torch.tensor(frame, dtype=torch.float32)

        # Load YOLO-format labels (class x_center y_center width height)
        targets = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()

            frame_labels = []
            for line in lines:
                parts = list(map(float, line.strip().split()))

                # If only bbox (5 values: class, x, y, w, h), pad with 10 dummy landmark coords
                if len(parts) == 5:
                    parts += [0.0] * 10  # add x1 y1 ... x5 y5 = zeros

                frame_labels.append(parts)

            frame_labels = np.array(frame_labels, dtype=np.float32)
            self.labels.append(frame_labels)
        # if os.path.exists(label_path):
        #     with open(label_path, "r") as f:
        #         for line in f.readlines():
        #             parts = [float(x) for x in line.strip().split()]
        #             targets.append(parts)
        targets = (
            torch.tensor(targets, dtype=torch.float32)
            if len(targets) > 0
            else torch.zeros((0, 15))
        )

        # Optionally apply transforms
        if self.transform:
            frame, targets = self.transform(frame, targets)

        return frame, targets


def yolo_video_collate(batch):
    """
    Collate function for VideoYOLODataset compatible with YOLOv5 train.py.
    Returns imgs, targets, paths, shapes.
    """
    imgs, targets, paths, shapes = [], [], [], []

    for i, (img, target) in enumerate(batch):
        imgs.append(img)
        paths.append(f"video_frame_{i}")  # arbitrary frame identifier
        shapes.append(torch.tensor(img.shape[1:], dtype=torch.float32))  # H,W

        if target.numel() > 0:
            b = torch.full((target.shape[0], 1), i)  # batch index
            targets.append(torch.cat((b, target), dim=1))

    imgs = torch.stack(imgs, 0)
    if targets:
        targets = torch.cat(targets, 0)
    else:
        targets = torch.zeros((0, 16))

    shapes = torch.stack(shapes, 0)  # [B,2]

    return imgs, targets, paths, shapes


def create_video_yolo_dataloader(
    video_root,
    label_root,
    imgsz=640,
    batch_size=16,
    workers=4,
    frame_skip=1,
    sample_frames=None,
    shuffle=True,
):
    dataset = VideoYOLODataset(
        video_root=video_root,
        label_root=label_root,
        img_size=imgsz,
        frame_skip=frame_skip,
        sample_frames=sample_frames,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        collate_fn=yolo_video_collate,
    )
