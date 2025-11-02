import os
import cv2
import random
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import tqdm
from multiprocessing.pool import ThreadPool

NUM_THREADS = min(8, os.cpu_count() or 4)


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
        cache_images=True,
        hyp=None,
    ):
        self.video_root = Path(video_root)
        self.label_root = Path(label_root)
        self.img_size = img_size
        self.frame_skip = frame_skip
        self.sample_frames = sample_frames
        self.transform = transform
        self.cache_images = cache_images
        self.hyp = hyp

        print(video_root, label_root)

        # Build index: mapping video -> [(video_path, frame_id, label_path), ...]
        self.video_index, self.index = self._build_indexes()

        print(f"⚙️ Preloading frames from {len(self.video_index)} videos into RAM...")

        # Parallel video loading
        with ThreadPool(NUM_THREADS) as pool:
            results = list(
                tqdm.tqdm(
                    pool.imap_unordered(
                        lambda i_v: self._load_video(
                            i_v[1], position=(i_v[0] % 20 + 1)
                        ),
                        enumerate(self.video_index.items()),
                    ),
                    total=len(self.video_index),
                    desc="🎬 Preloading videos",
                    ncols=100,
                )
            )

        # Flatten results
        self.frames, self.labels, self.shapes = [], [], []
        for frames, labels in results:
            if not frames:
                continue
            self.frames.extend(frames)
            self.labels.extend(labels)
            self.shapes.extend([(self.img_size, self.img_size)] * len(frames))

        self.shapes = np.array(self.shapes, dtype=np.float32)

        # Global shuffle for training randomness
        combined = list(zip(self.frames, self.labels, self.shapes))
        random.shuffle(combined)
        self.frames, self.labels, self.shapes = zip(*combined)
        self.frames = list(self.frames)
        self.labels = list(self.labels)
        self.shapes = np.array(self.shapes, dtype=np.float32)

        total_labels = sum(len(lbl) for lbl in self.labels)
        print(
            f"✅ Cached {len(self.index)} frames with {total_labels} bounding boxes from {len(self.video_index)} videos using {NUM_THREADS} threads."
        )

    # ----------------------------
    #   VIDEO LOADING FUNCTION
    # ----------------------------
    def _load_video(self, video_item, position=0):
        """
        Load all frames from one video sequentially, with a per-video progress bar.
        """
        video_path, frame_items = video_item
        frames, labels = [], []

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Could not open {video_path}")
            return [], []

        # Sort frames for sequential access
        frame_items.sort(key=lambda x: x[1])

        total_frames = len(frame_items)
        video_name = Path(video_path).name

        # tqdm progress bar for this video
        pbar = tqdm.tqdm(
            total=total_frames,
            desc=f"🎞️ Loading {video_name}",
            position=position,  # ensures unique row per video when run in parallel
            leave=False,  # keep console clean after completion
            ncols=100,
            dynamic_ncols=True,
        )

        for _, frame_id, label_file in frame_items:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if not ret:
                pbar.update(1)
                continue

            # Resize + normalize
            h0, w0 = frame.shape[:2]
            r = self.img_size / max(h0, w0)
            if r <= 1:
                frame = cv2.resize(
                    frame, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_AREA
                )
            else:
                raise ValueError(
                    f"The requested image size ({self.img_size}) is larger than the original image size {(w0, h0)}."
                )

            h_resized, w_resized = frame.shape[:2]

            frame, (dw, dh) = letterbox(
                frame, self.img_size
            )  # add padding to make image square
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            fliplr_random = random.random()
            if fliplr_random < self.hyp["fliplr"]:  # flip left right
                frame = np.fliplr(frame).copy()
            frame = frame.transpose(2, 0, 1)  # / 255.0
            frame = torch.tensor(frame, dtype=torch.float32)

            if self.cache_images:
                frames.append(frame)

            # Load labels
            if label_file.exists():
                with open(label_file, "r") as f:
                    lines = f.readlines()
                frame_labels = []
                for line in lines:
                    parts = list(map(float, line.strip().split()))
                    c, xc, yc, w, h = parts[:5]
                    xc = (xc * w_resized + dw) / self.img_size
                    yc = (yc * h_resized + dh) / self.img_size
                    w = w * w_resized / self.img_size
                    h = h * h_resized / self.img_size
                    parts[:5] = [c, xc, yc, w, h]
                    if len(parts) == 5:
                        parts += [
                            xc,
                            yc,
                            xc,
                            yc,
                            xc,
                            yc,
                            xc,
                            yc,
                            xc,
                            yc,
                        ]  # add dummy landmarks
                    frame_labels.append(parts)
                nL = len(frame_labels)  # number of frame_labels
                frame_labels = np.array(frame_labels, dtype=np.float32)
                if frame_labels.ndim == 1:
                    frame_labels = frame_labels.reshape(-1, 15)
                if nL and fliplr_random < self.hyp["fliplr"]:
                    frame_labels[:, 1] = 1 - frame_labels[:, 1]
            else:
                frame_labels = np.zeros((0, 15), dtype=np.float32)

            labels.append(frame_labels)
            pbar.update(1)

        pbar.close()
        cap.release()
        return frames, labels

    # ----------------------------
    #   INDEX BUILDER
    # ----------------------------
    def _build_indexes(self):
        """Build a mapping from each video file to its frame list."""
        video_index = {}
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
                if frame_id % self.frame_skip == 0:
                    video_index.setdefault(video_path, []).append(
                        (video_path, frame_id, label_file)
                    )
                    index.append((video_path, frame_id, label_file))

        return video_index, index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        """
        Retrieve one preloaded frame and its corresponding labels.
        Frames and labels are already preprocessed and normalized in __init__.
        """
        # Get preloaded frame and corresponding label array
        frame = self.frames[idx]  # torch.Tensor [3, H, W]
        frame_labels = self.labels[idx]  # np.ndarray [N, 15]

        # Convert labels to tensor
        targets = torch.tensor(frame_labels, dtype=torch.float32)

        # Apply optional transforms (e.g., augmentations)
        if self.transform:
            frame, targets = self.transform(frame, targets)

        # if frame_labels.shape[0] > 0:
        #     img = np.array(frame)
        #     img = img.transpose(1, 2, 0)  # * 255.0
        #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #     h, w = img.shape[:2]
        #     for lbl in frame_labels:
        #         _, xc, yc, bw, bh = lbl[:5]
        #         x1 = int((xc - bw / 2) * w)
        #         y1 = int((yc - bh / 2) * h)
        #         x2 = int((xc + bw / 2) * w)
        #         y2 = int((yc + bh / 2) * h)
        #         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        #         landmarks = lbl[5:].reshape(-1, 2)
        #         for lx, ly in landmarks:
        #             lx = int(lx * w)
        #             ly = int(ly * h)
        #             cv2.circle(img, (lx, ly), 2, (0, 0, 255), -1)

        #     save_dir = Path("debug_frames")
        #     save_dir.mkdir(exist_ok=True)
        #     save_path = save_dir / f"frame{idx}.png"
        #     cv2.imwrite(str(save_path), img)
        #     print(f"🖼️ Saved debug frame with boxes → {save_path}")

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

        # Get current shape (H, W)
        h, w = img.shape[1], img.shape[2]

        # Fake original and resized shapes (YOLO expects a tuple of tuples)
        shapes.append(((h, w), (h, w)))

        if target.numel() > 0:
            b = torch.full((target.shape[0], 1), i)  # batch index
            targets.append(torch.cat((b, target), dim=1))

    imgs = torch.stack(imgs, 0)

    if targets:
        targets = torch.cat(targets, 0)
    else:
        targets = torch.zeros((0, 16))

    return imgs, targets, paths, shapes


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Compute padding
    dh, dw = new_shape[0] - shape[0], new_shape[1] - shape[1]  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return img, (dw, dh)


def create_video_yolo_dataloader(
    video_root,
    label_root,
    imgsz=640,
    batch_size=16,
    workers=4,
    frame_skip=1,
    sample_frames=None,
    shuffle=True,
    cache_images=False,
    hyp=None,
):
    dataset = VideoYOLODataset(
        video_root=video_root,
        label_root=label_root,
        img_size=imgsz,
        frame_skip=frame_skip,
        sample_frames=sample_frames,
        cache_images=cache_images,
        hyp=hyp,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        collate_fn=yolo_video_collate,
    )
