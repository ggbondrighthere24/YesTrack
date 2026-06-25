# Copyright (c) Ruopeng Gao. All Rights Reserved.

import os
import glob
import torch
from collections import defaultdict

from .one_dataset import OneDataset
from .util import is_legal, append_annotation


class KittiMOT(OneDataset):
    def __init__(
            self,
            data_root: str = "./datasets/",
            sub_dir: str = "KITTI/training",
            split: str = "train",
            load_annotation: bool = True,
    ):
        super(KittiMOT, self).__init__(
            data_root=data_root,
            sub_dir=sub_dir,
            split=split,
            load_annotation=load_annotation,
        )

        # Prepare the data according to KITTI Tracking layout
        # Expected under self.data_dir:
        #   image_02/<seq>/<frame:06d>.png  (0-indexed)
        #   label_02/<seq>.txt              (0-indexed frame ids)
        self.sequence_names = self._get_sequence_names_for_split()
        self.sequence_infos = self._get_sequence_infos()
        self.image_paths = self._get_image_paths()
        if self.load_annotation:
            self.annotations = self._get_annotations()
        return

    def _get_all_sequence_names(self):
        image_root = os.path.join(self.data_dir, "image_02")
        if not os.path.isdir(image_root):
            raise FileNotFoundError(f"image_02 not found under {self.data_dir}")
        names = sorted([d for d in os.listdir(image_root) if os.path.isdir(os.path.join(image_root, d))])
        return names

    def _get_sequence_names_for_split(self):
        all_names = self._get_all_sequence_names()
        # Define split mapping by required sequence indices
        train_ids = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20]
        val_ids = [5, 11, 13]
        split_map = {
            "train": [f"{i:04d}" for i in train_ids],
            "val": [f"{i:04d}" for i in val_ids],
        }
        if self.split not in split_map:
            # If an unknown split is requested, default to using all sequences
            return all_names
        target_names = set(split_map[self.split])
        # Intersect with available names to be robust
        names = sorted([n for n in all_names if n in target_names])
        return names

    def _get_sequence_infos(self):
        sequence_infos = dict()
        for seq in self.sequence_names:
            seq_dir = os.path.join(self.data_dir, "image_02", seq)
            # count frames by files in image dir (0-indexed .png)
            frame_files = sorted(glob.glob(os.path.join(seq_dir, "*.png")))
            length = len(frame_files)
            # infer width/height from first frame if available, else fallback to KITTI default 1242x375
            width, height = 1242, 375
            if length > 0:
                try:
                    from PIL import Image
                    with Image.open(frame_files[0]) as im:
                        width, height = im.size
                except Exception:
                    pass
            sequence_infos[seq] = {
                "width": int(width),
                "height": int(height),
                "length": int(length),
                "is_static": False,
            }
        return sequence_infos

    def _get_image_paths(self):
        image_paths = defaultdict(list)
        for seq in self.sequence_names:
            seq_dir = os.path.join(self.data_dir, "image_02", seq)
            length = self.sequence_infos[seq]["length"]
            # Frames are 0-indexed and 6-digit zero padded
            for i in range(length):
                image_paths[seq].append(os.path.join(seq_dir, f"{i:06d}.png"))
        return image_paths

    def _init_annotations(self):
        annotations = dict()
        for seq in self.sequence_names:
            length = self.sequence_infos[seq]["length"]
            annotations[seq] = []
            for _ in range(length):
                annotations[seq].append({
                    "id": torch.zeros((0,), dtype=torch.int64),
                    "category": torch.zeros((0,), dtype=torch.int64),
                    "bbox": torch.zeros((0, 4), dtype=torch.float32),
                    "visibility": torch.zeros((0,), dtype=torch.float32),
                })
        return annotations

    def _get_annotations(self):
        annotations = self._init_annotations()
        label_root = os.path.join(self.data_dir, "label_02")
        for seq in self.sequence_names:
            label_file = os.path.join(label_root, f"{seq}.txt")
            if not os.path.isfile(label_file):
                # allow missing labels (e.g., test split); keep empty annotations
                continue
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 17:
                        # malformed line
                        continue
                    frame_id = int(parts[0])  # 0-indexed
                    track_id = int(parts[1])
                    obj_type = parts[2]
                    # Skip DontCare
                    if obj_type == "DontCare":
                        continue
                    # bbox as x1,y1,x2,y2 to x,y,w,h
                    x1 = float(parts[6]); y1 = float(parts[7]); x2 = float(parts[8]); y2 = float(parts[9])
                    w = max(0.0, x2 - x1)
                    h = max(0.0, y2 - y1)
                    bbox = [x1, y1, w, h]
                    # category unified to 0, visibility unified to 1
                    category = 0
                    visibility = 1.0
                    # Append to 0-indexed frame slot
                    if 0 <= frame_id < len(annotations[seq]):
                        annotations[seq][frame_id] = append_annotation(
                            annotation=annotations[seq][frame_id],
                            obj_id=track_id,
                            category=category,
                            bbox=bbox,
                            visibility=visibility,
                        )
            # Determine legality for each frame
            for i in range(self.sequence_infos[seq]["length"]):
                annotations[seq][i]["is_legal"] = is_legal(annotations[seq][i])
        return annotations


