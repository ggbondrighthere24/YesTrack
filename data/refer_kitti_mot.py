# Copyright (c) Ruopeng Gao. All Rights Reserved.

import os
import glob
import json
from collections import defaultdict

import torch

from .one_dataset import OneDataset
from .util import is_legal, append_annotation


class ReferKittiMOT(OneDataset):
    def __init__(
            self,
            data_root: str = "./datasets/",
            # sub_dir: str = "DanceTrack/training",
            sub_dir: str = "KITTI/training",
            split: str = "train",
            load_annotation: bool = True,
            expression_sub_dir: str = "expression",
            labels_with_ids_sub_dir: str = "labels_with_ids/image_02",
            # 可选：覆盖默认的训练/验证序列划分（使用 KITTI 序列 ID 列表）
            train_ids_override: list | None = None,
            val_ids_override: list | None = None,
    ):
        super(ReferKittiMOT, self).__init__(
            data_root=data_root,
            sub_dir=sub_dir,
            split=split,
            load_annotation=load_annotation,
        )
        # Also record the root dir for optional 'labels_with_ids'
        self._root_dir = os.path.dirname(self.data_dir)
        # 自定义划分覆盖（若提供）
        self._train_ids_override = train_ids_override
        self._val_ids_override = val_ids_override

        # Expected layout under data_root:
        #   KITTI/training/image_02/<seq>/<frame:06d>.png
        #   KITTI/training/label_02/<seq>.txt
        #   expression/<seq>/*.json    (each json has keys: 'sentence', 'label': {frame_id(str): [obj_ids...]})
        # Required:
        #   KITTI/labels_with_ids/<seq>/<frame:06d>.txt  (each line: cls id x y w h) normalized to [0,1]

        self.sequence_names = self._get_sequence_names_for_split()
        self.sequence_infos = self._get_sequence_infos()
        self.image_paths = self._get_image_paths()
        self.expression_root = os.path.join(data_root, expression_sub_dir)
        self.labels_with_ids_sub_dir = labels_with_ids_sub_dir
        if self.load_annotation:
            # Base annotations follow KittiMOT, with extra fields for referring:
            self.annotations = self._get_annotations_with_referring()
        return

    def _get_all_sequence_names(self):
        image_root = os.path.join(self.data_dir, "image_02")
        if not os.path.isdir(image_root):
            raise FileNotFoundError(f"image_02 not found under {self.data_dir}")
        names = sorted([d for d in os.listdir(image_root) if os.path.isdir(os.path.join(image_root, d))])
        return names
#不加referdance的版本
    def _get_sequence_names_for_split(self):
        all_names = self._get_all_sequence_names()
        # Follow KittiMOT split mapping
        # v1（默认）
        train_ids = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20]
        val_ids = [5, 11, 13]
        # 若外部提供覆盖，则使用覆盖（例如 v2：包含 0/17/19）
        if isinstance(self._train_ids_override, list) and len(self._train_ids_override) > 0:
            train_ids = [int(i) for i in self._train_ids_override]
        if isinstance(self._val_ids_override, list) and len(self._val_ids_override) > 0:
            val_ids = [int(i) for i in self._val_ids_override]
        split_map = {
            "train": [f"{i:04d}" for i in train_ids],
            "val": [f"{i:04d}" for i in val_ids],
        }
        if self.split not in split_map:
            return all_names
        target_names = set(split_map[self.split])
        names = sorted([n for n in all_names if n in target_names])
        return names


#加referdance的版本
    # def _get_sequence_names_for_split(self):
    #     all_names = self._get_all_sequence_names()

    #     # 若外部直接提供字符串序列名，优先使用
    #     if self.split == "train" and self._train_ids_override is not None:
    #         target = set(str(n) for n in self._train_ids_override)
    #         return sorted([n for n in all_names if n in target])
    #     if self.split == "val" and self._val_ids_override is not None:
    #         target = set(str(n) for n in self._val_ids_override)
    #         return sorted([n for n in all_names if n in target])

    #     # fallback：KITTI 默认数字划分
    #     train_ids = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20]
    #     val_ids = [5, 11, 13]
    #     split_map = {
    #         "train": [f"{i:04d}" for i in train_ids],
    #         "val":   [f"{i:04d}" for i in val_ids],
    #     }
    #     if self.split not in split_map:
    #         return all_names
    #     target = set(split_map[self.split])
    #     return sorted([n for n in all_names if n in target])




    def _get_sequence_infos(self):
        sequence_infos = dict()
        for seq in self.sequence_names:
            seq_dir = os.path.join(self.data_dir, "image_02", seq)
            frame_files = sorted(glob.glob(os.path.join(seq_dir, "*.png")))
            length = len(frame_files)
            width, height = 1242, 375
            if length > 0:
                # try:
                from PIL import Image
                with Image.open(frame_files[0]) as im:
                    width, height = im.size
                # except Exception:
                #     pass
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
                    "bbox": torch.zeros((0, 4), dtype=torch.float32),  # xywh
                    "visibility": torch.zeros((0,), dtype=torch.float32),
                    # Extra fields for referring:
                    # Per-frame candidate texts and mapping to referred IDs (per text)
                    "refer_texts": [],
                    "refer_ids_per_text": [],  # List[List[int]] aligned with refer_texts
                })
        return annotations

    def _load_expressions_for_sequence(self, seq: str):
        seq_expr_dir = os.path.join(self.expression_root, seq)
        if not os.path.isdir(seq_expr_dir):
            return []
        json_files = sorted(glob.glob(os.path.join(seq_expr_dir, "*.json")))
        expressions = []
        for jf in json_files:
            # try:
            with open(jf, "r") as f:
                data = json.load(f)
            sentence = data.get("sentence", "")
            labels = data.get("label", {})  # {frame_id(str): [ids...]}
            expressions.append({"sentence": sentence, "label": labels})
            # except Exception:
                # Skip malformed
                # continue
        return expressions

    def _get_annotations_with_referring(self):
        annotations = self._init_annotations()
        for seq in self.sequence_names:
            lwi_seq_dir = os.path.join(self._root_dir, self.labels_with_ids_sub_dir, seq)
            if not os.path.isdir(lwi_seq_dir):
                raise FileNotFoundError(f"labels_with_ids sequence dir not found: {lwi_seq_dir}")
            # Load per-frame labels from KITTI/labels_with_ids
            width = self.sequence_infos[seq]["width"]
            height = self.sequence_infos[seq]["height"]
            num_frames = self.sequence_infos[seq]["length"]
            for frame_id in range(num_frames):
                label_file = os.path.join(lwi_seq_dir, f"{frame_id:06d}.txt")
                if not os.path.isfile(label_file):
                    continue
                with open(label_file, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 6:
                            raise ValueError(f"Invalid line: expected >=6 fields, got {len(parts)}. Content: {parts}")
                        # Format: cls id x y w h (normalized [0,1], x,y are top-left)
                        # try:
                        track_id = int(float(parts[1]))
                        nx = float(parts[2]); ny = float(parts[3]); nw = float(parts[4]); nh = float(parts[5])
                        # except Exception:
                        #     continue
                        x = nx * width
                        y = ny * height
                        w = max(0.0, nw * width)
                        h = max(0.0, nh * height)
                        bbox = [x, y, w, h]
                        category = 0
                        visibility = 1.0
                        annotations[seq][frame_id] = append_annotation(
                            annotation=annotations[seq][frame_id],
                            obj_id=track_id,
                            category=category,
                            bbox=bbox,
                            visibility=visibility,
                        )
            # Attach legality flags
            for i in range(self.sequence_infos[seq]["length"]):
                annotations[seq][i]["is_legal"] = is_legal(annotations[seq][i])

            # Load expressions and align to frames by ids
            expressions = self._load_expressions_for_sequence(seq)
            if len(expressions) == 0:
                continue
            # Prepare per-frame referring maps
            for frame_idx in range(self.sequence_infos[seq]["length"]):
                refer_texts = []
                refer_ids_per_text = []
                # Build lookups for this frame
                for expr in expressions:
                    sentence = expr.get("sentence", "")
                    label_map = expr.get("label", {})
                    # support both 0-indexed and 1-indexed keys
                    key_0 = str(frame_idx)
                    # key_1 = str(frame_idx + 1)
                    ids = label_map.get(key_0, [])  # 只认准 0-indexed
                    try:
                        ids = [int(i) for i in ids]
                    except Exception:
                        raise ValueError(f"Invalid ids: {ids}")
                    refer_texts.append(sentence)
                    refer_ids_per_text.append(ids)
                annotations[seq][frame_idx]["refer_texts"] = refer_texts
                annotations[seq][frame_idx]["refer_ids_per_text"] = refer_ids_per_text
        return annotations


