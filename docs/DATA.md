# Data Notes

The main dataset loader is `data/refer_kitti_mot.py`. It expects a Refer-KITTI style directory and uses 0-indexed, six-digit frame names.

## Refer-KITTI Layout

```text
/path/to/refer_kitti/
├── KITTI/
│   ├── training/
│   │   ├── image_02/
│   │   │   ├── 0005/
│   │   │   │   ├── 000000.png
│   │   │   │   └── ...
│   │   │   └── ...
│   │   └── label_02/
│   │       ├── 0005.txt
│   │       └── ...
│   └── labels_with_ids/
│       └── image_02/
│           ├── 0005/
│           │   ├── 000000.txt
│           │   └── ...
│           └── ...
└── expression/
    ├── 0005/
    │   ├── query_001.json
    │   └── ...
    └── ...
```

## `labels_with_ids` Format

Each frame-level file should contain normalized boxes:

```text
cls_id track_id x y w h
```

`x`, `y`, `w`, and `h` are normalized to `[0, 1]`, with `x` and `y` representing the top-left corner.

## Expression JSON Format

Each expression file should contain:

```json
{
  "sentence": "black cars in the left lane",
  "label": {
    "0": [1, 4],
    "1": [1, 4]
  }
}
```

The `label` keys are 0-indexed frame ids. Values are the object ids referred to by the sentence in that frame.

## Default Splits

The built-in KITTI split mapping is:

```text
train: 0001, 0002, 0003, 0004, 0006, 0007, 0008, 0009, 0010, 0012, 0014, 0015, 0016, 0018, 0020
val:   0005, 0011, 0013
```

Refer-KITTI-V2 can be selected with `--dataset_version v2`; some scripts also adjust the data root automatically when their old lab defaults are used. For reproducibility, prefer passing `--data_root` explicitly.
