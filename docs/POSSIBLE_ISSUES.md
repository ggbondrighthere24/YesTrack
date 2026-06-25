# Possible Issues

Check this document first when a script fails, evaluation covers fewer expressions than expected, or the measured performance is below the paper result.

## Frame Index Alignment

Frame alignment should be the first diagnostic step. Refer-KITTI image files are commonly named from `000000.png`, MOT files are often 1-based, expression JSON keys may be 0-based or 1-based, and MOTChallenge GT is written as 1-based. A one-frame shift can keep the pipeline running while pairing boxes with the wrong images and significantly reducing performance.

### Inference Mapping

`refer_llm/llm_eval_from_mot.py` first applies `--frame_offset` to the MOT frame. For `results_root` input, the image index is then:

```text
image_index = raw_mot_frame + frame_offset - mot_frame_start_one + image_frame_start_one
```

`mot_frame_start_one` and `image_frame_start_one` are integer flags with values `0` or `1`.

| MOT first frame | First image | Recommended arguments |
| --- | --- | --- |
| `1` | `000000.png` | `--mot_frame_start_one 1 --image_frame_start_one 0 --frame_offset 0` |
| `0` | `000000.png` | `--mot_frame_start_one 0 --image_frame_start_one 0 --frame_offset 0` |
| `1` | `000001.png` | `--mot_frame_start_one 1 --image_frame_start_one 1 --frame_offset 0` |
| `0` | `000001.png` | `--mot_frame_start_one 0 --image_frame_start_one 1 --frame_offset 0` |

For `bytetrack` input, the script treats tracker frames as 0-based and does not subtract `mot_frame_start_one`. `--frame_offset` is still applied, and `--image_frame_start_one 1` adds one to the image filename index.

### GT and Metric Mapping

`tools/prepare_refer_kitti_motc_gt.py` always writes MOTChallenge GT frames as 1-based. Its `--label-key-mode` controls how expression JSON frame keys are read:

- `strict0`: JSON labels use keys `0, 1, 2, ...`.
- `strict1`: JSON labels use keys `1, 2, 3, ...`.
- `auto`: infer the convention from the presence of key `0` or key `1`; explicit `strict0` or `strict1` is safer when the convention is known.

`eval_refer_kitti_mot.py --offset` shifts prediction frames before TrackEval. With the default 1-based `results_root` input, `llm_eval_from_mot.py` keeps predictions 1-based; its ByteTrack path also converts output to 1-based. Both normally use `--offset 0`. If a 0-based `results_root` input is used, its output remains 0-based and metric evaluation needs `--offset 1`.

### Symptoms of a One-Frame Shift

- The pipeline completes, but HOTA, DetA, or IDF1 is unexpectedly low.
- Boxes appear on the correct object in adjacent frames rather than the current frame.
- The first or last frame is missing, duplicated, or reported as an unavailable image.
- Per-ID confidence changes do not match visible target changes.
- Results differ substantially after changing only an indexing flag.

### Visual Check

Overlay GT and predictions for a short frame range:

```bash
python tools/visualize_refer_seq.py \
  --gt outputs/refer_kitti_motc_gt/SEQUENCE__EXPRESSION/gt/gt.txt \
  --pred track_result/yestrack/SEQUENCE/EXPRESSION/predict_with_conf.txt \
  --image-root /path/to/refer_kitti/KITTI/training/image_02 \
  --seq-name SEQUENCE__EXPRESSION \
  --start-frame 1 \
  --end-frame 20 \
  --output-root visualize/alignment_check
```

The visualizer automatically shifts predictions that start at frame `0`. To override the decision, pass `--pred-frame-offset 0` for 1-based predictions or `--pred-frame-offset 1` for 0-based predictions.

Inspect the first several frames and at least one long trajectory. Confirm that the box, track ID, expression label, and image all refer to the same instant.

### Example AI Prompt

Use the following prompt with an AI coding assistant. Attach several source images and visualization outputs when possible.

```text
I am debugging a possible 0-based/1-based frame alignment issue in YesTrack.

Repository entry point: refer_llm/llm_eval_from_mot.py
Input mode: <results_root or bytetrack>
First image filenames: <for example, 000000.png, 000001.png, ...>
First five MOT rows: <paste rows here>
Expression JSON frame keys: <paste several keys here>
Current arguments:
  --mot_frame_start_one <0 or 1>
  --image_frame_start_one <0 or 1>
  --frame_offset <integer>
Metric argument:
  --offset <integer>

Please:
1. Trace the frame-index conversion in the inference, GT-generation, and metric scripts.
2. Build a table mapping the first five raw MOT frames to image filenames and GT frames.
3. Identify any off-by-one mismatch.
4. Recommend the exact argument values to use; do not change box coordinates or track IDs.
5. If visualizations are attached, check whether each box belongs to the displayed frame or an adjacent frame.
```

## Generate Evaluation GT

`tools/prepare_refer_kitti_motc_gt.py` generates the per-expression MOTChallenge-style GT and seqmap required by `eval_refer_kitti_mot.py`. Use `--dataset-version v1` for Refer-KITTI and `--dataset-version v2` for Refer-KITTI-V2; the script applies the corresponding train/validation sequence split.

### V1 Seqmap Protocol

The released Refer-KITTI V1 GT is converted directly from the dataset annotations. Its seqmap contains **158 evaluation entries**, which is more than the reduced seqmaps used by TransRMOT or TempRMOT. In our tests, the released 158-entry protocol is more challenging; evaluating YesTrack with the reduced seqmaps raises HOTA by approximately **0.5–1.0 points**.

Before investigating an apparent metric mismatch, confirm that every method was evaluated with exactly the same GT and seqmap. Results produced with different seqmaps are not directly comparable, even when all other inference and metric arguments are unchanged. Paper-aligned YesTrack evaluation should use the released 158-entry V1 seqmap.

For Refer-KITTI-V2, change the dataset paths, output folder, and version together:

```bash
python tools/prepare_refer_kitti_motc_gt.py \
  --kitti-root /path/to/refer_kitti_v2/KITTI/training \
  --labels-with-ids-root /path/to/refer_kitti_v2/KITTI/labels_with_ids/image_02 \
  --expression-root /path/to/refer_kitti_v2/expression \
  --out-gt-folder outputs/refer_kitti_motc_gt_v2 \
  --dataset-version v2 \
  --split val \
  --seqmap-val outputs/refer_kitti_motc_gt_v2/seqmaps/val.txt \
  --label-key-mode strict0
```

## Refer-KITTI-V2 Names Containing Commas

Some Refer-KITTI-V2 expression names contain commas. An unquoted seqmap treats those commas as CSV delimiters, and result directories whose normalized names omit commas no longer match the GT names. This can silently reduce the number of evaluated expressions.

New seqmaps generated by `prepare_refer_kitti_motc_gt.py` are CSV-safe. For existing inference results whose directory or file names lost commas, preview the repair plan first:

```bash
python tools/restore_commas_in_refer.py \
  --refer-root track_result/yestrack_v2 \
  --gt-root outputs/refer_kitti_motc_gt_v2 \
  --seqmap-file outputs/refer_kitti_motc_gt_v2/seqmaps/val.txt \
  --structure nested
```

Review the matched, ambiguous, unmatched, and conflict counts. Apply the renames only after the dry run looks correct:

```bash
python tools/restore_commas_in_refer.py \
  --refer-root track_result/yestrack_v2 \
  --gt-root outputs/refer_kitti_motc_gt_v2 \
  --seqmap-file outputs/refer_kitti_motc_gt_v2/seqmaps/val.txt \
  --structure nested \
  --commit
```

Run `eval_refer_kitti_mot.py` after the repaired result names match the GT seqmap.

## Environment Snapshot

The checked-in `requirements.txt` contains CUDA-specific packages and platform-specific `file://` entries. Review or replace those entries before installing the environment on another machine.
