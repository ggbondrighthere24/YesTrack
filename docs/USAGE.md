# Usage Notes

This repository contains research scripts rather than a packaged command-line app. Most scripts expose their configuration through `argparse`, so paths and checkpoints should be passed explicitly.

## Main Scripts

| Script | Purpose |
| --- | --- |
| `demo.py` | Run a custom sentence on one sequence using a MOT-format detection file. |
| `refer_llm/export_chat_dataset.py` | Export cropped Yes/No training samples from Refer-KITTI. |
| `refer_llm/llm_train.py` | Train LoRA adapters for MLLM Yes/No reasoning. |
| `refer_llm/llm_eval.py` | Evaluate the MLLM referring module on dataset annotations. |
| `refer_llm/llm_eval_from_mot.py` | Score tracker outputs with the referring module, TCP, and TRP. |
| `refer_llm/llm_eval_from_mot_noise.py` | Run robustness experiments with perturbed expressions. |
| `eval_refer_kitti_mot.py` | Convert referring outputs into TrackEval runs and report HOTA/CLEAR/Identity metrics. |
| `tools/prepare_refer_kitti_motc_gt.py` | Prepare per-expression MOTChallenge-style ground truth. |
| `tools/visualize_refer_seq.py` | Visualize ground truth and predictions on a sequence. |
| `tools/visualize_mot_dir.py` | Batch-render MOT result directories to frames or videos. |

## MOT Input Format

Inference from tracker outputs expects MOT-like rows:

```text
frame,id,x,y,w,h,score
```

Frames can be 0-indexed or 1-indexed depending on the script arguments. Use `--mot_frame_start_one`, `--image_frame_start_one`, or `--frame_offset` to align your tracker output with image filenames.

## TCP and TRP Controls

`refer_llm/llm_eval_from_mot.py` exposes the temporal controls used by YesTrack:

```bash
python refer_llm/llm_eval_from_mot.py \
  --infer_every_n_frames 4 \
  --stability_enable \
  --stability_window 6 \
  --stability_thresh 0.4 \
  --stability_boost 0.3
```

`--infer_every_n_frames` controls TRP-style sparse MLLM verification. The stability options implement the TCP-style online confidence boost.

## Evaluation

Before evaluating referring results, prepare MOTChallenge-style ground truth:

```bash
python tools/prepare_refer_kitti_motc_gt.py \
  --kitti-root /path/to/refer_kitti/KITTI/training \
  --labels-with-ids-root /path/to/refer_kitti/KITTI/labels_with_ids/image_02 \
  --expression-root /path/to/refer_kitti/expression \
  --out-gt-folder outputs/refer_kitti_motc_gt \
  --seqmap-val outputs/refer_kitti_motc_gt/seqmaps/val.txt
```

Then run:

```bash
python eval_refer_kitti_mot.py \
  --refer-root track_result/yestrack \
  --gt-folder outputs/refer_kitti_motc_gt \
  --seqmap-file outputs/refer_kitti_motc_gt/seqmaps/val.txt \
  --out-dir track_result/yestrack_eval
```

`outputs/` and `track_result/` are ignored by Git because they are generated experiment artifacts.
