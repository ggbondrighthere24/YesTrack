# Usage

YesTrack is a research codebase built around three main entry points:

| Stage | Entry point |
| --- | --- |
| Training | `refer_llm/llm_train.py` |
| Referring inference | `refer_llm/llm_eval_from_mot.py` |
| Tracking metrics | `eval_refer_kitti_mot.py` |

The examples below use explicit paths where they matter. Adjust dataset, checkpoint, tracker-result, and output paths for your environment.

If a command fails or the measured performance is lower than expected, check [`POSSIBLE_ISSUES.md`](./POSSIBLE_ISSUES.md) first. Frame-index alignment is the most important item to verify.

## Data and Environment

The checked-in `requirements.txt` records the validated environment. It includes CUDA-specific packages and platform-specific `file://` entries, so review or replace those entries before installing it on another machine.

Prepare Refer-KITTI following [`DATA.md`](./DATA.md). The main commands expect a dataset root with `KITTI/`, `expression/`, and `KITTI/labels_with_ids/` beneath it.

Tracker inputs use MOT-style rows:

```text
frame,id,x,y,w,h,score
```

Use `--mot_frame_start_one`, `--image_frame_start_one`, or `--frame_offset` when tracker frame indices do not match the image filenames.

## Released Resources

The trained Refer-KITTI V1/V2 LoRA weights, TempRMOT* pure tracking results, and pre-generated evaluation GT can be downloaded from [Baidu Netdisk](https://pan.baidu.com/s/1nwhcEIQshWk9TnNjhiYMrQ) with extraction code `z5kt`.

```text
Yestrack/
├── refer_kitti_motc_gt/
├── refer_kitti_motc_gt_v2/
├── v1best_track_result/
├── v1bestweight/
├── v2best_track_result/
└── v2bestweight/
```

- `refer_kitti_motc_gt` and `refer_kitti_motc_gt_v2` contain MOTChallenge-format evaluation GT for Refer-KITTI V1 and V2, respectively. Pass the matching folder to `eval_refer_kitti_mot.py --gt-folder`.
- `v1bestweight` and `v2bestweight` contain the LoRA adapters trained for Refer-KITTI V1 and V2, respectively.
- `v1best_track_result` and `v2best_track_result` contain the corresponding TempRMOT* candidate tracks. TempRMOT* is the paper's pure-tracking setting obtained by training TempRMOT without its text module.
- The tracking-result folders are inputs to YesTrack's referring stage, not final `predict_with_conf.txt` outputs.
- Keep GT, weights, tracker results, and the dataset on the same V1 or V2 protocol.

> [!IMPORTANT]
> The released Refer-KITTI V1 GT is converted directly from the dataset annotations and uses a seqmap with **158 evaluation entries**. It includes more evaluation tasks than the reduced TransRMOT/TempRMOT seqmaps and was more challenging in our tests. Running YesTrack on those reduced seqmaps typically produces HOTA scores approximately **0.5–1.0 points higher**. Results from different seqmaps are not directly comparable, so record the exact seqmap used for every reported result.

> [!NOTE]
> The released V2 weight was trained with negative-sample downsampling. Although this had little effect on overall performance in our tests, it changed score calibration. The best classification threshold is approximately `0.4` for the V1 weight and `0.6` for the V2 weight; validate around these values on the matching evaluation protocol.

Set `--lora_path` to the extracted adapter directory that contains `adapter_config.json`. If needed, locate it with:

```bash
find /path/to/v1bestweight -name adapter_config.json
```

Set `--results_root` to `/path/to/v1best_track_result` for V1 or `/path/to/v2best_track_result` for V2. These folders already use the nested `<sequence>/<expression>/predict.txt` structure expected by `mot_input_type=results_root`.

Use `/path/to/refer_kitti_motc_gt` as `--gt-folder` for V1 and `/path/to/refer_kitti_motc_gt_v2` for V2. You can alternatively regenerate either GT version with `tools/prepare_refer_kitti_motc_gt.py`.

## Training

Single-frame training is the default:

```bash
torchrun --nproc_per_node=2 refer_llm/llm_train.py \
  --data_root /path/to/refer_kitti \
  --dataset_version v1
```

Train with both single-frame and video-clip samples:

```bash
torchrun --nproc_per_node=2 refer_llm/llm_train.py \
  --data_root /path/to/refer_kitti \
  --dataset_version v1 \
  --enable_video_mode \
  --train_both_modes \
  --video_n_frames 3
```

Train with video clips only:

```bash
torchrun --nproc_per_node=2 refer_llm/llm_train.py \
  --data_root /path/to/refer_kitti \
  --dataset_version v1 \
  --enable_video_mode \
  --video_only \
  --video_n_frames 3
```

Training mode behavior:

- No video flags: use only single-frame samples.
- `--enable_video_mode --train_both_modes`: concatenate the single-frame and video datasets.
- `--enable_video_mode --video_only`: use only video samples.
- `--enable_video_mode` alone still trains on the single-frame dataset.
- `--train_both_modes` only takes effect together with `--enable_video_mode`.

Each run is stored under `runs/<timestamp>/`. LoRA checkpoints are written to `runs/<timestamp>/train/`, evaluation artifacts to `runs/<timestamp>/eval/`, and the resolved configuration to `runs/<timestamp>/config.json`.

Training-time evaluation is configured separately from training sample selection. Selective video refinement is controlled by `--re_refer_lower`, `--re_refer_thresh`, and `--prompt_video_tpl`, not by the training-side `--enable_video_mode` flag.

## Referring Inference

For `results_root` input, arrange tracker outputs as:

```text
/path/to/tracker_results/
└── <sequence>/
    └── <expression>/
        └── predict.txt
```

Run the validated TCP and TRP configuration:

```bash
python refer_llm/llm_eval_from_mot.py \
  --data_root /path/to/refer_kitti \
  --lora_path runs/<timestamp>/train/lora_step_<step> \
  --mot_input_type results_root \
  --results_root /path/to/tracker_results \
  --mot_filename predict.txt \
  --output_root track_result/yestrack \
  --stability_enable \
  --stability_window 6 \
  --stability_thresh 0.4 \
  --stability_boost 0.3 \
  --infer_every_n_frames 4
```

The output for each task is written to:

```text
track_result/yestrack/<sequence>/<expression>/predict_with_conf.txt
```

### TCP

- `--stability_enable` enables Temporal Confidence Prior.
- `--stability_window 6` stores the latest 6 final confidence scores for each track ID.
- `--stability_thresh 0.4` requires every score in the history window to be at least `0.4`.
- `--stability_boost 0.3` adds `0.3` to the current score when the history is stable, clipping the result to `1.0`.
- TCP only reads previous frames. Refined, propagated, and boosted output scores become history for later frames.

### TRP

`--infer_every_n_frames 4` performs a full inference pass every 4 processed frames. Intermediate frames reuse the latest confidence for each existing track ID, while a new ID is inferred immediately. Set the value to `1` to run full inference on every frame.

### Video Modes

Default inference uses a single-frame first stage and selectively re-evaluates scores in `[re_refer_lower, re_refer_thresh)` with a historical multi-frame clip. The defaults are `[0.2, 0.8)`.

Use full video-mode inference when every candidate should be scored from a multi-frame clip:

```bash
python refer_llm/llm_eval_from_mot.py \
  --data_root /path/to/refer_kitti \
  --lora_path /path/to/lora_checkpoint \
  --results_root /path/to/tracker_results \
  --output_root track_result/yestrack_video \
  --enable_video_mode \
  --video_n_frames 4 \
  --stability_enable \
  --infer_every_n_frames 4
```

In inference, `--enable_video_mode` switches the complete primary inference path to multi-frame inputs. This differs from training, where the flag only makes video samples available and must be combined with `--train_both_modes` or `--video_only` to change the training dataset.

Use `--disable_refine` for single-frame inference without selective video refinement.

## Tracking Metrics

First prepare per-expression MOTChallenge-style ground truth:

```bash
python tools/prepare_refer_kitti_motc_gt.py \
  --kitti-root /path/to/refer_kitti/KITTI/training \
  --labels-with-ids-root /path/to/refer_kitti/KITTI/labels_with_ids/image_02 \
  --expression-root /path/to/refer_kitti/expression \
  --out-gt-folder outputs/refer_kitti_motc_gt \
  --dataset-version v1 \
  --split val \
  --seqmap-val outputs/refer_kitti_motc_gt/seqmaps/val.txt \
  --label-key-mode strict0
```

Then evaluate the nested output generated by `llm_eval_from_mot.py`:

```bash
python eval_refer_kitti_mot.py \
  --refer-root track_result/yestrack \
  --refer-structure nested \
  --gt-folder outputs/refer_kitti_motc_gt \
  --seqmap-file outputs/refer_kitti_motc_gt/seqmaps/val.txt \
  --out-dir track_result/yestrack_eval \
  --thresholds 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7
```

The evaluator reports HOTA, DetA, AssA, MOTA, IDF1, and related metrics for each referring-confidence threshold.

For Refer-KITTI V1, the released seqmap contains 158 evaluation entries. Use that same seqmap for paper-aligned evaluation and method comparisons; changing to the reduced TransRMOT/TempRMOT seqmap may increase YesTrack's HOTA by approximately 0.5–1.0 points because it evaluates fewer, easier tasks.

## Demo

Use `demo.py` to run a custom expression on one validation sequence and render the result:

```bash
python demo.py \
  --seq 0005 \
  --sentence "black cars" \
  --mot-path /path/to/predict.txt \
  --data-root /path/to/refer_kitti \
  --lora-path /path/to/lora_checkpoint \
  --output-dir visualize/demo
```

Generated `runs/`, `outputs/`, `track_result/`, and visualization files are ignored by Git.
