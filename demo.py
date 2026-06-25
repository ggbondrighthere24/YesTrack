#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo: run referring with a custom sentence on a chosen validation sequence, using a MOT-format detection file.

Outputs:
  - Annotated images: <output_dir>/images/frame_XXXXXX.jpg
  - MP4 video:        <output_dir>/video.mp4
  - TXT (all dets):   <output_dir>/predict_with_conf.txt   (writes ALL detections with their refer conf)

Detection input format (MOT-like, comma/space separated):
  frame,id,x,y,w,h,score

Notes:
  - This demo does NOT use GT; it uses your provided detections.
  - It scores each detection with the same prompt templates as `refer_llm/llm_eval_from_mot.py`.
  - TXT writes ALL detections (with conf). Visualization is threshold-filtered: only show conf >= --out-conf-thresh.
  - Requires OpenCV for MP4 writing (raises explicit error if missing).
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import deque

import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from refer_llm.llm_eval import build_model_and_processor_for_eval, get_yes_no_token_ids
from refer_llm.crop_utils import crop_with_margin
from data.refer_kitti_mot import ReferKittiMOT


def _load_mot_file(
    mot_path: str,
    *,
    frame_offset: int = 0,
) -> Dict[int, List[Tuple[int, float, float, float, float, float]]]:
    """
    Read MOT-like detections, return {frame -> [(id,x,y,w,h,score), ...]}.
    Supports comma or whitespace separated lines.
    """
    frame_to_dets: Dict[int, List[Tuple[int, float, float, float, float, float]]] = {}
    off = int(frame_offset)
    with open(mot_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = [p for p in re.split(r"[, \t]+", s) if p]
            if len(parts) < 6:
                continue
            try:
                frame = int(float(parts[0])) + off
                tid = int(float(parts[1]))
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                score = float(parts[6]) if len(parts) > 6 else 1.0
            except Exception:
                continue
            if frame < 0:
                raise ValueError(f"Negative frame after offset: raw={parts[0]} offset={off} -> {frame}")
            frame_to_dets.setdefault(frame, []).append((tid, x, y, w, h, score))
    return frame_to_dets


@torch.no_grad()
def _forward_yes_probs_for_patches(
    model: torch.nn.Module,
    processor,
    device: torch.device,
    patches: List[Image.Image],
    prompts: List[str],
    yes_id: int,
    no_id: int,
) -> List[float]:
    """Return P(Yes) for each (patch,prompt)."""
    if len(patches) != len(prompts):
        raise ValueError(f"patches/prompts length mismatch: {len(patches)} vs {len(prompts)}")
    if len(patches) == 0:
        return []

    if processor.tokenizer.padding_side != "left":
        processor.tokenizer.padding_side = "left"

    batch_messages = []
    for img, pr in zip(patches, prompts):
        if img is None:
            raise RuntimeError("Invalid patch (None).")
        batch_messages.append([{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": pr}]}])

    inputs = processor.apply_chat_template(
        batch_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )
    inputs = inputs.to(device)
    outputs = model(**inputs)
    last_logits = outputs.logits[:, -1, :]
    two_logits = last_logits[:, [yes_id, no_id]]
    probs_yes = torch.softmax(two_logits, dim=-1)[:, 0].detach().cpu().tolist()
    return [float(p) for p in probs_yes]


@torch.no_grad()
def _forward_yes_probs_for_video_samples(
    model: torch.nn.Module,
    processor,
    device: torch.device,
    frames_imgs_per_sample: List[List[Image.Image]],
    prompts: List[str],
    yes_id: int,
    no_id: int,
) -> List[float]:
    """
    Video prompt: each sample has multiple frames (images) + one text prompt.
    Returns P(Yes) per sample.
    """
    if len(frames_imgs_per_sample) != len(prompts):
        raise ValueError(f"video samples/prompts length mismatch: {len(frames_imgs_per_sample)} vs {len(prompts)}")
    if len(frames_imgs_per_sample) == 0:
        return []

    if processor.tokenizer.padding_side != "left":
        processor.tokenizer.padding_side = "left"

    batch_messages = []
    for frames, pr in zip(frames_imgs_per_sample, prompts):
        if frames is None or len(frames) == 0:
            raise RuntimeError("Video sample has 0 frames after load/crop.")
        content = [{"type": "image", "image": im} for im in frames] + [{"type": "text", "text": pr}]
        batch_messages.append([{"role": "user", "content": content}])

    inputs = processor.apply_chat_template(
        batch_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )
    inputs = inputs.to(device)
    outputs = model(**inputs)
    last_logits = outputs.logits[:, -1, :]
    two_logits = last_logits[:, [yes_id, no_id]]
    probs_yes = torch.softmax(two_logits, dim=-1)[:, 0].detach().cpu().tolist()
    return [float(p) for p in probs_yes]


def _resolve_seq_image_dir(data_root: str, seq: str) -> Path:
    # Prefer {data_root}/KITTI/training/image_02/{seq}
    p1 = Path(data_root) / "KITTI" / "training" / "image_02" / seq
    if p1.is_dir():
        return p1
    # Alternate layout: {data_root}/training/image_02/{seq}
    p2 = Path(data_root) / "training" / "image_02" / seq
    if p2.is_dir():
        return p2
    raise FileNotFoundError(f"Cannot find image dir for seq={seq}. Tried: {p1} and {p2}")


def _ensure_seq_in_val_split(data_root: str, seq: str, dataset_version: str) -> None:
    # Only used as a safety check; will raise explicit error if user picks a non-val sequence.
    if str(dataset_version) == "v1":
        ds_val = ReferKittiMOT(data_root=data_root, split="val", load_annotation=False)
    elif str(dataset_version) == "v2":
        # v2 val often includes extra sequences; keep default val split mapping unless user customized dataset.
        ds_val = ReferKittiMOT(data_root=data_root, split="val", load_annotation=False)
    else:
        raise ValueError(f"Unknown dataset_version: {dataset_version}")
    if seq not in set(ds_val.sequence_names):
        raise ValueError(f"seq={seq} is not in dataset split=val (dataset_version={dataset_version}). val_seqs={ds_val.sequence_names}")


def _get_font(font_size: int = 18) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # Try common fonts; fall back to PIL default (explicitly, not silently).
    for fp in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(fp, font_size)
        except Exception:
            continue
    return ImageFont.load_default()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seq", default="0005", help="Sequence id, e.g. 0005 / 0011 / 0013")
    p.add_argument("--sentence", default="black cars", help="User custom referring sentence")
    p.add_argument("--mot-path", default="track_result/v1best_track_result/0005/black-cars-in-the-left/predict.txt", help="MOT-format detection file path (frame,id,x,y,w,h,score)")

    p.add_argument("--data-root", type=str, default="/data/sq_2023/refer_kitti")
    p.add_argument("--dataset-version", type=str, default="v1", choices=["v1", "v2"])

    p.add_argument("--model-name", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--lora-path", type=str, default="v1bestweight/lora_step_10000")
    p.add_argument("--use-4bit", action="store_true")
    p.add_argument("--fp16", action="store_true")

    p.add_argument("--frame-offset", type=int, default=0, help="frame := frame + frame_offset for det file")
    p.add_argument("--mot-frame-start-one", type=int, default=1, choices=[0, 1], help="Detection file frames start at 1 (1) or 0 (0)")
    p.add_argument("--frame-step", type=int, default=1, help="Process every N frames")
    p.add_argument("--max-frames", type=int, default=-1, help="Max frames to process (-1 = all present in det file)")

    p.add_argument("--image-size", type=int, default=320)
    p.add_argument("--margin-ratio", type=float, default=0.2)
    p.add_argument("--margin-px", type=int, default=None)
    p.add_argument("--min-side", type=int, default=8)

    p.add_argument("--score-thresh", type=float, default=0.0, help="Filter detections with det_score < thresh (uses det file score column)")
    p.add_argument("--out-conf-thresh", type=float, default=0.3, help="Final output threshold on refer confidence (keep all conf >= thresh)")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--output-dir", type=str, default="visualize/")
    # Match llm_eval_from_mot.py defaults exactly (including punctuation)
    p.add_argument(
        "--prompt-single-tpl",
        type=str,
        default="The normalized position of the car or person in the picture is <{coord}>.Determine whether this description matches this image: {sentence}. Answer Yes or No.",
        help="Single-frame prompt template. Supports {coord} and {sentence}.",
    )
    p.add_argument(
        "--prompt-video-tpl",
        type=str,
        default="This is a short video clip of a car or person at <{coord}> across frames. The target may include motion cues; consider background and temporal context when deciding if the description matches this target: {sentence}. Answer Yes or No.",
        help="Video prompt template (used by --enable-video-mode or refine). Supports {coord} and {sentence}.",
    )
    # Match llm_eval_from_mot.py refine logic (optional)
    p.add_argument("--enable-video-mode", action="store_true", help="Use video mode for ALL detections (multi-frame prompt)")
    p.add_argument("--video-n-frames", type=int, default=4, help="How many frames to include in video prompt (uses past frames up to current)")
    p.add_argument("--disable-refine", action="store_true", help="Disable refine-in-range step (single->video)")
    p.add_argument("--re-refer-thresh", type=float, default=0.8, help="Refine upper threshold (refine if conf in [lower, thresh))")
    p.add_argument("--re-refer-lower", type=float, default=0.2, help="Refine lower threshold")
    p.add_argument("--infer-every-n-frames", type=int, default=1, help="Stride inference: reuse last conf for existing IDs on intermediate frames")
    args = p.parse_args()

    seq = str(args.seq).strip()
    if len(seq) == 0:
        raise ValueError("Empty --seq")

    # Safety: ensure seq is in validation split
    _ensure_seq_in_val_split(args.data_root, seq, args.dataset_version)

    img_dir = _resolve_seq_image_dir(args.data_root, seq)
    mot_path = Path(args.mot_path)
    if not mot_path.is_file():
        raise FileNotFoundError(f"--mot-path not found: {mot_path}")

    out_dir = Path(args.output_dir)
    img_out_dir = out_dir / "images"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    video_out_path = out_dir / "video.mp4"

    # Load detections
    frame_to_dets = _load_mot_file(str(mot_path), frame_offset=int(args.frame_offset))
    if len(frame_to_dets) == 0:
        raise RuntimeError(f"No detections loaded from: {mot_path}")

    # Build id -> trajectory for video mode/refine: tid -> [(frame,(x,y,w,h))]
    id_to_traj: Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]] = {}
    for f, dets in frame_to_dets.items():
        for tid, x, y, w, h, _ in dets:
            id_to_traj.setdefault(int(tid), []).append((int(f), (x, y, w, h)))
    for tid in id_to_traj:
        id_to_traj[tid].sort(key=lambda z: z[0])

    # Setup model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, processor, _ = build_model_and_processor_for_eval(
        model_name=args.model_name,
        use_4bit=bool(args.use_4bit),
        bf16=not bool(args.fp16),
        lora_path=args.lora_path,
    )
    model.to(device)
    model.eval()
    yes_id, no_id = get_yes_no_token_ids(processor)

    # OpenCV for MP4 output (explicit requirement)
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "OpenCV (cv2) is required to write MP4 in this demo. "
            "Please install opencv-python or opencv-python-headless."
        ) from e

    font = _get_font(18)

    frames_sorted = sorted(frame_to_dets.keys())
    if int(args.frame_step) > 1:
        frames_sorted = [f for i, f in enumerate(frames_sorted) if (i % int(args.frame_step) == 0)]
    if int(args.max_frames) > 0:
        frames_sorted = frames_sorted[: int(args.max_frames)]

    video_writer = None
    wrote_any = False
    out_dir.mkdir(parents=True, exist_ok=True)
    out_txt_path = out_dir / "predict_with_conf.txt"
    out_lines: List[str] = []

    # For infer stride (match llm_eval_from_mot behavior)
    last_conf_by_tid: Dict[int, float] = {}
    infer_stride = max(1, int(getattr(args, "infer_every_n_frames", 1) or 1))

    for idx, frame in enumerate(frames_sorted):
        dets = frame_to_dets.get(frame, [])
        # Filter low det-score
        dets = [d for d in dets if float(d[5]) >= float(args.score_thresh)]
        frame_counter = idx
        do_full_infer = (infer_stride <= 1) or (frame_counter % infer_stride == 0)

        # Map det frame -> image index
        # MOT files are often 1-based; KITTI images are 0-based filenames.
        f_img = int(frame) - 1 if int(args.mot_frame_start_one) == 1 else int(frame)
        img_path = img_dir / f"{f_img:06d}.png"
        if not img_path.is_file():
            raise FileNotFoundError(f"Missing image for frame={frame} -> {img_path}")

        with Image.open(img_path) as im0:
            im = im0.convert("RGB")
        W, H = im.size

        # If no detections in this frame, still render header-only frame.
        boxes_xywh: List[Tuple[float, float, float, float]] = []
        tids: List[int] = []
        det_scores: List[float] = []
        for tid, x, y, w, h, s_det in dets:
            if w <= 0 or h <= 0:
                continue
            boxes_xywh.append((x, y, w, h))
            tids.append(int(tid))
            det_scores.append(float(s_det))

        probs_yes: List[float] = [0.0 for _ in boxes_xywh]

        # Inference selection for stride reuse
        infer_indices: List[int] = []
        if do_full_infer:
            infer_indices = list(range(len(boxes_xywh)))
        else:
            for i, tid in enumerate(tids):
                if tid in last_conf_by_tid:
                    probs_yes[i] = float(last_conf_by_tid[tid])
                else:
                    infer_indices.append(i)

        # Run inference on selected indices
        if len(infer_indices) > 0:
            if bool(getattr(args, "enable_video_mode", False)):
                # Video mode for all selected detections
                frames_imgs_per_sample: List[List[Image.Image]] = []
                prompts_video: List[str] = []
                for i in infer_indices:
                    tid = tids[i]
                    x, y, w, h = boxes_xywh[i]
                    # Build recent trajectory up to current frame
                    traj = id_to_traj.get(tid, [])
                    idx_in_traj = None
                    for k, (f_tr, _bb) in enumerate(traj):
                        if int(f_tr) == int(frame):
                            idx_in_traj = k
                            break
                    # Collect up to N frames ending at current
                    k0 = 0 if idx_in_traj is None else max(0, idx_in_traj - int(args.video_n_frames) + 1)
                    traj_slice = traj[(k0 if idx_in_traj is not None else 0) : (idx_in_traj + 1 if idx_in_traj is not None else 1)]
                    if len(traj_slice) == 0:
                        traj_slice = [(int(frame), (x, y, w, h))]

                    frames_imgs: List[Image.Image] = []
                    for f_tr, (xt, yt, wt, ht) in traj_slice:
                        f_img_t = int(f_tr) - 1 if int(args.mot_frame_start_one) == 1 else int(f_tr)
                        img_path_t = img_dir / f"{f_img_t:06d}.png"
                        if not img_path_t.is_file():
                            raise FileNotFoundError(f"Missing image for video frame={f_tr} -> {img_path_t}")
                        with Image.open(img_path_t) as imt0:
                            imt = imt0.convert("RGB")
                        patch_t = crop_with_margin(
                            image=imt,
                            bbox_xywh=(xt, yt, wt, ht),
                            margin_ratio=float(args.margin_ratio),
                            margin_px=(int(args.margin_px) if args.margin_px is not None else None),
                            min_side=int(args.min_side),
                        )
                        if patch_t is None:
                            raise RuntimeError(f"crop_with_margin returned None in video mode: frame={f_tr} tid={tid}")
                        if args.image_size is not None:
                            patch_t = patch_t.resize((int(args.image_size), int(args.image_size)))
                        frames_imgs.append(patch_t)

                    cx = x + 0.5 * w
                    cy = y + 0.5 * h
                    nx = max(0.0, min(1.0, cx / float(W)))
                    ny = max(0.0, min(1.0, cy / float(H)))
                    coord_str = f"{nx:.3f} {ny:.3f}"
                    pr_v = str(args.prompt_video_tpl).format(coord=coord_str, sentence=str(args.sentence))
                    frames_imgs_per_sample.append(frames_imgs)
                    prompts_video.append(pr_v)

                probs_new = _forward_yes_probs_for_video_samples(
                    model=model,
                    processor=processor,
                    device=device,
                    frames_imgs_per_sample=frames_imgs_per_sample,
                    prompts=prompts_video,
                    yes_id=yes_id,
                    no_id=no_id,
                )
                if len(probs_new) != len(infer_indices):
                    raise RuntimeError("Video inference output length mismatch.")
                for k, det_i in enumerate(infer_indices):
                    probs_yes[det_i] = float(probs_new[k])
                    last_conf_by_tid[tids[det_i]] = float(probs_yes[det_i])
            else:
                # Single-frame mode (like llm_eval_from_mot single prompt)
                patches: List[Image.Image] = []
                prompts: List[str] = []
                for i in infer_indices:
                    x, y, w, h = boxes_xywh[i]
                    patch = crop_with_margin(
                        image=im,
                        bbox_xywh=(x, y, w, h),
                        margin_ratio=float(args.margin_ratio),
                        margin_px=(int(args.margin_px) if args.margin_px is not None else None),
                        min_side=int(args.min_side),
                    )
                    if patch is None:
                        raise RuntimeError(f"crop_with_margin returned None: seq={seq} frame={frame} tid={tids[i]} bbox={(x,y,w,h)}")
                    if args.image_size is not None:
                        patch = patch.resize((int(args.image_size), int(args.image_size)))
                    cx = x + 0.5 * w
                    cy = y + 0.5 * h
                    nx = max(0.0, min(1.0, cx / float(W)))
                    ny = max(0.0, min(1.0, cy / float(H)))
                    coord_str = f"{nx:.3f} {ny:.3f}"
                    pr = str(args.prompt_single_tpl).format(coord=coord_str, sentence=str(args.sentence))
                    patches.append(patch)
                    prompts.append(pr)

                probs_new = _forward_yes_probs_for_patches(
                    model=model,
                    processor=processor,
                    device=device,
                    patches=patches,
                    prompts=prompts,
                    yes_id=yes_id,
                    no_id=no_id,
                )
                if len(probs_new) != len(infer_indices):
                    raise RuntimeError("Single inference output length mismatch.")
                for k, det_i in enumerate(infer_indices):
                    probs_yes[det_i] = float(probs_new[k])
                    last_conf_by_tid[tids[det_i]] = float(probs_yes[det_i])

        # Optional refine step: for conf in [lower, upper), run video prompt to re-score (like llm_eval_from_mot)
        if (not bool(getattr(args, "disable_refine", False))) and (not bool(getattr(args, "enable_video_mode", False))):
            lower = float(getattr(args, "re_refer_lower", 0.0))
            upper = float(getattr(args, "re_refer_thresh", 1.0))
            if not (upper > lower):
                raise ValueError(f"re_refer_thresh({upper}) must be > re_refer_lower({lower})")
            refine_indices = [i for i, p0 in enumerate(probs_yes) if (p0 >= lower and p0 < upper)]
            if len(refine_indices) > 0:
                frames_imgs_per_sample: List[List[Image.Image]] = []
                prompts_video: List[str] = []
                for i in refine_indices:
                    tid = tids[i]
                    x, y, w, h = boxes_xywh[i]
                    traj = id_to_traj.get(tid, [])
                    idx_in_traj = None
                    for k, (f_tr, _bb) in enumerate(traj):
                        if int(f_tr) == int(frame):
                            idx_in_traj = k
                            break
                    k0 = 0 if idx_in_traj is None else max(0, idx_in_traj - int(args.video_n_frames) + 1)
                    traj_slice = traj[(k0 if idx_in_traj is not None else 0) : (idx_in_traj + 1 if idx_in_traj is not None else 1)]
                    if len(traj_slice) == 0:
                        traj_slice = [(int(frame), (x, y, w, h))]

                    frames_imgs: List[Image.Image] = []
                    for f_tr, (xt, yt, wt, ht) in traj_slice:
                        f_img_t = int(f_tr) - 1 if int(args.mot_frame_start_one) == 1 else int(f_tr)
                        img_path_t = img_dir / f"{f_img_t:06d}.png"
                        if not img_path_t.is_file():
                            raise FileNotFoundError(f"Missing image for refine frame={f_tr} -> {img_path_t}")
                        with Image.open(img_path_t) as imt0:
                            imt = imt0.convert("RGB")
                        patch_t = crop_with_margin(
                            image=imt,
                            bbox_xywh=(xt, yt, wt, ht),
                            margin_ratio=float(args.margin_ratio),
                            margin_px=(int(args.margin_px) if args.margin_px is not None else None),
                            min_side=int(args.min_side),
                        )
                        if patch_t is None:
                            raise RuntimeError(f"crop_with_margin returned None in refine: frame={f_tr} tid={tid}")
                        if args.image_size is not None:
                            patch_t = patch_t.resize((int(args.image_size), int(args.image_size)))
                        frames_imgs.append(patch_t)

                    cx = x + 0.5 * w
                    cy = y + 0.5 * h
                    nx = max(0.0, min(1.0, cx / float(W)))
                    ny = max(0.0, min(1.0, cy / float(H)))
                    coord_str = f"{nx:.3f} {ny:.3f}"
                    pr_v = str(args.prompt_video_tpl).format(coord=coord_str, sentence=str(args.sentence))
                    frames_imgs_per_sample.append(frames_imgs)
                    prompts_video.append(pr_v)

                probs_refined = _forward_yes_probs_for_video_samples(
                    model=model,
                    processor=processor,
                    device=device,
                    frames_imgs_per_sample=frames_imgs_per_sample,
                    prompts=prompts_video,
                    yes_id=yes_id,
                    no_id=no_id,
                )
                if len(probs_refined) != len(refine_indices):
                    raise RuntimeError("Refine output length mismatch.")
                for k, det_i in enumerate(refine_indices):
                    probs_yes[det_i] = float(probs_refined[k])
                    last_conf_by_tid[tids[det_i]] = float(probs_yes[det_i])

        # Visualization threshold filter: show ONLY detections with conf >= out_conf_thresh
        keep = [i for i, p0 in enumerate(probs_yes) if (p0 >= float(args.out_conf_thresh))]
        # No "best" highlighting: all kept boxes use the same color.

        # Draw
        vis = im.copy()
        draw = ImageDraw.Draw(vis)
        # No header text; only draw bounding boxes on the original image.

        # Write ALL detections to txt (unfiltered)
        for i, ((x, y, w, h), p_yes, tid) in enumerate(zip(boxes_xywh, probs_yes, tids)):
            out_lines.append(f"{int(frame)},{int(tid)},{int(x)},{int(y)},{int(w)},{int(h)},{float(p_yes):.6f}")

        # Draw ONLY kept detections (same color, no best)
        for i in keep:
            x, y, w, h = boxes_xywh[i]
            p_yes = float(probs_yes[i])
            tid = int(tids[i])
            x1 = int(round(x))
            y1 = int(round(y))
            x2 = int(round(x + w))
            y2 = int(round(y + h))
            color = (255, 0, 0)  #统一使用红色框
            width = 3
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

        out_img_path = img_out_dir / f"frame_{idx:06d}.jpg"
        vis.save(out_img_path, quality=95)
        wrote_any = True

        # Init video writer using first frame size
        if video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(str(video_out_path), fourcc, float(args.fps), (W, H))
            if not video_writer.isOpened():
                raise RuntimeError(f"Failed to open VideoWriter for: {video_out_path}")

        # Write frame to video (convert RGB->BGR)
        frame_bgr = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)

    if video_writer is not None:
        video_writer.release()

    if not wrote_any:
        raise RuntimeError("No frames were written (maybe all frames filtered out).")

    # Write filtered text output
    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))

    print(f"[DONE] images_dir={img_out_dir}")
    print(f"[DONE] video={video_out_path}")
    print(f"[DONE] txt_all_dets={out_txt_path} (lines={len(out_lines)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())