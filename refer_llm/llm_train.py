from __future__ import annotations


import argparse
import json
import os
import sys
import random
from typing import Optional

import numpy as np

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

"""确保可以从项目根目录导入兄弟包（例如 data）"""
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from refer_llm.llm_eval import evaluate as external_evaluate
from refer_llm.llm_eval import evaluate_all_sequences as external_evaluate_all
from refer_llm.llm_eval import evaluate_all_sequences_multi_thresholds as external_evaluate_all_multi
from refer_llm.data.datasets import (
    build_refer_dataset,
    QwenReferYesNoDataset,
    QwenReferVideoYesNoDataset,
    FilteredSingleYesNoDataset,
    FilteredVideoYesNoDataset,
)
from refer_llm.data.collate import make_collate_fn
from refer_llm.modeling.build import build_model_and_processor
from data.refer_kitti_mot import ReferKittiMOT


def _load_config_file(config_path: str) -> dict:
    """Load config from json (preferred) or yaml (fallback)."""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()
    # try json first
    try:
        cfg = json.loads(content)
    except Exception:
        try:
            import yaml  # type: ignore

            cfg = yaml.safe_load(content)
        except Exception as e:  # pragma: no cover - optional dependency
            raise RuntimeError(f"Failed to parse config file: {config_path}") from e
    if not isinstance(cfg, dict):
        raise RuntimeError(f"Config file must contain a mapping/dict, got {type(cfg)}")
    return cfg


def _apply_config_defaults(parser: argparse.ArgumentParser, config_path: Optional[str]):
    """Set parser defaults from config so later CLI args still override."""
    if not config_path:
        return
    cfg = _load_config_file(config_path)
    valid_keys = {a.dest for a in parser._actions if a.dest != "help"}
    filtered = {k: v for k, v in cfg.items() if k in valid_keys}
    parser.set_defaults(**filtered)


def _save_config(args: argparse.Namespace, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2, sort_keys=True)


def _setup_distributed():
    """初始化分布式（torchrun）环境，返回 (rank, world_size, local_rank)。"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def _cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def _is_main_process() -> bool:
    return (not dist.is_initialized()) or (dist.get_rank() == 0)


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _save_batch_visualization(images, prompts, out_dir: str, global_step: int, max_items: int = 8):
    if images is None or prompts is None:
        return
    os.makedirs(out_dir, exist_ok=True)
    items = list(zip(images, prompts))[:max_items]
    if len(items) == 0:
        return
    tile_w, tile_h = items[0][0].size
    cols = min(4, len(items))
    rows = (len(items) + cols - 1) // cols
    pad = 8
    # 字体与测量器
    try:
        font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    # 先做一张临时图用于测量文字宽度
    _tmp_img = Image.new("RGB", (tile_w, tile_h), (255, 255, 255))
    _tmp_draw = ImageDraw.Draw(_tmp_img)
    def _text_width(s: str) -> int:
        if hasattr(_tmp_draw, "textlength"):
            return int(_tmp_draw.textlength(s, font=font))
        # 兼容：用 getbbox 估计
        try:
            bbox = font.getbbox(s)
            return int(bbox[2] - bbox[0])
        except Exception:
            return len(s) * 6
    def _wrap_lines(text: str, max_width: int, max_lines: int = 4):
        text = (text if isinstance(text, str) else str(text)).strip()
        words = text.split()
        lines = []
        cur = ""
        for w in words:
            cand = (cur + " " + w).strip() if cur else w
            if _text_width(cand) <= max_width or cur == "":
                cur = cand
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            # 尾行加省略号并保证不超宽
            ell = " ..."
            last = lines[-1]
            while _text_width(last + ell) > max_width and len(last) > 0:
                last = last[:-1]
            lines[-1] = (last + ell) if last else ell.strip()
        return lines
    # 逐项预换行，计算最大行数用于统一网格高度
    wrapped: list[list[str]] = []
    max_lines_used = 1
    for _, pr in items:
        lines = _wrap_lines(pr, max_width=tile_w, max_lines=4)
        wrapped.append(lines)
        if len(lines) > max_lines_used:
            max_lines_used = len(lines)
    # 行高估计
    try:
        lh = (font.getbbox("Ag")[3] - font.getbbox("Ag")[1]) + 2
    except Exception:
        lh = 14
    text_h = max_lines_used * lh + 4
    # 画布尺寸
    canvas_w = cols * tile_w + (cols + 1) * pad
    canvas_h = rows * (tile_h + text_h) + (rows + 1) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for idx, (im, _) in enumerate(items):
        c = idx % cols
        r = idx // cols
        x0 = pad + c * (tile_w + pad)
        y0 = pad + r * (tile_h + text_h + pad)
        canvas.paste(im, (x0, y0))
        # 多行逐行写入
        lines = wrapped[idx]
        y_text = y0 + tile_h + 4
        for line in lines:
            draw.text((x0, y_text), line, fill=(0, 0, 0), font=font)
            y_text += lh
    out_path = os.path.join(out_dir, f"vis_step_{global_step}.jpg")
    canvas.save(out_path, quality=90)


def _preview_training_batch(
    ds,
    processor,
    label_tokens=None,
    *,
    n_batches: int = 1,
    batch_size: int = 2,
    max_items: int = 2,
    decode_max_chars: int = 400,
):
    """Print a few samples (model inputs + labels) BEFORE training starts.

    Uses a separate dataloader (shuffle=False, num_workers=0) so it won't affect training order.
    """
    if ds is None or processor is None:
        return
    try:
        tok = getattr(processor, "tokenizer", None)
    except Exception:
        tok = None

    collate_fn = make_collate_fn(processor)
    preview_dl = DataLoader(
        ds,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        sampler=None,
        num_workers=0,
        pin_memory=False,
        collate_fn=lambda b: collate_fn(b),
    )

    def _shape(x):
        try:
            return tuple(x.shape)
        except Exception:
            return None

    n_batches = max(1, int(n_batches))
    it = iter(preview_dl)
    for bi in range(n_batches):
        try:
            batch = next(it)
        except StopIteration:
            if bi == 0:
                print("[preview] dataset is empty; nothing to preview.")
            break

        print("\n" + "=" * 90)
        print(f"[preview] Batch {bi + 1}/{n_batches} before training")
        if label_tokens is not None:
            print(f"[preview] label_tokens (Yes, No) = {label_tokens}")
        print(f"[preview] input_ids shape       : {_shape(batch.input_ids)}")
        print(f"[preview] attention_mask shape  : {_shape(batch.attention_mask)}")
        print(f"[preview] labels shape          : {_shape(batch.labels)}")
        print(f"[preview] pixel_values shape    : {_shape(batch.pixel_values)}")
        print(f"[preview] image_grid_thw shape  : {_shape(getattr(batch, 'image_grid_thw', None))}")

        bs = int(batch.input_ids.shape[0]) if hasattr(batch.input_ids, "shape") else 0
        k = min(max(0, int(max_items)), bs)
        for i in range(k):
            # Prefer the original prompt string (much more readable than decoding multimodal tokens)
            prompt_str = None
            dbg_prompts = getattr(batch, "debug_prompts", None)
            if isinstance(dbg_prompts, list) and i < len(dbg_prompts):
                prompt_str = dbg_prompts[i]
            if prompt_str is None and tok is not None:
                try:
                    ids = batch.input_ids[i].tolist()
                    prompt_str = tok.decode(ids, skip_special_tokens=True)
                except Exception:
                    prompt_str = None

            # Decode the supervised label region (labels != -100)
            label_decoded = None
            label_token_count = None
            try:
                lab = batch.labels[i]
                mask = (lab != -100)
                label_token_count = int(mask.long().sum().item()) if hasattr(mask, "long") else None
                if tok is not None:
                    lab_ids = lab[mask].tolist() if hasattr(lab, "__getitem__") else []
                    label_decoded = tok.decode(lab_ids, skip_special_tokens=True)
            except Exception:
                pass

            if isinstance(prompt_str, str) and len(prompt_str) > decode_max_chars:
                prompt_str = prompt_str[:decode_max_chars] + " ...[trunc]"
            if isinstance(label_decoded, str) and len(label_decoded) > decode_max_chars:
                label_decoded = label_decoded[:decode_max_chars] + " ...[trunc]"

            print("-" * 90)
            print(f"[preview] sample #{i}")
            if prompt_str is not None:
                print(f"[preview] prompt: {prompt_str}")
            if label_token_count is not None:
                print(f"[preview] label_token_count (labels!=-100): {label_token_count}")
            if label_decoded is not None:
                print(f"[preview] decoded label region: {label_decoded}")
        print("=" * 90 + "\n")


def train(args):
    # 初始化分布式
    rank, world_size, local_rank = _setup_distributed()
    # 设定随机种子（每个进程用 base_seed + rank，确保可复现）
    base_seed = int(getattr(args, "seed", 42))
    _set_seed(base_seed + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # 构建数据集
    dataset_version = getattr(args, "dataset_version", "v1")
    # 根据版本选择 data_root（若用户未手动覆盖）
    if dataset_version == "v2" and (args.data_root is None or args.data_root == "/data/sq_2023/refer_kitti"):
        args.data_root = "/data/sq_2023/refer_kitti_v2"
    # 版本到划分映射（如果需要可直接在此修改）
    if dataset_version == "v2":
        train_ids_override = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 14, 15, 16, 17, 18, 20]
        val_ids_override = [5, 11, 13, 19]
    elif dataset_version == "dancetrack":
        train_nums = [1,2,6,8,12,15,16,20,23,24,27,29,32,33,37,39,44,45,49,51,52,53,55,57,61,62,66,68,69,72,74,75,80,82,83,86,87,96,98,99]
        val_nums   = [4,5,7,10,14,18,19,25,26,30,34,35,41,43,47,58,63,65,73,77,79,81,90,94,97]
        train_ids_override = [f"dancetrack{i:04d}" for i in train_nums]
        val_ids_override   = [f"dancetrack{i:04d}" for i in val_nums]
    else:
        train_ids_override = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20]
        val_ids_override = [5, 11, 13]
    base_ds = build_refer_dataset(
        data_root=args.data_root,
        split="train",
        train_ids_override=train_ids_override,
        val_ids_override=val_ids_override,
    )
    # 若提供 filtered_root，则仅使用筛选数据进行训练；不做任何回退
    if getattr(args, "filtered_root", None):
        filtered_root = str(args.filtered_root)
        if not os.path.isdir(filtered_root):
            raise RuntimeError(f"--filtered_root 路径不存在或不可访问: {filtered_root}")
        video_only = bool(getattr(args, "video_only", False))
        enable_video = bool(getattr(args, "enable_video_mode", False))
        if video_only and not enable_video:
            raise RuntimeError("--video_only 需要同时开启 --enable_video_mode")
        if enable_video and bool(getattr(args, "train_both_modes", False)) and not video_only:
            filt_single = FilteredSingleYesNoDataset(filtered_root=filtered_root, image_size=args.image_size)
            filt_video = FilteredVideoYesNoDataset(filtered_root=filtered_root, refer_dataset=base_ds, image_size=args.image_size)
            ds = ConcatDataset([filt_single, filt_video])
        elif video_only:
            filt_video = FilteredVideoYesNoDataset(filtered_root=filtered_root, refer_dataset=base_ds, image_size=args.image_size)
            ds = filt_video
        else:
            # 未开启 both / video_only 时，只使用 single
            filt_single = FilteredSingleYesNoDataset(filtered_root=filtered_root, image_size=args.image_size)
            ds = filt_single
    else:
        # 原始数据集路径
        if getattr(args, "prompt_single_tpl", None) is None or len(str(args.prompt_single_tpl).strip()) == 0:
            raise RuntimeError("必须提供 --prompt_single_tpl")
        ds_single = QwenReferYesNoDataset(
            refer_dataset=base_ds,
            image_size=args.image_size,
            margin_ratio=args.margin_ratio,
            margin_px=args.margin_px,
            min_side=args.min_side,
            max_text_len=args.max_text_len,
            negative_downsample=args.negative_downsample,
            coord_mode=args.coord_mode,
            coord_decimals=args.coord_decimals,
            prompt_single_tpl=args.prompt_single_tpl,
            oversample_seq=("0016" if bool(getattr(args, "oversample_pedestrian", False)) else None),
            oversample_factor=int(getattr(args, "oversample_factor", 4)),
        )
        video_only = bool(getattr(args, "video_only", False))
        enable_video = bool(getattr(args, "enable_video_mode", False))
        if video_only and not enable_video:
            raise RuntimeError("--video_only 需要同时开启 --enable_video_mode")
        if enable_video:
            if getattr(args, "prompt_video_tpl", None) is None or len(str(args.prompt_video_tpl).strip()) == 0:
                raise RuntimeError("启用视频模式时必须提供 --prompt_video_tpl")
            ds_video = QwenReferVideoYesNoDataset(
                refer_dataset=base_ds,
                image_size=args.image_size,
                margin_ratio=args.margin_ratio,
                margin_px=args.margin_px,
                min_side=args.min_side,
                max_text_len=args.max_text_len,
                negative_downsample=args.negative_downsample,
                video_n_frames=getattr(args, "video_n_frames", 4),
                prompt_video_tpl=args.prompt_video_tpl,
                oversample_seq=("0016" if bool(getattr(args, "oversample_pedestrian", False)) else None),
                oversample_factor=int(getattr(args, "oversample_factor", 4)),
                coord_mode=args.coord_mode,
                coord_decimals=args.coord_decimals,
            )
            if video_only:
                ds = ds_video
            elif getattr(args, "train_both_modes", False):
                ds = ConcatDataset([ds_single, ds_video])
            else:
                # 未开启 both 时只使用 single（即便启用了视频模式）
                ds = ds_single
        else:
            if video_only:
                raise RuntimeError("--video_only 需要同时开启 --enable_video_mode")
            ds = ds_single

    # 模型与处理器
    model, processor, label_tokens = build_model_and_processor(
        model_name=args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_4bit=args.use_4bit,
        bf16=not args.fp16,
    )
    model.to(device)
    # DDP 包裹
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # DataLoader
    collate_fn = make_collate_fn(processor)
    if world_size > 1:
        train_sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
        shuffle_flag = False
    else:
        train_sampler = None
        shuffle_flag = True
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=bool(getattr(args, "pin_memory", True)),
        persistent_workers=bool(getattr(args, "persistent_workers", True)) if args.num_workers > 0 else False,
        prefetch_factor=int(getattr(args, "prefetch_factor", 2)) if args.num_workers > 0 else None,
        collate_fn=lambda b: collate_fn(b),
    )

    # 训练开始前预览若干样本（仅主进程；单独 dataloader，不影响训练顺序）
    if _is_main_process() and int(getattr(args, "preview_batches", 1) or 0) > 0:
        _preview_training_batch(
            ds,
            processor,
            label_tokens=label_tokens,
            n_batches=int(getattr(args, "preview_batches", 1) or 1),
            batch_size=min(int(args.batch_size), int(getattr(args, "preview_batch_size", 2) or 2)),
            max_items=int(getattr(args, "preview_max_items", 2) or 2),
            decode_max_chars=int(getattr(args, "preview_decode_max_chars", 400) or 400),
        )

    # 优化器
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)

    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    # 统一 runs/ 时间戳目录（主进程创建并广播）
    run_root = None
    if _is_main_process():
        ts = torch.tensor(0, dtype=torch.int32)  # 占位，后续广播用不到数值
        run_ts = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
        run_root = os.path.join("runs", run_ts)
        train_out_dir = os.path.join(run_root, "train")
        eval_out_dir = os.path.join(run_root, "eval")
        os.makedirs(train_out_dir, exist_ok=True)
        os.makedirs(eval_out_dir, exist_ok=True)
    else:
        train_out_dir = None
        eval_out_dir = None
    if world_size > 1 and dist.is_initialized():
        dir_info = [run_root, train_out_dir, eval_out_dir]
        dist.broadcast_object_list(dir_info, src=0)
        run_root, train_out_dir, eval_out_dir = dir_info
    # 将输出定向到 runs/ 时间戳子目录
    args.output_dir = train_out_dir
    args.eval_output_dir = eval_out_dir
    # 保存本次运行的最终配置（仅主进程）
    config_save_path = os.path.join(run_root, "config.json") if run_root else None
    if _is_main_process() and config_save_path:
        _save_config(args, config_save_path)

    # 评估与保存步数：使用“全局步数=每卡本地步数×卡数”，单一参数 every_steps
    local_trigger_interval = 0
    if getattr(args, "every_steps", 0):
        if world_size > 1 and (int(args.every_steps) % world_size != 0):
            raise ValueError(f"every_steps({args.every_steps}) 需要能被 world_size({world_size}) 整除，以便平分到多卡。")
        local_trigger_interval = int(max(1, int(args.every_steps) // max(1, world_size)))
    # 可视化频率（每 steps/5）
    local_vis_interval = max(1, local_trigger_interval // 5) if local_trigger_interval > 0 else 0

    local_step = 0  # 每卡本地步数
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if _is_main_process():
            pbar = tqdm(dl, total=len(dl), desc=f"Epoch {epoch + 1}/{args.epochs}", dynamic_ncols=True)
            iterator = pbar
        else:
            pbar = None
            iterator = dl
        for batch in iterator:
            pixel_values = batch.pixel_values
            if pixel_values is None:
                raise RuntimeError("pixel_values is None - images not properly processed in training batch")
            pixel_values = pixel_values.to(device)
            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            labels = batch.labels.to(device)
            image_grid_thw = batch.image_grid_thw
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(device)

            with torch.cuda.amp.autocast(enabled=args.fp16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    labels=labels,
                )
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)

            if _is_main_process() and (local_step % args.log_every == 0) and pbar is not None:
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "bs": args.batch_size})

            local_step += 1

            # 按步保存当前模型（LoRA 适配器）
            if local_trigger_interval > 0 and (local_step % local_trigger_interval == 0):
                # 仅主进程保存；目录名按“全局步数”
                global_total_step = local_step * max(1, world_size)
                if _is_main_process() and args.output_dir:
                    os.makedirs(args.output_dir, exist_ok=True)
                    just_saved_dir = os.path.join(args.output_dir, f"lora_step_{global_total_step}")
                    to_save = model.module if isinstance(model, DDP) else model
                    to_save.save_pretrained(just_saved_dir)
            
            # 可视化保存（每 steps/5）
            if local_vis_interval > 0 and (local_step % local_vis_interval == 0):
                global_total_step = local_step * max(1, world_size)
                if _is_main_process() and args.output_dir:
                    vis_dir = os.path.join(args.output_dir, "vis")
                    imgs = getattr(batch, "debug_images", None)
                    prs = getattr(batch, "debug_prompts", None)
                    _save_batch_visualization(imgs, prs, vis_dir, global_total_step, max_items=8)

            # 按步数进行简单评估
            if local_trigger_interval > 0 and (local_step % local_trigger_interval == 0):
                # 与保存同步，以全局步数触发
                global_total_step = local_step * max(1, world_size)
                # 评估前同步
                if world_size > 1 and dist.is_initialized():
                    dist.barrier()
                try:
                    model.eval()
                    # 统一使用当前内存中的模型进行评估（不加载保存点）
                    # 校验评估所需的 prompt 模板
                    if getattr(args, "prompt_single_tpl", None) is None or len(str(args.prompt_single_tpl).strip()) == 0:
                        raise RuntimeError("评估需要提供 --prompt_single_tpl")
                    if getattr(args, "re_refer_lower", None) is None:
                        raise RuntimeError("评估需要提供 --re_refer_lower（无回退）")
                    eval_re_lower = float(args.re_refer_lower)
                    eval_re_thresh = float(args.re_refer_thresh)
                    eval_enable_refine = True
                    # video_only: 全部走视频精炼，不再按区间筛选
                    if bool(getattr(args, "video_only", False)):
                        eval_re_lower = 0.0
                        eval_re_thresh = 1.0
                        eval_enable_refine = True
                    if eval_re_thresh is not None and float(eval_re_thresh) > float(eval_re_lower):
                        if getattr(args, "prompt_video_tpl", None) is None or len(str(args.prompt_video_tpl).strip()) == 0:
                            raise RuntimeError("二阶段评估开启时需要提供 --prompt_video_tpl")

                    # 多阈值评估：一次前向，多个阈值统计
                    thresholds = args.eval_threshold if isinstance(getattr(args, "eval_threshold", [0.4]), (list, tuple)) else [args.eval_threshold]
                    if str(args.eval_sequence).strip().lower() == "all":
                        eval_summary = external_evaluate_all_multi(
                            model=model,
                            processor=processor,
                            label_tokens=label_tokens,
                            ds=ReferKittiMOT(
                                data_root=args.data_root,
                                split="val",
                                load_annotation=True,
                                expression_sub_dir="expression",
                                labels_with_ids_sub_dir="labels_with_ids/image_02",
                                train_ids_override=train_ids_override,
                                val_ids_override=val_ids_override,
                            ),
                            thresholds=[float(t) for t in thresholds],
                            image_size=args.image_size,
                            margin_ratio=args.margin_ratio,
                            margin_px=args.margin_px,
                            min_side=args.min_side,
                            coord_mode=args.coord_mode,
                            coord_decimals=args.coord_decimals,
                            batch_size=args.eval_batch_size,
                            device=device,
                            output_dir=args.eval_output_dir,
                            global_step=global_total_step,
                            show_tqdm=bool(args.eval_show_tqdm) and _is_main_process(),
                            re_refer_thresh=eval_re_thresh,
                            re_refer_lower=eval_re_lower,
                            video_n_frames=getattr(args, "video_n_frames", 4),
                            prompt_single_tpl=args.prompt_single_tpl,
                            prompt_video_tpl=args.prompt_video_tpl,
                            enable_refine=eval_enable_refine,
                            rank=rank,
                            world_size=world_size,
                            max_texts_per_seq=(int(args.eval_limit_texts_per_seq) if int(getattr(args, "eval_limit_texts_per_seq", 0) or 0) > 0 else None),
                            preprocess_workers=int(getattr(args, "eval_preprocess_workers", 0)),
                            infer_every_n_frames=int(getattr(args, "eval_infer_every_n_frames", 1) or 1),
                        )
                    else:
                        # 保持单序列路径（可后续扩展多阈值版本）
                        if _is_main_process():
                            eval_summary = external_evaluate(
                                model=model,
                                processor=processor,
                                label_tokens=label_tokens,
                                data_root=args.data_root,
                                dataset_version=dataset_version,
                                sequence=args.eval_sequence,
                                image_size=args.image_size,
                                margin_ratio=args.margin_ratio,
                                margin_px=args.margin_px,
                                min_side=args.min_side,
                                coord_mode=args.coord_mode,
                                coord_decimals=args.coord_decimals,
                                threshold=float(thresholds[0]),
                                batch_size=args.eval_batch_size,
                                device=device,
                                output_dir=args.eval_output_dir,
                                global_step=global_total_step,
                                show_tqdm=bool(args.eval_show_tqdm),
                                re_refer_thresh=eval_re_thresh,
                                re_refer_lower=eval_re_lower,
                                video_n_frames=getattr(args, "video_n_frames", 4),
                                prompt_single_tpl=args.prompt_single_tpl,
                                prompt_video_tpl=args.prompt_video_tpl,
                                enable_refine=eval_enable_refine,
                                max_texts_per_seq=(int(args.eval_limit_texts_per_seq) if int(getattr(args, "eval_limit_texts_per_seq", 0) or 0) > 0 else None),
                                preprocess_workers=int(getattr(args, "eval_preprocess_workers", 0)),
                                infer_every_n_frames=int(getattr(args, "eval_infer_every_n_frames", 1) or 1),
                            )
                        else:
                            eval_summary = {"sequence": args.eval_sequence, "overall_acc": 0.0, "num_texts": 0}

                    # 简要打印评估摘要（仅主进程）
                    if _is_main_process() and pbar is not None:
                        pbar.write(f"Eval@{global_total_step} | seq={args.eval_sequence} | thresholds={getattr(args, 'eval_threshold', None)}")
                finally:
                    model.train()
                # 评估后同步
                if world_size > 1 and dist.is_initialized():
                    dist.barrier()

        # 每个 epoch 保存
        if _is_main_process() and args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            to_save = model.module if isinstance(model, DDP) else model
            to_save.save_pretrained(os.path.join(args.output_dir, f"lora_epoch_{epoch}"))

    # 清理分布式环境
    _cleanup_distributed()

def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None, help="从配置文件加载参数（json/yaml），CLI 仍可覆盖")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--data_root", type=str, default="/data/sq_2023/refer_kitti")
    p.add_argument("--dataset_version", type=str, default="v1", choices=["v1", "v2", "dancetrack"], help="选择数据集版本：v1 或 v2 或 dancetrack")
    # 过滤数据接入
    p.add_argument("--filtered_root", type=str, default=None, help="筛选器输出根目录（包含 single/ 与 video/ 子目录）")
    p.add_argument("--every_steps", type=int, default=10000, help="全局步数间隔：保存与评估同时触发（需可被卡数整除）")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--seed", type=int, default=2025, help="全局随机种子（分布式将按 seed+rank 设置）")
    p.add_argument("--num_workers", type=int, default=12, help="DataLoader worker 数（>0 开启多进程数据加载）")
    p.add_argument("--prefetch_factor", type=int, default=2, help="DataLoader 每 worker 预取批次数（num_workers>0 时生效）")
    p.add_argument("--persistent_workers", type=int, default=1, help="是否启用持久 worker（1=True,0=False，num_workers>0 时生效）")
    p.add_argument("--pin_memory", type=int, default=1, help="是否 pin_memory 以加速 H2D 拷贝")
    p.add_argument("--log_every", type=int, default=50)
    # 训练开始前样本预览
    p.add_argument("--preview_batches", type=int, default=0, help="训练开始前预览打印 batch 数（0=关闭）")
    p.add_argument("--preview_batch_size", type=int, default=2, help="预览 dataloader 的 batch_size（与训练无关）")
    p.add_argument("--preview_max_items", type=int, default=2, help="每个预览 batch 打印的样本条数")
    p.add_argument("--preview_decode_max_chars", type=int, default=400, help="解码文本最大显示字符数（超出截断）")
    # 图像/裁剪
    p.add_argument("--image_size", type=int, default=384)
    p.add_argument("--margin_ratio", type=float, default=0.2)
    p.add_argument("--margin_px", type=int, default=None)
    p.add_argument("--min_side", type=int, default=8)
    # 文本与采样
    p.add_argument("--max_text_len", type=int, default=100)
    p.add_argument("--negative_downsample", type=float, default=0.5)
    p.add_argument("--coord_mode", type=str, default="xy", choices=["xy", "xywh"])
    p.add_argument("--coord_decimals", type=int, default=3)
    # LoRA
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0)
    # 精度与量化
    p.add_argument("--use_4bit", action="store_true")
    p.add_argument("--fp16", action="store_true")
    # 评估
    p.add_argument("--eval_threshold", type=float, nargs="+", default=[0.2,0.3,0.4,0.5,0.6,0.7,0.8], help="可传多个阈值，例如: --eval_threshold 0.3 0.4 0.5")
    p.add_argument("--eval_sequence", type=str, default="all")
    p.add_argument("--eval_batch_size", type=int, default=2)
    p.add_argument("--eval_show_tqdm", type=int, default=1)
    p.add_argument("--eval_limit_texts_per_seq", type=int, default=0, help="训练期评估：每个验证序列仅评估前 N 条表达式(0=不限制)")
    p.add_argument("--eval_preprocess_workers", type=int, default=4, help="训练期评估图像预处理线程数（0 表示单线程）")
    p.add_argument("--eval_infer_every_n_frames", type=int, default=5, help="训练期评估跳帧推理间隔 N（1=每帧都推理；N>1 启用跳帧复用 confidence）")
    # 视频样本训练模式
    p.add_argument("--enable_video_mode", action="store_true", help="构建视频片段训练数据；需与 --train_both_modes 或 --video_only 联用才会改变训练样本组成")
    p.add_argument("--train_both_modes", action="store_true", help="同时使用单帧与视频样本训练（需同时开启 --enable_video_mode）")
    p.add_argument("--video_only", action="store_true", help="仅使用视频样本进行训练（需同时开启 --enable_video_mode）")
    p.add_argument("--video_n_frames", type=int, default=3, help="视频模式取前 N 帧（不足则取可用帧）")
    p.add_argument("--re_refer_thresh", type=float, default=0.8, help="二阶段精炼阈值上限（在 [threshold, re_refer_thresh) 范围内触发视频精炼）")
    p.add_argument("--re_refer_lower", type=float, default=0.2, help="二阶段精炼触发下限（必须显式提供；与 re_refer_thresh 构成区间）")
    # 行人过采样（过采样 0016 序列）
    p.add_argument("--oversample_pedestrian", action="store_true", help="开启时过采样 0016 序列以缓解行人样本稀少")
    p.add_argument("--oversample_factor", type=int, default=4, help="0016 序列过采样倍数（>1 生效）")
    # # Prompt 模板
    # p.add_argument("--prompt_single_tpl", type=str, default="The normalized position (0-1) of the car or person in the picture is <{coord}>.Determine whether this description matches this image: \"{sentence}\". Answer Yes or No.", help="单帧模板，必须提供，支持 {coord}, {sentence}")
    # p.add_argument("--prompt_video_tpl", type=str, default="The normalized position (0-1) of the car or person in the video clip is <{coord}>.Determine whether this description matches this image: \"{sentence}\". Answer Yes or No.", help="视频模板，启用视频/二阶段时必须提供，支持 {sentence}, {coord}")

    # p.add_argument(
    #     "--prompt_single_tpl",
    #     type=str,
    #     default=(
    #         "You are a person/car re-identification assistant.\n"
    #         "The coordinates <{coord}> are normalized to the range [0, 1], representing the relative location of the image patch.\n"
    #         "Analyze whether the image patch at this location matches the target described in \"{sentence}\".\n"
    #         "Answer strictly with one word: Yes or No."
    #     ),
    #     help="单帧模板，必须提供，支持 {coord}, {sentence}",
    # )

    

    # p.add_argument(
    #     "--prompt_video_tpl",
    #     type=str,
    #     default=(
    #         "You are a person/car re-identification assistant.\n"
    #         "Analyze whether the sequence of image patches located at the normalized position <{coord}> "
    #         "across multiple consecutive frames matches the target described in the sentence \"{sentence}\".\n"
    #         "The coordinate <{coord}> is normalized to the range [0, 1].\n"
    #          "Use all temporal cues - appearance consistency, motion continuity, pose changes, and occlusion handling to make your judgment.\n"
    #         "Answer strictly with one word: Yes or No."
    #     ),
    #     help="视频模板，支持 {coord}, {sentence}"
    # )



    p.add_argument("--prompt_single_tpl", type=str, default="The normalized position of the car or person in the picture is <{coord}>.Determine whether this description matches this image: {sentence}. Answer Yes or No. The normalized position of the car or person in the picture is <{coord}>.Determine whether this description matches this image: {sentence}. Answer Yes or No.", help="单帧模板，必须提供，支持 {coord}, {sentence}")
    p.add_argument("--prompt_video_tpl", type=str, default="This is a short video clip of a car or person at <{coord}> across frames. The target may include motion cues; consider background and temporal context when deciding if the description matches this target: {sentence}. Answer Yes or No.", help="视频模板，启用视频/二阶段时必须提供，支持 {sentence}, {coord}")

    # p.add_argument("--prompt_video_tpl", type=str, default="This is a short video clip of a car or person at <{coord}> across frames.The target may include motion cues; consider background and temporal context when making your decision.Pay attention to the person or vehicle near the center region of the video, and if the target is a person, consider gender appearance (male or female) when deciding if the description matches this target: {sentence}.Answer Yes or No.", help="视频模板，启用二阶段时使用，支持 {sentence}, {coord}")
    # p.add_argument("--prompt_video_tpl", type=str, default="This is a short sequence of consecutive cropped and resized square images showing a car or person at <{coord}> across frames. Observe background changes to judge whether the target is moving or stationary. Consider temporal consistency and gender when applicable when deciding if the description matches this target: {sentence}. Answer Yes or No.", help="视频模板，启用二阶段时使用，支持 {sentence}, {coord}")
    return p


def parse_args(argv=None):
    """Two-pass parsing: read --config first, then allow CLI override."""
    parser = build_argparser()
    first_pass, _ = parser.parse_known_args(argv)
    _apply_config_defaults(parser, getattr(first_pass, "config", None))
    args = parser.parse_args(argv)
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
