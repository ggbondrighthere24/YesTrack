from __future__ import annotations

import os
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

# 确保可从项目根目录导入兄弟包
import sys
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.refer_kitti_mot import ReferKittiMOT
from refer_llm.crop_utils import crop_with_margin
from transformers import AutoProcessor


def build_and_forward_yes_probs(
    model: torch.nn.Module,
    processor: AutoProcessor,
    images: List[Image.Image],
    prompts: List[str],
    device: torch.device,
    yes_id: int,
    no_id: int,
) -> List[float]:
    """将若干 (image, prompt) 样本打包，前向一次并返回每个样本的 Yes 概率。"""
    if len(images) == 0:
        return []

    # 构造 Batch Messages
    messages_batch = []
    for img, pr_ in zip(images, prompts):
        messages_batch.append([
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": pr_},
                ],
            }
        ])
    
    # 确保 Padding Side 为 Left，以便 logits[:, -1, :] 取到最后一个 Token
    if hasattr(processor.tokenizer, "padding_side"):
        processor.tokenizer.padding_side = "left"

    # 批量应用 Chat Template
    inputs = processor.apply_chat_template(
        messages_batch,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    pixel_values = inputs.get("pixel_values")
    image_grid_thw = inputs.get("image_grid_thw")
    
    if pixel_values is not None:
        pixel_values = pixel_values.to(device)
    if image_grid_thw is not None:
        image_grid_thw = image_grid_thw.to(device)

    # 如果模型（如 Qwen-VL）在没有图像时可能报错或行为不一，需注意
    # 但此处 images 列表非空，且每个样本都含图像，通常 pixel_values 不会为 None

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
    )
    
    # 取最后一个 token 的 logits
    last_logits = outputs.logits[:, -1, :]
    two_logits = last_logits[:, [yes_id, no_id]]
    two_probs = F.softmax(two_logits, dim=-1)
    p_yes = two_probs[:, 0].detach().cpu().tolist()
    return p_yes


def _build_video_images_for_seq(
    ds: ReferKittiMOT,
    seq: str,
    frame_idx: int,
    obj_id: int,
    current_patch: Image.Image,
    image_size: Optional[int],
    margin_ratio: float,
    margin_px: Optional[int],
    min_side: int,
    video_n_frames: int,
) -> Tuple[List[Image.Image], str]:
    """为二阶段精炼构建多帧图像与最近一帧坐标字符串。"""
    frames: List[Image.Image] = []
    coords_each: List[str] = []
    start_f = max(0, frame_idx - int(video_n_frames) + 1)
    for fidx in range(start_f, frame_idx + 1):
        ann_f = ds.annotations[seq][fidx]
        ids_f = ann_f["id"].tolist()
        try:
            idx_in_f = ids_f.index(obj_id)
        except ValueError:
            continue
        x_f, y_f, w_f, h_f = ann_f["bbox"][idx_in_f].tolist()
        img_path_f = ds.image_paths[seq][fidx]
        try:
            image_f = Image.open(img_path_f).convert("RGB")
        except Exception:
            continue
        patch_f = crop_with_margin(
            image=image_f,
            bbox_xywh=(x_f, y_f, w_f, h_f),
            margin_ratio=margin_ratio,
            margin_px=margin_px,
            min_side=min_side,
        )
        if patch_f is None:
            patch_f = image_f.crop((int(x_f), int(y_f), int(x_f + w_f), int(y_f + h_f)))
        if image_size is not None:
            patch_f = patch_f.resize((image_size, image_size))
        frames.append(patch_f)
        Wf, Hf = image_f.size
        cxf = x_f + 0.5 * w_f
        cyf = y_f + 0.5 * h_f
        nxf = max(0.0, min(1.0, cxf / float(Wf)))
        nyf = max(0.0, min(1.0, cyf / float(Hf)))
        coords_each.append(f"{nxf:.3f} {nyf:.3f}")
    if len(frames) == 0:
        frames.append(current_patch)
    coords_str = f"<{coords_each[-1]}>" if len(coords_each) > 0 else ""
    return frames, coords_str


def refine_probs_with_video_for_indices(
    model: torch.nn.Module,
    processor: AutoProcessor,
    ds: ReferKittiMOT,
    seq: str,
    batch_metas: List[Tuple[int, int, str, int, int, int, int]],
    batch_images: List[Image.Image],
    p_yes: List[float],
    sentence: str,
    prompt_video_tpl: str,
    device: torch.device,
    yes_id: int,
    no_id: int,
    video_n_frames: int,
    image_size: Optional[int],
    margin_ratio: float,
    margin_px: Optional[int],
    min_side: int,
    lower_bound: float,
    re_refer_thresh: float,
) -> List[float]:
    """对 [lower_bound, re_refer_thresh) 内的样本进行视频多帧精炼，并更新 p_yes。"""
    refine_indices = [i for i, p in enumerate(p_yes) if (p >= lower_bound and p < re_refer_thresh)]
    if len(refine_indices) == 0:
        return p_yes

    messages_batch_v = []
    
    for i in refine_indices:
        fidx, oid, coord_s, bx, by, bw, bh = batch_metas[i]
        vid_frames, coords_s = _build_video_images_for_seq(
            ds=ds,
            seq=seq,
            frame_idx=fidx,
            obj_id=oid,
            current_patch=batch_images[i],
            image_size=image_size,
            margin_ratio=margin_ratio,
            margin_px=margin_px,
            min_side=min_side,
            video_n_frames=video_n_frames,
        )
        try:
            txt_v = prompt_video_tpl.format(sentence=sentence, coord=coord_s, coords=coords_s)
        except Exception as e:
            raise RuntimeError(f"视频 prompt 模板格式化失败: {e}")
        
        # 构造单条消息
        messages_batch_v.append([
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": im} for im in vid_frames],
                    {"type": "text", "text": txt_v},
                ],
            }
        ])

    # 确保 Padding Side 为 Left
    if hasattr(processor.tokenizer, "padding_side"):
        processor.tokenizer.padding_side = "left"

    # 批量处理
    inputs_v = processor.apply_chat_template(
        messages_batch_v,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )

    input_ids_v = inputs_v["input_ids"].to(device)
    attention_mask_v = inputs_v["attention_mask"].to(device)
    pixel_values_v = inputs_v.get("pixel_values")
    image_grid_thw_v = inputs_v.get("image_grid_thw")
    
    if pixel_values_v is not None:
        pixel_values_v = pixel_values_v.to(device)
    if image_grid_thw_v is not None:
        image_grid_thw_v = image_grid_thw_v.to(device)

    outputs_v = model(
        input_ids=input_ids_v,
        attention_mask=attention_mask_v,
        pixel_values=pixel_values_v,
        image_grid_thw=image_grid_thw_v,
    )
    
    last_logits_v = outputs_v.logits[:, -1, :]
    two_logits_v = last_logits_v[:, [yes_id, no_id]]
    two_probs_v = F.softmax(two_logits_v, dim=-1)
    p_yes_v = two_probs_v[:, 0].detach().cpu().tolist()
    
    # 更新回原数组
    for idx_loc, p_new in zip(refine_indices, p_yes_v):
        p_yes[idx_loc] = p_new
        
    return p_yes
