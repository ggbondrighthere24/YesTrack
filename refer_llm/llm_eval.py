from __future__ import annotations

import os
import re
import json
from collections import OrderedDict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.distributed as dist
from PIL import Image

from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration,Gemma3ForConditionalGeneration,Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

# 确保可从项目根目录导入兄弟包
import sys
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.refer_kitti_mot import ReferKittiMOT
from refer_llm.crop_utils import crop_with_margin
from refer_llm.infer_utils import build_and_forward_yes_probs, refine_probs_with_video_for_indices


def _iter_samples_for_sentence(
    ds: ReferKittiMOT,
    sequence: str,
    sentence: str,
    label_map: Dict[str, Any],
    image_size: int,
    margin_ratio: float,
    margin_px: Optional[int],
    min_side: int,
    coord_mode: str,
    fmt: str,
    prompt_single_tpl: str,
    show_tqdm: bool,
    preprocess_workers: int,
) -> Any:
    """Yield (patch, prompt, target, meta) for all frames in a sentence."""
    num_frames = ds.sequence_infos[sequence]["length"]
    frame_iter = range(num_frames)
    if show_tqdm:
        from tqdm import tqdm as _tqdm

        frame_iter = _tqdm(frame_iter, total=num_frames, leave=False, desc="frames")

    def _process_frame(fidx: int):
        samples: List[Tuple[Any, str, int, Tuple[int, int, str, int, int, int, int]]] = []
        ann = ds.annotations[sequence][fidx]
        M = ann["bbox"].shape[0]
        if M == 0:
            return samples

        img_path = ds.image_paths[sequence][fidx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            return samples
        W, H = image.size

        ids = label_map.get(str(fidx), [])
        try:
            pos_ids = set(int(i) for i in ids)
        except Exception:
            pos_ids = set()

        for ann_idx in range(M):
            x, y, w, h = ann["bbox"][ann_idx].tolist()
            obj_id = int(ann["id"][ann_idx].item())
            y_true = 1 if obj_id in pos_ids else 0

            patch = crop_with_margin(
                image=image,
                bbox_xywh=(x, y, w, h),
                margin_ratio=margin_ratio,
                margin_px=margin_px,
                min_side=min_side,
            )
            if patch is None:
                patch = image.crop((int(x), int(y), int(x + w), int(y + h)))
            if image_size is not None:
                patch = patch.resize((image_size, image_size))

            cx = x + 0.5 * w
            cy = y + 0.5 * h
            nx1 = max(0.0, min(1.0, cx / float(W)))
            ny1 = max(0.0, min(1.0, cy / float(H)))
            if coord_mode == "xywh":
                nw = max(0.0, min(1.0, w / float(W)))
                nh = max(0.0, min(1.0, h / float(H)))
                coord_str = f"{fmt.format(nx1)} {fmt.format(ny1)} {fmt.format(nw)} {fmt.format(nh)}"
            else:
                coord_str = f"{fmt.format(nx1)} {fmt.format(ny1)}"
            try:
                pr = prompt_single_tpl.format(coord=coord_str, sentence=sentence)
            except Exception as e:  # pragma: no cover - format errors should surface
                raise RuntimeError(f"单帧 prompt 模板格式化失败: {e}")

            samples.append(
                (
                    patch,
                    pr,
                    y_true,
                    (fidx, obj_id, coord_str, int(x), int(y), int(w), int(h)),
                )
            )
        return samples

    if preprocess_workers and int(preprocess_workers) > 0:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=int(preprocess_workers)) as ex:
            for samples in ex.map(_process_frame, frame_iter):
                for item in samples:
                    yield item
    else:
        for fidx in frame_iter:
            for item in _process_frame(fidx):
                yield item


DEFAULT_PROMPT_SINGLE = (
    "The normalized position of the car in the picture is <{coord}>."
    "Determine whether this description matches this image: {sentence}. Answer Yes or No."
)
DEFAULT_PROMPT_VIDEO = (
    "This is a short video clip of a car or person at <{coord}> across frames. "
    "The target may include motion cues; consider background and temporal context when deciding if the description matches this target: {sentence}. "
    "Answer Yes or No."
)

def _split_overrides_by_version(version: str) -> Tuple[list, list]:
    """返回给 ReferKittiMOT 的 train/val 划分覆盖（基于数据集版本）"""
    v = (version or "v1").lower()
    if v == "v2":
        # v2: 包含 0/17/19
        return (
            [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 14, 15, 16, 17, 18, 20],
            [5, 11, 13, 19],
        )
    # v1（默认）
    return (
        [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20],
        [5, 11, 13],
    )

def setup_distributed():
    """初始化分布式环境"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """判断是否为主进程"""
    return not dist.is_initialized() or dist.get_rank() == 0


def gather_results(data: Any, world_size: int) -> List[Any]:
    """收集所有进程的结果到主进程"""
    if world_size == 1:
        return [data]
    
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, data)
    return gathered


def get_yes_no_token_ids(processor: AutoProcessor) -> Tuple[int, int]:
    yes_id = processor.tokenizer.convert_tokens_to_ids("Yes")
    no_id = processor.tokenizer.convert_tokens_to_ids("No")
    if yes_id is None or yes_id == processor.tokenizer.unk_token_id:
        yes_id = processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
    if no_id is None or no_id == processor.tokenizer.unk_token_id:
        no_id = processor.tokenizer.encode("No", add_special_tokens=False)[0]
    return int(yes_id), int(no_id)

 

def build_model_and_processor_for_eval(
    model_name: str,
    use_4bit: bool = False,
    bf16: bool = True,
    lora_path: Optional[str] = None,
):
    quant_config = None
    torch_dtype = torch.bfloat16 if bf16 else torch.float16
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        quantization_config=quant_config,
        trust_remote_code=True,
    )

    # 可选加载 LoRA 适配器
    if lora_path and os.path.isdir(lora_path):
        model = PeftModel.from_pretrained(model, lora_path)

    return model, processor, get_yes_no_token_ids(processor)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    processor: AutoProcessor,
    label_tokens: Tuple[int, int],
    data_root: str,
    sequence: str = "0005",
    dataset_version: str = "v1",
    image_size: int = 336,
    margin_ratio: float = 0.1,
    margin_px: Optional[int] = None,
    min_side: int = 8,
    coord_mode: str = "xy",
    coord_decimals: int = 3,
    threshold: float = 0.5,
    batch_size: int = 64,
    device: torch.device = torch.device("cpu"),
    output_dir: Optional[str] = None,
    global_step: Optional[int] = None,
    show_tqdm: bool = True,
    re_refer_thresh: float = 1.0,
    re_refer_lower: Optional[float] = None,
    video_n_frames: int = 4,
    prompt_single_tpl: Optional[str] = None,
    prompt_video_tpl: Optional[str] = None,
    enable_refine: bool = True,
    max_texts_per_seq: Optional[int] = None,
    preprocess_workers: int = 0,
    infer_every_n_frames: int = 1,
) -> Dict[str, Any]:
    model.eval()
    # 强制要求外部显式提供 prompt（无回退）
    if prompt_single_tpl is None or len(str(prompt_single_tpl).strip()) == 0:
        raise ValueError("evaluate 需要提供 prompt_single_tpl（无回退）")
    if bool(enable_refine):
        if re_refer_lower is None:
            raise ValueError("enable_refine=True 时必须显式提供 re_refer_lower（无回退）")
        if not (float(re_refer_thresh) > float(re_refer_lower)):
            raise ValueError(f"re_refer_thresh({re_refer_thresh}) 必须大于 re_refer_lower({re_refer_lower})")
    lower_bound = float(re_refer_lower) if (re_refer_lower is not None) else None
    if bool(enable_refine) and re_refer_thresh is not None and float(re_refer_thresh) > float(lower_bound):
        if prompt_video_tpl is None or len(str(prompt_video_tpl).strip()) == 0:
            raise ValueError("evaluate 二阶段精炼需要提供 prompt_video_tpl（无回退）")
    yes_id, no_id = int(label_tokens[0]), int(label_tokens[1])

    train_ids_override, val_ids_override = _split_overrides_by_version(dataset_version)
    ds = ReferKittiMOT(
        data_root=data_root,
        split="val",
        load_annotation=True,
        expression_sub_dir="expression",
        labels_with_ids_sub_dir="labels_with_ids/image_02",
        train_ids_override=train_ids_override,
        val_ids_override=val_ids_override,
    )

    # 单序列运行目录（用于“写一个保存一个”）
    run_dir = None
    mot_dir = None
    refer_dir = None
    if output_dir and sequence != "all":
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        step_str = f"step{global_step}" if global_step is not None else "nostep"
        run_dir = os.path.join(output_dir, f"eval_{sequence}_{step_str}_{ts}")
        mot_dir = os.path.join(run_dir, "mot")
        refer_dir = os.path.join(run_dir, "refer")
        os.makedirs(mot_dir, exist_ok=True)
        os.makedirs(refer_dir, exist_ok=True)

        # 写出该序列的 MOT 文件（一次性）
        try:
            mot_lines: List[str] = []
            num_frames_seq = ds.sequence_infos[sequence]["length"]
            for fidx in range(num_frames_seq):
                ann_ = ds.annotations[sequence][fidx]
                M_ = ann_["bbox"].shape[0]
                if M_ == 0:
                    continue
                for ann_idx_ in range(M_):
                    x_, y_, w_, h_ = ann_["bbox"][ann_idx_].tolist()
                    obj_id_ = int(ann_["id"][ann_idx_].item())
                    mot_lines.append(f"{fidx},{obj_id_},{int(x_)},{int(y_)},{int(w_)},{int(h_)},{1.0}")
            with open(os.path.join(mot_dir, f"{sequence}.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(mot_lines))
        except Exception:
            pass

    if sequence == "all":
        return evaluate_all_sequences(
            model=model,
            processor=processor,
            label_tokens=label_tokens,
            ds=ds,
            image_size=image_size,
            margin_ratio=margin_ratio,
            margin_px=margin_px,
            min_side=min_side,
            coord_mode=coord_mode,
            coord_decimals=coord_decimals,
            threshold=threshold,
            batch_size=batch_size,
            device=device,
            output_dir=output_dir,
            global_step=global_step,
            show_tqdm=show_tqdm,
            re_refer_thresh=re_refer_thresh,
            re_refer_lower=lower_bound,
            video_n_frames=video_n_frames,
            prompt_single_tpl=prompt_single_tpl,
            prompt_video_tpl=prompt_video_tpl,
            enable_refine=enable_refine,
            preprocess_workers=preprocess_workers,
            infer_every_n_frames=infer_every_n_frames,
        )

    if sequence not in ds.image_paths:
        return {
            "sequence": sequence,
            "threshold": threshold,
            "overall_acc": 0.0,
            "num_texts": 0,
            "per_text": [],
        }

    expressions = ds._load_expressions_for_sequence(sequence)
    if isinstance(max_texts_per_seq, int) and max_texts_per_seq > 0:
        expressions = expressions[: int(max_texts_per_seq)]
    num_frames = ds.sequence_infos[sequence]["length"]

    per_text_stats: List[Dict[str, Any]] = []
    overall_correct = 0
    overall_total = 0

    fmt = "{:." + str(max(0, int(coord_decimals))) + "f}"

    # 文件名安全化
    def _sanitize_filename(text: str) -> str:
        text = (text or "").strip().lower()
        text = re.sub(r"\s+", "_", text)
        text = re.sub(r"[^a-z0-9_\-]+", "", text)
        return text[:128] if len(text) > 128 else text

    iterator = enumerate(expressions)
    if show_tqdm:
        from tqdm import tqdm as _tqdm
        iterator = _tqdm(list(iterator), total=len(expressions), desc=f"Eval {sequence} texts", dynamic_ncols=True)
    for sentence_idx, expr in iterator:
        sentence = expr.get("sentence", "")
        label_map = expr.get("label", {})

        batch_images: List[Image.Image] = []
        batch_prompts: List[str] = []
        batch_targets: List[int] = []
        batch_metas: List[Tuple[int, int, str, int, int, int, int]] = []  # (frame_idx, obj_id, coord_str, x, y, w, h)

        n_correct = 0
        n_total = 0

        # 当前表达对应的 refer 行集合，按“写一个保存一个”在句末落盘
        refer_positive_lines: List[str] = []
        infer_stride = max(1, int(infer_every_n_frames or 1))
        last_conf_by_oid: Dict[int, float] = {}

        sample_iter = _iter_samples_for_sentence(
            ds=ds,
            sequence=sequence,
            sentence=sentence,
            label_map=label_map,
            image_size=image_size,
            margin_ratio=margin_ratio,
            margin_px=margin_px,
            min_side=min_side,
            coord_mode=coord_mode,
            fmt=fmt,
            prompt_single_tpl=prompt_single_tpl,
            show_tqdm=show_tqdm,
            preprocess_workers=preprocess_workers,
        )

        for patch, pr, y_true, meta in sample_iter:
            fidx_, oid_, coord_s_, bx, by, bw, bh = meta
            do_full_infer = (infer_stride <= 1) or (int(fidx_) % infer_stride == 0)
            oid_i = int(oid_)
            if (not do_full_infer) and (oid_i in last_conf_by_oid):
                prob = float(last_conf_by_oid[oid_i])
                y_pred = 1 if prob >= threshold else 0
                n_correct += int(y_pred == int(y_true))
                n_total += 1
                refer_positive_lines.append(f"{fidx_},{oid_},{bx},{by},{bw},{bh},{prob:.6f}")
                continue

            batch_images.append(patch)
            batch_prompts.append(pr)
            batch_targets.append(y_true)
            batch_metas.append(meta)

            if len(batch_images) >= batch_size:
                    p_yes = build_and_forward_yes_probs(
                        model=model,
                        processor=processor,
                        images=batch_images,
                        prompts=batch_prompts,
                        device=device,
                        yes_id=yes_id,
                        no_id=no_id,
                    )
                    if bool(enable_refine) and re_refer_thresh is not None and float(re_refer_thresh) > float(lower_bound):
                        p_yes = refine_probs_with_video_for_indices(
                            model=model,
                            processor=processor,
                            ds=ds,
                            seq=sequence,
                            batch_metas=batch_metas,
                            batch_images=batch_images,
                            p_yes=p_yes,
                            sentence=sentence,
                            prompt_video_tpl=prompt_video_tpl,
                            device=device,
                            yes_id=yes_id,
                            no_id=no_id,
                            video_n_frames=video_n_frames,
                            image_size=image_size,
                            margin_ratio=margin_ratio,
                            margin_px=margin_px,
                            min_side=min_side,
                            lower_bound=float(lower_bound),
                            re_refer_thresh=float(re_refer_thresh),
                        )
                    for prob, y_t, meta in zip(p_yes, batch_targets, batch_metas):
                        y_pred = 1 if prob >= threshold else 0
                        n_correct += int(y_pred == y_t)
                        n_total += 1
                        # 记录像素框（不做阈值过滤，便于后续分析）
                        fidx_, oid_, coord_s_, bx, by, bw, bh = meta
                        last_conf_by_oid[int(oid_)] = float(prob)
                        refer_positive_lines.append(f"{fidx_},{oid_},{bx},{by},{bw},{bh},{prob:.6f}")

                    batch_images.clear()
                    batch_prompts.clear()
                    batch_targets.clear()
                    batch_metas.clear()

        if len(expressions) > 0 and len(batch_images) > 0:
            p_yes = build_and_forward_yes_probs(
                model=model,
                processor=processor,
                images=batch_images,
                prompts=batch_prompts,
                device=device,
                yes_id=yes_id,
                no_id=no_id,
            )
            if bool(enable_refine) and re_refer_thresh is not None and float(re_refer_thresh) > float(lower_bound):
                p_yes = refine_probs_with_video_for_indices(
                    model=model,
                    processor=processor,
                    ds=ds,
                    seq=sequence,
                    batch_metas=batch_metas,
                    batch_images=batch_images,
                    p_yes=p_yes,
                    sentence=sentence,
                    prompt_video_tpl=prompt_video_tpl,
                    device=device,
                    yes_id=yes_id,
                    no_id=no_id,
                    video_n_frames=video_n_frames,
                    image_size=image_size,
                    margin_ratio=margin_ratio,
                    margin_px=margin_px,
                    min_side=min_side,
                    lower_bound=float(lower_bound),
                    re_refer_thresh=float(re_refer_thresh),
                )

            for prob, y_t, meta in zip(p_yes, batch_targets, batch_metas):
                y_pred = 1 if prob >= threshold else 0
                n_correct += int(y_pred == y_t)
                n_total += 1
                fidx_, oid_, coord_s_, bx, by, bw, bh = meta
                last_conf_by_oid[int(oid_)] = float(prob)
                refer_positive_lines.append(f"{fidx_},{oid_},{bx},{by},{bw},{bh},{prob:.6f}")

            batch_images.clear()
            batch_prompts.clear()
            batch_targets.clear()
            batch_metas.clear()
            batch_metas.clear()

        # 单条表达写文件（写一个保存一个）
        if refer_dir is not None:
            safe_name = _sanitize_filename(sentence) or f"text_{sentence_idx}"
            try:
                with open(os.path.join(refer_dir, f"{safe_name}.txt"), "w", encoding="utf-8") as f:
                    f.write("\n".join(refer_positive_lines))
            except Exception:
                pass

        acc = float(n_correct) / float(n_total) if n_total > 0 else 0.0
        per_text_stats.append({
            "sentence_idx": sentence_idx,
            "sentence": sentence,
            "num_samples": n_total,
            "acc": acc,
            "correct": n_correct,
        })
        overall_correct += n_correct
        overall_total += n_total

    overall_acc = float(overall_correct) / float(overall_total) if overall_total > 0 else 0.0
    summary: Dict[str, Any] = {
        "sequence": sequence,
        "threshold": threshold,
        "overall_acc": overall_acc,
        "num_texts": len(per_text_stats),
        "overall_total": overall_total,
        "overall_correct": overall_correct,
        "per_text": per_text_stats,
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        step_str = f"step{global_step}" if global_step is not None else "nostep"
        out_path = os.path.join(output_dir, f"eval_{sequence}_{step_str}_{ts}.json")
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    return summary


@torch.no_grad()
def evaluate_all_sequences(
    model: torch.nn.Module,
    processor: AutoProcessor,
    label_tokens: Tuple[int, int],
    ds: ReferKittiMOT,
    image_size: int = 336,
    margin_ratio: float = 0.1,
    margin_px: Optional[int] = None,
    min_side: int = 8,
    coord_mode: str = "xy",
    coord_decimals: int = 3,
    threshold: float = 0.5,
    batch_size: int = 64,
    device: torch.device = torch.device("cpu"),
    output_dir: Optional[str] = None,
    global_step: Optional[int] = None,
    show_tqdm: bool = True,
    re_refer_thresh: float = 1.0,
    re_refer_lower: Optional[float] = None,
    video_n_frames: int = 4,
    prompt_single_tpl: Optional[str] = None,
    prompt_video_tpl: Optional[str] = None,
    enable_refine: bool = True,
    rank: int = 0,
    world_size: int = 1,
    max_texts_per_seq: Optional[int] = None,
    preprocess_workers: int = 0,
    infer_every_n_frames: int = 1,
) -> Dict[str, Any]:
    yes_id, no_id = int(label_tokens[0]), int(label_tokens[1])
    # 参数校验
    if prompt_single_tpl is None or len(str(prompt_single_tpl).strip()) == 0:
        raise ValueError("evaluate_all_sequences 需要提供 prompt_single_tpl（无回退）")
    if bool(enable_refine):
        if re_refer_lower is None:
            raise ValueError("enable_refine=True 时必须显式提供 re_refer_lower（无回退）")
        if not (float(re_refer_thresh) > float(re_refer_lower)):
            raise ValueError(f"re_refer_thresh({re_refer_thresh}) 必须大于 re_refer_lower({re_refer_lower})")
    lower_bound = float(re_refer_lower) if (re_refer_lower is not None) else None
    if bool(enable_refine) and re_refer_thresh is not None and float(re_refer_thresh) > float(lower_bound):
        if prompt_video_tpl is None or len(str(prompt_video_tpl).strip()) == 0:
            raise ValueError("evaluate_all_sequences 二阶段精炼需要提供 prompt_video_tpl（无回退）")

    run_dir = None
    mot_dir = None
    refer_root_dir = None
    if output_dir and rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        step_str = f"step{global_step}" if global_step is not None else "nostep"
        run_dir = os.path.join(output_dir, f"eval_all_{step_str}_{ts}")
        mot_dir = os.path.join(run_dir, "mot")
        refer_root_dir = os.path.join(run_dir, "refer")
        os.makedirs(mot_dir, exist_ok=True)
        os.makedirs(refer_root_dir, exist_ok=True)
    
    # 同步确保主进程创建完目录
    if world_size > 1:
        if dist.is_initialized():
            dist.barrier()
        # 广播 run_dir 给所有进程
        if rank == 0:
            dir_info = [run_dir, mot_dir, refer_root_dir]
        else:
            dir_info = [None, None, None]
        if dist.is_initialized():
            dist.broadcast_object_list(dir_info, src=0)
        run_dir, mot_dir, refer_root_dir = dir_info

    def _sanitize_filename(text: str) -> str:
        text = (text or "").strip().lower()
        text = re.sub(r"\s+", "_", text)
        text = re.sub(r"[^a-z0-9_\-]+", "", text)
        return text[:128] if len(text) > 128 else text

    fmt = "{:." + str(max(0, int(coord_decimals))) + "f}"

    # 所有进程处理所有序列，但在表达式级别分配任务
    all_seqs = ds.sequence_names
    
    if rank == 0:
        print(f"总序列数: {len(all_seqs)}, 使用 {world_size} 个进程并行处理（表达式级并行）")

    all_sequences_stats: List[Dict[str, Any]] = []
    total_correct_all = 0
    total_samples_all = 0
    total_texts_all = 0

    seq_iter = all_seqs
    if show_tqdm and len(seq_iter) > 1 and rank == 0:
        from tqdm import tqdm as _tqdm
        seq_iter = _tqdm(seq_iter, desc=f"Rank {rank} Eval sequences", dynamic_ncols=True)

    for seq in seq_iter:
        if seq not in ds.image_paths:
            all_sequences_stats.append({
                "sequence": seq,
                "overall_acc": 0.0,
                "num_texts": 0,
                "overall_total": 0,
                "overall_correct": 0,
                "per_text": [],
            })
            continue

        if mot_dir is not None:
            mot_lines: List[str] = []
            num_frames_seq = ds.sequence_infos[seq]["length"]
            for fidx in range(num_frames_seq):
                ann = ds.annotations[seq][fidx]
                M = ann["bbox"].shape[0]
                if M == 0:
                    continue
                for ann_idx in range(M):
                    x, y, w, h = ann["bbox"][ann_idx].tolist()
                    obj_id = int(ann["id"][ann_idx].item())
                    mot_lines.append(f"{fidx},{obj_id},{int(x)},{int(y)},{int(w)},{int(h)},{1.0}")
            try:
                with open(os.path.join(mot_dir, f"{seq}.txt"), "w", encoding="utf-8") as f:
                    f.write("\n".join(mot_lines))
            except Exception:
                pass

        expressions = ds._load_expressions_for_sequence(seq)
        if isinstance(max_texts_per_seq, int) and max_texts_per_seq > 0:
            expressions = expressions[: int(max_texts_per_seq)]
        num_frames = ds.sequence_infos[seq]["length"]

        per_text_stats: List[Dict[str, Any]] = []
        overall_correct = 0
        overall_total = 0

        iterator = enumerate(expressions)
        if show_tqdm:
            from tqdm import tqdm as _tqdm
            iterator = _tqdm(list(iterator), total=len(expressions), desc=f"Eval {seq} texts", dynamic_ncols=True)
        for sentence_idx, expr in iterator:
            # 按表达式维度进行多卡切分：不同 rank 处理不同的 sentence
            if world_size > 1 and (sentence_idx % world_size != rank):
                continue
            sentence = expr.get("sentence", "")
            label_map = expr.get("label", {})

            refer_positive_lines: List[str] = []

            batch_images: List[Image.Image] = []
            batch_prompts: List[str] = []
            batch_targets: List[int] = []
            batch_metas: List[Tuple[int, int, str, int, int, int, int]] = []  # (frame,obj_id,coord_str,x,y,w,h)
            infer_stride = max(1, int(infer_every_n_frames or 1))
            last_conf_by_oid: Dict[int, float] = {}

            n_correct = 0
            n_total = 0

            sample_iter = _iter_samples_for_sentence(
                ds=ds,
                sequence=seq,
                sentence=sentence,
                label_map=label_map,
                image_size=image_size,
                margin_ratio=margin_ratio,
                margin_px=margin_px,
                min_side=min_side,
                coord_mode=coord_mode,
                fmt=fmt,
                prompt_single_tpl=prompt_single_tpl,
                show_tqdm=show_tqdm and rank == 0,
                preprocess_workers=preprocess_workers,
            )

            for patch, pr, y_true, meta in sample_iter:
                fidx, oid, coord_s, bx, by, bw, bh = meta
                do_full_infer = (infer_stride <= 1) or (int(fidx) % infer_stride == 0)
                oid_i = int(oid)
                if (not do_full_infer) and (oid_i in last_conf_by_oid):
                    prob = float(last_conf_by_oid[oid_i])
                    y_pred = 1 if prob >= threshold else 0
                    n_correct += int(y_pred == int(y_true))
                    n_total += 1
                    refer_positive_lines.append(f"{fidx},{oid},{bx},{by},{bw},{bh},{prob:.6f}")
                    continue

                batch_images.append(patch)
                batch_prompts.append(pr)
                batch_targets.append(y_true)
                batch_metas.append(meta)

                if len(batch_images) >= batch_size:
                        p_yes = build_and_forward_yes_probs(
                            model=model,
                            processor=processor,
                            images=batch_images,
                            prompts=batch_prompts,
                            device=device,
                            yes_id=yes_id,
                            no_id=no_id,
                        )
                        if bool(enable_refine) and re_refer_thresh is not None and float(re_refer_thresh) > float(lower_bound):
                            p_yes = refine_probs_with_video_for_indices(
                                model=model,
                                processor=processor,
                                ds=ds,
                                seq=seq,
                                batch_metas=batch_metas,
                                batch_images=batch_images,
                                p_yes=p_yes,
                                sentence=sentence,
                                prompt_video_tpl=prompt_video_tpl,
                                device=device,
                                yes_id=yes_id,
                                no_id=no_id,
                                video_n_frames=video_n_frames,
                                image_size=image_size,
                                margin_ratio=margin_ratio,
                                margin_px=margin_px,
                                min_side=min_side,
                                lower_bound=float(lower_bound),
                                re_refer_thresh=float(re_refer_thresh),
                            )

                        for prob, y_t, meta in zip(p_yes, batch_targets, batch_metas):
                            y_pred = 1 if prob >= threshold else 0
                            n_correct += int(y_pred == y_t)
                            n_total += 1
                            fidx, oid, coord_s, bx, by, bw, bh = meta
                            last_conf_by_oid[int(oid)] = float(prob)
                            refer_positive_lines.append(f"{fidx},{oid},{bx},{by},{bw},{bh},{prob:.6f}")

                        batch_images.clear()
                        batch_prompts.clear()
                        batch_targets.clear()
                        batch_metas.clear()

            if len(batch_images) > 0:
                p_yes = build_and_forward_yes_probs(
                    model=model,
                    processor=processor,
                    images=batch_images,
                    prompts=batch_prompts,
                    device=device,
                    yes_id=yes_id,
                    no_id=no_id,
                )
                if bool(enable_refine) and re_refer_thresh is not None and float(re_refer_thresh) > float(lower_bound):
                    p_yes = refine_probs_with_video_for_indices(
                        model=model,
                        processor=processor,
                        ds=ds,
                        seq=seq,
                        batch_metas=batch_metas,
                        batch_images=batch_images,
                        p_yes=p_yes,
                        sentence=sentence,
                        prompt_video_tpl=prompt_video_tpl,
                        device=device,
                        yes_id=yes_id,
                        no_id=no_id,
                        video_n_frames=video_n_frames,
                        image_size=image_size,
                        margin_ratio=margin_ratio,
                        margin_px=margin_px,
                        min_side=min_side,
                        lower_bound=float(lower_bound),
                        re_refer_thresh=float(re_refer_thresh),
                    )

                for prob, y_t, meta in zip(p_yes, batch_targets, batch_metas):
                    y_pred = 1 if prob >= threshold else 0
                    n_correct += int(y_pred == y_t)
                    n_total += 1
                    fidx, oid, coord_s, bx, by, bw, bh = meta
                    last_conf_by_oid[int(oid)] = float(prob)
                    refer_positive_lines.append(f"{fidx},{oid},{bx},{by},{bw},{bh},{prob:.6f}")

                batch_images.clear()
                batch_prompts.clear()
                batch_targets.clear()
                batch_metas.clear()

            if refer_root_dir is not None:
                seq_ref_dir = os.path.join(refer_root_dir, seq)
                os.makedirs(seq_ref_dir, exist_ok=True)
                safe_name = _sanitize_filename(sentence) or f"text_{sentence_idx}"
                try:
                    with open(os.path.join(seq_ref_dir, f"{safe_name}.txt"), "w", encoding="utf-8") as f:
                        f.write("\n".join(refer_positive_lines))
                except Exception:
                    pass

            acc = float(n_correct) / float(n_total) if n_total > 0 else 0.0
            per_text_stats.append({
                "sentence_idx": sentence_idx,
                "sentence": sentence,
                "num_samples": n_total,
                "acc": acc,
                "correct": n_correct,
            })
            overall_correct += n_correct
            overall_total += n_total

        overall_acc = float(overall_correct) / float(overall_total) if overall_total > 0 else 0.0
        seq_summary: Dict[str, Any] = {
            "sequence": seq,
            "threshold": threshold,
            "overall_acc": overall_acc,
            "num_texts": len(per_text_stats),
            "overall_total": overall_total,
            "overall_correct": overall_correct,
            "per_text": per_text_stats,
        }
        all_sequences_stats.append(seq_summary)
        total_correct_all += overall_correct
        total_samples_all += overall_total
        total_texts_all += len(per_text_stats)

    # 同步所有进程
    if world_size > 1 and dist.is_initialized():
        dist.barrier()
    
    # 收集所有进程的结果
    all_proc_stats = gather_results(all_sequences_stats, world_size)
    
    # 主进程汇总（合并相同序列和文本的统计）
    if rank == 0:
        # 按 (sequence, text) 合并统计
        merged_stats: Dict[str, Dict[str, Any]] = {}  # seq -> per_text_stats dict
        
        for proc_stats in all_proc_stats:
            for seq_stat in proc_stats:
                seq_name = seq_stat["sequence"]
                if seq_name not in merged_stats:
                    merged_stats[seq_name] = {
                        "sequence": seq_name,
                        "threshold": threshold,
                        "per_text": {},  # sentence_idx -> stats
                    }
                
                # 合并每个文本的统计
                for text_stat in seq_stat.get("per_text", []):
                    sent_idx = text_stat["sentence_idx"]
                    if sent_idx not in merged_stats[seq_name]["per_text"]:
                        merged_stats[seq_name]["per_text"][sent_idx] = {
                            "sentence_idx": sent_idx,
                            "sentence": text_stat["sentence"],
                            "num_samples": 0,
                            "correct": 0,
                        }
                    merged_stats[seq_name]["per_text"][sent_idx]["num_samples"] += text_stat.get("num_samples", 0)
                    merged_stats[seq_name]["per_text"][sent_idx]["correct"] += text_stat.get("correct", 0)
        
        # 转换为列表格式并计算准确率
        final_sequences_stats: List[Dict[str, Any]] = []
        total_correct_all = 0
        total_samples_all = 0
        total_texts_all = 0
        
        for seq_name in sorted(merged_stats.keys()):
            seq_data = merged_stats[seq_name]
            per_text_list = []
            seq_correct = 0
            seq_total = 0
            
            for sent_idx in sorted(seq_data["per_text"].keys()):
                text_stat = seq_data["per_text"][sent_idx]
                acc = float(text_stat["correct"]) / float(text_stat["num_samples"]) if text_stat["num_samples"] > 0 else 0.0
                per_text_list.append({
                    "sentence_idx": text_stat["sentence_idx"],
                    "sentence": text_stat["sentence"],
                    "num_samples": text_stat["num_samples"],
                    "acc": acc,
                    "correct": text_stat["correct"],
                })
                seq_correct += text_stat["correct"]
                seq_total += text_stat["num_samples"]
            
            seq_acc = float(seq_correct) / float(seq_total) if seq_total > 0 else 0.0
            final_sequences_stats.append({
                "sequence": seq_name,
                "threshold": threshold,
                "overall_acc": seq_acc,
                "num_texts": len(per_text_list),
                "overall_total": seq_total,
                "overall_correct": seq_correct,
                "per_text": per_text_list,
            })
            
            total_correct_all += seq_correct
            total_samples_all += seq_total
            total_texts_all += len(per_text_list)
        
        all_sequences_stats = final_sequences_stats
        
        overall_acc_all = float(total_correct_all) / float(total_samples_all) if total_samples_all > 0 else 0.0
        seq_overall = []
        for s in sorted(all_sequences_stats, key=lambda x: x.get("sequence", "")):
            seq_overall.append({
                "sequence": s.get("sequence"),
                "overall_acc": s.get("overall_acc", 0.0),
                "overall_total": s.get("overall_total", 0),
                "overall_correct": s.get("overall_correct", 0),
                "num_texts": s.get("num_texts", 0),
            })
        best_seq = sorted(seq_overall, key=lambda x: (-x.get("overall_acc", 0.0), -x.get("overall_total", 0)))[:3]
        worst_seq = sorted(seq_overall, key=lambda x: (x.get("overall_acc", 0.0), -x.get("overall_total", 0)))[:3]
        summary: Dict[str, Any] = OrderedDict()
        summary["sequence"] = "all"
        summary["threshold"] = threshold
        summary["overall_acc"] = overall_acc_all
        summary["num_texts"] = total_texts_all
        summary["overall_total"] = total_samples_all
        summary["overall_correct"] = total_correct_all
        summary["sequences_overall"] = seq_overall
        summary["best_sequences"] = best_seq
        summary["worst_sequences"] = worst_seq
        summary["sequences"] = sorted(all_sequences_stats, key=lambda x: x.get("sequence", ""))
    else:
        summary: Dict[str, Any] = OrderedDict()
        summary["sequence"] = "all"
        summary["threshold"] = threshold
        summary["overall_acc"] = 0.0
        summary["num_texts"] = 0
        summary["overall_total"] = 0
        summary["overall_correct"] = 0
        summary["sequences_overall"] = []
        summary["best_sequences"] = []
        summary["worst_sequences"] = []
        summary["sequences"] = []

    # 只有主进程保存结果
    if rank == 0:
        if run_dir is not None:
            try:
                with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
        if output_dir:
            try:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                step_str = f"step{global_step}" if global_step is not None else "nostep"
                out_path = os.path.join(output_dir, f"eval_all_{step_str}_{ts}.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

    return summary


@torch.no_grad()
def evaluate_all_sequences_multi_thresholds(
    model: torch.nn.Module,
    processor: AutoProcessor,
    label_tokens: Tuple[int, int],
    ds: ReferKittiMOT,
    thresholds: List[float],
    image_size: int = 336,
    margin_ratio: float = 0.1,
    margin_px: Optional[int] = None,
    min_side: int = 8,
    coord_mode: str = "xy",
    coord_decimals: int = 3,
    batch_size: int = 64,
    device: torch.device = torch.device("cpu"),
    output_dir: Optional[str] = None,
    global_step: Optional[int] = None,
    show_tqdm: bool = True,
    re_refer_thresh: float = 1.0,
    re_refer_lower: Optional[float] = None,
    video_n_frames: int = 4,
    prompt_single_tpl: Optional[str] = None,
    prompt_video_tpl: Optional[str] = None,
    enable_refine: bool = True,
    rank: int = 0,
    world_size: int = 1,
    max_texts_per_seq: Optional[int] = None,
    preprocess_workers: int = 0,
    infer_every_n_frames: int = 1,
) -> Dict[str, Any]:
    """一次前向，返回所有阈值的评估结果并分别落盘。概率文件仅写一次。"""
    yes_id, no_id = int(label_tokens[0]), int(label_tokens[1])
    if prompt_single_tpl is None or len(str(prompt_single_tpl).strip()) == 0:
        raise ValueError("evaluate_all_sequences_multi_thresholds 需要提供 prompt_single_tpl")
    if bool(enable_refine):
        if re_refer_lower is None:
            raise ValueError("enable_refine=True 时必须显式提供 re_refer_lower（无回退）")
        if not (float(re_refer_thresh) > float(re_refer_lower)):
            raise ValueError(f"re_refer_thresh({re_refer_thresh}) 必须大于 re_refer_lower({re_refer_lower})")
    lower_bound = float(re_refer_lower) if (re_refer_lower is not None) else None
    if bool(enable_refine) and re_refer_thresh is not None and float(re_refer_thresh) > float(lower_bound):
        if prompt_video_tpl is None or len(str(prompt_video_tpl).strip()) == 0:
            raise ValueError("evaluate_all_sequences_multi_thresholds 二阶段精炼需要提供 prompt_video_tpl（无回退）")

    # 目录
    run_dir = None
    mot_dir = None
    refer_root_dir = None
    if output_dir and rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        step_str = f"step{global_step}" if global_step is not None else "nostep"
        run_dir = os.path.join(output_dir, f"eval_all_{step_str}_{ts}")
        # 概率只写一次
        mot_dir = os.path.join(run_dir, "probs", "mot")
        refer_root_dir = os.path.join(run_dir, "probs", "refer")
        os.makedirs(mot_dir, exist_ok=True)
        os.makedirs(refer_root_dir, exist_ok=True)
        # 为每个阈值预创建目录
        for th in thresholds:
            os.makedirs(os.path.join(run_dir, f"th_{float(th):.3f}"), exist_ok=True)
    # 同步并广播
    if world_size > 1:
        if dist.is_initialized():
            dist.barrier()
        if rank == 0:
            dir_info = [run_dir, mot_dir, refer_root_dir]
        else:
            dir_info = [None, None, None]
        if dist.is_initialized():
            dist.broadcast_object_list(dir_info, src=0)
        run_dir, mot_dir, refer_root_dir = dir_info

    # 准备结构
    fmt = "{:." + str(max(0, int(coord_decimals))) + "f}"
    all_seqs = ds.sequence_names
    if rank == 0:
        print(f"总序列数: {len(all_seqs)}, 使用 {world_size} 个进程并行处理（表达式级并行）")

    # 为每个阈值维护统计
    th_list: List[float] = [float(t) for t in thresholds]
    per_th_all_sequences_stats: Dict[float, List[Dict[str, Any]]] = {t: [] for t in th_list}

    seq_iter = all_seqs
    if show_tqdm and len(seq_iter) > 1 and rank == 0:
        from tqdm import tqdm as _tqdm
        seq_iter = _tqdm(seq_iter, desc=f"Rank {rank} Eval sequences", dynamic_ncols=True)

    for seq in seq_iter:
        if seq not in ds.image_paths:
            for t in th_list:
                per_th_all_sequences_stats[t].append({
                    "sequence": seq,
                    "threshold": t,
                    "overall_acc": 0.0,
                    "num_texts": 0,
                    "overall_total": 0,
                    "overall_correct": 0,
                    "per_text": [],
                })
            continue

        # 写 MOT 一次
        if mot_dir is not None:
            mot_lines: List[str] = []
            num_frames_seq = ds.sequence_infos[seq]["length"]
            for fidx in range(num_frames_seq):
                ann = ds.annotations[seq][fidx]
                M = ann["bbox"].shape[0]
                if M == 0:
                    continue
                for ann_idx in range(M):
                    x, y, w, h = ann["bbox"][ann_idx].tolist()
                    obj_id = int(ann["id"][ann_idx].item())
                    mot_lines.append(f"{fidx},{obj_id},{int(x)},{int(y)},{int(w)},{int(h)},{1.0}")
            try:
                with open(os.path.join(mot_dir, f"{seq}.txt"), "w", encoding="utf-8") as f:
                    f.write("\n".join(mot_lines))
            except Exception:
                pass

        expressions = ds._load_expressions_for_sequence(seq)
        if isinstance(max_texts_per_seq, int) and max_texts_per_seq > 0:
            expressions = expressions[: int(max_texts_per_seq)]
        num_frames = ds.sequence_infos[seq]["length"]

        # 为每个阈值准备本序列聚合容器
        per_th_per_text_stats: Dict[float, List[Dict[str, Any]]] = {t: [] for t in th_list}
        per_th_overall_correct: Dict[float, int] = {t: 0 for t in th_list}
        per_th_overall_total: Dict[float, int] = {t: 0 for t in th_list}

        iterator = enumerate(expressions)
        if show_tqdm:
            from tqdm import tqdm as _tqdm
            iterator = _tqdm(list(iterator), total=len(expressions), desc=f"Eval {seq} texts", dynamic_ncols=True)
        for sentence_idx, expr in iterator:
            # 表达式级切分
            if world_size > 1 and (sentence_idx % world_size != rank):
                continue
            sentence = expr.get("sentence", "")
            label_map = expr.get("label", {})

            # 概率文件
            refer_positive_lines: List[str] = []

            batch_images: List[Image.Image] = []
            batch_prompts: List[str] = []
            batch_targets: List[int] = []
            batch_metas: List[Tuple[int, int, str, int, int, int, int]] = []
            infer_stride = max(1, int(infer_every_n_frames or 1))
            last_conf_by_oid: Dict[int, float] = {}

            # 为每个阈值准备计数器
            per_th_correct = {t: 0 for t in th_list}
            per_th_total = {t: 0 for t in th_list}

            sample_iter = _iter_samples_for_sentence(
                ds=ds,
                sequence=seq,
                sentence=sentence,
                label_map=label_map,
                image_size=image_size,
                margin_ratio=margin_ratio,
                margin_px=margin_px,
                min_side=min_side,
                coord_mode=coord_mode,
                fmt=fmt,
                prompt_single_tpl=prompt_single_tpl,
                show_tqdm=show_tqdm and rank == 0,
                preprocess_workers=preprocess_workers,
            )

            for patch, pr, y_true, meta in sample_iter:
                fidx, oid, bx, by, bw, bh = meta[0], meta[1], meta[3], meta[4], meta[5], meta[6]
                do_full_infer = (infer_stride <= 1) or (int(fidx) % infer_stride == 0)
                oid_i = int(oid)
                if (not do_full_infer) and (oid_i in last_conf_by_oid):
                    prob = float(last_conf_by_oid[oid_i])
                    for t in th_list:
                        y_pred = 1 if prob >= t else 0
                        per_th_correct[t] += int(y_pred == int(y_true))
                        per_th_total[t] += 1
                    refer_positive_lines.append(f"{fidx},{oid},{bx},{by},{bw},{bh},{prob:.6f}")
                    continue

                batch_images.append(patch)
                batch_prompts.append(pr)
                batch_targets.append(y_true)
                batch_metas.append(meta)

                if len(batch_images) >= batch_size:
                        p_yes = build_and_forward_yes_probs(
                            model=model,
                            processor=processor,
                            images=batch_images,
                            prompts=batch_prompts,
                            device=device,
                            yes_id=yes_id,
                            no_id=no_id,
                        )
                        if bool(enable_refine) and re_refer_thresh is not None and float(re_refer_thresh) > float(lower_bound):
                            p_yes = refine_probs_with_video_for_indices(
                                model=model,
                                processor=processor,
                                ds=ds,
                                seq=seq,
                                batch_metas=batch_metas,
                                batch_images=batch_images,
                                p_yes=p_yes,
                                sentence=sentence,
                                prompt_video_tpl=prompt_video_tpl,
                                device=device,
                                yes_id=yes_id,
                                no_id=no_id,
                                video_n_frames=video_n_frames,
                                image_size=image_size,
                                margin_ratio=margin_ratio,
                                margin_px=margin_px,
                                min_side=min_side,
                                lower_bound=float(lower_bound),
                                re_refer_thresh=float(re_refer_thresh),
                            )

                        # 累计到所有阈值
                        for prob, y_t, meta in zip(p_yes, batch_targets, batch_metas):
                            last_conf_by_oid[int(meta[1])] = float(prob)
                            for t in th_list:
                                y_pred = 1 if prob >= t else 0
                                per_th_correct[t] += int(y_pred == y_t)
                                per_th_total[t] += 1
                            fidx, oid, bx, by, bw, bh = meta[0], meta[1], meta[3], meta[4], meta[5], meta[6]
                            refer_positive_lines.append(f"{fidx},{oid},{bx},{by},{bw},{bh},{prob:.6f}")

                        batch_images.clear()
                        batch_prompts.clear()
                        batch_targets.clear()
                        batch_metas.clear()

            if len(batch_images) > 0:
                p_yes = build_and_forward_yes_probs(
                    model=model,
                    processor=processor,
                    images=batch_images,
                    prompts=batch_prompts,
                    device=device,
                    yes_id=yes_id,
                    no_id=no_id,
                )
                if bool(enable_refine) and re_refer_thresh is not None and float(re_refer_thresh) > float(lower_bound):
                    p_yes = refine_probs_with_video_for_indices(
                        model=model,
                        processor=processor,
                        ds=ds,
                        seq=seq,
                        batch_metas=batch_metas,
                        batch_images=batch_images,
                        p_yes=p_yes,
                        sentence=sentence,
                        prompt_video_tpl=prompt_video_tpl,
                        device=device,
                        yes_id=yes_id,
                        no_id=no_id,
                        video_n_frames=video_n_frames,
                        image_size=image_size,
                        margin_ratio=margin_ratio,
                        margin_px=margin_px,
                        min_side=min_side,
                        lower_bound=float(lower_bound),
                        re_refer_thresh=float(re_refer_thresh),
                    )

                for prob, y_t, meta in zip(p_yes, batch_targets, batch_metas):
                    last_conf_by_oid[int(meta[1])] = float(prob)
                    for t in th_list:
                        y_pred = 1 if prob >= t else 0
                        per_th_correct[t] += int(y_pred == y_t)
                        per_th_total[t] += 1
                    fidx, oid, bx, by, bw, bh = meta[0], meta[1], meta[3], meta[4], meta[5], meta[6]
                    refer_positive_lines.append(f"{fidx},{oid},{bx},{by},{bw},{bh},{prob:.6f}")

                batch_images.clear()
                batch_prompts.clear()
                batch_targets.clear()
                batch_metas.clear()

            # 写该 expression 的概率文件（一次）
            if refer_root_dir is not None:
                seq_ref_dir = os.path.join(refer_root_dir, seq)
                os.makedirs(seq_ref_dir, exist_ok=True)
                safe_name = re.sub(r"\s+", "_", (sentence or "").strip().lower())
                safe_name = re.sub(r"[^a-z0-9_\-]+", "", safe_name)[:128] or f"text_{sentence_idx}"
                try:
                    with open(os.path.join(seq_ref_dir, f"{safe_name}.txt"), "w", encoding="utf-8") as f:
                        f.write("\n".join(refer_positive_lines))
                except Exception:
                    pass

            # 完成该 expression 后，累加到序列级（各阈值）
            for t in th_list:
                acc = float(per_th_correct[t]) / float(per_th_total[t]) if per_th_total[t] > 0 else 0.0
                per_th_per_text_stats[t].append({
                    "sentence_idx": sentence_idx,
                    "sentence": sentence,
                    "num_samples": per_th_total[t],
                    "acc": acc,
                    "correct": per_th_correct[t],
                })
                per_th_overall_correct[t] += per_th_correct[t]
                per_th_overall_total[t] += per_th_total[t]

        # 序列结束后，构造每个阈值的序列摘要
        for t in th_list:
            overall_acc = float(per_th_overall_correct[t]) / float(per_th_overall_total[t]) if per_th_overall_total[t] > 0 else 0.0
            seq_summary = {
                "sequence": seq,
                "threshold": t,
                "overall_acc": overall_acc,
                "num_texts": len(per_th_per_text_stats[t]),
                "overall_total": per_th_overall_total[t],
                "overall_correct": per_th_overall_correct[t],
                "per_text": per_th_per_text_stats[t],
            }
            per_th_all_sequences_stats[t].append(seq_summary)

    # 同步并收集
    if world_size > 1 and dist.is_initialized():
        dist.barrier()
    all_proc_stats_per_th = {t: gather_results(per_th_all_sequences_stats[t], world_size) for t in th_list}

    # 主进程合并并落盘
    results: Dict[str, Any] = {}
    if rank == 0:
        for t in th_list:
            # 合并
            merged_stats: Dict[str, Dict[str, Any]] = {}
            for proc_stats in all_proc_stats_per_th[t]:
                for seq_stat in proc_stats:
                    seq_name = seq_stat["sequence"]
                    if seq_name not in merged_stats:
                        merged_stats[seq_name] = {"per_text": {}}
                    for text_stat in seq_stat.get("per_text", []):
                        sent_idx = text_stat["sentence_idx"]
                        if sent_idx not in merged_stats[seq_name]["per_text"]:
                            merged_stats[seq_name]["per_text"][sent_idx] = {
                                "sentence_idx": sent_idx,
                                "sentence": text_stat["sentence"],
                                "num_samples": 0,
                                "correct": 0,
                            }
                        merged_stats[seq_name]["per_text"][sent_idx]["num_samples"] += text_stat.get("num_samples", 0)
                        merged_stats[seq_name]["per_text"][sent_idx]["correct"] += text_stat.get("correct", 0)

            final_sequences_stats: List[Dict[str, Any]] = []
            total_correct_all = 0
            total_samples_all = 0
            total_texts_all = 0
            for seq_name in sorted(merged_stats.keys()):
                per_text_list = []
                seq_correct = 0
                seq_total = 0
                for sent_idx in sorted(merged_stats[seq_name]["per_text"].keys()):
                    st = merged_stats[seq_name]["per_text"][sent_idx]
                    acc = float(st["correct"]) / float(st["num_samples"]) if st["num_samples"] > 0 else 0.0
                    per_text_list.append({
                        "sentence_idx": st["sentence_idx"],
                        "sentence": st["sentence"],
                        "num_samples": st["num_samples"],
                        "acc": acc,
                        "correct": st["correct"],
                    })
                    seq_correct += st["correct"]
                    seq_total += st["num_samples"]
                seq_acc = float(seq_correct) / float(seq_total) if seq_total > 0 else 0.0
                final_sequences_stats.append({
                    "sequence": seq_name,
                    "threshold": t,
                    "overall_acc": seq_acc,
                    "num_texts": len(per_text_list),
                    "overall_total": seq_total,
                    "overall_correct": seq_correct,
                    "per_text": per_text_list,
                })
                total_correct_all += seq_correct
                total_samples_all += seq_total
                total_texts_all += len(per_text_list)

            seq_overall = []
            for s in sorted(final_sequences_stats, key=lambda x: x.get("sequence", "")):
                seq_overall.append({
                    "sequence": s.get("sequence"),
                    "overall_acc": s.get("overall_acc", 0.0),
                    "overall_total": s.get("overall_total", 0),
                    "overall_correct": s.get("overall_correct", 0),
                    "num_texts": s.get("num_texts", 0),
                })
            best_seq = sorted(seq_overall, key=lambda x: (-x.get("overall_acc", 0.0), -x.get("overall_total", 0)))[:3]
            worst_seq = sorted(seq_overall, key=lambda x: (x.get("overall_acc", 0.0), -x.get("overall_total", 0)))[:3]
            summary = OrderedDict()
            summary["sequence"] = "all"
            summary["threshold"] = t
            summary["overall_acc"] = float(total_correct_all) / float(total_samples_all) if total_samples_all > 0 else 0.0
            summary["num_texts"] = total_texts_all
            summary["overall_total"] = total_samples_all
            summary["overall_correct"] = total_correct_all
            summary["sequences_overall"] = seq_overall
            summary["best_sequences"] = best_seq
            summary["worst_sequences"] = worst_seq
            summary["sequences"] = sorted(final_sequences_stats, key=lambda x: x.get("sequence", ""))
            results[f"{t:.3f}"] = summary

            # 保存该阈值的 summary.json
            if run_dir is not None:
                try:
                    th_dir = os.path.join(run_dir, f"th_{t:.3f}")
                    os.makedirs(th_dir, exist_ok=True)
                    with open(os.path.join(th_dir, "summary.json"), "w", encoding="utf-8") as f:
                        json.dump(summary, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass
            if output_dir:
                try:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    step_str = f"step{global_step}" if global_step is not None else "nostep"
                    out_path = os.path.join(output_dir, f"eval_all_{step_str}_th{t:.3f}_{ts}.json")
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(summary, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass

    return results


def _build_argparser():
    import argparse
    p = argparse.ArgumentParser()
    # 模型/精度
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--use_4bit", action="store_true")
    p.add_argument("--fp16", action="store_true")
    #注意只放lora权重的目录
    p.add_argument("--lora_path", type=str, default='llm_outputs/qwenvl_lora_v2/lora_step_20000_piconly')

    # 数据与评估
    p.add_argument("--data_root", type=str, default="/data/sq_2023/refer_kitti_v2")
    p.add_argument("--dataset_version", type=str, default="v2", choices=["v1", "v2"], help="选择数据集版本：v1 或 v2")
    p.add_argument("--sequence", type=str, default="0019")
    p.add_argument("--image_size", type=int, default=336)
    p.add_argument("--margin_ratio", type=float, default=0.2)
    p.add_argument("--margin_px", type=int, default=None)
    p.add_argument("--min_side", type=int, default=8)
    p.add_argument("--coord_mode", type=str, default="xy", choices=["xy", "xywh"])
    p.add_argument("--coord_decimals", type=int, default=3)
    p.add_argument("--threshold", type=float, default=0.4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--eval_output_dir", type=str, default="./llm_eval")
    p.add_argument("--eval_show_tqdm", type=int, default=1)
    p.add_argument("--preprocess_workers", type=int, default=4, help="评估时用于图像裁剪/构造的线程数（0 表示单线程）")
    # 二阶段与 Prompt 模板
    p.add_argument("--re_refer_thresh", type=float, default=1.0, help="二阶段精炼阈值上限（与 re_refer_lower 构成触发区间 [re_refer_lower, re_refer_thresh)）")
    p.add_argument("--re_refer_lower", type=float, default=None, help="二阶段精炼触发下限（必须显式提供；无回退）")
    p.add_argument("--video_n_frames", type=int, default=4, help="视频精炼阶段使用的历史帧数")
    p.add_argument("--prompt_single_tpl", type=str, default="The normalized position of the car or person in the picture is <{coord}>.Determine whether this description matches this image: {sentence}. Answer Yes or No.", help="单帧模板，必须提供，支持 {coord}, {sentence}")
    p.add_argument("--prompt_video_tpl", type=str, default="This is a short video clip of a car or person at <{coord}> across frames. The target may include motion cues; consider background and temporal context when deciding if the description matches this target: {sentence}. Answer Yes or No.", help="视频模板，启用视频/二阶段时必须提供，支持 {sentence}, {coord}")
    # p.add_argument("--prompt_video_tpl", type=str, default="This is a short video clip of a car or person at <{coord}> across frames.The target may include motion cues; consider background and temporal context when making your decision.Pay attention to the person or vehicle near the center region of the video, and if the target is a person, consider gender appearance (male or female) when deciding if the description matches this target: {sentence}.Answer Yes or No.", help="视频模板，启用二阶段时使用，支持 {sentence}, {coord}")
    # p.add_argument("--prompt_video_tpl", type=str, default="This is a short sequence of consecutive cropped and resized square images showing a car or person at <{coord}> across frames. Observe background changes to judge whether the target is moving or stationary. Consider temporal consistency and gender when applicable when deciding if the description matches this target: {sentence}. Answer Yes or No.", help="视频模板，启用二阶段时使用，支持 {sentence}, {coord}")
    p.add_argument("--disable_refine", action="store_true", help="不使用二阶段精炼（忽略 re_refer_thresh 与视频模板）")
    # 分布式
    p.add_argument("--local_rank", type=int, default=-1, help="DDP: local rank")
    return p


def main():
    args = _build_argparser().parse_args()
    
    # 初始化分布式环境
    rank, world_size, local_rank = setup_distributed()
    
    # 设置设备
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if is_main_process():
        print("正在加载模型...")
    
    model, processor, label_tokens = build_model_and_processor_for_eval(
        model_name=args.model_name,
        use_4bit=bool(args.use_4bit),
        bf16=not bool(args.fp16),
        lora_path=args.lora_path,
    )
    model.to(device)
    
    if is_main_process():
        print("模型加载完成")

    # 若用户未显式修改 data_root，则根据版本切换默认路径
    if args.dataset_version == "v1" and args.data_root == "/data/sq_2023/refer_kitti_v2":
        args.data_root = "/data/sq_2023/refer_kitti"
    if args.dataset_version == "v2" and args.data_root == "/data/sq_2023/refer_kitti":
        args.data_root = "/data/sq_2023/refer_kitti_v2"
    train_ids_override, val_ids_override = _split_overrides_by_version(args.dataset_version)

    if args.sequence == "all":
        ds = ReferKittiMOT(
            data_root=args.data_root,
            split="val",
            load_annotation=True,
            expression_sub_dir="expression",
            labels_with_ids_sub_dir="labels_with_ids/image_02",
            train_ids_override=train_ids_override,
            val_ids_override=val_ids_override,
        )
        summary = evaluate_all_sequences(
            model=model,
            processor=processor,
            label_tokens=label_tokens,
            ds=ds,
            image_size=args.image_size,
            margin_ratio=args.margin_ratio,
            margin_px=args.margin_px,
            min_side=args.min_side,
            coord_mode=args.coord_mode,
            coord_decimals=args.coord_decimals,
            threshold=args.threshold,
            batch_size=args.batch_size,
            device=device,
            output_dir=args.eval_output_dir,
            global_step=None,
            show_tqdm=bool(args.eval_show_tqdm),
            re_refer_thresh=getattr(args, "re_refer_thresh", 1.0),
            re_refer_lower=getattr(args, "re_refer_lower", None),
            video_n_frames=getattr(args, "video_n_frames", 4),
            prompt_single_tpl=getattr(args, "prompt_single_tpl", None),
            prompt_video_tpl=getattr(args, "prompt_video_tpl", None),
            enable_refine=not bool(getattr(args, "disable_refine", False)),
            rank=rank,
            world_size=world_size,
            preprocess_workers=int(getattr(args, "preprocess_workers", 0)),
        )
    else:
        if world_size > 1 and is_main_process():
            print("警告: 单序列评估不支持多卡并行，将只使用 rank 0")
        
        if rank == 0:
            summary = evaluate(
                model=model,
                processor=processor,
                label_tokens=label_tokens,
                data_root=args.data_root,
                sequence=args.sequence,
                dataset_version=args.dataset_version,
                image_size=args.image_size,
                margin_ratio=args.margin_ratio,
                margin_px=args.margin_px,
                min_side=args.min_side,
                coord_mode=args.coord_mode,
                coord_decimals=args.coord_decimals,
                threshold=args.threshold,
                batch_size=args.batch_size,
                device=device,
                output_dir=args.eval_output_dir,
                global_step=None,
                show_tqdm=bool(args.eval_show_tqdm),
                re_refer_thresh=getattr(args, "re_refer_thresh", 1.0),
                re_refer_lower=getattr(args, "re_refer_lower", None),
                video_n_frames=getattr(args, "video_n_frames", 4),
                prompt_single_tpl=getattr(args, "prompt_single_tpl", None),
                prompt_video_tpl=getattr(args, "prompt_video_tpl", None),
                enable_refine=not bool(getattr(args, "disable_refine", False)),
                preprocess_workers=int(getattr(args, "preprocess_workers", 0)),
            )
        else:
            summary = {"sequence": args.sequence, "overall_acc": 0.0, "num_texts": 0, "overall_total": 0}

    # 控制台轻量显示 (只有主进程)
    if is_main_process():
        print(json.dumps({
            "sequence": summary.get("sequence"),
            "overall_acc": summary.get("overall_acc"),
            "num_texts": summary.get("num_texts"),
            "overall_total": summary.get("overall_total"),
        }, ensure_ascii=False))
    
    # 清理分布式环境
    cleanup_distributed()


if __name__ == "__main__":
    main()


# 供训练时直接按保存目录加载评估的便捷函数
@torch.no_grad()
def load_and_evaluate(
    model_name: str,
    lora_path: Optional[str],
    data_root: str,
    sequence: str = "all",
    dataset_version: str = "v1",
    image_size: int = 336,
    margin_ratio: float = 0.1,
    margin_px: Optional[int] = None,
    min_side: int = 8,
    coord_mode: str = "xy",
    coord_decimals: int = 3,
    threshold: float = 0.5,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
    output_dir: Optional[str] = None,
    global_step: Optional[int] = None,
    show_tqdm: bool = True,
) -> Dict[str, Any]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor, label_tokens = build_model_and_processor_for_eval(
        model_name=model_name,
        use_4bit=False,
        bf16=True,
        lora_path=lora_path,
    )
    model.to(device)
    if sequence == "all":
        train_ids_override, val_ids_override = _split_overrides_by_version(dataset_version)
        ds = ReferKittiMOT(
            data_root=data_root,
            split="val",
            load_annotation=True,
            expression_sub_dir="expression",
            labels_with_ids_sub_dir="labels_with_ids/image_02",
            train_ids_override=train_ids_override,
            val_ids_override=val_ids_override,
        )
        return evaluate_all_sequences(
            model=model,
            processor=processor,
            label_tokens=label_tokens,
            ds=ds,
            image_size=image_size,
            margin_ratio=margin_ratio,
            margin_px=margin_px,
            min_side=min_side,
            coord_mode=coord_mode,
            coord_decimals=coord_decimals,
            threshold=threshold,
            batch_size=batch_size,
            device=device,
            output_dir=output_dir,
            global_step=global_step,
            show_tqdm=show_tqdm,
        )
    else:
        return evaluate(
            model=model,
            processor=processor,
            label_tokens=label_tokens,
            data_root=data_root,
            sequence=sequence,
            dataset_version=dataset_version,
            image_size=image_size,
            margin_ratio=margin_ratio,
            margin_px=margin_px,
            min_side=min_side,
            coord_mode=coord_mode,
            coord_decimals=coord_decimals,
            threshold=threshold,
            batch_size=batch_size,
            device=device,
            output_dir=output_dir,
            global_step=global_step,
            show_tqdm=show_tqdm,
        )


