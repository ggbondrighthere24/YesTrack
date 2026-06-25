from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


@dataclass
class CollateOut:
    pixel_values: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    image_grid_thw: Optional[torch.Tensor] = None
    debug_images: Optional[List[Any]] = None
    debug_prompts: Optional[List[str]] = None


def make_collate_fn(processor):
    def _collate(batch: List[Dict[str, Any]]) -> CollateOut:
        prompts = [b["text"] for b in batch]
        answers = ["Yes" if b["label"] == 1 else "No" for b in batch]

        prompt_inputs_list = []
        answer_token_ids_list = []
        pixel_values_list = []
        image_grid_thw_list = []
        dbg_images: List[Any] = []
        dbg_prompts: List[str] = []
        
        for sample, pr, ans in zip(batch, prompts, answers):
            imgs = sample.get("images") if isinstance(sample, dict) else None
            if imgs is None:
                img_single = sample.get("image") if isinstance(sample, dict) else None
                imgs = [img_single]
            # 收集少量可视化样本
            if len(dbg_images) < 8:
                if imgs and imgs[0] is not None:
                    dbg_images.append(imgs[0])
                    dbg_prompts.append(pr)
            user_messages = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": im} for im in imgs if im is not None],
                        {"type": "text", "text": pr},
                    ],
                }
            ]
            prompt_inp = processor.apply_chat_template(
                user_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            expected_n = len([im for im in imgs if im is not None])
            actual_n = None
            if "image_grid_thw" in prompt_inp and prompt_inp["image_grid_thw"] is not None:
                thw = prompt_inp["image_grid_thw"]
                if hasattr(thw, "dim"):
                    if thw.dim() == 3:
                        actual_n = int(thw.shape[1])
                    elif thw.dim() == 2:
                        actual_n = int(thw.shape[0])
            if actual_n is None and "pixel_values" in prompt_inp and prompt_inp["pixel_values"] is not None:
                pv = prompt_inp["pixel_values"]
                if hasattr(pv, "dim"):
                    if pv.dim() == 5:
                        actual_n = int(pv.shape[1])
                    elif pv.dim() == 4:
                        actual_n = 1
            if expected_n > 1:
                if actual_n is None:
                    raise RuntimeError("多图样本无法验证被模型接收的图像数量（缺少 image_grid_thw 或无法从 pixel_values 推断）")
                if int(actual_n) != int(expected_n):
                    raise RuntimeError(f"多图样本接收数量不一致: expected={expected_n}, actual={actual_n}")
            prompt_inputs_list.append(prompt_inp)
            
            ans_ids = processor.tokenizer.encode(ans, add_special_tokens=False)
            answer_token_ids_list.append(ans_ids)
        
        max_prompt_len = max(inp["input_ids"].shape[1] for inp in prompt_inputs_list)
        max_ans_len = max(len(ans_ids) for ans_ids in answer_token_ids_list)
        max_len = max_prompt_len + max_ans_len + 1
        
        batch_size = len(prompt_inputs_list)
        input_ids = torch.full((batch_size, max_len), processor.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        prompt_lens: List[int] = []
        
        for i, (prompt_inp, ans_ids) in enumerate(zip(prompt_inputs_list, answer_token_ids_list)):
            prompt_len = prompt_inp["input_ids"].shape[1]
            input_ids[i, :prompt_len] = prompt_inp["input_ids"][0]
            attention_mask[i, :prompt_len] = prompt_inp["attention_mask"][0]
            
            ans_len = len(ans_ids)
            input_ids[i, prompt_len:prompt_len + ans_len] = torch.tensor(ans_ids, dtype=torch.long)
            attention_mask[i, prompt_len:prompt_len + ans_len] = 1
            
            input_ids[i, prompt_len + ans_len] = processor.tokenizer.eos_token_id
            attention_mask[i, prompt_len + ans_len] = 1
            
            prompt_lens.append(prompt_len)
            
            if "pixel_values" in prompt_inp and prompt_inp["pixel_values"] is not None:
                pixel_values_list.append(prompt_inp["pixel_values"])
            if "image_grid_thw" in prompt_inp and prompt_inp["image_grid_thw"] is not None:
                image_grid_thw_list.append(prompt_inp["image_grid_thw"])
        
        pixel_values = torch.cat(pixel_values_list, dim=0) if pixel_values_list else None
        image_grid_thw = torch.cat(image_grid_thw_list, dim=0) if image_grid_thw_list else None

        labels = input_ids.clone()
        labels[:] = torch.where(
            attention_mask.bool(), labels, torch.full_like(labels, -100)
        )
        for i, pl in enumerate(prompt_lens):
            if pl > 0:
                labels[i, :pl] = -100

        return CollateOut(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            image_grid_thw=image_grid_thw,
            debug_images=dbg_images if len(dbg_images) > 0 else None,
            debug_prompts=dbg_prompts if len(dbg_prompts) > 0 else None,
        )

    return _collate


