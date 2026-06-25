from __future__ import annotations

from typing import Tuple

import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration,Qwen2_5_VLForConditionalGeneration,Qwen3_5ForConditionalGeneration
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def build_model_and_processor(
        model_name: str,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0,
        use_4bit: bool = False,
        bf16: bool = True,
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

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "up_proj", "down_proj", "gate_proj",
        ],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    yes_id = processor.tokenizer.convert_tokens_to_ids("Yes")
    no_id = processor.tokenizer.convert_tokens_to_ids("No")
    
    # if yes_id is None or yes_id == processor.tokenizer.unk_token_id:
    #     yes_id = processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
    # if no_id is None or no_id == processor.tokenizer.unk_token_id:
    #     no_id = processor.tokenizer.encode("No", add_special_tokens=False)[0]

    return model, processor, (yes_id, no_id)

if __name__ == "__main__":
    import sys

    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-VL-2B"
    
    print(f"\n{'='*50}")
    print(f"验证模型: {model_name}")
    print(f"{'='*50}")

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = processor.tokenizer

    words = ["Yes", "No", "yes", "no", "YES", "NO","ggbond"]

    print(f"\n{'Word':<10} {'convert_tokens_to_ids':>22} {'encode()':>12} {'is_unk':>8} {'token_count':>12}")
    print("-" * 70)

    unk_id = tokenizer.unk_token_id

    for word in words:
        direct_id   = tokenizer.convert_tokens_to_ids(word)
        encoded_ids = tokenizer.encode(word, add_special_tokens=False)
        is_unk      = (direct_id == unk_id)
        token_count = len(encoded_ids)

        flag = "⚠️ MULTI" if token_count > 1 else ("❌ UNK" if is_unk else "✅")
        print(f"{word:<10} {str(direct_id):>22} {str(encoded_ids):>12} {str(is_unk):>8}   {token_count}  {flag}")

    print(f"\nunk_token_id = {unk_id}")
    print(f"unk_token    = {tokenizer.unk_token!r}")

    # 推荐做法：找到正确的单 token id
    print(f"\n--- 推荐方式 ---")
    for word in ["Yes", "No"]:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if len(ids) == 1 and ids[0] != unk_id:
            print(f"{word!r:>5} → single token ✅  id={ids[0]}  "
                  f"decoded={tokenizer.decode(ids)!r}")
        else:
            print(f"{word!r:>5} → ⚠️  ids={ids}, 需要特殊处理")
            # 尝试带空格前缀
            for candidate in [f" {word}", f"▁{word}"]:
                cids = tokenizer.encode(candidate, add_special_tokens=False)
                if len(cids) == 1 and cids[0] != unk_id:
                    print(f"       候选 {candidate!r} → id={cids[0]} ✅")
                    break
