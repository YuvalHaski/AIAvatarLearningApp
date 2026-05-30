"""QLoRA fine-tune of the Stage 2 feedback model.

Trains a small instruct model (default Phi-3.5-mini) on the
(ErrorReport -> spoken feedback) pairs from generate_dataset.py, then merges the
LoRA adapter into the base weights so the result can be served directly by vLLM.

Run on a cloud GPU (A10 / A100):
    pip install -r training/requirements.txt
    python training/train.py --data training/data --out training/out

Serve the merged model:
    python -m vllm.entrypoints.openai.api_server \
        --model training/out/merged --served-model-name feedback-model --port 8000
Then set FEEDBACK_MODEL_URL=http://<gpu-host>:8000/v1 for the app.
"""
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="microsoft/Phi-3.5-mini-instruct")
    parser.add_argument("--data", default="training/data")
    parser.add_argument("--out", default="training/out")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    out_dir = Path(args.out)
    adapter_dir = out_dir / "adapter"
    merged_dir = out_dir / "merged"

    dataset = load_dataset(
        "json",
        data_files={
            "train": f"{args.data}/train.jsonl",
            "validation": f"{args.data}/val.jsonl",
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4-bit quantized base for memory-efficient QLoRA training.
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # Use Phi-3's fused module names so the LoRA survives GGUF conversion.
        # Targeting q_proj/k_proj/v_proj separately causes convert_lora_to_gguf.py
        # to silently drop them (it can't fuse 3 LoRAs into the GGUF's attn_qkv).
        target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    )

    sft_config = SFTConfig(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        bf16=True,
        # The dataset has a "messages" column; SFTTrainer applies the chat
        # template and masks everything except the assistant turn.
        max_length=1024,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=lora_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(adapter_dir))
    print(f"saved LoRA adapter -> {adapter_dir}")

    # Merge the adapter into fp16 base weights for serving with vLLM.
    del model
    torch.cuda.empty_cache()
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.float16,
        device_map="auto", trust_remote_code=True,
    )
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    # Phi-3 sets _tied_weights_keys=["lm_head.weight"] (list) but transformers'
    # save_pretrained calls .keys() on it. The naive list->{k:k} fix dedupes
    # lm_head.weight out of the state_dict and produces garbage GGUF.
    # Phi-3.5-mini has tie_word_embeddings=False, so clearing it is safe.
    for m in merged.modules():
        if hasattr(m, "_tied_weights_keys"):
            m._tied_weights_keys = {}
    merged.save_pretrained(str(merged_dir), safe_serialization=False)
    tokenizer.save_pretrained(str(merged_dir))
    AutoTokenizer.from_pretrained(args.base_model, use_fast=False).save_pretrained(
        str(merged_dir)
    )
    print(f"saved merged model -> {merged_dir}  (serve this with vLLM)")


if __name__ == "__main__":
    main()
