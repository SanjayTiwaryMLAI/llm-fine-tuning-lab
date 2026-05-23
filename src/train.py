"""QLoRA Fine-Tuning for Mistral-7B / LLaMA-3
Usage: python src/train.py --model mistralai/Mistral-7B-v0.1
"""
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer


def get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )


def get_lora_config():
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )


def format_alpaca(ex):
    if ex.get("input"):
        return f"### Instruction:\n{ex['instruction']}\n\n### Input:\n{ex['input']}\n\n### Response:\n{ex['output']}"
    return f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['output']}"


def train(model_name, dataset_name, output_dir, epochs=3):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=get_bnb_config(), device_map="auto"
    )
    model = get_peft_model(model, get_lora_config())
    model.print_trainable_parameters()
    ds = load_dataset(dataset_name, split="train[:5000]").map(lambda x: {"text": format_alpaca(x)})
    args = TrainingArguments(
        output_dir=output_dir, num_train_epochs=epochs,
        per_device_train_batch_size=4, gradient_accumulation_steps=4,
        learning_rate=2e-4, fp16=True, logging_steps=50, save_steps=200,
        warmup_ratio=0.05, lr_scheduler_type="cosine",
    )
    SFTTrainer(
        model=model, tokenizer=tok, train_dataset=ds,
        dataset_text_field="text", max_seq_length=2048, args=args,
    ).train()
    model.save_pretrained(output_dir)
    tok.save_pretrained(output_dir)
    print(f"\n✅ Model saved to {output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",      default="mistralai/Mistral-7B-v0.1")
    p.add_argument("--dataset",    default="tatsu-lab/alpaca")
    p.add_argument("--output_dir", default="./outputs/finetuned")
    p.add_argument("--epochs",     type=int, default=3)
    a = p.parse_args()
    train(a.model, a.dataset, a.output_dir, a.epochs)
