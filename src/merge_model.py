"""Merge LoRA adapter into base model for deployment"""
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge(base, adapter, output):
    print(f"Merging {adapter} -> {base} ...")
    tok   = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, device_map="cpu")
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.save_pretrained(output)
    tok.save_pretrained(output)
    print(f"✅ Merged model saved to {output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base_model",   required=True)
    p.add_argument("--adapter_path", required=True)
    p.add_argument("--output_path",  required=True)
    a = p.parse_args()
    merge(a.base_model, a.adapter_path, a.output_path)
