import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
from collections import OrderedDict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to the original base model (e.g., v4) to get the config.")
    parser.add_argument("--lora_model_path", type=str, required=True, help="Path to the LoRA-trained model directory containing the pytorch_model.bin with prefixed keys.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the purified, standard-format model.")
    args = parser.parse_args()

    print(f"Loading base model config from: {args.base_model_path}")
    # We only need the config from the base model to create a 'shell'
    model = AutoModelForCausalLM.from_pretrained(args.base_model_path)

    print(f"Loading LoRA model's state dictionary from: {args.lora_model_path}")
    lora_state_dict = torch.load(os.path.join(args.lora_model_path, 'pytorch_model.bin'), map_location='cpu')

    # This is the core purification logic
    new_state_dict = OrderedDict()
    for key, value in lora_state_dict.items():
        if key.startswith("base_model.model."):
            new_key = key[len("base_model.model."):]  # Strip the prefix
            new_state_dict[new_key] = value
        else:
            # Keep other keys like lm_head.weight if they exist without prefix
            new_state_dict[key] = value

    print("Loading purified state dictionary into the model structure...")
    model.load_state_dict(new_state_dict)

    print(f"Saving purified model to: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)

    # Also copy the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.lora_model_path)
    tokenizer.save_pretrained(args.output_dir)
    print("Purification complete.")

if __name__ == "__main__":
    main()
