#!/usr/bin/env python
import os
import argparse
import json
import gc
import math
from copy import deepcopy
from typing import List

# --- Setup Environment ---
CACHE_DIR = "/BS/DApt/work/huggingface_cache"
os.environ["TORCH_HOME"] = CACHE_DIR
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm

from imagecorruptions import corrupt
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

# =========================================================================
# LoRA Class & Loading Functions (These are universal and need no changes)
# =========================================================================
class LoRALinear(nn.Module):
    def __init__(self, original_layer: nn.Linear, r: int, lora_alpha: int):
        super().__init__()
        self.original_layer = original_layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
        device = original_layer.weight.device
        dtype = torch.float32
        self.lora_A = nn.Linear(original_layer.in_features, r, bias=False, device=device, dtype=dtype)
        self.lora_B = nn.Linear(r, original_layer.out_features, bias=False, device=device, dtype=dtype)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.scaling = lora_alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_output = self.original_layer(x)
        x_float32 = x.to(torch.float32)
        # Temporarily cast the LoRA layers to float32 for the calculation
        lora_delta = self.lora_B.to(torch.float32)(self.lora_A.to(torch.float32)(x_float32)) # <--- FIXED LINE
        return original_output + (lora_delta * self.scaling).to(original_output.dtype)

def apply_lora_to_llm(model: nn.Module, r: int, lora_alpha: int, target_modules: list, layer_start: int, layer_end: int):
    modifications = []
    num_layers = len(model.model.layers)
    if layer_end == -1: layer_end = num_layers
    layer_end = min(layer_end, num_layers)
    for layer_idx in range(layer_start, layer_end):
        layer = model.model.layers[layer_idx]
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear) and any(t in name for t in target_modules):
                parent_name, child_name = name.rsplit('.', 1) if '.' in name else ('', name)
                parent_module = layer.get_submodule(parent_name) if parent_name else layer
                modifications.append((parent_module, child_name, module))
    for parent_module, child_name, module in modifications:
        setattr(parent_module, child_name, LoRALinear(module, r, lora_alpha))
    print(f"Applied LoRA to {len(modifications)} linear layers in LLM (Layers {layer_start}-{layer_end-1}).")

def apply_lora_to_vision_tower(model: nn.Module, r: int, lora_alpha: int, target_modules: list, layer_start: int, layer_end: int):
    modifications = []
    vision_tower = model.get_vision_tower().vision_tower
    num_layers = len(vision_tower.vision_model.encoder.layers)
    if layer_end == -1: layer_end = num_layers
    layer_end = min(layer_end, num_layers)
    for layer_idx in range(layer_start, layer_end):
        layer = vision_tower.vision_model.encoder.layers[layer_idx]
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear) and any(t in name for t in target_modules):
                parent_name, child_name = name.rsplit('.', 1) if '.' in name else ('', name)
                parent_module = layer.get_submodule(parent_name) if parent_name else layer
                modifications.append((parent_module, child_name, module))
    for parent_module, child_name, module in modifications:
        setattr(parent_module, child_name, LoRALinear(module, r, lora_alpha))
    print(f"Applied LoRA to {len(modifications)} linear layers in Vision Tower (Layers {layer_start}-{layer_end-1}).")

def load_lora_weights(model: nn.Module, lora_weights_path: str):
    if not os.path.exists(lora_weights_path):
        raise FileNotFoundError(f"LoRA weights file not found: '{lora_weights_path}'")
    lora_state_dict = torch.load(lora_weights_path, map_location='cpu')
    incompatible_keys = model.load_state_dict(lora_state_dict, strict=False)
    missing_lora_keys = [k for k in incompatible_keys.missing_keys if 'lora' in k]
    if missing_lora_keys:
        print("Missing Keys:", missing_lora_keys)
        raise ValueError("CRITICAL ERROR: LoRA parameters missing from weights file. The model structure does not match the checkpoint. Check your --adapt flags and layer ranges.")
    if incompatible_keys.unexpected_keys:
        print("Warning: Unexpected keys found in checkpoint:", incompatible_keys.unexpected_keys)
    return model

# =========================================================================
# Main Inference Logic
# =========================================================================
def main(args):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TARGET_DTYPE = torch.float16 if DEVICE.type == 'cuda' else torch.float32

    print("--- Universal LLaVA TTT-LoRA Captioning Inference ---")
    
    path_to_lora_corruption_weights = os.path.join(args.lora_weights_parent_dir, f"{args.corruption_name}_{args.corruption_severity}")
    
    output_base_dir = f"{args.lora_weights_parent_dir}_captions"
    run_name = os.path.basename(os.path.normpath(args.lora_weights_parent_dir))
    output_dir = os.path.join(output_base_dir, "INFER", run_name, f"{args.corruption_name}_{args.corruption_severity}")
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint_name_safe = os.path.splitext(args.checkpoint_name)[0]
    jsonl_output_file_path = os.path.join(output_dir, f"captions_{checkpoint_name_safe}.jsonl")

    print(f"Adapters Source Dir: {path_to_lora_corruption_weights}")
    print(f"Caption Output File: {jsonl_output_file_path}")

    if not os.path.isdir(path_to_lora_corruption_weights):
        print(f"FATAL: Adapter directory not found: '{path_to_lora_corruption_weights}'")
        return
        
    adapted_image_stems = [d for d in os.listdir(path_to_lora_corruption_weights) if os.path.isdir(os.path.join(path_to_lora_corruption_weights, d))]
    
    if not adapted_image_stems:
        print(f"FATAL: No adapted image folders found in '{path_to_lora_corruption_weights}'.")
        return
        
    print(f"Discovered {len(adapted_image_stems)} images with existing LoRA adapters.")

    disable_torch_init()
    tokenizer, original_llava_model, image_processor, _ = load_pretrained_model(
        args.model_path, model_base=None, model_name=get_model_name_from_path(args.model_path),
        device_map=None, torch_dtype=TARGET_DTYPE, load_4bit=False, load_8bit=False
    )
    original_llava_model.eval()

    with open(jsonl_output_file_path, "w") as f_out: pass

    for image_stem in tqdm(adapted_image_stems, desc=f"Generating Captions ({checkpoint_name_safe})"):
        lora_checkpoint_path = os.path.join(path_to_lora_corruption_weights, image_stem, args.checkpoint_name)
        image_filename = f"{image_stem}.jpg"
        image_full_path = os.path.join(args.image_dir, image_filename)

        result_entry = {"image_id": image_stem, "image_filename": image_filename, "lora_checkpoint_used": lora_checkpoint_path, "caption": None, "error": None}

        if not os.path.exists(lora_checkpoint_path) or not os.path.exists(image_full_path):
            result_entry["error"] = f"Missing file. LoRA exists: {os.path.exists(lora_checkpoint_path)}. Image exists: {os.path.exists(image_full_path)}."
            with open(jsonl_output_file_path, "a") as f_out: f_out.write(json.dumps(result_entry) + "\n")
            continue
        
        llava_model = None
        try:
            llava_model = deepcopy(original_llava_model)
            
            if args.adapt_llm:
                apply_lora_to_llm(llava_model, r=args.lora_llm_r, lora_alpha=args.lora_llm_alpha,
                                  target_modules=args.lora_target_modules, 
                                  layer_start=args.lora_llm_start_layer, 
                                  layer_end=args.lora_llm_end_layer)
            if args.adapt_vision:
                apply_lora_to_vision_tower(llava_model, r=args.lora_vision_r, lora_alpha=args.lora_vision_alpha,
                                           target_modules=args.lora_target_modules, 
                                           layer_start=args.lora_vision_start_layer, 
                                           layer_end=args.lora_vision_end_layer)

            llava_model = load_lora_weights(llava_model, lora_checkpoint_path)
            llava_model.to(device=DEVICE, dtype=TARGET_DTYPE)
            llava_model.eval()

            image_pil = Image.open(image_full_path).convert('RGB')
            corrupted_np = corrupt(np.array(image_pil), corruption_name=args.corruption_name, severity=args.corruption_severity)
            image_tensor = process_images([Image.fromarray(corrupted_np)], image_processor, llava_model.config).to(llava_model.device, dtype=TARGET_DTYPE)
            
            conv = conv_templates[args.conv_mode].copy()
            prompt = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + "Please describe this image in detail.") # other prompt: "Generate a short caption of the image."
            conv.append_message(conv.roles[0], prompt); conv.append_message(conv.roles[1], None)
            input_ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(llava_model.device)

            with torch.inference_mode():
                output_ids = llava_model.generate(
                    input_ids, images=image_tensor, image_sizes=[image_pil.size],
                    do_sample=False, temperature=0, max_new_tokens=args.max_new_tokens, use_cache=True
                )
            raw_output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            result_entry["caption"] = raw_output.split('ASSISTANT:')[-1].strip()
            print(result_entry["caption"])

        except Exception as e:
            tqdm.write(f"  ERROR processing '{image_filename}': {e}")
            result_entry["error"] = str(e)
        finally:
            if llava_model is not None: del llava_model
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        with open(jsonl_output_file_path, "a") as f_out: f_out.write(json.dumps(result_entry) + "\n")

    print(f"\n--- ✅ Inference complete. Results saved to: {jsonl_output_file_path} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal LLaVA TTT-LoRA Captioning Inference")
    # --- Universal Path Arguments ---
    parser.add_argument("--lora-weights-parent-dir", type=str, required=True, help="Path to the specific training run's output directory.")
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--image-dir", type=str, default="/BS/DApt/work/LLaVA/playground/data/eval/pope/val2014/")
    
    # --- Universal Inference Arguments ---
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--checkpoint-name", type=str, required=True)
    parser.add_argument("--corruption-name", type=str, required=True)
    parser.add_argument("--corruption-severity", type=int, required=True)
    
    # --- Universal LoRA Arguments ---
    parser.add_argument('--lora-target-modules', nargs='+', default=["q_proj", "v_proj"])

    # --- LLM LoRA Specific Arguments ---
    parser.add_argument('--adapt-llm', action='store_true', help="Enable this flag if the checkpoint contains LLM LoRA weights.")
    parser.add_argument('--lora-llm-r', type=int, default=8)
    parser.add_argument('--lora-llm-alpha', type=int, default=16)
    parser.add_argument('--lora-llm-start-layer', type=int, default=0)
    parser.add_argument('--lora-llm-end-layer', type=int, default=-1)

    # --- Vision LoRA Specific Arguments ---
    parser.add_argument('--adapt-vision', action='store_true', help="Enable this flag if the checkpoint contains Vision LoRA weights.")
    parser.add_argument('--lora-vision-r', type=int, default=8)
    parser.add_argument('--lora-vision-alpha', type=int, default=16)
    parser.add_argument('--lora-vision-start-layer', type=int, default=0)
    parser.add_argument('--lora-vision-end-layer', type=int, default=6)

    args = parser.parse_args()
    if not args.adapt_llm and not args.adapt_vision:
        raise ValueError("You must specify at least one module to adapt: --adapt-llm and/or --adapt-vision")
    main(args)