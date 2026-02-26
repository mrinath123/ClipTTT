#!/usr/bin/env python

import os
import torch
import random
import numpy as np
from PIL import Image
import torch.nn.functional as F
import gc
from copy import deepcopy
import json
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import sys
import nltk

# --- Set Environment Variables at the VERY TOP ---
CACHE_DIR = "/BS/DApt/work/huggingface_cache"
os.environ["TORCH_HOME"] = CACHE_DIR
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
os.environ["CLIP_CACHE_PATH"] = CACHE_DIR

# --- Model/Util Imports ---
from transformers import AutoProcessor, LlavaForConditionalGeneration
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, get_model_name_from_path, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from imagecorruptions import corrupt
from transformers import get_cosine_schedule_with_warmup,get_constant_schedule_with_warmup
import torch.nn as nn
import clip
import torchvision.transforms as T

# =========================================================================
# 1. LoRA MODULES AND HELPERS (UNCHANGED)
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
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))   #nn.init.normal_(self.lora_A.weight, std=1 / r)
        nn.init.zeros_(self.lora_B.weight)
        self.scaling = lora_alpha / r
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_output = self.original_layer(x)
        x_float32 = x.to(torch.float32)
        lora_output = self.lora_B(self.lora_A(x_float32))
        lora_output = lora_output.to(original_output.dtype)
        return original_output + lora_output * self.scaling

def apply_lora_to_llm(model: nn.Module, r: int, lora_alpha: int, target_modules: list, layer_start: int = 5, layer_end: int = 18):
    llm_layers_container = None
    if hasattr(model, 'language_model') and hasattr(model.language_model, 'model'):
        print("GOD MODE 👁️‍🗨️: (LoRA) Detected HF model structure -> `model.language_model.model`")
        llm_layers_container = model.language_model.model
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        print("GOD MODE 👁️‍🗨️: (LoRA) Detected original LLaVA model structure -> `model.model`")
        llm_layers_container = model.model
    else:
        raise TypeError("Could not determine the LLM structure to apply LoRA.")
    
    modifications = []
    num_layers = len(llm_layers_container.layers)
    if layer_end == -1: layer_end = num_layers
    layer_end = min(layer_end, num_layers)
    
    for layer_idx in range(layer_start, layer_end):
        layer = llm_layers_container.layers[layer_idx]
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear) and any(t in name for t in target_modules):
                parent_name, child_name = name.rsplit('.', 1) if '.' in name else ('', name)
                modifications.append((layer_idx, parent_name, child_name, module))

    for layer_idx, parent_name, child_name, module in modifications:
        parent_module = llm_layers_container.layers[layer_idx].get_submodule(parent_name) if parent_name else llm_layers_container.layers[layer_idx]
        setattr(parent_module, child_name, LoRALinear(module, r, lora_alpha))
        
    print(f"  Applied LoRA to {len(modifications)} linear layers.")
    total_params = sum(p.numel() for p in model.parameters())
    lora_params = sum(p.numel() for name, p in model.named_parameters() if 'lora_' in name)
    
    if hasattr(model, 'get_vision_tower'):
        vision_tower = model.get_vision_tower()
    else:
        vision_tower = model.vision_tower
        
    vision_tower_params = sum(p.numel() for p in vision_tower.parameters())
    llm_params = total_params - vision_tower_params
    percentage = (lora_params / total_params) * 100
    print("\n  --- 📊 Parameter Statistics ---")
    print(f"  Total Model Parameters:  {total_params / 1e6:.2f} M")
    print(f"  Trainable LoRA Params:   {lora_params / 1e3:.2f} K")
    print(f"  LoRA as % of Total:      {percentage:.4f} % \n")

# =========================================================================
# 2. UTILITY AND CORE LOGIC FUNCTIONS (INSTRUMENTED)
# =========================================================================
def save_lora_weights(model, file_path):
    lora_state_dict = { name: param.cpu() for name, param in model.named_parameters() if 'lora' in name }
    torch.save(lora_state_dict, file_path)
    # print(f"  💾 Saved LoRA weights to {file_path}") # Silenced to reduce log spam

def save_loss_plot(loss_history, file_path, title):
    plt.figure(figsize=(10, 6)); plt.plot(loss_history, label='Cross-Entropy Loss')
    plt.title(title); plt.xlabel("Adaptation Step"); plt.ylabel("Loss")
    plt.legend(); plt.grid(True); plt.savefig(file_path); plt.close()
    print(f"  📊 Saved loss plot to {file_path}")

def update_ema_teacher(teacher_model, student_model, decay):
    student_params = dict(student_model.named_parameters())
    for name, teacher_param in teacher_model.named_parameters():
        if 'lora' in name:
            student_param = student_params[name]
            teacher_param.data.mul_(decay).add_(student_param.data.to(teacher_param.device), alpha=1 - decay)



def calculate_clip_score_v2(student_model, tokenizer, image_tensor, corrupted_pil, prompt_ids, clip_model, clip_preprocess, device, args, processor, prompt_str):
    """
    Generates a caption from the student model and computes its CLIP score against the image.
    This function is robust to both HF-native and original LLaVA model types.
    """
    student_model.eval()
    with torch.no_grad():
        # --- 1. Generate Caption from Student Model ---
        # Handle both HF-native and original LLaVA model types
        if isinstance(student_model, LlavaForConditionalGeneration):
            inputs = processor(text=prompt_str, images=corrupted_pil, return_tensors="pt").to(device, dtype=student_model.dtype)
            current_gen_ids = student_model.generate(
                **inputs, 
                do_sample=False, 
                max_new_tokens=args.max_new_tokens, 
                use_cache=True, 
                pad_token_id=tokenizer.eos_token_id
            )
        else:
            current_gen_ids = student_model.generate(
                prompt_ids, 
                images=image_tensor, 
                image_sizes=[corrupted_pil.size], 
                do_sample=False, 
                max_new_tokens=args.max_new_tokens, 
                use_cache=True, 
                pad_token_id=tokenizer.eos_token_id
            )

        # --- 2. Decode and Clean the Caption ---
        full_caption = tokenizer.decode(current_gen_ids[0], skip_special_tokens=True).strip()
        # Remove the prompt part to isolate the generated response
        caption = full_caption.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in full_caption else full_caption

        # Safety check: If the model generates an empty string, return a score of 0.
        if not caption or not caption.strip():
            return 0.0

        # --- 3. Chunk Long Captions for CLIP ---
        # CLIP's context length is 77 tokens. We chunk longer text to get a more stable score.
        words = caption.split()
        chunk_size, stride = 50, 25 # Use a sliding window for chunks
        if len(words) <= chunk_size:
            text_chunks = [' '.join(words)]
        else:
            text_chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words) - chunk_size + 1, stride)]
            # Add the final part of the text if it was missed by the sliding window
            if (len(words) - chunk_size) % stride != 0:
                text_chunks.append(' '.join(words[-chunk_size:]))
        
        # Safety check: If chunking results in an empty list, return 0
        if not text_chunks or all(not chunk for chunk in text_chunks):
            return 0.0

        # --- 4. Calculate CLIP Score ---
        # Preprocess image and text for CLIP
        image_input_clip = clip_preprocess(corrupted_pil).unsqueeze(0).to(device)
        text_inputs_clip = clip.tokenize(text_chunks, truncate=True).to(device)

        # Encode features and normalize
        image_features = clip_model.encode_image(image_input_clip)
        text_features = clip_model.encode_text(text_inputs_clip)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Calculate similarity scores
        all_scores = (100.0 * image_features @ text_features.T).squeeze()

        import pdb;pdb.set_trace()
        
        # Average the scores if there were multiple text chunks, otherwise just use the single score
        final_score = all_scores.mean().item() if all_scores.numel() > 1 else all_scores.item()
        
        return final_score




def calculate_clip_scorev3(student_model, tokenizer, image_tensor, corrupted_pil, prompt_ids, clip_model, clip_preprocess, device, args, processor, prompt_str):
    student_model.eval()
    with torch.no_grad():
        # --- 1. Generate Caption ---
        if isinstance(student_model, LlavaForConditionalGeneration):
            inputs = processor(text=prompt_str, images=corrupted_pil, return_tensors="pt").to(device, dtype=student_model.dtype)
            current_gen_ids = student_model.generate(
                **inputs, 
                do_sample=False, 
                max_new_tokens=args.max_new_tokens, 
                use_cache=True, 
                pad_token_id=tokenizer.eos_token_id
            )
        else:
            current_gen_ids = student_model.generate(
                prompt_ids, 
                images=image_tensor, 
                image_sizes=[corrupted_pil.size], 
                do_sample=False, 
                max_new_tokens=args.max_new_tokens, 
                use_cache=True, 
                pad_token_id=tokenizer.eos_token_id
            )

        # --- 2. Decode and Clean ---
        full_caption = tokenizer.decode(current_gen_ids[0], skip_special_tokens=True).strip()
        caption = full_caption.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in full_caption else full_caption

        if not caption:
            return 0.0

        # --- 3. Sentence-Based Chunking ---
        # Use NLTK to split into actual sentences
        raw_sentences = nltk.sent_tokenize(caption)
        
        # Filter out empty strings or single punctuation marks
        text_chunks = [s.strip() for s in raw_sentences if len(s.strip()) > 2]

        if not text_chunks:
            return 0.0

        # --- 4. Calculate CLIP Score ---
        image_input_clip = clip_preprocess(corrupted_pil).unsqueeze(0).to(device)
        
        # clip.tokenize handles a list of strings. 
        text_inputs_clip = clip.tokenize(text_chunks).to(device)

        # Encode features
        image_features = clip_model.encode_image(image_input_clip)
        text_features = clip_model.encode_text(text_inputs_clip)

        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Calculate cosine similarity
        # image_features: [1, dim], text_features: [num_sentences, dim]
        # result: [num_sentences]
        similarities = (100.0 * image_features @ text_features.T).squeeze(0)
        # Average the scores across all sentences
        #import pdb;pdb.set_trace()
        final_score = similarities.mean().item() 
        return final_score



def get_pseudo_gt_and_targetsv2(teacher_model, tokenizer, image_tensor, image_size, prompt_ids, clip_model, clip_preprocess, pil_image, device, args, processor, prompt_str):
    teacher_model.eval()
    with torch.no_grad():
        if isinstance(teacher_model, LlavaForConditionalGeneration):
            inputs = processor(text=prompt_str, images=pil_image, return_tensors="pt").to(device, dtype=teacher_model.dtype)
            candidate_output_ids = teacher_model.generate(**inputs, do_sample=True, num_return_sequences=args.num_candidate_captions, temperature=0.8, top_p=0.9, max_new_tokens=args.max_new_tokens, use_cache=True, pad_token_id=tokenizer.eos_token_id)
        else:
            candidate_output_ids = teacher_model.generate(prompt_ids, images=image_tensor, image_sizes=[image_size], do_sample=True, num_return_sequences=args.num_candidate_captions, temperature=0.8, top_p=0.9, max_new_tokens=args.max_new_tokens, use_cache=True, pad_token_id=tokenizer.eos_token_id)

        candidate_captions = [tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in candidate_output_ids]
        image_input_clip = clip_preprocess(pil_image).unsqueeze(0).to(device)
        text_inputs_clip = clip.tokenize(candidate_captions, truncate=True).to(device)
        image_features = clip_model.encode_image(image_input_clip)
        text_features = clip_model.encode_text(text_inputs_clip)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity_scores = (100.0 * image_features @ text_features.T).squeeze()
        
        best_full_caption = candidate_captions[similarity_scores.argmax()]

        gt_caption = best_full_caption.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in best_full_caption else best_full_caption

        gt_caption_ids = tokenizer(gt_caption, return_tensors='pt', add_special_tokens=False).input_ids.to(device)
        if gt_caption_ids.nelement() > 0 and gt_caption_ids[0, -1] != tokenizer.eos_token_id:
            eos_tensor = torch.tensor([[tokenizer.eos_token_id]], device=device)
            gt_caption_ids = torch.cat([gt_caption_ids, eos_tensor], dim=1)

        gt_sequence_ids = torch.cat([prompt_ids, gt_caption_ids], dim=1)
        gt_input_ids_for_model = gt_sequence_ids[:, :-1]
        attention_mask = torch.ones_like(gt_input_ids_for_model)
        
        return gt_caption, gt_input_ids_for_model, gt_sequence_ids, attention_mask



def calculate_clip_score(student_model, tokenizer, image_tensor, corrupted_pil, prompt_ids, clip_model, clip_preprocess, device, args, processor, prompt_str):
    student_model.eval()
    with torch.no_grad():
        # --- 1. Generate Caption ---
        if isinstance(student_model, LlavaForConditionalGeneration):
            inputs = processor(text=prompt_str, images=corrupted_pil, return_tensors="pt").to(device, dtype=student_model.dtype)
            current_gen_ids = student_model.generate(
                **inputs, 
                do_sample=False, 
                max_new_tokens=args.max_new_tokens, 
                use_cache=True, 
                pad_token_id=tokenizer.eos_token_id
            )
        else:
            current_gen_ids = student_model.generate(
                prompt_ids, 
                images=image_tensor, 
                image_sizes=[corrupted_pil.size], 
                do_sample=False, 
                max_new_tokens=args.max_new_tokens, 
                use_cache=True, 
                pad_token_id=tokenizer.eos_token_id
            )

        # --- 2. Decode and Clean ---
        full_caption = tokenizer.decode(current_gen_ids[0], skip_special_tokens=True).strip()
        caption = full_caption.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in full_caption else full_caption

        if not caption:
            return 0.0

        # --- 3. Sentence-Based Chunking ---
        raw_sentences = nltk.sent_tokenize(caption)
        text_chunks = [s.strip() for s in raw_sentences if len(s.strip()) > 2]

        if not text_chunks:
            return 0.0

        # --- 4. Calculate CLIP Score (Q10 Logic) ---
        image_input_clip = clip_preprocess(corrupted_pil).unsqueeze(0).to(device)
        text_inputs_clip = clip.tokenize(text_chunks, truncate=True).to(device)

        image_features = clip_model.encode_image(image_input_clip)
        text_features = clip_model.encode_text(text_inputs_clip)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        similarities = (100.0 * image_features @ text_features.T).squeeze(0)
        
        # --- UPDATED: Use Q10 instead of Mean ---
        q10_score = torch.quantile(similarities.float(), 0.1).item()
        #mean = similarities.mean().item()
        return q10_score


def get_pseudo_gt_and_targets(teacher_model, tokenizer, image_tensor, image_size, prompt_ids, clip_model, clip_preprocess, pil_image, device, args, processor, prompt_str):
    teacher_model.eval()
    with torch.no_grad():
        # --- 1. Generate Candidates ---
        if isinstance(teacher_model, LlavaForConditionalGeneration):
            inputs = processor(text=prompt_str, images=pil_image, return_tensors="pt").to(device, dtype=teacher_model.dtype)
            candidate_output_ids = teacher_model.generate(**inputs, do_sample=True, num_return_sequences=args.num_candidate_captions, temperature=0.8, top_p=0.9, max_new_tokens=args.max_new_tokens, use_cache=True, pad_token_id=tokenizer.eos_token_id)
        else:
            candidate_output_ids = teacher_model.generate(prompt_ids, images=image_tensor, image_sizes=[image_size], do_sample=True, num_return_sequences=args.num_candidate_captions, temperature=0.8, top_p=0.9, max_new_tokens=args.max_new_tokens, use_cache=True, pad_token_id=tokenizer.eos_token_id)

        # Decode all candidates
        candidate_captions = [tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in candidate_output_ids]
        
        # Prepare Image for CLIP
        image_input_clip = clip_preprocess(pil_image).unsqueeze(0).to(device)
        image_features = clip_model.encode_image(image_input_clip)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # --- UPDATED: Sentence-level Q10 Scoring for each candidate ---
        candidate_q10_scores = []

        for cap in candidate_captions:
            # Clean caption (handle ASSISTANT tag)
            clean_cap = cap.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in cap else cap
            
            # Split into sentences
            raw_sents = nltk.sent_tokenize(clean_cap)
            text_chunks = [s.strip() for s in raw_sents if len(s.strip()) > 2]
            
            if not text_chunks:
                candidate_q10_scores.append(-1.0) # Penalty for empty
                continue
                
            # CLIP encode individual sentences
            text_tokens = clip.tokenize(text_chunks, truncate=True).to(device)
            text_features = clip_model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # Sentence-level similarities
            sims = (100.0 * image_features @ text_features.T).squeeze(0)
            #mean = similarities.mean().item()
            
            # Q10 aggregation (the hallucination detector)
            q10_val = torch.quantile(sims.float(), 0.1).item()
            candidate_q10_scores.append(q10_val)

        # Select best candidate based on Q10 scores
        best_idx = np.argmax(candidate_q10_scores)
        best_full_caption = candidate_captions[best_idx]
        
        # --- Continue with original sequence prep logic ---
        gt_caption = best_full_caption.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in best_full_caption else best_full_caption

        gt_caption_ids = tokenizer(gt_caption, return_tensors='pt', add_special_tokens=False).input_ids.to(device)
        if gt_caption_ids.nelement() > 0 and gt_caption_ids[0, -1] != tokenizer.eos_token_id:
            eos_tensor = torch.tensor([[tokenizer.eos_token_id]], device=device)
            gt_caption_ids = torch.cat([gt_caption_ids, eos_tensor], dim=1)

        gt_sequence_ids = torch.cat([prompt_ids, gt_caption_ids], dim=1)
        gt_input_ids_for_model = gt_sequence_ids[:, :-1]
        attention_mask = torch.ones_like(gt_input_ids_for_model)
        
        return gt_caption, gt_input_ids_for_model, gt_sequence_ids, attention_mask


def parse_args():
    # This function is identical to your original
    parser = argparse.ArgumentParser(description="LLaVA LLM Adaptation with EMA Teacher enabled by default")
    parser.add_argument('--llava-model-path', type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument('--json-path', type=str, required=True)
    parser.add_argument('--image-dir', type=str, required=True)
    parser.add_argument('--output-base-dir', type=str, default="./ttt_llm_lora_output_ema")
    parser.add_argument('--image-start-index', type=int, default=0)
    parser.add_argument('--num-images-to-process', type=int, default=1)
    parser.add_argument('--corruption-name', type=str, default='gaussian_noise')
    parser.add_argument('--corruption-severity', type=int, default=5)
    parser.add_argument('--adaptation-steps', type=int, default=50)
    parser.add_argument('--num-candidate-captions', type=int, default=16)
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup-steps', type=int, default=0)
    parser.add_argument('--reevaluate-gt-every', type=int, default=20)
    parser.add_argument('--lora-r', type=int, default=8)
    parser.add_argument('--lora-alpha', type=int, default=16)
    parser.add_argument('--lora-target-modules', nargs='+', default=["q_proj", "v_proj"])
    parser.add_argument('--disable-ema-teacher', action='store_false', dest='use_ema_teacher', help="Disable the EMA teacher (use frozen teacher instead).")
    parser.add_argument('--ema-decay', type=float, default=0.999)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log-every', type=int, default=15)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    run_params_str = f"lora_r{args.lora_r}_lr{args.lr}_steps{args.adaptation_steps}_reeval{args.reevaluate_gt_every}"
    if args.use_ema_teacher: run_params_str += f"_ema{args.ema_decay}"
    else: run_params_str += "_frozen_teacher"
    corruption_output_dir = os.path.join(args.output_base_dir, run_params_str, f"{args.corruption_name}_{args.corruption_severity}")
    os.makedirs(corruption_output_dir, exist_ok=True)

    print("--- ⚙️ CONFIGURATION ---"); print(vars(args)); print("-----------------------")
    
    print("\n--- 🧠 LOADING BASE MODELS (ONCE) ---")
    disable_torch_init()

        # Robustly detect if the model is HF-native or requires original LLaVA loading scripts
    is_hf_model = False  # Default to False for liuhaotian models

    # Only set to True if explicitly HF-compatible
    if '-hf' in args.llava_model_path or "tiny-llava" in args.llava_model_path:
        is_hf_model = True
    elif 'liuhaotian' not in args.llava_model_path:
        # Assume non-liuhaotian models are HF-compatible
        is_hf_model = True

    processor = None
    if is_hf_model:
        print(f"Detected HF-compatible model. Loading with Auto-classes: '{args.llava_model_path}'")
        base_model = LlavaForConditionalGeneration.from_pretrained(args.llava_model_path, torch_dtype=target_dtype, low_cpu_mem_usage=True, device_map='auto', cache_dir=CACHE_DIR)
        processor = AutoProcessor.from_pretrained(args.llava_model_path, cache_dir=CACHE_DIR)
        tokenizer, image_processor = processor.tokenizer, processor.image_processor
    else:
        print(f"Using original LLaVA loader for model: '{args.llava_model_path}'")
        model_name = get_model_name_from_path(args.llava_model_path)
        tokenizer, base_model, image_processor, _ = load_pretrained_model(model_path=args.llava_model_path, model_base=None, model_name=model_name, load_8bit=False, load_4bit=False, device_map='auto', torch_dtype=target_dtype, cache_dir=CACHE_DIR)
    
    base_model.eval()
    clip_model, clip_preprocess = clip.load("ViT-L/14", device=device); clip_model.eval()
    print("  ✅ Base models loaded.")
    
    print("\n--- 🖼️ PREPARING DATA & PROMPT ---")
    with open(args.json_path, 'r') as f: image_entries = [json.loads(line) for line in f]
    selected_image_entries = image_entries[args.image_start_index : args.image_start_index + args.num_images_to_process]

    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + ' '); conv.append_message(conv.roles[1], None)
    prompt_str = conv.get_prompt()
    prompt_ids_for_generate = tokenizer_image_token(prompt_str, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
    input_token_len = prompt_ids_for_generate.shape[1]

    print("\n--- 🚀 BEGINNING DYNAMIC ADAPTATION PROCESS 🚀 ---")
    for image_entry in (pbar := tqdm(selected_image_entries, desc="Adapting Images")):
        image_filename = image_entry['image']
        image_stem = os.path.splitext(image_filename)[0]
        pbar.set_postfix_str(image_filename)
        image_output_dir = os.path.join(corruption_output_dir, image_stem)
        os.makedirs(image_output_dir, exist_ok=True)

        student_model = deepcopy(base_model)
        apply_lora_to_llm(student_model, r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=args.lora_target_modules)
        
        if args.use_ema_teacher:
            teacher_model = deepcopy(student_model); teacher_model.requires_grad_(False)
        else:
            teacher_model = base_model
        
        lora_params_to_update = [p for name, p in student_model.named_parameters() if 'lora' in name and p.requires_grad]
        optimizer = torch.optim.AdamW(lora_params_to_update, lr=args.lr)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.adaptation_steps)

        image_path = os.path.join(args.image_dir, image_filename)
        pil_image_original = Image.open(image_path).convert("RGB")
        corrupted_pil = Image.fromarray(corrupt(np.array(pil_image_original), corruption_name=args.corruption_name, severity=args.corruption_severity))
        image_tensor = image_processor(corrupted_pil, return_tensors="pt")['pixel_values'].to(student_model.dtype)
        
        student_augment_transform = T.Compose([T.RandomHorizontalFlip(p=0.5), T.RandomRotation(degrees=5), T.RandomResizedCrop(size=corrupted_pil.size, scale=(0.95, 1.0), ratio=(0.95, 1.05))])
        loss_history = []; current_gt_caption = ""
        best_loss, best_clip_score = float('inf'), -float('inf')

        for i in (inner_pbar := tqdm(range(args.adaptation_steps), desc=f"Adapting {image_stem[:15]}...", leave=False)):
            
            # --- STAGE 1: GET TRAINING DATA ---
            if i == 0 or (args.reevaluate_gt_every > 0 and i % args.reevaluate_gt_every == 0):
                new_gt_caption, gt_input_ids_for_model, gt_sequence_ids, attention_mask = get_pseudo_gt_and_targets(
                    teacher_model, tokenizer, image_tensor, corrupted_pil.size, prompt_ids_for_generate,
                    clip_model, clip_preprocess, corrupted_pil, device, args, processor, prompt_str
                )
                if new_gt_caption != current_gt_caption:
                    inner_pbar.write(f"  Step {i}: New pseudo-GT found: '{new_gt_caption[:80]}...'")
                    current_gt_caption = new_gt_caption

            # --- STAGE 2 & 3: FORWARD PASS & LOSS CALCULATION ---
            augmented_pil = student_augment_transform(corrupted_pil)
            student_model.train()
            optimizer.zero_grad()

            if is_hf_model:

                # 1. Prepare the full input sequence using the processor.
                full_training_text = prompt_str + current_gt_caption
                inputs = processor(
                    text=full_training_text, 
                    images=augmented_pil, 
                    return_tensors="pt", 
                    padding="longest"
                ).to(device)

                # 2. Create the labels tensor by cloning the input_ids.
                labels = inputs.input_ids.clone()

                prompt_ids = tokenizer(prompt_str, return_tensors="pt").input_ids
                prompt_len = prompt_ids.shape[1]

                labels[:, :prompt_len] = -100

                student_outputs = student_model(**inputs, labels=labels)
                ce_loss = student_outputs.loss


            else:
                # For original LLaVA models, use the manual tensor preparation.
                augmented_image_tensor = image_processor(augmented_pil, return_tensors="pt")['pixel_values'].to(student_model.dtype)
                student_outputs = student_model(input_ids=gt_input_ids_for_model, images=augmented_image_tensor, image_sizes=[corrupted_pil.size])
                student_logits = student_outputs.logits

                # Manually calculate the cross-entropy loss on the response part.
                target_labels = gt_sequence_ids[:, 1:].contiguous()
                response_start_index = input_token_len - 1
                student_response_logits = student_logits[:, response_start_index:]
                response_labels = target_labels[:, response_start_index:]
                seq_len = min(student_response_logits.size(1), response_labels.size(1))
                ce_loss = F.cross_entropy(
                    student_response_logits[:, :seq_len, :].reshape(-1, student_logits.size(-1)),
                    response_labels[:, :seq_len].reshape(-1)
                )

            loss_item = ce_loss.item()
            if not math.isfinite(loss_item):
                inner_pbar.write(f"!!! WARNING: Non-finite loss ({loss_item}) at step {i}. Skipping step.")
                continue

            # --- STAGE 4: BACKWARD PASS & OPTIMIZER STEP ---
            ce_loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params_to_update, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            if args.use_ema_teacher: 
                update_ema_teacher(teacher_model, student_model, args.ema_decay)

            # --- STAGE 5: EVALUATION ---
            score = calculate_clip_score(student_model, tokenizer, image_tensor, corrupted_pil, prompt_ids_for_generate, clip_model, clip_preprocess, device, args, processor, prompt_str)
            
            # --- STAGE 6: LOGGING & SAVING ---
            loss_history.append(loss_item)
            if loss_item < best_loss:
                best_loss = loss_item
                save_lora_weights(student_model, os.path.join(image_output_dir, 'lora_iter_best.pth'))
            if score > best_clip_score:
                inner_pbar.write(f"  Step {i}: New best CLIP score! {best_clip_score:.2f} -> {score:.2f}")
                best_clip_score = score
                save_lora_weights(student_model, os.path.join(image_output_dir, 'lora_iter_max_clip.pth'))
            
            inner_pbar.set_postfix(loss=f"{loss_item:.4f}", clip=f"{score:.2f}", best_clip=f"{best_clip_score:.2f}")
            
        # --- Save final artifacts for the image ---
        save_lora_weights(student_model, os.path.join(image_output_dir, 'lora_final.pth'))
        save_loss_plot(loss_history, os.path.join(image_output_dir, 'loss_curve.png'), f"Loss Curve for {image_filename}")
        
        # Cleanup memory before next image
        del student_model, teacher_model, optimizer, scheduler, lora_params_to_update
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    print("\n--- ✅ SCRIPT COMPLETE ✅ ---")
    print(f"All artifacts for this job saved in: {corruption_output_dir}")



if __name__ == "__main__":
    main()

