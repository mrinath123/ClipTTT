#!/bin/bash
#SBATCH -p gpu22,gpu24,gpu17
#SBATCH --gres=gpu:1
#SBATCH -t 2:00:00
#SBATCH -o /BS/DApt/work/LLaVA/slurm_logs/ddp/run-%A_%a.out # Unique log file for each job task
#SBATCH -e /BS/DApt/work/LLaVA/slurm_logs/ddp/run-%A_%a.err # Unique error file for each job task
#SBATCH --job-name=lora_clip_inference
#SBATCH --array=0-14

#================================================================================#
# SCRIPT CONFIGURATION - EDIT THIS SECTION
#================================================================================#
# --- 1. SET THE PATH TO YOUR PYTHON SCRIPT ---
PYTHON_SCRIPT_PATH="/BS/DApt/work/LLaVA/llava/eval/llava_lora_clipscore_cptn_gen.py"
# === CONFIGURATION FOR: LLM + Vision LoRA ===
LORA_WEIGHTS_PARENT_DIR="/BS/DApt/work/LLaVA/Vistext_LLAVA_7b_clipscore_iter_70_CLIPScore/lora_r8_lr5e-05_steps70_reeval20_ema0.999"
ADAPT_LLM=true
ADAPT_VISION=false   # IMPORTANT: The training script did NOT adapt the vision tower.
LORA_LLM_START=0     # Must match the training script 
LORA_LLM_END=-1  
#\\\winfs-inf.mpi-inf.mpg.de\BS\DApt\work\LLaVA\Vistext_LLAVA_7b_clipscore_iter_70_CLIPScore\lora_r8_lr5e-05_steps70_reeval20_ema0.999\brightness_5\COCO_val2014_000000007320
# --- 3. UNIVERSAL PARAMETERS (Usually don't need to change) ---
FIXED_CORRUPTION_SEVERITY=5
MODEL_PATH="liuhaotian/llava-v1.5-7b"
IMAGE_DIR="/BS/DApt/work/LLaVA/playground/data/eval/pope/val2014/"
CONV_MODE="vicuna_v1"
MAX_NEW_TOKENS=512
LORA_R=8
LORA_ALPHA=16

#================================================================================#
# SLURM JOB SETUP (No need to edit below this line)
#================================================================================#
CORRUPTIONS=("gaussian_noise" "shot_noise" "impulse_noise" "defocus_blur" "glass_blur" "motion_blur" "zoom_blur" "snow" "frost" "fog" "brightness" "contrast" "elastic_transform" "pixelate" "jpeg_compression")
CHECKPOINT_FILES=("lora_iter_max_clip.pth")

NUM_CHECKPOINTS=${#CHECKPOINT_FILES[@]}
CORRUPTION_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_CHECKPOINTS))
CHECKPOINT_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_CHECKPOINTS))
CURRENT_CORRUPTION_TYPE=${CORRUPTIONS[$CORRUPTION_INDEX]}
CURRENT_CHECKPOINT_NAME=${CHECKPOINT_FILES[$CHECKPOINT_INDEX]}

echo "--- Universal LoRA Inference Launcher ---"
echo "SLURM Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Evaluating Experiment: ${LORA_WEIGHTS_PARENT_DIR}"
echo "Corruption: ${CURRENT_CORRUPTION_TYPE}"
echo "Checkpoint: ${CURRENT_CHECKPOINT_NAME}"

# --- Build the command dynamically based on the configuration ---
CMD="python ${PYTHON_SCRIPT_PATH} \
  --lora-weights-parent-dir '${LORA_WEIGHTS_PARENT_DIR}' \
  --model-path '${MODEL_PATH}' \
  --image-dir '${IMAGE_DIR}' \
  --conv-mode '${CONV_MODE}' \
  --max-new-tokens ${MAX_NEW_TOKENS} \
  --corruption-name '${CURRENT_CORRUPTION_TYPE}' \
  --corruption-severity ${FIXED_CORRUPTION_SEVERITY} \
  --checkpoint-name '${CURRENT_CHECKPOINT_NAME}'"

if [ "$ADAPT_LLM" = true ]; then
  CMD+=" --adapt-llm"
  CMD+=" --lora-llm-r ${LORA_R}"
  CMD+=" --lora-llm-alpha ${LORA_ALPHA}"
  CMD+=" --lora-llm-start-layer ${LORA_LLM_START}"
  CMD+=" --lora-llm-end-layer ${LORA_LLM_END}"
fi

# This block will now be skipped, which is the correct behavior.
if [ "$ADAPT_VISION" = true ]; then
  CMD+=" --adapt-vision"
  CMD+=" --lora-vision-r ${LORA_R}"
  CMD+=" --lora-vision-alpha ${LORA_ALPHA}"
  CMD+=" --lora-vision-start-layer ${LORA_VISION_START}"
  CMD+=" --lora-vision-end-layer ${LORA_VISION_END}"
fi

# --- Execute the command ---
echo ""
echo "Executing Command:"
echo "${CMD}"
echo ""

eval ${CMD}

EXIT_CODE=$?
if [ ${EXIT_CODE} -eq 0 ]; then
  echo "SUCCESS: Task completed."
else
  echo "FAILURE: Task failed with exit code ${EXIT_CODE}."
fi