#!/bin/bash
#SBATCH --partition=gpu24             # Target partition
#SBATCH --gres=gpu:1                  # Allocate 1 GPU per job
#SBATCH -t 3:00:00                    # Walltime for 50 images 
#SBATCH -o /BS/DApt/work/LLaVA/slurm_logs/baseline_qwen/run-%A_%a.out
#SBATCH -e /BS/DApt/work/LLaVA/slurm_logs/baseline_qwen/run-%A_%a.err
#SBATCH --job-name=tta_100_n # A descriptive name for your job array
#SBATCH --array=0-149%10         
#================================================================================#
# SCRIPT CONFIGURATION
#================================================================================#

# --- MODIFIABLE ---
IMAGES_PER_JOB=50       # How many images each job will process.
NUM_TOTAL_IMAGES=500    # Total number of images for one corruption.
NUM_CORRUPTIONS=15      # Number of different corruption types.

# Auto-calculate number of chunks/jobs per corruption
NUM_CHUNKS_PER_CORRUPTION=$((NUM_TOTAL_IMAGES / IMAGES_PER_JOB))  # 500 / 50 = 10 chunks

# List of all 15 ImageNet-C corruptions
corruptions=(
  "brightness" "gaussian_noise" "shot_noise" "impulse_noise" "defocus_blur" "glass_blur"
  "motion_blur" "zoom_blur" "snow" "frost" "fog" "contrast"
  "elastic_transform" "pixelate" "jpeg_compression"
)

#================================================================================#
# DYNAMIC TASK CALCULATION
#================================================================================#

# Determine corruption type and image chunk for this specific job
CORRUPTION_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_CHUNKS_PER_CORRUPTION))
CHUNK_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_CHUNKS_PER_CORRUPTION))

# Calculate starting image index for Python script
IMAGE_START_INDEX=$((CHUNK_INDEX * IMAGES_PER_JOB))

# Get the string name of the corruption
corruption=${corruptions[$CORRUPTION_INDEX]}

#================================================================================#
# LOGGING AND ENVIRONMENT SETUP
#================================================================================#

echo "========================================================"
echo "SLURM Job: $SLURM_JOB_ID, Array Task: $SLURM_ARRAY_TASK_ID"
echo "Running on host: $(hostname)"
echo "--------------------------------------------------------"
echo "This task will process:"
echo "  -> Corruption:      $corruption (Index: $CORRUPTION_INDEX)"
echo "  -> Image Chunk:     $CHUNK_INDEX (Images $IMAGE_START_INDEX to $((IMAGE_START_INDEX + IMAGES_PER_JOB - 1)))"
echo "========================================================"

export OMP_NUM_THREADS=1
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1


ADAPTATION_STEPS=70
REEVAL_EVERY=1


#================================================================================#
# PATHS AND HYPERPARAMETERS
#================================================================================#

# --- IMPORTANT: Set your paths here ---
PYTHON_SCRIPT_PATH="/BS/DApt/work/LLaVA/llava/eval/llava_vistext_clipscore_normal_dist.py"
IMAGE_DIR="/BS/DApt/work/LLaVA/playground/data/eval/pope/val2014"
JSON_PATH="/BS/DApt/work/LLaVA/playground/data/coco_pope_chat_popular.json"

# --- Adaptation Hyperparameters ---
CORRUPTION_SEVERITY=5

LEARNING_RATE=5e-5
MAX_NEW_TOKENS=512
CANDIDATE_CAPTIONS=16
LOG_EVERY=2
SEED=42
OUTPUT_BASE_DIR="/BS/DApt/work/LLaVA/Vistext_LLAVA_7b_clipscore_iter_${ADAPTATION_STEPS}_normal_dist"
# --- LoRA Hyperparameters ---
LORA_R=8
LORA_ALPHA=16

# --- EMA Teacher Hyperparameters ---
EMA_DECAY=0.999
# model names
# llava-v1.6-vicuna-13b
# bczhou/tiny-llava-v1-hf
# liuhaotian/llava-v1.5-7b
#================================================================================#
# EXECUTION COMMAND
#================================================================================#

python "$PYTHON_SCRIPT_PATH" \
    --llava-model-path "liuhaotian/llava-v1.5-7b" \
    --image-dir "$IMAGE_DIR" \
    --json-path "$JSON_PATH" \
    --output-base-dir "$OUTPUT_BASE_DIR" \
    --image-start-index "$IMAGE_START_INDEX" \
    --num-images-to-process "$IMAGES_PER_JOB" \
    --corruption-name "$corruption" \
    --corruption-severity "$CORRUPTION_SEVERITY" \
    --adaptation-steps "$ADAPTATION_STEPS" \
    --lr "$LEARNING_RATE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --num-candidate-captions "$CANDIDATE_CAPTIONS" \
    --reevaluate-gt-every "$REEVAL_EVERY" \
    --log-every "$LOG_EVERY" \
    --seed "$SEED" \
    --lora-r "$LORA_R" \
    --lora-alpha "$LORA_ALPHA" \
    --ema-decay "$EMA_DECAY"

EXIT_CODE=$?

echo "--------------------------------------------------------"
echo "Python script finished with exit code $EXIT_CODE"
echo "--- End of SLURM Task $SLURM_ARRAY_TASK_ID ---"

exit $EXIT_CODE
