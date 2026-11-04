#!/bin/bash
# ============================================================================
# Machine 2: Batch generate synthetic data using Wan 2.1 14B (8-GPU parallel)
# Uses Ulysses Sequence Parallelism: all 8 GPUs collaborate on each video.
# Supports resume: skips already-generated samples.
# Starts from index 100000 to avoid overlap with Machine 1.
# ============================================================================
set -e

MODEL_DIR=/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/video2/Wan2.1-T2V-14B
OUTPUT_DIR=/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/video/synthetic_data_machine2
NUM_GPUS=8
SIZE="832*480"
FRAME_NUM=81
SAMPLE_STEPS=50
SEED_START=100000

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/.cache/huggingface

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "  RDD-Wan Data Generation - Machine 2"
echo "  Model: Wan 2.1 14B (${NUM_GPUS}-GPU Ulysses SP)"
echo "  Output: $OUTPUT_DIR"
echo "  Resolution: $SIZE, Frames: $FRAME_NUM"
echo "  Each video uses all $NUM_GPUS GPUs"
echo "  Index starts at $SEED_START (no overlap with Machine 1)"
echo "=============================================="

# Same prompts as machine 1 (different seeds produce different videos)
PROMPTS=(
"A golden retriever running through a field of wildflowers in slow motion, sunlight filtering through the petals."
"Timelapse of a bustling city intersection at night with streaks of car headlights and neon reflections."
"An astronaut floating in zero gravity inside a space station, reaching for a floating water droplet."
"Ocean waves crashing against rocky cliffs at sunset, with seagulls flying overhead."
"A dancer performing a graceful ballet routine on an empty stage with dramatic spotlight."
"Snow falling gently over a quiet Japanese garden with a stone lantern and koi pond."
"A chef preparing sushi in a traditional Japanese restaurant, carefully slicing fresh salmon."
"Aerial drone footage of a winding river cutting through autumn-colored forests."
"A mechanical clock being assembled piece by piece in stop-motion style."
"Lightning storm over a vast desert landscape at twilight with cactus silhouettes."
"A hummingbird hovering near a bright red flower, wings beating rapidly."
"Steam rising from a hot cup of coffee on a rainy window sill, droplets sliding down the glass."
"A group of children launching colorful kites on a windy beach."
"Northern lights dancing across a starry sky above a frozen lake in Iceland."
"An artist painting a large canvas with bold brushstrokes of vibrant colors."
"Underwater footage of a coral reef teeming with tropical fish and sea turtles."
"A vintage steam train chugging through a mountain pass with smoke billowing."
"Cherry blossom petals falling in slow motion along a peaceful walkway in Tokyo."
"A blacksmith forging a sword, sparks flying in a dimly lit workshop."
"Time-lapse of clouds rolling over mountain peaks at sunrise with golden light."
"A cat gracefully leaping from one rooftop to another in a European village."
"Fireworks exploding over a city skyline reflected in calm harbor waters."
"A surfer riding a massive wave in slow motion, water spraying dramatically."
"Raindrops creating ripples in a still pond surrounded by lily pads."
"A robot hand carefully picking up a delicate glass ornament."
"Hot air balloons rising over the valleys of Cappadocia at dawn."
"A pianist hands moving rapidly over piano keys during a passionate concerto."
"Butterflies emerging from chrysalises in a greenhouse garden."
"A candle flame flickering in the dark, casting dancing shadows on the wall."
"Two wolves running through deep snow in a pine forest during winter."
"Lava flowing from a volcanic eruption meeting the ocean, creating steam clouds."
"A paper airplane being folded and launched from a skyscraper rooftop."
"Schools of fish forming mesmerizing swirling patterns in deep blue ocean."
"A woodpecker drumming on a tree trunk in a misty morning forest."
"Time-lapse of a flower blooming from bud to full bloom over several hours."
"A skateboarder performing tricks in an empty swimming pool at golden hour."
"Fog rolling through a dense redwood forest with shafts of sunlight breaking through."
"A street musician playing violin on a cobblestone street as people walk by."
"Ice crystals forming on a window pane in extreme cold, intricate patterns emerging."
"A falcon diving at high speed to catch prey, captured in slow motion."
"Ships passing through a canal lock system with water levels changing."
"A glass of red wine being poured in slow motion with light passing through."
"Wildfire spreading across dry grassland with thick smoke rising into the sky."
"A spider weaving an intricate web between branches, dew drops forming."
"Aerial view of rice terraces in Bali with farmers working in knee-deep water."
"A marble rolling through an elaborate Rube Goldberg machine."
"Northern sea otters floating on their backs, cracking shellfish with rocks."
"A time-lapse of an ice sculpture melting under warm studio lights."
"Bioluminescent plankton glowing blue in ocean waves crashing on a dark beach."
"A flock of starlings performing a murmuration at dusk, swirling across the sky."
)

NUM_PROMPTS=${#PROMPTS[@]}
SAMPLE_IDX=$SEED_START
TOTAL_GENERATED=0

echo "Total prompts available: $NUM_PROMPTS (will cycle indefinitely)"
echo "Press Ctrl+C to stop at any time."
echo ""

while true; do
    # Cycle through prompts
    PROMPT_IDX=$(( (SAMPLE_IDX - SEED_START) % NUM_PROMPTS))
    PROMPT="${PROMPTS[$PROMPT_IDX]}"
    SAVE_FILE="$OUTPUT_DIR/sample_$(printf '%06d' $SAMPLE_IDX).mp4"
    
    # Skip if already exists (resume support)
    if [ -f "$SAVE_FILE" ]; then
        echo "[Skip] $SAVE_FILE already exists"
        SAMPLE_IDX=$((SAMPLE_IDX + 1))
        continue
    fi
    
    CURRENT_SEED=$SAMPLE_IDX
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[$(date '+%H:%M:%S')] Generating sample $SAMPLE_IDX (seed=$CURRENT_SEED)"
    echo "  Prompt: ${PROMPT:0:80}..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Run 8-GPU inference with Ulysses SP
    torchrun --nproc_per_node=$NUM_GPUS \
        /home/tiger/multiscale_diffusion/Wan2.1/generate.py \
        --task t2v-14B \
        --size "$SIZE" \
        --frame_num $FRAME_NUM \
        --ckpt_dir "$MODEL_DIR" \
        --ulysses_size $NUM_GPUS \
        --sample_steps $SAMPLE_STEPS \
        --sample_guide_scale 5.0 \
        --base_seed $CURRENT_SEED \
        --prompt "$PROMPT" \
        --save_file "$SAVE_FILE"
    
    if [ $? -eq 0 ]; then
        TOTAL_GENERATED=$((TOTAL_GENERATED + 1))
        echo "[Done] Generated $TOTAL_GENERATED videos total. Latest: $SAVE_FILE"
    else
        echo "[Error] Failed to generate sample $SAMPLE_IDX, skipping..."
    fi
    
    SAMPLE_IDX=$((SAMPLE_IDX + 1))
done
