export CUDA_VISIBLE_DEVICES=0,1,2,3
IFS=',' read -ra ADDR <<< "$CUDA_VISIBLE_DEVICES"
NUM_CHUNKS=${#ADDR[@]}

HS_PATH="output/hidden_states"  # hidden_states saved path
CHECKPOINT_PATH="output/checkpoint/hire_editor"    # path to save editor checkpoint
DIRECTION_SAVE_PATH="output/direction"  # path to save hallucinatory direction
TYPE="dci"  # dataset type

# Model Hyperparameters
H_DIM=4096
I_DIM="2048 1024"
EDIT_LAYER=$(seq -s ' ' 0 2 62)

# Training Hyperparameters
LEARNING_RATE=1e-2
MIN_LR=1e-3
MAX_LEN=300
WEIGHT_DECAY=0
EPOCHS=1
SEED=39
DATA_SIZE=2000
ACCUMULATE_STEPS=2

python train_hire/train_editor.py \
    --hidden-states-path "$HS_PATH" \
    --learning-rate "$LEARNING_RATE" \
    --accumulate-steps "$ACCUMULATE_STEPS" \
    --min-lr "$MIN_LR" \
    --max-len "$MAX_LEN" \
    --weight-decay "$WEIGHT_DECAY" \
    --epoch "$EPOCHS" \
    --seed "$SEED" \
    --data-size "$DATA_SIZE" \
    --h-dim "$H_DIM" \
    --i-dim $I_DIM \
    --edit-layer $EDIT_LAYER \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --num-chunks "$NUM_CHUNKS" \
    --direction-save-path "$DIRECTION_SAVE_PATH" \
    --type "$TYPE"