# first step
export CUDA_VISIBLE_DEVICES=0,1,2,3

IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#GPUS[@]}

EDITOR_MODEL_PATH="output/checkpoint/hire_editor/hire_editor_final.pth" # path to editor checkpoint
MODEL_PATH="path/to/llava-v1.5-7b" # path to llava
IMAGE_FOLDER="path/to/coco2014/train2014"
ANNO_FOLDER="path/to/instances_train2014.json"
OUTPUT_FILE="output/chair/router_dpo.json"  # generate captions path
CHAIR_PATH="path/to/Chair2014.pkl"  # see https://github.com/Maxlinn/CHAIR-metric-standalone
POS_PATH="output/hidden_states/router_pos_1"    # path to save pos idden_states
NEG_PATH="output/hidden_states/router_neg_1"    # path to save neg idden_states

DATA_SIZE=8000

# Model Hyperparameters
H_DIM=4096
I_DIM="2048 1024"
EDIT_LAYER=$(seq -s ' ' 0 2 62)  # 0,2,4,...,62

for ((i=0; i<NUM_GPUS; i++)); do
    echo "Starting process $i on GPU $i..."
    
    python train_hire/train_router_step1.py \
        --editor-model-path "$EDITOR_MODEL_PATH" \
        --model-path "$MODEL_PATH" \
        --image-folder "$IMAGE_FOLDER" \
        --anno-folder "$ANNO_FOLDER" \
        --output-file "$OUTPUT_FILE" \
        --chair-path "$CHAIR_PATH" \
        --hs-path "$HS_PATH" \
        --data-size "$DATA_SIZE" \
        --pos-path "$POS_PATH" \
        --neg-path "$NEG_PATH" \
        --h-dim "$H_DIM" \
        --i-dim $I_DIM \
        --edit-layer $EDIT_LAYER \
        --num-chunks "$NUM_GPUS" \
        --chunk-idx "$i" &
done


python train_hire/train_router_step2.py \
    --data-path output/hidden_states/router_\<type\>_1 \
    --save-path output/RL_router \
    --batch-size 32 \
    --num-epochs 100 \
    --save-epoch 20 \
    --learning-rate 1e-2 \
    --eta-min 1e-3 \
    --p-dropout 0.5 \
    --beta 0.1 \
    --accumulate-steps 1 \
    --weight-decay 0 \
    --seed 39 \
    --h_dim 2048 1024 \
    --gpus 0 1 2 3 \