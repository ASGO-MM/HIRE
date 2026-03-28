export CUDA_VISIBLE_DEVICES=0,1,2,3
IFS=',' read -ra ADDR <<< "$CUDA_VISIBLE_DEVICES"
num_chunks=${#ADDR[@]}

CHECKPOINT_PATH="output/checkpoint/hire_editor"
edit_strength=1.0

for ((chunk_idx=0; chunk_idx<num_chunks; chunk_idx++))
do 
    CUDA_VISIBLE_DEVICES=${ADDR[$chunk_idx]}
    python train_hire/generate_captions_router.py \
        --editor-path "${CHECKPOINT_PATH}/hire_editor_final.pth" \
        --router-path output/RL_router/router_llava_epoch100_new_sgd_banlance.pth \
        --model-path path/to/llava-v1.5-7b \
        --image-folder path/to/coco2014/val2014 \
        --anno-folder path/to/coco2014/annotations/instances_val2014.json \
        --output-file output/chair/llava_hire.json \
        --data-size 500 \
        --num-chunks $num_chunks \
        --edit-strength $edit_strength \
        --batch-size 16 \
        --chunk-idx $chunk_idx &
done
