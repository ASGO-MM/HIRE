export CUDA_VISIBLE_DEVICES=0,1,2,3

IFS=',' read -ra ADDR <<< "$CUDA_VISIBLE_DEVICES"
num_chunks=${#ADDR[@]}

model_path="path/to/llava"  # model path
data_path="path/to/densely_captioned_images"    # data path
hidden_states_path="output/hidden_states"   #  path to save hidden_states for further training
data_size=2000

for ((chunk_idx=0; chunk_idx<num_chunks; chunk_idx++))
do
    CUDA_VISIBLE_DEVICES=${ADDR[$chunk_idx]}
    python train_hire/extract_hidden_states.py \
        --model-path $model_path \
        --data-path $data_path \
        --hs-path $hidden_states_path \
        --data-size $data_size \
        --num-chunks $num_chunks \
        --chunk-idx $chunk_idx &
done