#!/bin/bash

# Configuration parameters
data_path="Health"
input_len=24
target="OT"  # Change target according to Health dataset
feature="S"  # M: multivariate, S: univariate, MS: multivariate forecasting univariate
llm_model="GPT2"
batch_size=1
device="cuda"

# Create log directories if they don't exist
mkdir -p ./Results/emb_logs/
mkdir -p ./Embeddings_TimeCMA/

echo "Starting embedding generation for dataset $data_path..."

log_file="./Results/emb_logs/${data_path}.log"

python llm/generate_embedding.py \
    --data_path "$data_path" \
    --input_len "$input_len" \
    --target "$target" \
    --feature "$feature" \
    --llm_model "$llm_model" \
    --batch_size "$batch_size" \
    --device "$device" \
    --emb_saved_path "./Embeddings_TimeCMA" \
    > "$log_file" 2>&1

if [ $? -eq 0 ]; then
    echo "✓ Completed - Log: $log_file"
else
    echo "✗ Error processing - Check log: $log_file"
fi

echo "Finished generating embeddings for all splits!"