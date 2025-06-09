#!/bin/bash

# Configuration parameters
input_len=24
feature="S"  # M: multivariate, S: univariate, MS: multivariate forecasting univariate
llm_model="GPT2"
batch_size=1
device="cuda"

# Dataset configurations (data_path:target)
declare -A datasets=(
    ["Agriculture"]="OT"
    ["Climate"]="OT"
    ["Economy"]="OT"
    ["Energy"]="OT"
    ["Environment"]="OT"
    ["Health"]="OT"
    ["Security"]="OT"
    ["SocialGood"]="OT"
    ["Traffic"]="OT"
)

# Create log directories if they don't exist
mkdir -p ./Results/emb_logs/
mkdir -p ./Embeddings_TimeCMA/

echo "Starting embedding generation for all datasets..."
echo "Found ${#datasets[@]} datasets to process"

# Track success/failure
success_count=0
error_count=0
failed_datasets=()

# Process each dataset
for data_path in "${!datasets[@]}"; do
    target="${datasets[$data_path]}"

    echo ""
    echo "=================================================="
    echo "Processing dataset: $data_path"
    echo "Target: $target"
    echo "=================================================="

    log_file="./Results/emb_logs/${data_path}.log"

    # Run the embedding generation
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

    # Check if successful
    if [ $? -eq 0 ]; then
        echo "✓ SUCCESS: $data_path - Log: $log_file"
        ((success_count++))
    else
        echo "✗ ERROR: $data_path - Check log: $log_file"
        ((error_count++))
        failed_datasets+=("$data_path")
    fi
done

# Summary
echo ""
echo "=================================================="
echo "PROCESSING SUMMARY"
echo "=================================================="
echo "Total datasets: ${#datasets[@]}"
echo "Successful: $success_count"
echo "Failed: $error_count"

if [ $error_count -gt 0 ]; then
    echo ""
    echo "Failed datasets:"
    for dataset in "${failed_datasets[@]}"; do
        echo "  - $dataset"
    done
fi

echo ""
echo "All logs saved in: ./Results/emb_logs/"
echo "Embeddings saved in: ./Embeddings_TimeCMA/"
echo "Finished processing all datasets!"