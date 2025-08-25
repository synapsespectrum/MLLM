#!/bin/bash

# Configuration parameters
feature="S"  # M: multivariate, S: univariate, MS: multivariate forecasting univariate
llm_model="GPT2"
batch_size=1
device="cuda"

# Define datasets with their corresponding input lengths
declare -A dataset_input_lens
dataset_input_lens["Environment"]=84
dataset_input_lens["Energy"]=36
dataset_input_lens["Health"]=36
dataset_input_lens["Agriculture"]=8
dataset_input_lens["Climate"]=8
dataset_input_lens["Economy"]=8
dataset_input_lens["Security"]=8
dataset_input_lens["SocialGood"]=8
dataset_input_lens["Traffic"]=8

# Create log directories if they don't exist
mkdir -p ./log-emb/text_emb_logs/${llm_model}/
mkdir -p ./Embeddings/

echo "Starting embedding generation for all datasets with different input lengths..."

# Count total datasets
dataset_count=${#dataset_input_lens[@]}
echo "Found $dataset_count datasets to process"

# Track success/failure
success_count=0
error_count=0
failed_datasets=""

# Process each dataset with its specific input length
for data_path in "${!dataset_input_lens[@]}"; do
    input_len=${dataset_input_lens[$data_path]}

    # Default target (you can customize this per dataset if needed)
    target="OT"

    echo ""
    echo "=================================================="
    echo "Processing dataset: $data_path"
    echo "Input length: $input_len"
    echo "Target: $target"
    echo "=================================================="

    log_file="./log-emb/text_emb_logs/${llm_model}/${data_path}_len${input_len}.log"

    # Run the embedding generation
    python generate_embedding.py \
        --embedding_mode 0 \
        --data_path "$data_path" \
        --target "$target" \
        --feature "$feature" \
        --llm_model "$llm_model" \
        --batch_size "$batch_size" \
        --device "$device" \
        --input_len "$input_len" \
        --emb_saved_path "./Embeddings" \
         > "$log_file" 2>&1

    # Check if successful
    if [ $? -eq 0 ]; then
        echo "✓ SUCCESS: $data_path (input_len=$input_len) - Log: $log_file"
        success_count=$((success_count + 1))
        echo "Embeddings saved in ./Embeddings/text/${llm_model}/$data_path/"
    else
        echo "✗ ERROR: $data_path (input_len=$input_len) - Check log: $log_file"
        error_count=$((error_count + 1))
        if [ -z "$failed_datasets" ]; then
            failed_datasets="$data_path(len=$input_len)"
        else
            failed_datasets="$failed_datasets $data_path(len=$input_len)"
        fi
    fi
done

# Summary
echo ""
echo "=================================================="
echo "PROCESSING SUMMARY"
echo "=================================================="
echo "Total datasets: $dataset_count"
echo "Successful: $success_count"
echo "Failed: $error_count"

if [ $error_count -gt 0 ]; then
    echo ""
    echo "Failed datasets:"
    for dataset in $failed_datasets; do
        echo "  - $dataset"
    done
fi

# Show successful configurations
if [ $success_count -gt 0 ]; then
    echo ""
    echo "Successfully processed configurations:"
    for data_path in "${!dataset_input_lens[@]}"; do
        input_len=${dataset_input_lens[$data_path]}
        embedding_path="./Embeddings/text/${llm_model}/$data_path/"
        if [ -d "$embedding_path" ]; then
            echo "  - $data_path: input_len=$input_len, pred_len=$pred_len"
        fi
    done
fi

echo ""
echo "All logs saved in: ./log-emb/text_emb_logs/${llm_model}/"
echo "Embeddings saved in: ./Embeddings/text/${llm_model}/"
echo "Finished processing all datasets with their specific input lengths!"