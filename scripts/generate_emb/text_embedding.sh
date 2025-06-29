
#!/bin/bash

# Configuration parameters
feature="S"  # M: multivariate, S: univariate, MS: multivariate forecasting univariate
llm_model="GPT2"
batch_size=1
device="cuda"

# List of datasets (you can modify this list as needed)
datasets="Agriculture Climate Economy Energy Environment Health Security SocialGood Traffic"

# Create log directories if they don't exist
mkdir -p ./Results/text_emb_logs/${llm_model}/
mkdir -p ./Embeddings/

echo "Starting embedding generation for all datasets..."

# Count total datasets
dataset_count=0
for dataset in $datasets; do
    dataset_count=$((dataset_count + 1))
done

echo "Found $dataset_count datasets to process"

# Track success/failure
success_count=0
error_count=0
failed_datasets=""

# Process each dataset
for data_path in $datasets; do
    # Default target (you can customize this per dataset if needed)
    target="OT"
    
    echo ""
    echo "=================================================="
    echo "Processing dataset: $data_path"
    echo "Target: $target"
    echo "=================================================="
    
    log_file="./Results/text_emb_logs/${llm_model}/${data_path}.log"
    # Run the embedding generation
    python generate_embedding.py \
        --embedding_mode 0 \
        --data_path "$data_path" \
        --target "$target" \
        --feature "$feature" \
        --llm_model "$llm_model" \
        --batch_size "$batch_size" \
        --device "$device" \
        --emb_saved_path "./Embeddings" \
         > "$log_file" 2>&1
    
    # Check if successful
    if [ $? -eq 0 ]; then
        echo "✓ SUCCESS: $data_path - Log: $log_file"
        success_count=$((success_count + 1))
        echo "Embeddings saved in ./Embeddings/text/${llm_model}/$data_path/"
    else
        echo "✗ ERROR: $data_path - Check log: $log_file"
        error_count=$((error_count + 1))
        if [ -z "$failed_datasets" ]; then
            failed_datasets="$data_path"
        else
            failed_datasets="$failed_datasets $data_path"
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

echo ""
echo "All logs saved in: ./Results/emb_logs/"
echo "Finished processing all datasets!"