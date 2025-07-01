#!/bin/bash

# Mapping datasets to their frequency
declare -A dataset_freq=(
    [Environment]="daily"
    [Energy]="weekly"
    [Health]="weekly"
    [Economy]="monthly"
    [Agriculture]="monthly"
    [Climate]="monthly"
    [Security]="monthly"
    [SocialGood]="monthly"
    [Traffic]="monthly"
)

# Parameter sets for each frequency
declare -A seq_len_map=( [daily]=96 [weekly]=36 [monthly]=8 )
declare -A label_len_map=( [daily]=48 [weekly]=18 [monthly]=4 )
declare -A pred_len_choices
pred_len_choices[daily]="48 96 192 336"
pred_len_choices[weekly]="12 24 36 48"
pred_len_choices[monthly]="6 8 10 12"

feature="S"
llm_model="GPT2"
batch_size=1
device="cuda"
datasets="Agriculture Climate Economy Energy Environment Health Security SocialGood Traffic"

output_dir="./Embeddings_TimeCMA"
log_dir="./Results/emb_logs"
mkdir -p "$log_dir"
mkdir -p "$output_dir"

echo "Starting embedding generation for all datasets..."

dataset_count=0
for dataset in $datasets; do
    dataset_count=$((dataset_count + 1))
done

echo "Found $dataset_count datasets to process"

success_count=0
error_count=0
failed_datasets=""

for data_path in $datasets; do
    freq="${dataset_freq[$data_path]}"
    seq_len="${seq_len_map[$freq]}"
    label_len="${label_len_map[$freq]}"
    # Pick the first pred_len for each dataset, or loop for all if needed
    for pred_len in ${pred_len_choices[$freq]}; do
        target="OT"
        log_file="$log_dir/${data_path}_${pred_len}.log"
        echo ""
        echo "=================================================="
        echo "Processing dataset: $data_path (freq: $freq, pred_len: $pred_len)"
        echo "Target: $target"
        echo "=================================================="
        python generate_embedding.py \
            --data_path "$data_path" \
            --input_len "$seq_len" \
            --label_len "$label_len" \
            --pred_len "$pred_len" \
            --target "$target" \
            --feature "$feature" \
            --llm_model "$llm_model" \
            --batch_size "$batch_size" \
            --device "$device" \
            --emb_saved_path "$output_dir" \
            > "$log_file" 2>&1
        if [ $? -eq 0 ]; then
            echo "✓ SUCCESS: $data_path (pred_len: $pred_len) - Log: $log_file"
            success_count=$((success_count + 1))
        else
            echo "✗ ERROR: $data_path (pred_len: $pred_len) - Check log: $log_file"
            error_count=$((error_count + 1))
            if [ -z "$failed_datasets" ]; then
                failed_datasets="$data_path($pred_len)"
            else
                failed_datasets="$failed_datasets $data_path($pred_len)"
            fi
        fi
    done
done

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
echo "All logs saved in: $log_dir"
echo "Embeddings saved in: $output_dir"
echo "Finished processing all datasets!"