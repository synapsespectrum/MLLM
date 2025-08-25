#!/bin/bash

# Configuration parameters
feature="S"  # M: multivariate, S: univariate, MS: multivariate forecasting univariate
llm_model="GPT2"
batch_size=1
device="cuda"

# Define datasets with their corresponding input lengths (multiple lengths per dataset)
declare -A dataset_input_lens
dataset_input_lens["Environment"]="84 24"
dataset_input_lens["Energy"]="36 24"
dataset_input_lens["Health"]="36 24"
dataset_input_lens["Agriculture"]="8 24"
dataset_input_lens["Climate"]="8 24"
dataset_input_lens["Economy"]="8 24"
dataset_input_lens["Security"]="8 24"
dataset_input_lens["SocialGood"]="8 24"
dataset_input_lens["Traffic"]="8 24"

# Create log directories if they don't exist
mkdir -p ./log-emb/text_emb_logs/${llm_model}/
mkdir -p ./Embeddings/

echo "Starting embedding generation for all datasets with multiple input lengths..."

# Count total combinations
total_combinations=0
for data_path in "${!dataset_input_lens[@]}"; do
    input_lens_array=(${dataset_input_lens[$data_path]})
    total_combinations=$((total_combinations + ${#input_lens_array[@]}))
done

echo "Found $total_combinations dataset-input_length combinations to process"

# Track success/failure
success_count=0
error_count=0
failed_datasets=""
successful_configs=""

# Process each dataset with all its input lengths
for data_path in "${!dataset_input_lens[@]}"; do
    # Convert string to array
    input_lens_array=(${dataset_input_lens[$data_path]})

    echo ""
    echo "🔄 Processing dataset: $data_path"
    echo "   Input lengths to process: ${input_lens_array[*]}"
    echo "   Total variations: ${#input_lens_array[@]}"

    # Process each input length for this dataset
    for input_len in "${input_lens_array[@]}"; do
        # Default target (you can customize this per dataset if needed)
        target="OT"

        echo ""
        echo "=================================================="
        echo "Processing: $data_path with input_len=$input_len"
        echo "Target: $target"
        echo "=================================================="

        log_file="./log-emb/text_emb_logs/${llm_model}/${data_path}_len${input_len}.log"

        # Expected output path
        expected_output="./Embeddings/text/${llm_model}/$data_path/$input_len/"
        echo "Expected output: $expected_output"

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
        if [ $? -eq 0 ] && [ -d "$expected_output" ]; then
            echo "✅ SUCCESS: $data_path (input_len=$input_len) - Log: $log_file"
            echo "   Embeddings saved in: $expected_output"
            success_count=$((success_count + 1))

            # Add to successful configs
            if [ -z "$successful_configs" ]; then
                successful_configs="$data_path($input_len)"
            else
                successful_configs="$successful_configs $data_path($input_len)"
            fi

            # Show created files
            echo "   Created files:"
            for file in "$expected_output"*.h5; do
                if [ -f "$file" ]; then
                    echo "     - $(basename "$file")"
                fi
            done
        else
            echo "❌ ERROR: $data_path (input_len=$input_len) - Check log: $log_file"
            error_count=$((error_count + 1))
            if [ -z "$failed_datasets" ]; then
                failed_datasets="$data_path($input_len)"
            else
                failed_datasets="$failed_datasets $data_path($input_len)"
            fi
        fi
    done

    echo ""
    echo "📊 Completed processing $data_path with ${#input_lens_array[@]} input lengths"
done

# Final Summary
echo ""
echo "=================================================="
echo "🏁 FINAL PROCESSING SUMMARY"
echo "=================================================="
echo "Total combinations processed: $total_combinations"
echo "Successful: $success_count"
echo "Failed: $error_count"
echo "Success rate: $(awk "BEGIN {printf \"%.1f\", $success_count/$total_combinations*100}")%"

if [ $error_count -gt 0 ]; then
    echo ""
    echo "❌ Failed combinations:"
    for dataset in $failed_datasets; do
        echo "  - $dataset"
    done
fi

# Show successful configurations grouped by dataset
if [ $success_count -gt 0 ]; then
    echo ""
    echo "✅ Successfully processed configurations:"
    for data_path in "${!dataset_input_lens[@]}"; do
        input_lens_array=(${dataset_input_lens[$data_path]})
        echo "📁 $data_path:"
        for input_len in "${input_lens_array[@]}"; do
            embedding_path="./Embeddings/text/${llm_model}/$data_path/$input_len/"
            if [ -d "$embedding_path" ]; then
                h5_count=$(find "$embedding_path" -name "*.h5" 2>/dev/null | wc -l)
                echo "   ✓ input_len=$input_len ($h5_count files) - $embedding_path"
            else
                echo "   ❌ input_len=$input_len - Missing"
            fi
        done
        echo ""
    done
fi

echo ""
echo "📂 Directory structure created:"
echo "  ./Embeddings/text/${llm_model}/"
for data_path in "${!dataset_input_lens[@]}"; do
    input_lens_array=(${dataset_input_lens[$data_path]})
    echo "  ├── $data_path/"
    for input_len in "${input_lens_array[@]}"; do
        if [ -d "./Embeddings/text/${llm_model}/$data_path/$input_len/" ]; then
            echo "  │   ├── $input_len/ ✅"
        else
            echo "  │   ├── $input_len/ ❌"
        fi
    done
done

echo ""
echo "📝 All logs saved in: ./log-emb/text_emb_logs/${llm_model}/"
echo "💾 Base embeddings path: ./Embeddings/text/${llm_model}/"
echo ""
echo "🎉 Finished processing all datasets with their multiple input lengths!"