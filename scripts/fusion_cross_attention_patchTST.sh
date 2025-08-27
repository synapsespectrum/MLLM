
#!/bin/bash

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

all_models=("PatchTST")
start_index=$1
end_index=$2

# Define datasets with their specific configurations
declare -A dataset_configs

# Environment datasets: input_len 84,24 → pred_lengths 24,96,192,336
dataset_configs["Environment"]="84,24:24,96,192,336"

# Energy, Health datasets: input_len 36,24 → pred_lengths 12,24,36,48
dataset_configs["Energy"]="36,24:12,24,36,48"
dataset_configs["Health"]="36,24:12,24,36,48"

# Agriculture, Climate, Economy, Security, Social Good, Traffic: input_len 8,24 → pred_lengths 6,8,10,12
dataset_configs["Agriculture"]="8,24:6,8,10,12"
dataset_configs["Climate"]="8,24:6,8,10,12"
dataset_configs["Economy"]="8,24:6,8,10,12"
dataset_configs["Security"]="8,24:6,8,10,12"
dataset_configs["SocialGood"]="8,24:6,8,10,12"
dataset_configs["Traffic"]="8,24:6,8,10,12"

# Function to get patch_len based on input_len (lookback)
get_patch_len() {
    local input_len=$1
    case $input_len in
        84) echo "16" ;;
        36) echo "12" ;;
        8)  echo "2" ;;
        24) echo "3" ;;
        *)  echo "8" ;; # default value
    esac
}

# Define datasets and paths
root_path="/home/andrew/github/pb/TaTS/data"
datasets=("Agriculture" "Climate" "Economy" "Energy" "Environment" "Health" "Security" "SocialGood" "Traffic")

seeds=(2025)
llm_model="GPT2"
use_fullmodel=0
prompt_weight=0.5
embedding_path="/home/andrew/github/MLLM/Embeddings/text"
experiment_name="fusion_cross_attention"

# Create folder for saving results
mkdir -p "./logs/${experiment_name}"

echo "Starting experiments with dataset-specific configurations..."

for seed in "${seeds[@]}"
do
  for model_name in "${all_models[@]}"
  do
    for dataset in "${datasets[@]}"
    do
      # Get configuration for this dataset
      config=${dataset_configs[$dataset]}
      input_lens=$(echo $config | cut -d':' -f1)
      pred_lens=$(echo $config | cut -d':' -f2)

      # Convert comma-separated values to arrays
      IFS=',' read -ra input_len_array <<< "$input_lens"
      IFS=',' read -ra pred_len_array <<< "$pred_lens"

      # Create directory for the dataset if it doesn't exist
      mkdir -p "./logs/${experiment_name}/${dataset}"

      # Run experiments for each input_len and pred_len combination
      for input_len in "${input_len_array[@]}"
      do
        # Get patch_len based on input_len
        patch_len=$(get_patch_len $input_len)

        for pred_len in "${pred_len_array[@]}"
        do
          data_path="${dataset}.csv"
          model_id=$(basename ${root_path})

          # Adjust batch size for larger datasets
          if [[ "$dataset" == "Security" ]]; then
            batch_size=16
          else
            batch_size=32
          fi

          echo "Running experiment: Dataset=$dataset, Model=$model_name, Input_len=$input_len, Pred_len=$pred_len, Patch_len=$patch_len"
          echo "Log saved to ./logs/${experiment_name}/${dataset}/terminal_il${input_len}_pl${pred_len}_patch${patch_len}.log"

          python -u run.py \
            --experiment_name $experiment_name \
            --task_name long_term_forecast \
            --is_training 1 \
            --tracking_mlflow 1 \
            --batch_size $batch_size \
            --root_path $root_path \
            --data_path $data_path \
            --model_id "${model_id}_${dataset}_il${input_len}_pl${pred_len}_patch${patch_len}" \
            --model $model_name \
            --data $dataset \
            --features S \
            --seq_len $input_len \
            --label_len 0 \
            --pred_len $pred_len \
            --patch_len $patch_len \
            --embedding_path $embedding_path \
            --des "Exp_il${input_len}_pl${pred_len}_patch${patch_len}" \
            --seed $seed \
            --type_tag "#F#" \
            --text_len 4 \
            --prompt_weight $prompt_weight \
            --pool_type "avg" \
            --save_name "results_il${input_len}_pl${pred_len}.txt" \
            --llm_model $llm_model \
            --huggingface_token "${HUGGINGFACE_TOKEN}" \
            --train_epochs 100 \
            --patience 20 | ts '[%Y-%m-%d %H:%M:%S]' >> "./logs/${experiment_name}/${dataset}/terminal_il${input_len}_pl${pred_len}_patch${patch_len}.log" 2>&1

          # Check if experiment completed successfully
          if [ $? -eq 0 ]; then
            echo "✓ Completed: $dataset - $model_name (input_len=$input_len, pred_len=$pred_len, patch_len=$patch_len)"
          else
            echo "✗ Failed: $dataset - $model_name (input_len=$input_len, pred_len=$pred_len, patch_len=$patch_len)"
          fi
        done
      done
    done
  done
done

echo ""
echo "=================================================="
echo "All fusion cross attention experiments completed!"
echo "=================================================="
echo "Results are saved in ./logs/${experiment_name}/"
echo ""
echo "Summary of experiments run:"
for dataset in "${datasets[@]}"
do
  config=${dataset_configs[$dataset]}
  input_lens=$(echo $config | cut -d':' -f1)
  pred_lens=$(echo $config | cut -d':' -f2)
  echo "- $dataset: input_len=[$input_lens], pred_len=[$pred_lens]"

  # Show patch_len mapping for this dataset
  IFS=',' read -ra input_len_array <<< "$input_lens"
  echo "  Patch lengths:"
  for input_len in "${input_len_array[@]}"
  do
    patch_len=$(get_patch_len $input_len)
    echo "    input_len=$input_len → patch_len=$patch_len"
  done
done