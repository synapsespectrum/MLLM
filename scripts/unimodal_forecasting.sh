#!/bin/bash

# Define models to run
all_models=("Autoformer" "Transformer" "Informer" "Crossformer" "iTransformer" "PatchTST")

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

# Define datasets and paths
root_path="./data"
datasets=("Agriculture" "Climate" "Economy" "Energy" "Environment" "Health" "Security" "SocialGood" "Traffic")

# Seeds for reproducibility
seeds=(2025)

# Create folder for saving results
mkdir -p "./benchmarks/logs"

echo "Starting experiments with custom configurations for each dataset..."

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
      mkdir -p "./benchmarks/logs/${dataset}"

      # Run experiments for each input_len and pred_len combination
      for input_len in "${input_len_array[@]}"
      do
        for pred_len in "${pred_len_array[@]}"
        do
          data_path="${dataset}.csv"

          # Adjust batch size for larger datasets
          if [[ "$dataset" == "Security" ]]; then
            batch_size=16
          else
            batch_size=32
          fi

          label_len=$((input_len / 2))

          echo "Running experiment: Dataset=$dataset, Model=$model_name, Input_len=$input_len, Pred_len=$pred_len, Seed=$seed"
          echo "Log saved to ./benchmarks/logs/${dataset}/terminal_${model_name}_il${input_len}_pl${pred_len}.log"

          # Run the experiment
          python -u run_unimodal.py \
            --is_training 1 \
            --batch_size $batch_size \
            --root_path $root_path \
            --data_path $data_path \
            --model_id "${dataset}_${input_len}_${pred_len}" \
            --model $model_name \
            --data $dataset \
            --features S \
            --seq_len $input_len \
            --label_len $label_len \
            --pred_len $pred_len \
            --des "Exp_il${input_len}_pl${pred_len}" \
            --seed $seed \
            --train_epochs 50 \
            --patience 20 >> "./benchmarks/logs/${dataset}/terminal_${model_name}_il${input_len}_pl${pred_len}.log" 2>&1

          # Check if experiment completed successfully
          if [ $? -eq 0 ]; then
            echo "✓ Completed: $dataset - $model_name (input_len=$input_len, pred_len=$pred_len)"
          else
            echo "✗ Failed: $dataset - $model_name (input_len=$input_len, pred_len=$pred_len)"
          fi
        done
      done
    done
  done
done

echo ""
echo "=================================================="
echo "All experiments completed!"
echo "=================================================="
echo "Results are saved in ./benchmarks/logs/"
echo ""
echo "Summary of experiments run:"
for dataset in "${datasets[@]}"
do
  config=${dataset_configs[$dataset]}
  input_lens=$(echo $config | cut -d':' -f1)
  pred_lens=$(echo $config | cut -d':' -f2)
  echo "- $dataset: input_len=[$input_lens], pred_len=[$pred_lens]"
done