#!/bin/bash

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi


all_models=("iTransformer" )
start_index=$1
end_index=$2
# models=("${all_models[@]:$start_index:$end_index-$start_index+1}")
# Expanded to include all datasets
root_paths=(
  "/home/andrew/github/pb/TaTS/data"
  "/home/andrew/github/pb/TaTS/data"
  "/home/andrew/github/pb/TaTS/data"
  "/home/andrew/github/pb/TaTS/data"
  "/home/andrew/github/pb/TaTS/data"
  "/home/andrew/github/pb/TaTS/data"
  "/home/andrew/github/pb/TaTS/data"
  "/home/andrew/github/pb/TaTS/data"
  "/home/andrew/github/pb/TaTS/data"
)

data_paths=(
    "Agriculture.csv"
    "Climate.csv"
    "Economy.csv"
    "Energy.csv"
    "Environment.csv"
    "Health.csv"
    "Security.csv"
    "SocialGood.csv"
    "Traffic.csv"
)

input_len=24
pred_lengths=(12 )
seeds=(2025)
llm_model="GPT2"
use_fullmodel=0
prompt_weight=0.5
length=${#root_paths[@]}
embedding_path="/home/andrew/github/MLLM/Embeddings/text"
experiment_name="fusion_cross_attention"

# create folder for saving results
mkdir -p "./logs/${experiment_name}"

for seed in "${seeds[@]}"
do
  for model_name in "${all_models[@]}"
  do
    for ((i=0; i<$length; i++))
    do
      for pred_len in "${pred_lengths[@]}"
       do
        root_path=${root_paths[$i]}
        data_path=${data_paths[$i]}
        model_id=$(basename ${root_path})
        dataset=$(basename ${data_path} .csv)  # Extract dataset name from data_path
        # Create a directory for the dataset if it doesn't exist
        mkdir -p "./logs/${experiment_name}/${dataset}"
        # if SocialGood.csv, use batch size 16
        if [[ "$data_path" == *"SocialGood.csv"* ]]; then
          batch_size=16
        else
          batch_size=32
        fi

        echo "Running model $model_name with root $root_path, data $data_path, pred_len $pred_len, and prompt_weight $prompt_weight"
        echo "Log saved to ./logs/${experiment_name}/${dataset}/terminal.log" # training logs
        python -u run.py \
        --experiment_name $experiment_name \
        --task_name long_term_forecast \
        --is_training 1 \
        --tracking_mlflow 1 \
        --batch_size $batch_size \
        --root_path $root_path \
        --data_path $data_path \
         --model_id $model_id \
        --model $model_name \
        --data $dataset \
        --features S \
        --seq_len $input_len \
        --label_len 0 \
        --pred_len $pred_len \
        --embedding_path $embedding_path \
        --des 'Exp' \
        --seed $seed \
        --type_tag "#F#" \
        --text_len 4 \
        --prompt_weight $prompt_weight \
        --pool_type "avg" \
        --save_name results.txt \
        --llm_model $llm_model \
        --huggingface_token "${HUGGINGFACE_TOKEN}" \
        --train_epochs 50 \
        --patience 20 | ts '[%Y-%m-%d %H:%M:%S]' >> "./logs/${experiment_name}/${dataset}/terminal.log" 2>&1
      done
    done
  done
done