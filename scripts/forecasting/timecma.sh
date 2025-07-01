#!/bin/bash

echo "==================================================="
echo "RUNNING ALL DATASET EXPERIMENTS"
echo "==================================================="

export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# Dataset to frequency mapping
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
declare -A pred_len_map
pred_len_map[daily]="48 96 192 336"
pred_len_map[weekly]="12 24 36 48"
pred_len_map[monthly]="6 8 10 12"

# Default parameters
learning_rate=1e-4
channel=64
e_layer=1
d_layer=2
dropout_n=0.7
seed=2025

datasets="Health Climate Economy Energy Environment Agriculture Security SocialGood Traffic"
embedding_dir="./data/embeddings"
results_dir="./Results"

run_dataset_experiment() {
    data_path=$1
    seq_len=$2
    pred_len=$3
    batch_size=$4

    echo "---------------------------------------------------"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: $data_path (seq_len: $seq_len, pred_len: $pred_len, batch_size: $batch_size)"
    echo "---------------------------------------------------"

    log_path="./Results/${data_path}/"
    mkdir -p $log_path
    log_file="${log_path}i${seq_len}_o${pred_len}_lr${learning_rate}_c${channel}_el${e_layer}_dl${d_layer}_dn${dropout_n}_bs${batch_size}.log"
    embedding_dir="${embedding_dir}/${data_path}/${seq_len}/"

    python train.py \
      --data_path $data_path \
      --embedding_dir $embedding_dir \
      --batch_size $batch_size \
      --num_nodes 1 \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --epochs 100 \
      --seed $seed \
      --channel $channel \
      --learning_rate $learning_rate \
      --dropout_n $dropout_n \
      --e_layer $e_layer \
      --d_layer $d_layer \
      --target "OT" > $log_file 2>&1

    if [ $? -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Completed: $data_path"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ Failed: $data_path"
        echo "Check log file: $log_file for details"
    fi
}

echo "Configuration: Per-dataset/frequency"
echo ""

mkdir -p $results_dir

current=0
total=9

for dataset in $datasets; do
    freq="${dataset_freq[$dataset]}"
    seq_len="${seq_len_map[$freq]}"
    # Use the first pred_len for each dataset (customize as needed)
    pred_len=$(echo ${pred_len_map[$freq]} | awk '{print $1}')
    # Custom batch size for Security
    if [ "$dataset" = "Security" ]; then
        batch_size=16
    else
        batch_size=32
    fi
    current=$((current+1))
    echo "Progress: $current/$total"
    run_dataset_experiment "$dataset" "$seq_len" "$pred_len" "$batch_size"
done

echo ""
echo "==================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🎉 ALL EXPERIMENTS COMPLETED!"
echo "==================================================="
echo "Results saved in $results_dir directory"