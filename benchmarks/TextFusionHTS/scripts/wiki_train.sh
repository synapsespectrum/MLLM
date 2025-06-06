#!/bin/bash

# Activate Python environment (if applicable)
# source /path/to/your/venv/bin/activate

# Set CUDA device (if needed)
export CUDA_VISIBLE_DEVICES=0

# Define arguments
IS_TRAINING=1
MODEL_ID="wiki_experiment"
MODEL="TFHTS"
DATA="custom"
ROOT_PATH=$(pwd)
DATA_PATH="$ROOT_PATH/datasets/Wiki-People_en/"
OUTPUT_PATH="$ROOT_PATH/output/"
DATA_TS_FILENAME="train_1_people_en_filtered.csv"
DATA_TX_FILENAME="txt_avg_emb.npz"
SEQ_LEN=7
LABEL_LEN=0
BATCH_SIZE=256
LEARNING_RATE=0.001
TRAIN_EPOCHS=100
LOSS="MSE"
LR_ADJ="type2"
D_MODEL=128
N_HEADS=16
E_LAYERS=1
D_FF=64
DROPOUT=0.2
FUSION_HEAD=8
USE_GPU="--use_gpu"
USE_MULTI_GPU=""  # Leave empty if not using multi-GPU
DEVICES="0"

# Loop through PRED_LEN values
for PRED_LEN in 14 21 28 35
do
  echo "Running experiment with PRED_LEN=$PRED_LEN"
  python main_wiki.py \
    --is_training $IS_TRAINING \
    --model_id $MODEL_ID \
    --model $MODEL \
    --data $DATA \
    --root_path $ROOT_PATH \
    --data_path $DATA_PATH \
    --output_path $OUTPUT_PATH \
    --data_ts_filename $DATA_TS_FILENAME \
    --data_tx_filename $DATA_TX_FILENAME \
    --seq_len $SEQ_LEN \
    --label_len $LABEL_LEN \
    --pred_len $PRED_LEN \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --train_epochs $TRAIN_EPOCHS \
    --loss $LOSS \
    --lr_adj $LR_ADJ \
    --d_model $D_MODEL \
    --n_heads $N_HEADS \
    --e_layers $E_LAYERS \
    --d_ff $D_FF \
    --dropout $DROPOUT \
    --fusion_head $FUSION_HEAD \
    $USE_GPU \
    $USE_MULTI_GPU \
    --devices $DEVICES
done