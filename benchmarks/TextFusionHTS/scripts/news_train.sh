#!/bin/bash

# Activate Python environment (if applicable)
# source /path/to/your/venv/bin/activate

# Set CUDA device (if needed)
export CUDA_VISIBLE_DEVICES=0

# Define arguments
IS_TRAINING=1
MODEL_ID="news_1"
MODEL="TFHTS"
DATA="custom"
ROOT_PATH=$(pwd)
DATA_PATH="$ROOT_PATH/datasets/News/"
OUTPUT_PATH="$ROOT_PATH/output/"
DATA_TS_FILENAME="Facebook_Obama_transpose_wDate.csv"
DATA_TX_FILENAME="news_avg_emb.npz"
SEQ_LEN=9
LABEL_LEN=1
BATCH_SIZE=256
LEARNING_RATE=0.001
TRAIN_EPOCHS=100

# Run the script
for PRED_LEN in 1 3 9 12 15
do
    echo "Running experiment with PRED_LEN=$PRED_LEN"
    python main_news.py \
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
      --train_epochs $TRAIN_EPOCHS
done