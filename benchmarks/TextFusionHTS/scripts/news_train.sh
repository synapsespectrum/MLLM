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
DATA_TS_FILENAME="Facebook_Obama_transpose_final.csv"
DATA_TX_FILENAME="Facebook_Obama_txt_final.csv"
SEQ_LEN=9
LABEL_LEN=1
PRED_LEN=1
BATCH_SIZE=256
LEARNING_RATE=0.001
TRAIN_EPOCHS=100

# Run the script
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