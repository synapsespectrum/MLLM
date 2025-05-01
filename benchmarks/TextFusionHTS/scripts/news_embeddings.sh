#!/bin/bash

# Run text embeddings script for News dataset
python emb_learning.py \
  --txtfile_path="$PWD/datasets/News/Facebook_Obama_txt_final.csv" \
  --tsfile_path="$PWD/datasets/News/Facebook_Obama_transpose_final.csv" \
  --dataset_type=news \
  --text_column=text \
  --id_column=id \
  --batch_size=64 \
  --max_length=512 \
  --output_prefix=news

# If you want to run only the alignment part (skipping model processing)
# Add --skip_model flag to the command above