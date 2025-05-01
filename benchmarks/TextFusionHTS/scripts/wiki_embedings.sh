#!/bin/bash

# Run text embeddings script for Wiki-People dataset
python text_embedding.py \
  --txtfile_path="$PWD/datasets/Wiki-People_en/train_1_people_summaries_en_filtered_sameshape.csv" \
  --tsfile_path="$PWD/datasets/Wiki-People_en/train_1_people_en_filtered.csv" \
  --dataset_type=wiki \
  --text_column=summary \
  --id_column=article \
  --batch_size=64 \
  --max_length=512 \
  --output_prefix=wiki

# If you want to run only the alignment part (skipping model processing)
# Add --skip_model flag to the command above