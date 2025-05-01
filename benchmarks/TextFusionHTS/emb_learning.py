import os
import pandas as pd
import argparse
import torch
import numpy as np
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def parser():
    parser = argparse.ArgumentParser(description='text embedding learning')
    parser.add_argument('--root_path', type=str, default=os.getcwd(), help='root path')
    parser.add_argument('--txtfile_path', type=str, default='', help='file path')
    parser.add_argument('--tsfile_path', type=str, default='', help='ts file path')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size for processing')
    parser.add_argument('--max_length', type=int, default=512, help='maximum token length')
    parser.add_argument('--dataset_type', type=str, default='news', choices=['news', 'wiki'],
                        help='dataset type: news or wiki')
    parser.add_argument('--text_column', type=str, default='text', help='column name containing the text data')
    parser.add_argument('--id_column', type=str, default='id', help='column name for identifying entries')
    parser.add_argument('--output_prefix', type=str, default='output', help='prefix for output files')
    parser.add_argument('--huggingface_token', type=str, default='', help='Hugging Face token')
    parser.add_argument('--model_id', type=str, default='meta-llama/Meta-Llama-3.1-8B', help='Model ID to use')
    parser.add_argument('--skip_model', action='store_true',
                        help='Skip model loading and processing, just align existing embeddings')
    return parser.parse_args()


def align_order_wiki(ts_file, df, output_prefix, part=''):
    """Alignment function for Wiki-People dataset"""
    df_ts = pd.read_csv(ts_file)
    print(f"TS file shape: {df_ts.shape}")

    list_column = []
    for col in df_ts.columns[1:]:
        list_column.append('_'.join(col.split('_')[:-3]))

    new_txt_emb_list = []
    for lc in list_column:
        emb = df[df['article'] == lc]['embs'].values
        # transfer string to list
        emb = eval(emb[0])[0]
        new_txt_emb_list.append(emb)

    np.savez(f'{output_prefix}_{part}_emb.npz', new_txt_emb_list)
    np.save(f'{output_prefix}_{part}_emb.h5', new_txt_emb_list)
    print(f"Saved {output_prefix}_{part}_emb files")


def align_order_news(ts_file, df, output_prefix, part=''):
    """Alignment function for News dataset"""
    df_ts = pd.read_csv(ts_file)
    print(f"TS file shape: {df_ts.shape}")

    # convert the column names to a list (int64)
    list_column = df_ts.columns[1:].astype(float).tolist()
    df[df.columns[0]].astype(float)  # Convert ID column to float for comparison

    new_txt_emb_list = []
    for lc in list_column:
        emb = df[df[df.columns[0]] == lc]['embs'].values
        # transfer string to list
        emb = eval(emb[0])[0]
        new_txt_emb_list.append(emb)

    np.savez(f'{output_prefix}_{part}_emb.npz', new_txt_emb_list)
    np.save(f'{output_prefix}_{part}_emb.h5', new_txt_emb_list)
    print(f"Saved output files: \n"
          f"{output_prefix}_{part}_emb.npz \n"
          f"{output_prefix}_{part}_emb.h5")


def batch_process_texts(texts, tokenizer, model, batch_size, max_length, device):
    """Process texts in batches to improve efficiency"""
    all_avg_embs, all_cls_embs, all_bos_embs, all_eos_embs = [], [], [], []

    # Move the model to the correct device
    model.to(device)

    # Create batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch_texts = texts[i:i + batch_size]

        # Tokenize batch
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            max_length=max_length,
            padding='max_length',
            truncation=True
        )

        # Move inputs to the same device as the model
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Process batch
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Extract embeddings
        last_hidden_states = outputs.hidden_states[-1]

        # Get different embedding types for each text in batch
        for j in range(len(batch_texts)):
            # Only consider non-padding tokens for this specific text
            attention_mask = inputs['attention_mask'][j]
            actual_tokens = attention_mask.sum().item()

            # Get embeddings
            text_embeddings = last_hidden_states[j, :actual_tokens, :]

            # Calculate different embedding types
            avg_emb = text_embeddings.mean(dim=0).cpu().numpy()[np.newaxis, :]
            cls_emb = last_hidden_states[j, 0, :].cpu().numpy()[np.newaxis, :]
            bos_emb = last_hidden_states[j, 1, :].cpu().numpy()[np.newaxis, :] if actual_tokens > 1 else cls_emb
            eos_emb = last_hidden_states[j, actual_tokens - 1, :].cpu().numpy()[np.newaxis,
                      :] if actual_tokens > 1 else cls_emb

            all_avg_embs.append(avg_emb)
            all_cls_embs.append(cls_emb)
            all_bos_embs.append(bos_emb)
            all_eos_embs.append(eos_emb)

        # Clear CUDA cache after each batch to prevent memory issues
        torch.cuda.empty_cache()

    return all_avg_embs, all_cls_embs, all_bos_embs, all_eos_embs


def main():
    args = parser()

    # Setup output prefix if not provided
    if args.output_prefix == 'output':
        args.output_prefix = 'wiki' if args.dataset_type == 'wiki' else 'news'

    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Using CUDA:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU instead.")

    # Select the appropriate alignment function based on dataset type
    align_function = align_order_wiki if args.dataset_type == 'wiki' else align_order_news

    # If not skipping model loading and processing
    if not args.skip_model:
        # Login to Hugging Face if token is provided
        if args.huggingface_token:
            print(f"Logging in to Hugging Face...")
            login(args.huggingface_token)

        print(f"Loading model: {args.model_id}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        tokenizer.pad_token = tokenizer.eos_token
        llama = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16,  # Use half precision
            device_map="auto",  # Automatically manage model placement
            low_cpu_mem_usage=True
        )

        # Load the data
        print("Loading data...")
        df = pd.read_csv(args.txtfile_path)
        text = list(df[args.text_column].values)
        print(f"Total texts to process: {len(text)}")

        # Process all texts in batches
        print("Starting batch processing...")
        avg_embs, cls_embs, bos_embs, eos_embs = batch_process_texts(
            text, tokenizer, llama, args.batch_size, args.max_length, device
        )

        # Create copies only when needed
        print("Saving embeddings...")
        df1 = df.copy()
        df2 = df.copy()
        df3 = df.copy()

        # Save the embeddings
        df['embs'] = [str(x.tolist()) for x in avg_embs]
        df1['embs'] = [str(x.tolist()) for x in cls_embs]
        df2['embs'] = [str(x.tolist()) for x in bos_embs]
        df3['embs'] = [str(x.tolist()) for x in eos_embs]

        # Save the embeddings to csv in case of memory issues
        output_avg = f"{args.output_prefix}_avg_embs.csv"
        output_cls = f"{args.output_prefix}_cls_embs.csv"
        output_bos = f"{args.output_prefix}_bos_embs.csv"
        output_eos = f"{args.output_prefix}_eos_embs.csv"

        df.to_csv(output_avg, index=False)
        df1.to_csv(output_cls, index=False)
        df2.to_csv(output_bos, index=False)
        df3.to_csv(output_eos, index=False)

        print(
            f"All embeddings saved: "
            f"{args.output_prefix}_avg_embs.csv, "
            f"{args.output_prefix}_cls_embs.csv, "
            f"{args.output_prefix}_bos_embs.csv, "
            f"{args.output_prefix}_eos_embs.csv")

    # Align embeddings
    print("Aligning embedding order...")

    # Read from files (these should exist from earlier in the script or from a previous run)
    df = pd.read_csv(f"{args.output_prefix}_avg_embs.csv")
    print(f"Average embeddings shape: {df.shape}")
    align_function(args.tsfile_path, df, args.output_prefix, 'avg')

    df1 = pd.read_csv(f"{args.output_prefix}_cls_embs.csv")
    print(f"CLS embeddings shape: {df1.shape}")
    align_function(args.tsfile_path, df1, args.output_prefix, 'cls')

    df2 = pd.read_csv(f"{args.output_prefix}_bos_embs.csv")
    align_function(args.tsfile_path, df2, args.output_prefix, 'bos')

    df3 = pd.read_csv(f"{args.output_prefix}_eos_embs.csv")
    align_function(args.tsfile_path, df3, args.output_prefix, 'eos')

    print('Processing completed!')


if __name__ == "__main__":
    main()
