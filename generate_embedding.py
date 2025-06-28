import torch
import os
import time
import h5py
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_provider.embedding_dataloader import Dataset_Custom
from llm.time_series_prompt_embedder import GenPromptEmb
from llm.text_embedder import TextEmbedder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="")
    parser.add_argument("--num_workers", type=int, default=min(10, os.cpu_count()))

    # Embedding parameters
    parser.add_argument('--embedding_mode', type=int, default=0,
                        help='0 for generating embeddings based on text data, '
                             '1 for generating embeddings based on time series data'
                             '2 for generating embeddings based on both text and time series data')
    parser.add_argument("--input_len", type=int, default=24)
    # parser.add_argument("--output_len", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=1)

    #  LLM parameters
    parser.add_argument('--llm_model', type=str, default='GPT2',
                        help='LLM model')  # LLAMA2, LLAMA3, GPT2, BERT, GPT2M, GPT2L, GPT2XL, Doc2Vec, ClosedLLM
    parser.add_argument('--llm_dim', type=int, default='768',
                        help='LLM model dimension')  # LLama7b:4096; GPT2-small:768; BERT-base:768
    parser.add_argument('--llm_layers', type=int, default=6)
    parser.add_argument('--type_tag', type=str, default="#F#")
    parser.add_argument('--text_len', type=int, default=3)
    parser.add_argument('--learning_rate2', type=float, default=1e-2, help='mlp learning rate')
    parser.add_argument('--learning_rate3', type=float, default=1e-3, help='proj learning rate')
    parser.add_argument('--prompt_weight', type=float, default=0.01,
                        help='prompt weight')  # please tune this hyperparameter for combining
    parser.add_argument('--pool_type', type=str, default='avg', help='pooling type')  # avg min max attention
    parser.add_argument('--date_name', type=str, default='end_date', help='matching date name in csv')  # mlp linear
    parser.add_argument('--addHisRate', type=float, default=0.5, help='add historical rate')
    parser.add_argument('--init_method', type=str, default='normal', help='init method of combined weight')
    parser.add_argument('--learning_rate_weight', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument('--save_name', type=str, default='result_longterm_forecast', help='save name')
    parser.add_argument('--use_fullmodel', type=int, default=0, help='use full model or just encoder')
    parser.add_argument('--use_closedllm', type=int, default=0, help='use closedllm or not')
    parser.add_argument('--huggingface_token', type=str, help='your token of huggingface; need for llama3')

    # Dataset parameters
    parser.add_argument("--data_path", type=str, default="Environment",
                        help="Dataset name, e.g., 'Farm', 'Health', 'Environment', etc.")
    parser.add_argument("--emb_saved_path", type=str, default="./Embeddings")
    parser.add_argument("--target", type=str, default="OT", help="Target variable for the dataset")
    parser.add_argument("--feature", type=str, default="M", help="Feature type for the dataset")

    args = parser.parse_args()

    if args.llm_model == "BERT" or args.llm_model == "GPT2":
        args.llm_dim = 768
    elif args.llm_model == "LLAMA2" or args.llm_model == "LLAMA3":
        args.llm_dim = 4096
    elif args.llm_model == "GPT2M":
        args.llm_dim = 1024
    elif args.llm_model == "GPT2L":
        args.llm_dim = 1280
    elif args.llm_model == "GPT2XL":
        args.llm_dim = 1600
    elif args.llm_model == "Doc2Vec":
        args.llm_dim = 64
    elif args.llm_model == "ClosedLLM":
        args.llm_model = "BERT"  # just for encoding
        args.llm_dim = 768
        args.use_closedllm = 1

    # print all the arguments in args with formatting
    print("Arguments: ")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("-" * 50)

    return args


def get_dataset(data_path, flag, input_len, target, feature):
    dataset_class = Dataset_Custom
    return dataset_class(flag=flag,
                         seq_len=input_len,
                         features='S',  # Feature type: M, S, MS, set to 'S' for univariate time series
                         target=target,
                         data_path=data_path)


def save_embeddings(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_set = get_dataset(args.data_path, 'train', args.input_len, args.target, args.feature)
    test_set = get_dataset(args.data_path, 'test', args.input_len, args.target, args.feature)
    val_set = get_dataset(args.data_path, 'val', args.input_len, args.target, args.feature)
    print(f"Train set length: {len(train_set)}")
    print(f"Test set length: {len(test_set)}")
    print(f"Validation set length: {len(val_set)}")
    data_loaders = {
        'train': DataLoader(train_set, batch_size=args.batch_size, shuffle=False, drop_last=False,
                            num_workers=args.num_workers),
        'test': DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False,
                           num_workers=args.num_workers),
        'val': DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False,
                          num_workers=args.num_workers)
    }
    if args.embedding_mode == 0:
        # Using text data to generate embeddings
        gen_prompt_emb = TextEmbedder(
            # Embedding text data into prompts for LLMs
            model_name=args.llm_model,
            llm_layers=args.llm_layers,
            device=device,
            hug_token=args.huggingface_token
        ).to(device)

        for flag, data_loader in data_loaders.items():
            print("=" * 50)
            print(f"Processing {flag} set with {len(data_loader)} batches")
            # save_path = f"./Embeddings_TimeCMA/{args.data_path}/{args.divide}/"
            save_path = f"{args.emb_saved_path}/text/{args.llm_model}/{args.data_path}/{flag}"
            os.makedirs(save_path, exist_ok=True)

            data_embedding = gen_prompt_emb(data_loader.dataset.text)
            # save the embeddings to h5 files
            file_path = f"{save_path}.h5"
            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('embeddings', data=data_embedding.cpu().numpy())
            print(f"Embeddings saved to {file_path}")

    else:
        gen_prompt_emb = GenPromptEmb(
            # Embedding Time series data into prompts for LLMs
            device=device,
            input_len=args.input_len,
            data_path=args.data_path,
            model_name=args.llm_model,
            d_model=args.llm_dim,
            layer=args.llm_layers
        ).to(device)
        for flag, data_loader in data_loaders.items():
            print("=" * 50)
            print(f"Processing {flag} set with {len(data_loader)} batches")
            # save_path = f"./Embeddings_TimeCMA/{args.data_path}/{args.divide}/"
            save_path = f"{args.emb_saved_path}/ts_txt/{args.llm_model}/{args.data_path}/{flag}/"
            os.makedirs(save_path, exist_ok=True)

            for i, (x, x_mark, seq_text) in tqdm(enumerate(data_loader),
                                                 total=len(data_loader),
                                                 desc="Generating embeddings"):

                if args.embedding_mode == 1:  # Using time series data to generate embeddings
                    embeddings = gen_prompt_emb.generate_embeddings(x.to(device), x_mark.to(device))
                else:  # Using both text and time series data to generate embeddings
                    embeddings = gen_prompt_emb.generate_embeddings(x.to(device), x_mark.to(device), seq_text)

                file_path = f"{save_path}{i}.h5"
                with h5py.File(file_path, 'w') as hf:
                    hf.create_dataset('embeddings', data=embeddings.cpu().numpy())


if __name__ == "__main__":
    args = parse_args()
    t1 = time.time()
    save_embeddings(args)
    t2 = time.time()
    print(f"Total time spent: {(t2 - t1) / 60:.4f} minutes")
