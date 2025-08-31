from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from transformers import AutoTokenizer
import os
import time
import datetime
import warnings
import numpy as np
from model import TSLLMFusionModel


def norm(input_emb):
    input_emb = input_emb - input_emb.mean(1, keepdim=True).detach()
    input_emb = input_emb / torch.sqrt(
        torch.var(input_emb, dim=1, keepdim=True, unbiased=False) + 1e-5)

    return input_emb


class MLP(nn.Module):
    def __init__(self, layer_sizes, dropout_rate=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x


warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        configs = args
        self.text_path = configs.text_path
        self.prompt_weight = configs.prompt_weight
        self.attribute = "final_sum"
        self.type_tag = configs.type_tag
        self.text_len = configs.text_len
        self.d_llm = configs.llm_dim
        self.pred_len = configs.pred_len
        self.text_embedding_dim = configs.text_emb
        self.pool_type = configs.pool_type
        self.use_fullmodel = configs.use_fullmodel
        self.hug_token = configs.huggingface_token
        mlp_sizes = [self.d_llm, int(self.d_llm / 8), self.text_embedding_dim]
        self.Doc2Vec = False
        if mlp_sizes is not None:
            self.mlp = MLP(mlp_sizes, dropout_rate=0.3)
        else:
            self.mlp = None
        mlp_sizes2 = [self.text_embedding_dim + self.args.pred_len, self.args.pred_len]
        if mlp_sizes2 is not None:
            self.mlp_proj = MLP(mlp_sizes2, dropout_rate=0.3)

        if configs.llm_model == 'LLAMA2':
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'LLAMA3':
            # Automatically load the configuration, model, and tokenizer for LLaMA-3-8B
            llama3_path = "meta-llama/Meta-Llama-3-8B-Instruct"
            cache_path = "./"

            # Load the configuration with custom adjustments
            self.config = LlamaConfig.from_pretrained(llama3_path, token=self.hug_token, cache_dir=cache_path)

            self.config.num_hidden_layers = configs.llm_layers
            self.config.output_attentions = True
            self.config.output_hidden_states = True

            self.llm_model = LlamaModel.from_pretrained(
                llama3_path,
                config=self.config,
                token=self.hug_token, cache_dir=cache_path
            )
            self.tokenizer = AutoTokenizer.from_pretrained(llama3_path, use_auth_token=self.hug_token,
                                                           cache_dir=cache_path)
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2M':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2-medium')

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2-medium',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2-medium',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2-medium',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2-medium',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2L':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2-large')

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2-large',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2-large',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2-large',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2-large',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2XL':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2-xl')

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2-xl',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2-xl',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2-xl',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2-xl',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )

        else:
            raise Exception('LLM model is not defined')

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False
        self.llm_model = self.llm_model.to(self.device)
        # print the model architecture
        print("LLM Model Architecture:")
        print(self.llm_model)  # Assuming the LLM model is stored as 'llm_model' attribute

        if args.init_method == 'uniform':
            self.weight1 = nn.Embedding(1, self.args.pred_len)
            self.weight2 = nn.Embedding(1, self.args.pred_len)
            nn.init.uniform_(self.weight1.weight)
            nn.init.uniform_(self.weight2.weight)
            self.weight1.weight.requires_grad = True
            self.weight2.weight.requires_grad = True
        elif args.init_method == 'normal':
            self.weight1 = nn.Embedding(1, self.args.pred_len)
            self.weight2 = nn.Embedding(1, self.args.pred_len)
            nn.init.normal_(self.weight1.weight)
            nn.init.normal_(self.weight2.weight)
            self.weight1.weight.requires_grad = True
            self.weight2.weight.requires_grad = True
        else:
            raise ValueError('Unsupported initialization method')

        # Build the main model
        self.model = TSLLMFusionModel(configs=self.args).to(self.device)
        print("Model Architecture:")
        print(self.model)

        # Add a projection layer to map text embeddings to output dimension
        self.text_to_output_proj = nn.Linear(self.text_embedding_dim, self.args.c_out).to(self.device)

        self.learning_rate2 = 1e-2
        self.learning_rate3 = 1e-3

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_optimizer_mlp(self):
        model_optim = optim.Adam(self.mlp.parameters(), lr=self.args.learning_rate2)
        return model_optim

    def _select_optimizer_proj(self):
        model_optim = optim.Adam(self.mlp_proj.parameters(), lr=self.args.learning_rate3)
        return model_optim

    def _select_optimizer_weight(self):
        model_optim = optim.Adam([{'params': self.weight1.parameters()},
                                  {'params': self.weight2.parameters()}], lr=self.args.learning_rate_weight)
        return model_optim

    def _select_optimizer_cross(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate2)
        return model_optim

    def _select_optimizer_text_proj(self):
        model_optim = optim.Adam(self.text_to_output_proj.parameters(), lr=self.args.learning_rate3)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                prior_y = torch.from_numpy(vali_data.get_prior_y(index)).float().to(self.device)

                prompt_emb = vali_data.get_text_embeddings(index).float().to(self.device)
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                outputs, _ = self.model(batch_x, batch_x_mark,
                                        prompt_emb,
                                        dec_inp, batch_y_mark,
                                        prior_y)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            # Start epoch tracking
            self.metrics_tracker.start_epoch(epoch)

            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                # 0523
                prior_y = torch.from_numpy(train_data.get_prior_y(index)).float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                prompt_emb = train_data.get_text_embeddings(index).float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # TS-LLM Fusion Model Forward Pass
                outputs, _ = self.model(batch_x, batch_x_mark,
                                        prompt_emb,
                                        dec_inp, batch_y_mark,
                                        prior_y)

                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                # Log iteration metrics
                self.metrics_tracker.log_iteration_metrics({
                    'train_loss': loss.item(),
                    'learning_rate': model_optim.param_groups[0]['lr']
                })

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            self.metrics_tracker.log_epoch_metrics({
                'train_loss': train_loss,
                'validation_loss': vali_loss,
                'test_loss': test_loss
            })

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, os.path.join(self.args.output_path, self.args.setting))
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = self.args.output_path + '/' + self.args.setting + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))  # for testing preparation

        return self.model

    def analyze_train_attention(self, batch_index=0):
        """
        Sau khi train xong, load best model và analyze attention patterns trên train data
        Args:
            setting: Model setting string
            batch_index: Index của batch muốn analyze (0-based)
            num_batches: Số lượng batches muốn analyze
        """
        print(f"🔍 Starting Getting Train Attention Analysis...")
        print(f"📊 Analyzing batch {batch_index} from train dataset with batch-size is {self.args.batch_size}")

        # # Load best model if weight is not provided
        # if not self.args.checkpoints:
        #     if not os.path.exists(self.args.checkpoints):
        #         print(f"❌ Model checkpoint not found at: {self.args.checkpoints}")
        #         return
        #     print(f"📥 Loading best model from: {self.args.checkpoints}")
        #     self.model.load_state_dict(torch.load(self.args.checkpoints))
        self.model.eval()

        # Get train data loader
        train_data, train_loader = self._get_data(flag='train')

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(train_loader):
                # Skip to desired batch index
                if i < batch_index:
                    continue

                # Move data to device
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                prompt_emb = train_data.get_text_embeddings(index).float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                _, attention_weights = self.model(batch_x, batch_x_mark,
                                                  prompt_emb,
                                                  dec_inp, batch_y_mark)

                # Log attention maps
                self.metrics_tracker.log_attention_maps(
                    attention_weights=attention_weights,
                    batch_idx=i,
                    sample_indices=index,
                    prefix_name='Train'
                )
                break

    def test(self, test=0):
        if test:
            print('Loading model from checkpoint: ', self.args.checkpoints)
            self.model.load_state_dict(torch.load(self.args.checkpoints))

        preds = []
        trues = []
        # # extract the cross-attention map from a trained dataset
        # self.analyze_train_attention()

        self.model.eval()

        print('Start testing phase...')
        test_data, test_loader = self._get_data(flag='test')

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                prior_y = torch.from_numpy(test_data.get_prior_y(index)).float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                prompt_emb = test_data.get_text_embeddings(index).float().to(self.device)
                # prompt_emb = self.mlp(prompt_emb)
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                outputs, attention_weights = self.model(batch_x, batch_x_mark,
                                                        prompt_emb,
                                                        dec_inp, batch_y_mark,
                                                        prior_y)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                preds.append(pred)
                trues.append(true)
                if i % 1 == 0:
                    try:
                        input = batch_x.detach().cpu().numpy()
                        if test_data.scale and self.args.inverse:
                            shape = input.shape
                            input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                        gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                        pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                        visual(gt, pd, os.path.join(self.args.log_path, str(i) + '.pdf'))
                        # self.metrics_tracker.log_attention_maps(attention_weights=attention_weights, batch_idx=i,
                        #                                         sample_indices=index)
                    except Exception as e:
                        print(f"Visualization skipped due to shape mismatch: {e}")
                        pass

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        self.metrics_tracker.log_test_metrics(mae=mae, mse=mse, rmse=rmse, mape=mape, mspe=mspe)
        print('mse:{}, mae:{}'.format(mse, mae))
        result_path = self.args.output_path + '/' + self.args.save_name
        # write datetime and metrics to TXT file
        f = open(result_path, 'a')
        f.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ")
        f.write(self.args.setting + "  \n")
        f.write('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, rmse, mape, mspe))
        f.write('\n')
        f.write('\n')
        f.close()
        # store metrics and predictions

        np.save(os.path.join(self.args.log_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
        np.save(os.path.join(self.args.log_path, 'pred.npy'), preds)
        np.save(os.path.join(self.args.log_path, 'true.npy'), trues)

        return mse
