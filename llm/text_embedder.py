import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from transformers import AutoConfig, AutoModel, AutoTokenizer, LlamaForCausalLM


class TextEmbedder(nn.Module):
    def __init__(self, model_name="gpt2",
                 llm_layers=6, device='cuda:0', pool_type='avg', hug_token=None):
        super(TextEmbedder, self).__init__()
        self.device = device

        # Model parameters
        self.llm_layers = llm_layers
        self.hug_token = hug_token  # for LLaMA-3 models that require authentication
        self.llm_model = None
        self.tokenizer = None
        self.pool_type = pool_type.lower()
        self.llama_config = None
        self.gpt2_config = None
        self.bert_config = None

        self._llm(model_name)  # Load the specified LLM model

    def _llm(self, llm_model_name):
        """
        Load the specified LLM model.
        """

        if llm_model_name == 'LLAMA2':
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = self.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            # except EnvironmentError:  # downloads model from HF is not already done
            except:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            # except EnvironmentError:  # downloads the tokenizer from HF if not already done
            except:  # downloads model from HF is not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif llm_model_name == 'LLAMA3':
            # Automatically load the configuration, model, and tokenizer for LLaMA-3-8B
            llama3_path = "meta-llama/Meta-Llama-3-8B-Instruct"
            cache_path = "./"

            # Load the configuration with custom adjustments
            self.config = LlamaConfig.from_pretrained(llama3_path, token=self.hug_token, cache_dir=cache_path)

            self.config.num_hidden_layers = self.llm_layers
            self.config.output_attentions = True
            self.config.output_hidden_states = True

            self.llm_model = LlamaModel.from_pretrained(
                llama3_path,
                config=self.config,
                token=self.hug_token, cache_dir=cache_path
            )
            self.tokenizer = AutoTokenizer.from_pretrained(llama3_path, use_auth_token=self.hug_token,
                                                           cache_dir=cache_path)
        elif llm_model_name == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')

            self.gpt2_config.num_hidden_layers = self.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except:  # downloads model from HF is not already done
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
            except:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif llm_model_name == 'GPT2M':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2-medium')

            self.gpt2_config.num_hidden_layers = self.llm_layers
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
        elif llm_model_name == 'GPT2L':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2-large')

            self.gpt2_config.num_hidden_layers = self.llm_layers
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
        elif llm_model_name == 'GPT2XL':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2-xl')

            self.gpt2_config.num_hidden_layers = self.llm_layers
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
        elif llm_model_name == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

            self.bert_config.num_hidden_layers = self.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            # except EnvironmentError:  # downloads model from HF is not already done
            except:  # downloads model from HF is not already done
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
            # except EnvironmentError:  # downloads the tokenizer from HF if not already done
            except:  # downloads model from HF is not already done
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
            param.requires_grad = False  # Freeze the LLM model parameters
        self.llm_model = self.llm_model.to(self.device)

        print(f"Using model: {llm_model_name} on device: {self.device}")
        print("LLM Model Architecture:")
        print(self.llm_model)

    def forward(self, input_texts):
        text_flattened = input_texts.reshape(-1).tolist()
        tokenized_output = self.tokenizer(
            text_flattened,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        )

        language_max_len = tokenized_output['input_ids'].shape[1]
        input_ids = tokenized_output['input_ids'].to(self.llm_model.device)
        attn_mask = tokenized_output['attention_mask'].to(self.llm_model.device)
        text_embeddings = self.llm_model.get_input_embeddings()(input_ids)  # [num_text, max_input_model, d_model]
        expanded_mask = attn_mask.unsqueeze(-1).expand_as(text_embeddings)

        print("Information about text embeddings:")
        print("Shape of text_embeddings:", text_embeddings.shape)
        print("Shape of expanded_mask:", expanded_mask.shape)
        print("Language max length:", language_max_len)
        print("attn_mask shape:", attn_mask.shape)

        if self.pool_type == "avg":
            # Mask the embeddings by setting padded tokens to 0
            masked_emb = text_embeddings * expanded_mask
            valid_counts = expanded_mask.sum(dim=1, keepdim=True).clamp(min=1)
            pooled_emb = masked_emb.sum(dim=1) / valid_counts.squeeze(1)
            text_embeddings = pooled_emb

        elif self.pool_type == "max":
            # Mask the embeddings by setting padded tokens to a very small value
            masked_emb = text_embeddings.masked_fill(expanded_mask == 0, float('-inf'))
            pooled_emb, _ = masked_emb.max(dim=1)
            text_embeddings = pooled_emb

        elif self.pool_type == "min":
            # Mask the embeddings by setting padded tokens to a very large value
            masked_emb = text_embeddings.masked_fill(expanded_mask == 0, float('inf'))
            pooled_emb, _ = masked_emb.min(dim=1)
            text_embeddings = pooled_emb

        return text_embeddings
