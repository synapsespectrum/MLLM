from typing import Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F

from ts import PatchTST, iTransformer


class MLP(nn.Module):
    """Multi-Layer Perceptron for dimension translation and forecasting head"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 2, dropout: float = 0.1, activation: str = 'relu'):
        super(MLP, self).__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        layers = []

        if num_layers == 1:
            # Single layer case
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            # Multi-layer case
            # Input layer
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                self._get_activation(activation),
                nn.Dropout(dropout)
            ])

            # Hidden layers
            for _ in range(num_layers - 2):
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    self._get_activation(activation),
                    nn.Dropout(dropout)
                ])

            # Output layer
            layers.append(nn.Linear(hidden_dim, output_dim))

        self.layers = nn.Sequential(*layers)

    def _get_activation(self, activation: str):
        """Get activation function"""
        if activation.lower() == 'relu':
            return nn.ReLU()
        elif activation.lower() == 'gelu':
            return nn.GELU()
        elif activation.lower() == 'tanh':
            return nn.Tanh()
        else:
            return nn.ReLU()  # Default

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class CrossModal(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super(CrossModal, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        # Multi-head cross attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, ts_features: torch.Tensor, text_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            ts_features: [batch_size, seq_len, d_model] - Time series features
            text_features: [batch_size, text_len, d_model] - Text features

        Returns:
            output: [batch_size, seq_len, d_model] - Fused features
            attention_weights: [batch_size, n_heads, seq_len, text_len] - Attention weights
        """
        # Cross-attention: time series as query, text as key/value
        attended_output, attention_weights = self.cross_attention(
            query=ts_features,  # [batch_size, seq_len, d_model]
            key=text_features,  # [batch_size, text_len, d_model]
            value=text_features  # [batch_size, text_len, d_model]
        )

        # Residual connection and layer norm
        output = self.norm1(ts_features + self.dropout(attended_output))

        # Feed-forward with residual connection
        ffn_output = self.ffn(output)
        output = self.norm2(output + self.dropout(ffn_output))

        return output, attention_weights


class TSLLMFusionModel(nn.Module):
    """
    Time Series - LLM Fusion Model using Cross-Attention

    Architecture:
    1. Time Series Encoder (iTransformer)
    2. Translator MLPs (align dimensions)
    3. Cross-Attention Fusion
    4. Forecasting Head (MLP)
    """

    def __init__(self, configs):
        super(TSLLMFusionModel, self).__init__()
        self.model_dict = {
            # 'TimesNet': TimesNet,
            # 'Autoformer': Autoformer,
            # 'Transformer': Transformer,
            # 'Nonstationary_Transformer': Nonstationary_Transformer,
            # 'DLinear': DLinear,
            # 'FEDformer': FEDformer,
            # 'Informer': Informer,
            # 'LightTS': LightTS,
            # 'Reformer': Reformer,
            # 'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            # 'Pyraformer': Pyraformer,
            # 'MICN': MICN,
            # 'Crossformer': Crossformer,
            # 'FiLM': FiLM,
            'iTransformer': iTransformer,
            # 'Koopa': Koopa,
            # 'TiDE': TiDE,
            # 'FreTS': FreTS,
            # 'TimeMixer': TimeMixer,
            # 'TSMixer': TSMixer,
            # 'SegRNN': SegRNN
        }

        # Configuration
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.llm_dim = configs.llm_dim
        self.fusion_dim = getattr(configs, 'fusion_dim', 256)  # Common dimension after translation
        self.enc_in = configs.enc_in

        # Time series backbone (iTransformer)
        self.ts_model = self._build_ts_model()

        # Translator MLPs to align dimensions
        self.ts_translator = MLP(
            input_dim=self.d_model,
            hidden_dim=self.fusion_dim,
            output_dim=self.fusion_dim,
            num_layers=2,
            dropout=configs.dropout,
            activation='gelu'
        )

        self.text_translator = MLP(
            input_dim=self.llm_dim,
            hidden_dim=self.fusion_dim,
            output_dim=self.fusion_dim,
            num_layers=2,
            dropout=configs.dropout,
            activation='gelu'
        )

        # Cross-attention fusion
        self.cross_attention = CrossModal(
            d_model=self.fusion_dim,
            n_heads=getattr(configs, 'fusion_heads', 4),
            dropout=configs.dropout
        )

        # Forecasting head
        self.forecasting_head = MLP(
            input_dim=self.fusion_dim,
            hidden_dim=self.fusion_dim * 2,
            output_dim=self.pred_len,
            num_layers=3,
            dropout=configs.dropout,
            activation='gelu'
        )

        # Optional: Direct time series path for residual connection
        self.use_residual = getattr(configs, 'use_residual', True)
        if self.use_residual:
            self.ts_direct_proj = nn.Linear(self.d_model, self.pred_len)
            self.fusion_weight = nn.Parameter(torch.tensor(self.configs.prompt_weight))

        # Sequence length alignment
        self.seq_align_method = getattr(configs, 'seq_align_method', 'mean')  # 'mean', 'max', 'adaptive'

    def _build_ts_model(self):
        """Build time series model (iTransformer, PatchTST, etc.)"""
        try:
            model = self.model_dict[self.configs.model].Model(self.configs).float()

            if self.configs.use_multi_gpu and self.configs.use_gpu:
                model = nn.DataParallel(model, device_ids=self.configs.device_ids)

            return model
        except ImportError:
            print("Warning: iTransformer not found. Using placeholder.")
            # Placeholder implementation
            return nn.Sequential(
                nn.Linear(self.configs.enc_in, self.configs.d_model),
                nn.LayerNorm(self.configs.d_model),
                nn.ReLU(),
                nn.Linear(self.configs.d_model, self.configs.d_model)
            )

    def forward(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor,
                prompt_emb: torch.Tensor, x_dec: Optional[torch.Tensor] = None,
                x_mark_dec: Optional[torch.Tensor] = None, prior_y=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            x_enc: [batch_size, seq_len, enc_in] - Time series input
            x_mark_enc: [batch_size, seq_len, mark_dim] - Time features
            prompt_emb: [batch_size, text_len, llm_dim] - LLM embeddings

        Returns:
            forecast_output: [batch_size, pred_len, enc_in] - Forecasted values
            attention_weights: [batch_size, n_heads, seq_len, text_len] - Attention weights
        """
        batch_size, seq_len, n_vars = x_enc.shape

        # 1. Get time series features from backbone model
        if hasattr(self.ts_model, 'forecast'):
            # For iTransformer
            ts_output = self.ts_model.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            # ts_output: [batch_size, n_vars, d_model]

            # Reshape for processing: [batch_size, n_vars, d_model] -> [batch_size, seq_len, d_model]
            if ts_output.dim() == 3 and ts_output.size(1) == n_vars:
                # Expand to sequence length dimension
                ts_output = ts_output.unsqueeze(2).expand(-1, -1, seq_len, -1)  # [batch_size, n_vars, seq_len, d_model]
                ts_output = ts_output.mean(dim=1)  # Average across variables: [batch_size, seq_len, d_model]
        else:
            # For placeholder model
            ts_output = self.ts_model(x_enc.mean(dim=-1, keepdim=True))  # [batch_size, seq_len, d_model]

        # 2. Translate both modalities to common dimension
        ts_translated = self.ts_translator(ts_output)  # [batch_size, seq_len, fusion_dim]
        text_translated = self.text_translator(prompt_emb)  # [batch_size, text_len, fusion_dim]

        # 3. Align sequence lengths
        # ts_aligned, text_aligned = self._align_sequence_length(ts_translated, text_translated)

        # 4. Cross-attention fusion
        fused_output, attention_weights = self.cross_attention(ts_translated, text_translated)
        # fused_output: [batch_size, seq_len, fusion_dim]

        # 5. Forecasting head
        # Aggregate sequence dimension for forecasting
        if hasattr(self.configs, 'pooling_method') and self.configs.pooling_method == 'last':
            fused_features = fused_output[:, -1, :]  # Take last timestep
        else:
            fused_features = fused_output.mean(dim=1)  # Average pooling

        forecast_output = self.forecasting_head(fused_features)  # [batch_size, pred_len]

        # 6. Optional residual connection
        if self.use_residual and hasattr(self, 'ts_direct_proj'):
            ts_direct_features = ts_output.mean(dim=1)  # [batch_size, d_model]
            ts_direct = self.ts_direct_proj(ts_direct_features)  # [batch_size, pred_len]
            if prior_y is not None:
                ts_direct = ts_direct + prior_y

            # Weighted combination
            fusion_weight = torch.sigmoid(self.fusion_weight)  # Ensure 0-1 range
            forecast_output = fusion_weight * forecast_output + (1 - fusion_weight) * ts_direct

        # 7. Reshape for compatibility: [batch_size, pred_len, n_vars]
        forecast_output = forecast_output.unsqueeze(-1).expand(-1, -1, n_vars)

        return forecast_output, attention_weights

    def _align_sequence_length(self, ts_features: torch.Tensor, text_features: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """Align sequence lengths between time series and text features"""
        batch_size = ts_features.size(0)
        ts_seq_len = ts_features.size(1)
        text_seq_len = text_features.size(1)

        if ts_seq_len == text_seq_len:
            return ts_features, text_features

        # Align text to time series length
        if self.seq_align_method == 'mean':
            # Simple mean pooling/expansion
            if text_seq_len > ts_seq_len:
                # Downsample text
                text_features = F.adaptive_avg_pool1d(
                    text_features.transpose(1, 2), ts_seq_len
                ).transpose(1, 2)
            else:
                # Upsample text (repeat)
                repeat_factor = ts_seq_len // text_seq_len
                remainder = ts_seq_len % text_seq_len
                text_features = text_features.repeat(1, repeat_factor, 1)
                if remainder > 0:
                    text_features = torch.cat([
                        text_features,
                        text_features[:, :remainder, :]
                    ], dim=1)

        elif self.seq_align_method == 'adaptive':
            # Use adaptive pooling
            text_features = F.adaptive_avg_pool1d(
                text_features.transpose(1, 2), ts_seq_len
            ).transpose(1, 2)

        return ts_features, text_features