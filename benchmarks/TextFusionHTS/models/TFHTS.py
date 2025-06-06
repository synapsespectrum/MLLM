import torch
import torch.nn as nn
from models.basic_model import BasicModel
from layers.Attention_Family import CrossAttention

class FlattenHead(nn.Module):
    def __init__(self, nf, target_window, head_dropout=0):
        super().__init__()
        # self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        print(x.shape)
        x = self.flatten(x)
        print(x.shape)
        x = self.linear(x)
        print(x.shape)
        x = self.dropout(x)
        return x

class Model(BasicModel):
    def __init__(self, configs, device, projection_dims=[768, 768], phase='fusion'):
        '''
        project_dim = dimension of pre-extract text and image embeddings
        phase = 'ts_only' (giai đoạn 1) hoặc 'fusion' (giai đoạn 2)
        '''
        super().__init__(configs, device)
        self.projection_dims = projection_dims
        self.phase = phase  # Thêm tham số phase để xác định giai đoạn
        self.flatten = nn.Flatten(start_dim=-2)
        self.head_nf = configs.d_model * \
                       int((configs.seq_len - configs.patch_len) /  2)
        self.head = FlattenHead(self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)
        # Text embeddings
        self.txt_fc = nn.Linear(configs.d_txt, projection_dims[0])
        # Adaptors
        self.relu = nn.ReLU()
        self.projection_mlp = nn.Sequential(
            nn.Linear(self.projection_dims[0], self.projection_dims[1]),
            nn.ReLU(),
            nn.Linear(self.projection_dims[0], self.projection_dims[1])
        )
        # Linear layer
        self.linear = nn.Linear(configs.d_model, configs.pred_len)
        self.linear_ts = nn.Linear(configs.d_model, configs.pred_len)
        self.reduce_patch = nn.Linear(configs.seq_len, 1)
        # Fusion
        self.crossatn = CrossAttention(configs.d_model, self.projection_dims[1])

        self.current_epoch = 0
        self.freeze_epochs = 3  # Số epoch để freeze ts_model

    def forward(self, txt_enc, x_enc, x_mark_enc, x_dec, x_mark_dec, x_date, y_date):
        '''
        txt_enc: embedding of text information from freezed LLM: llama2 7b
        '''
        # Normalization [bs x seq_len x ndim=1]
        if self.args.revin:
            x_enc = self.rev_in(x_enc, 'norm').to(self.device)
        else:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x_enc = x_enc / stdev

        # Encoder - Time series model
        ts_emb = self.ts_model(x_enc, x_mark_enc, x_dec, x_mark_dec)  # [bs x ndim=1 x patch_num x d_model]
        # print("ts_emb shape: ", ts_emb.shape)
        if self.phase == 'ts_only':
            # Giai đoạn 1: Chỉ huấn luyện ts_model
            ts_emb = ts_emb.squeeze(1)
            dec_out = self.linear_ts(ts_emb)
            # print("dec_out shape: ", dec_out.shape)
            # Giảm số lượng patch từ patch_num (12) về 1
            patch_out = dec_out.permute(0, 2, 1)  # [256, 14, 12]
            outputs = self.reduce_patch(patch_out)  # [256, 14, 1]
            # print("dec_out shape: ", dec_out.shape)
        elif self.phase == 'fusion':
            # Giai đoạn 2: Fusion với text
            if self.current_epoch < self.freeze_epochs:
                with torch.no_grad():  # Freeze ts_model in first few epochs
                    ts_emb = self.ts_model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            else:
                ts_emb = self.ts_model(x_enc, x_mark_enc, x_dec, x_mark_dec)

            txt_emb = self.txt_fc(txt_enc)  # [bs x d_txt=4096] -> [bs x projection_dim=768]
            txt_emb = self.projection_mlp(txt_emb)  # [bs x projection_dim]
            txt_emb = txt_emb.reshape(txt_emb.shape[0], 1, txt_emb.shape[1])  # [bs x 1 x projection_dim]
            # Fusion với cross-attention
            enc_out, _ = self.crossatn(ts_emb, txt_emb)  # [bs x ndim=1 x d_model]
            dec_out = self.linear(enc_out)
            dec_out = dec_out.permute(0, 2, 1)
            outputs = dec_out[:, -self.args.pred_len:, :]

        # De-Normalization
        if self.args.revin:
            outputs = self.rev_in(outputs, 'denorm').to(self.device)
        else:
            outputs = outputs * stdev
            outputs = outputs + means

        return outputs