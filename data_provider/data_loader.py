import os
import numpy as np
import pandas as pd
import torch
import h5py
from torch.utils.data import Dataset
from utils.timefeatures import time_features
import warnings
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


class Dataset_Custom(Dataset):
    """
    Dataset for loading time series data and text embeddings from h5 files
    """

    def __init__(self, root_path="dataset/", flag='train',
                 seq_len=24, label_len=24, pred_len=24,
                 features='S', data_path='Environment',
                 target='OT', scale=True, timeenc=0, freq='h',
                 patch_len=16, percent=100,
                 text_name="fact", use_embeddings=False,
                 embedding_path="Embeddings/text/", embedding_model="GPT2"):
        # size [seq_len, label_len, pred_len]
        # info
        self.percent = percent
        self.patch_len = patch_len

        # Set sequence lengths
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.text_name = text_name
        self.flag = flag

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq  # 'h' for hourly data, 'd' for daily data, 'w' for weekly data, etc.

        self.root_path = root_path
        self.data_path = data_path

        # Embedding settings
        self.use_embeddings = use_embeddings
        self.embedding_path = embedding_path
        self.embedding_model = embedding_model

        if not data_path.endswith('.csv'):
            data_path += '.csv'
        self.data_path = data_path

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')

        # Check if prior_history_avg column exists
        has_prior = 'prior_history_avg' in df_raw.columns
        has_dates = 'start_date' in df_raw.columns and 'end_date' in df_raw.columns
        if self.text_name in cols:
            cols.remove(self.text_name)
        if has_prior and has_dates:
            cols.remove('prior_history_avg')
            cols.remove('start_date')
            cols.remove('end_date')
            df_raw = df_raw[['date'] + cols + [self.target] + ['prior_history_avg'] + ['start_date'] + ['end_date']]
        elif has_prior:
            cols.remove('prior_history_avg')
            df_raw = df_raw[['date'] + cols + [self.target] + ['prior_history_avg']]
        elif has_dates:
            cols.remove('start_date')
            cols.remove('end_date')
            df_raw = df_raw[['date'] + cols + [self.target] + ['start_date'] + ['end_date']]
        else:
            df_raw = df_raw[['date'] + cols + [self.target]]

        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
            if has_prior:
                df_data_prior = df_raw[['prior_history_avg']]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
            if has_prior:
                data_prior = self.scaler.transform(df_data_prior.values[:,-1].reshape(-1, 1))
        else:
            data = df_data.values
            if has_prior:
                data_prior = df_data_prior.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        if has_prior:
            self.data_prior = data_prior[border1:border2]
        self.data_stamp = data_stamp

        # Store dates if available
        self.date = df_raw[['date']][border1:border2].values
        if has_dates:
            self.start_date = df_raw[['start_date']][border1:border2].values
            self.end_date = df_raw[['end_date']][border1:border2].values

        # Process text data
        if self.text_name in df_raw.columns:
            self.text = df_raw[[self.text_name]][border1:border2].values
            for i in range(len(self.text)):
                if pd.isnull(self.text[i][0]):
                    self.text[i][0] = 'No information available'
            print("text shape:", self.text.shape)

        # Load pre-embedded text data if specified
        if self.use_embeddings:
            self.load_embeddings()

        print("data_x shape:", self.data_x.shape)
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def load_embeddings(self):
        """Load pre-embedded text data from h5 files"""
        embedding_dir = os.path.join(self.embedding_path, self.embedding_model, self.data_path.replace('.csv', ''), self.flag)

        try:
            print(f"Loading embeddings from {embedding_dir}")

            # Check if embeddings are stored in a single file or multiple files
            if os.path.exists(f"{embedding_dir}.h5"):
                # Single file format
                with h5py.File(f"{embedding_dir}.h5", 'r') as f:
                    self.text_embeddings = torch.tensor(f['embeddings'][:])
                print(f"Loaded embeddings with shape: {self.text_embeddings.shape}")
            else:
                # Multiple files format
                h5_files = [f for f in os.listdir(embedding_dir) if f.endswith('.h5')]
                if not h5_files:
                    print(f"No h5 files found in {embedding_dir}")
                    return

                embeddings_list = []
                for file_name in sorted(h5_files, key=lambda x: int(x.split('.')[0])):
                    file_path = os.path.join(embedding_dir, file_name)
                    with h5py.File(file_path, 'r') as f:
                        embedding = f['embeddings'][:]
                        embeddings_list.append(embedding)

                # Concatenate all embeddings
                self.text_embeddings = torch.tensor(np.concatenate(embeddings_list, axis=0))
                print(f"Loaded embeddings with shape: {self.text_embeddings.shape}")
        except FileNotFoundError:
            # If the embedding directory does not exist, raise an error
            print(f"Embedding directory {embedding_dir} does not exist. Please check the path or generate embeddings first.")
            raise FileNotFoundError(f"Embedding directory {embedding_dir} does not exist.")


    def get_prior_y(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()

        s_begins = indices % self.tot_len
        s_ends = s_begins + self.seq_len
        r_begins = s_ends
        r_ends = r_begins + self.pred_len
        prior_y = np.array([self.data_prior[r_beg:r_end] for r_beg, r_end in zip(r_begins, r_ends)])
        return prior_y

    def get_prior_y_for_imputation(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()

        s_begins = indices % self.tot_len
        s_ends = s_begins + self.seq_len
        prior_y = np.array([self.data_prior[s_beg:s_end] for s_beg, s_end in zip(s_begins, s_ends)])
        return prior_y

    def get_text(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()

        s_begins = indices % self.tot_len
        s_ends = s_begins + self.seq_len
        text = np.array([self.text[s_end - self.seq_len: s_end] for s_end in s_ends])
        return text

    def get_text_embeddings(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()

        s_begins = indices % self.tot_len
        s_ends = s_begins + self.seq_len
        bsz = len(s_begins)
        # return tensor
        text_embeddings = torch.cat([self.text_embeddings[s_end - self.seq_len: s_end] for s_end in s_ends], dim=0).view(bsz, self.seq_len, -1)
        return text_embeddings

    def get_date(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()

        s_begins = indices % self.tot_len
        s_ends = s_begins + self.seq_len
        r_begins = s_ends - self.label_len
        r_ends = r_begins + self.label_len + self.pred_len

        x_start_dates = np.array([self.start_date[s_beg:s_end] for s_beg, s_end in zip(s_begins, s_ends)])
        x_end_dates = np.array([self.end_date[s_beg:s_end] for s_beg, s_end in zip(s_begins, s_ends)])

        return x_start_dates, x_end_dates

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1] if len(self.data_x.shape) > 1 else self.data_x[s_begin:s_end].reshape(-1, 1)
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1] if len(self.data_y.shape) > 1 else self.data_y[r_begin:r_end].reshape(-1, 1)

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        # If using embeddings, return them
        if self.use_embeddings:
            return seq_x, seq_y, seq_x_mark, seq_y_mark, index
        else:
            # Otherwise return text data
            if hasattr(self, 'text'):
                seq_text = self.text[s_begin:s_end]
                seq_text_list = [str(text[0]) for text in seq_text]
                return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_text_list, index
            else:
                return seq_x, seq_y, seq_x_mark, seq_y_mark, index

    def __len__(self):
        if hasattr(self, 'tot_len') and self.tot_len > 0:
            return self.tot_len
        else:
            # For backward compatibility with the original implementation
            return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
