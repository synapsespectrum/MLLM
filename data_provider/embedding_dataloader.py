import os
import pandas as pd
from torch.utils.data import Dataset
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Dataset_Custom(Dataset):
    """
    Dataset for generating embeddings from txt
    """

    def __init__(self, root_path="dataset/", flag='train', seq_len=24,
                 features='S', data_path='Environment',
                 target='OT', timeenc=0, freq='h',
                 patch_len=16, percent=100,
                 text_name="fact"):
        # size [seq_len, label_len, pred_len]
        # info
        self.percent = percent
        self.patch_len = patch_len
        self.seq_len = seq_len
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.text_name = text_name

        self.features = features
        self.target = target
        self.timeenc = timeenc
        self.freq = freq  # 'h' for hourly data, 'd' for daily data, 'w' for weekly data, etc.

        self.root_path = root_path
        self.data_path = data_path

        if not data_path.endswith('.csv'):
            data_path += '.csv'
        self.data_path = data_path

        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
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

        data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['year'] = df_stamp.date.apply(lambda row: row.year)
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday())
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        # text processing
        self.text = df_raw[[self.text_name]][border1:border2].values
        for i in range(len(self.text)):
            if pd.isnull(self.text[i][0]):
                self.text[i][0] = 'No information available'

        print("data_x shape:", self.data_x.shape)
        print("text shape:", self.text.shape)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        # r_begin = s_end
        # r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        # seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]

        # Convert text data to a tensor-compatible format
        seq_text = self.text[s_begin:s_end]

        # Create a list of strings from the numpy array and convert to a list
        # This approach avoids the need for a string tensor, which PyTorch doesn't support natively
        seq_text_list = [str(text[0]) for text in seq_text]

        # Return the text as a list, which is compatible with PyTorch's DataLoader
        return seq_x, seq_x_mark, seq_text_list

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1
