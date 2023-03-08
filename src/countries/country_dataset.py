import pandas as pd
import torch
from pytrends.request import TrendReq
from sklearn.preprocessing import StandardScaler
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm

class Dataset(torch.utils.data.Dataset):
    def __init__(self, input_length: int = 10, countries = None):
        DATA_PATH = Path(__file__).parent.parent / 'time_series.csv'
        if os.path.exists(DATA_PATH):
            self.dataframe = pd.read_csv(DATA_PATH, parse_dates=['date']).set_index('date')
        else:
            self.dataframe = self.raw_dataset()
            self.dataframe.to_csv(DATA_PATH)

        self.scaler = StandardScaler()
        # self.dataframe = self.dataframe.pct_change().drop("2018-03-11").dropna(axis=1).replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna(axis=0)
        if not countries is None:
            self.dataframe = self.dataframe[countries]
        self.dataframe[self.dataframe.columns] = self.scaler.fit_transform(self.dataframe)
        self.dataframe = self.dataframe.ewm(span = 20).mean()

        self.input_length = input_length
        self.X, self.y = self.generate_graph_seq2seq_io_data(self.dataframe)

    def raw_dataset(self):
        pytrends = TrendReq(hl='en-US', tz=360)
        country_df = pd.read_csv(Path(__file__).parent.parent / 'country.csv', delimiter='\t')

        dfs=[]
        for i in tqdm(range(0, len(country_df)-5, 5)):
            countries = country_df.iloc[i:min(len(country_df)-1, i+5)]['name'].to_list()
            pytrends.build_payload(countries, cat=16, timeframe='today 8-y', gprop='', geo='FR')
            df = pytrends.interest_over_time()[countries]
            dfs.append(df)

        df = pd.concat(dfs, axis=1)
        return df

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x = self.X[index]
        y = self.y[index]

        x = torch.tensor(x).T.float()
        y = torch.tensor(y)
        return x, y
    
    def generate_graph_seq2seq_io_data(
            self, df
    ):
        x_offsets = np.sort(
                np.concatenate((np.arange(-self.input_length, 1, 1),))
            )
        y_offsets = np.sort(np.arange(1, self.input_length, 1))

        num_samples, num_nodes = df.shape
        data = np.expand_dims(df.values, axis=-1)
        # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
        x, y = [], []
        # t is the index of the last observation.
        min_t = abs(min(x_offsets))
        max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
        for t in range(min_t, max_t):
            x_t = data[t + x_offsets, ...]
            y_t = data[t + y_offsets, ...]
            x.append(x_t)
            y.append(y_t)
        x = np.stack(x, axis=0)
        y = np.stack(y, axis=0)
        return x, y