import pandas as pd
import torch
from pytrends.request import TrendReq
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from pathlib import Path

class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.dataframe = self.raw_dataset()
        # self.dataframe = pd.read_csv('time_series.csv')[['United States','China','France','Ukraine','Russia']]
        self.scaler = MinMaxScaler()

        self.input_length = 10
        self.X, self.y = self.generate_graph_seq2seq_io_data(self.dataframe)

    def raw_dataset(self):
        pytrends = TrendReq(hl='en-US', tz=360)
        country_df = pd.read_csv(Path(__file__).parent.parent / 'country.csv', delimiter='\t')

        # TODO: countries 
        countries = ["United States", "China", "France", "Ukraine", "Russia"] # country_df['name'].sample(5).to_list()
        pytrends.build_payload(countries, cat=0, timeframe='today 5-y', geo='US', gprop='')

        df = pytrends.interest_over_time()[countries]
        df[countries] = self.scaler.fit_transform(df)

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