import pandas as pd
import torch
from pytrends.request import TrendReq
from sklearn.preprocessing import StandardScaler
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
from time import sleep
from config import COUNTRIES, SUPERHIGHWAY

class WhaleDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            x_input_length: int = 10, 
            y_input_length: int = 10, 
            dataframe: pd.DataFrame = None
        ):

        self.A_superhighway = torch.FloatTensor([
            [int(SUPERHIGHWAY[country]==SUPERHIGHWAY[country_b]) for country_b in COUNTRIES] for country in COUNTRIES
        ]) - torch.eye(len(COUNTRIES))

        DATA_PATH = Path(__file__).parent.parent.parent / 'whales.csv'
        if dataframe is None:
            if os.path.exists(DATA_PATH):
                dataframe = pd.read_csv(DATA_PATH, parse_dates=['date']).set_index('date')
            else:
                dataframe = self.combine_df()
                dataframe.to_csv(DATA_PATH)
            # else:
            #     dataframe = self.raw_dataset()
            #     dataframe.to_csv(DATA_PATH)

        self.dataframe = self.process_df(dataframe) 

        self.x_input_length = x_input_length
        self.y_input_length = y_input_length
        self.X, self.y = self.generate_graph_seq2seq_io_data(self.dataframe)

    def process_df(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        # self.scaler = StandardScaler()
        # self.dataframe = self.dataframe.pct_change().drop(self.dataframe.index[0]).replace(np.nan, 0).replace(np.inf, 0).replace(-np.inf, 0).dropna(axis=0)
        # if not COUNTRIES is None:
        #     dataframe = dataframe[COUNTRIES]
        # dataframe[dataframe.columns] = self.scaler.fit_transform(dataframe)
        #dataframe = dataframe.ewm(span = 20).mean()
        dataframe = dataframe.resample('6H').interpolate(method='linear')
        dataframe = dataframe / 100
        return dataframe

    def raw_dataset(self):
        pytrends = TrendReq(hl='en-US', tz=360, retries=2, backoff_factor=0.1, requests_args={'verify':False})

        df = pd.DataFrame()

        for place in tqdm(COUNTRIES):
            sleep(30)
            pytrends.build_payload(["/g/121dcm9p"], timeframe='today 5-y', gprop='', geo=place)
            df[place] = pytrends.interest_over_time()["/g/121dcm9p"]

        return df

    def combine_df(self) -> pd.DataFrame:
        df = pd.DataFrame()
        for country in COUNTRIES:
            country_df = pd.read_csv(Path(__file__).parent.parent / f"data/whales/{country}.csv", parse_dates=['Semaine'], skiprows=2)
            country_df = country_df.rename(columns={"Semaine": "date"}).set_index('date')
            df[country] = country_df
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
                np.concatenate((np.arange(-self.x_input_length, 1, 1),))
            )
        y_offsets = np.sort(np.arange(1, self.y_input_length, 1))
        num_samples, num_nodes = df.shape
        data = np.expand_dims(df.values, axis=-1)
        x, y = [], []
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
    
if __name__ == "__main__":
    whale_dataset = WhaleDataset()