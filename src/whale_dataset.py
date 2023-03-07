import pandas as pd
import torch
from pytrends.request import TrendReq
from sklearn.preprocessing import StandardScaler
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm

class WhaleDataset(torch.utils.data.Dataset):
    def __init__(self, input_length: int = 10):

        self.places = {
            # n°1
            "CA-NL": ("Canada", "Terre-Neuve-et-Labrador", (53.973902576875076, -61.14130912420827)),
            "CA-BC": ("Canada", "Colombie-Britannique", (54.67666698427657, -127.16718803964187)),
            "US-ME": ("United States", "Maine", (45.067796619231636, -69.22708572491548)),
            "US-AK": ("United States", "Alaska", (64.83098113826678, -150.97816874378356)),
            "US-OR": ("United States", "Oregon", (43.90272489383347, -122.1894971630876)),
            "US-CA": ("United States", "California", (36.324916117015206, -119.76960988843537)),
            "US-WA": ("United States", "Washington", (47.43812673867954, -121.59094374903619)),
            "US-MA": ("United States", "Massachusetts", (42.351943549747745, -71.17098431423099)),
            "US-FL": ("United States", "Floride", (27.800555414741435, -81.39651633437465)),
            "MX-BCS": ("Mexico", "Basse-Californie du Sud", (25.1606668804671, -111.34380582183239)),
            "US-HI": ("United States", "Hawaï", (19.706895802596467, -155.48697573111843)),
            # n°2
            "AU-WA": ("Australia", "Australie-Occidentale", (-27.791471267645093, 117.61185851642362)),
            "AU-TAS": ("Australia", "Tasmanie", (-42.177412581844116, 146.77411121219245)),
            "AU-SA": ("Australia", "Australie-Méridionale", (-31.912932850861168, 134.26634437421387)),
            "AU-VIC": ("Australia", "Victoria", (-38.027893928254066, 145.258141138917)),
            "AU-NSW": ("Australia", "Nouvelle-Galles du Sud", (-33.88917636971855, 150.70111378879918)),
            "AU-QLD": ("Australia", "Queensland", (-20.180951308717574, 147.15448763440028)),
            "NZ": ("NZ", "", (-43.759889092670825, 170.24110322231667)),
            "ID": ("Indonesia", "", (-7.310066925311799, 110.8644323575184)),
            # n°3
            "IS": ("Iceland", "", (64.74511674339414, -18.365096006071855)),
            "NO": ("Norway", "", (61.758375002335534, 6.257610579134723)),
            "IE": ("Ireland", "", (52.64110013708303, -8.182270477616262)),
            "GB": ("Great-Britain", "", (57.14207782828509, -4.307196072343293)),
            "ES-CN": ("Spain", "Canarias", (28.326509820198524, -16.555759112470028)),
            "PT": ("Portugal", "", (38.77432060348665, -9.344699339591637)),
            "FR": ("France", "", (48.05900873853055, -0.4358186810291075))
        }

        DATA_PATH = Path(__file__).parent.parent / 'whales.csv'
        if os.path.exists(DATA_PATH):
            self.dataframe = pd.read_csv(DATA_PATH, parse_dates=['date']).set_index('date')
        else:
            self.dataframe = self.raw_dataset()
            self.dataframe.to_csv(DATA_PATH)

        self.dataframe = self.process_df() 

        self.input_length = input_length
        self.X, self.y = self.generate_graph_seq2seq_io_data(self.dataframe)

    def process_df(self):
        self.scaler = StandardScaler()
        # self.dataframe = self.dataframe.pct_change().drop("2018-03-11").dropna(axis=1).replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna(axis=0)
        if not self.places is None:
            self.dataframe = self.dataframe[self.places]
        self.dataframe[self.dataframe.columns] = self.scaler.fit_transform(self.dataframe)
        self.dataframe = self.dataframe.ewm(span = 20).mean()
        return self.dataframe

    def raw_dataset(self):
        pytrends = TrendReq(hl='en-US', tz=360)

        df = pd.DataFrame()

        for place in tqdm(self.places.keys()):
            pytrends.build_payload(["/g/121dcm9p"], timeframe='today 5-y', gprop='', geo=place)
            df[place] = pytrends.interest_over_time()["/g/121dcm9p"]

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
    
if __name__ == "__main__":
    whale_dataset = WhaleDataset()