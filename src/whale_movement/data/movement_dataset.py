import pandas as pd
import torch
import h3
import geopandas as gpd

class Dataset(torch.utils.data.Dataset):
    def __init__(self, res: int = 10):
        self.hexagon_res = res

        DATA_PATH = 'Azores Great Whales Satellite Telemetry Program .csv'
        self.dataframe = pd.read_csv(DATA_PATH, nrows=200)
        self.features = self.process(self.dataframe)

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        data['date'] = pd.to_datetime(data['timestamp'])
        data = data.set_index('date').rename(columns={'location-lat': 'lat', 'location-long': 'lon'})
        
        data[f'h3_{self.hexagon_res}'] = data[['lat', 'lon']].apply(lambda x: h3.geo_to_h3(x.lat, x.lon, resolution=self.hexagon_res), axis=1)

        return pd.get_dummies(data[f'h3_{self.hexagon_res}'])

    def __len__(self):
        return len(self.dataframe) - 1

    def __getitem__(self, index):
        x = self.features.iloc[index].values
        y = self.features.iloc[index+1].values

        x = torch.tensor(x).float().unsqueeze(-1)
        y = torch.tensor(y).float()
        return x, y
    
