import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, LinearConstraint
import requests
import networkx as nx
import contextily as cx
from matplotlib.pyplot import imshow
from dataclasses import dataclass

API_URL = 'https://whaletrips.org/map/js/getallcountries.php'

ANIMALS = [
    'Orca',
    'Glattwal',
    'Buckelwal',
    'Blauwal',
    'Pottwal',
    'Finnwal',
    'Beluga',
    'Narwal',
    'Zwergwal',
    'Grauwal'
]

SEASONS = [
    'frühjahr',
    'sommer',
    'herbst',
    'winter',
]

DE2EN = {
    'Orca': 'Orca', 
    'Glattwal': 'Right Whale', 
    'Buckelwal': 'Humpback Whale', 
    'Blauwal': 'Blue Whale', 
    'Pottwal': 'Sperm Whale', 
    'Finnwal': 'Fin Whale', 
    'Beluga': 'Beluga',
    'Narwal': 'Narwhal',
    'Zwergwal': 'Minke Whale',
    'Grauwal': 'Gray Whale',

    'frühjahr': 'spring',
    'sommer': 'summer',
    'herbst': 'autumn',
    'winter': 'winter'
}

NAME2POS = {
'Alaska': (65.27653117955406, -152.41077157051282),
 'Argentinien': (-38.70492886148706, -63.58603920444051),
 'Australien (Ostküste)': (-27.95574169454626, 153.2790930111902),
 'Australien (Südküste)': (-35.48228040467213, 139.32732533931767),
 'Azoren': (37.81235158733328, -25.23087444852324),
 'British Columbia': (54.80490677820256, -127.11831474481993),
 'Chile': (-35.65668189577454, -71.79391930681011),
 'Grönland': (73.71782154445522, -37.98037125236898),
 'Hawaii': (19.630236329146488, -155.59669360466808),
 'Island': (64.8848181942907, -18.266145843541555),
 'Kalifornien': (36.30497745622242, -121.38597230951882),
 'Manitoba': (57.73362383762877, -93.9699327663503),
 'Mexiko': (26.324617792523977, -112.35502933964605),
 'Neuseeland': (-42.297428190414344, 172.74240263377402),
 'New England': (43.59454594629038, -70.5336380447341),
 'Norwegen': (61.61164478986404, 6.772261553459945),
 'Nunavut': (66.64046364215476, -96.35212735843194),
 'Quebec': (49.663905324352115, -70.41791589978648),
 'Südafrika': (-32.24566560173534, 23.56118282567003),
 'Norwegen': (68.75631464541773, 17.846480071334334),
 'Washington': (47.4176104699817, -123.62853179230028)
}

ANIMAL2COL = {
    x: i for i, x in enumerate(ANIMALS)
}

SEASON2COL = {
    x: i for i, x in enumerate(SEASONS)
}


@dataclass
class WhaleTripData():
    df: pd.DataFrame
    size: int

def collect_data() -> WhaleTripData:
    results = requests.get(API_URL)
    columns = [
        DE2EN[x] + '_' + DE2EN[y] for x in ANIMALS for y in SEASONS
    ]

    rows = [
        result['countryName'] for result in results.json()
    ]

    name = [x['countryName'] for x in results.json()]

    values = np.zeros((len(rows), len(columns)))

    for i, result in enumerate(results.json()):
        for x in result['filter']:
            name = x['name']
            col = 4*ANIMAL2COL[name]
            seasons = x['properties']
            if type(seasons) == dict:
                seasons = x['properties'].values()
            for season in seasons:
                values[i][col+SEASON2COL[season['season']]] = 1

    df = pd.DataFrame(
        index=rows,
        columns=columns,
    )
    df[columns] = values

    return WhaleTripData(
        df=df,
        size=len(df)
    )