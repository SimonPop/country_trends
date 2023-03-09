# Graph Structure Learning to the test of Whale Search Trends. 🐋

![Whale Pathways](imgs/whales_highway.avif)

## 🎯 Objective & motivation

The objective of this project is to show the graph structure learning capabilities of GNNs.

We will try demonstrating that using the MTGNN model from [Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks](https://arxiv.org/abs/2005.11650), a time-series forecasting paper learning a graph on the given multi-series.

To showcase that ability, we will use [Google Trends](https://trends.google.com/trends/explore?date=today%205-y&q=%2Fg%2F121dcm9p) data on the search frequency of the subject: `Whale (Animal)`.

This should give a proxy on whale apparitions on the coasts of different countries.

We hope the model can learn to predict search frequency in different locations and learn a relevant underlying graph structure along the way.

## 📈 The Data

> The data we use is both noisy and a naive approximation of real whale apparition frequency due to both bias in Google use in different places, accessibility and popularity of whale observation. However, the data is rich enough and allows for a fun exploration of Graph Structure Learning 😄

### Data Acquisition

Google Trends data is publicly available and can be scrapped using its API. 
We use the [PyTrends](https://pypi.org/project/pytrends/) library to facilitate the gathering of this data in a pandas data frame.

The frequency of acquisition varies and averages around a point every 5 days.

### Data Processing

Data from Google Trends is already pre-processed for taking values in a [0-100] interval.

For our learning objective, we will add some more processing:

1. Standardization: We standardize the series location by location, not only to center data around 0, but also in an attempt to remove bias over different Google usage at different locations.
2. Moving average for smoothing values.
2. Upsampling: we artificially increase the number of ticks in the dataset from a ~5d frequency to an hourly frequency.

## 🤖 The Model

Model in use is the [MTGNN](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#torch_geometric_temporal.nn.attention.mtgnn.MTGNN) as implemented in the [PyTorch Geometric Temporal](https://pytorch-geometric-temporal.readthedocs.io/en/latest/index.html) library.

PyTorch Geometric Temporal is *"a temporal graph neural network extension library for PyTorch Geometric."*. It provides direct access to this model, from its catalog.

### Hyperparameters

One of the biggest challenge for us is overfitting. The model, by default, is quite large compared to our dataset. 

We will therefore use different methods to try and regularize our model:
- Adam optimizer with `weights decay` (L2 regularization)
- Large `dropout` value
- `Upsampling` of the data as mentioned above
- Small `hidden dimensions`

## 📊 Results