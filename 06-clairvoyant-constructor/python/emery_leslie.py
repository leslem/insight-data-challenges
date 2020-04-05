# # Construction materials price forecasting

# 2020-03-31

# Leslie Emery

# # Summary

# ## Challenge
# The challenge is to forecast prices for core construction materials up to 6 months in the future. The manager of the construction company wants to be able to anticipate upcoming increases in material unit prices.

# ## Approach
# - Plot the data to view annual, monthly, weekly, and daily patterns
# - Decompose time series to identify trends and cyclic/seasonal patterns
# - Fit an ARIMA model to predict prices for last 20% of data

# ## Takeaways
# - All five construction material prices can be forecast well using an ARIMA model
# - Based on forecast prices, very large increases in price can be escaped by buying during predicted lows, including before the following major increases:
#     - Steel increases in late 2016 and late 2017
#     - Rebar increases in early 2018 and late 2016
#     - Multiple concrete increases from 2016 - 2018
#     - Plywood increase in 2017
# - Glass prices don't spike as much and are more steady so this approach will have less benefit for glass costs


# +
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.io as pio
import seaborn as sns
import statsmodels.api as sm

from IPython.display import display

pio.templates.default = "plotly_white"
sns.set(style="whitegrid", font_scale=1.25)
plt.figure(figsize=(12.8, 9.6), dpi=400)
# -

# +
data_dir = '~/devel/insight-data-challenges/06-clairvoyant-constructor/data'
output_dir = '~/devel/insight-data-challenges/06-clairvoyant-constructor/output'
# -

# ## Read in and clean the data

# +
df = pd.read_csv(
    os.path.join(os.path.expanduser(data_dir), 'construction_material_prices_2008_2018.csv'),
    parse_dates=['Unnamed: 0']
)

display(df.info())

with pd.option_context('display.max_columns', 100):
    print(df.head(15))
# -

# ### Name the date column

# +
df = df.rename(columns={'Unnamed: 0': 'Date'})
display(df.info())
# -

# ### Check data values

# The range of each price looks reasonable

# +
display(df.describe())
# -

# Range of dates also looks reasonable

# +
display(df['Date'].min(), df['Date'].max())
# -

# ### Reshape the data for plotting

# +
# df2 = pd.wide_to_long(df, stubnames='price', i='Date', j='material')
price_cols = df.columns[df.columns.str.startswith('price_')]
df_longer = df.melt(var_name='Material', value_name='Price', id_vars=['Date'], value_vars=price_cols)
df_longer['Material'] = df_longer['Material'].str.replace('price_', '')
display(df_longer)
# -


# ### Remove initial NaN rows

# Many of the rows in the data are NaN, but I need to keep them because I want to be able to set the period of the pandas time series index. So here I'm just going to remove the all NaN rows from the beginoing of the series.

# +
old_shape = df.shape
first_valid_idx = df[price_cols].isna().any(axis=1).idxmin()
df = df.loc[first_valid_idx:]
print('{} rows were removed'.format(old_shape[0] - df.shape[0]))
# -


# ## Add variables for date components

# I'm going to use these variables for plotting.

# +
df['year'] = df['Date'].dt.year.astype('category')
df['month'] = df['Date'].dt.month.astype('category')
df['day'] = df['Date'].dt.day
df['dayofyear'] = df['Date'].dt.dayofyear
df['dayofweek'] = df['Date'].dt.dayofweek.astype('category')
df['isweekday'] = df['dayofweek'].isin(range(5))
df['week'] = df['Date'].dt.week.astype('category')
display(df.info())
# -

# +
df_longer['year'] = df_longer['Date'].dt.year.astype('category')
df_longer['month'] = df_longer['Date'].dt.month.astype('category')
df_longer['day'] = df_longer['Date'].dt.day
df_longer['dayofyear'] = df_longer['Date'].dt.dayofyear
df_longer['dayofweek'] = df_longer['Date'].dt.dayofweek.astype('category')
df_longer['isweekday'] = df_longer['dayofweek'].isin(range(5))
df_longer['week'] = df_longer['Date'].dt.week.astype('category')
display(df_longer.info())
# -

# ### Examine sample size by material type

# +
df_longer.groupby('Material')['Date'].count()
df_longer.groupby(['Material', 'year'])['Date'].count()
# -

# There are some year/material combinations with far fewer data points


# ## Plot everything

# +
df[price_cols].hist(bins=20)
plt.show()
# -

# +
fig = px.box(df_longer, y='Price', x='Material', color='year')
fig.show()

fig = px.box(df_longer.sort_values('month'), y='Price', x='Material', color='month')
fig.show()

fig = px.box(df_longer.sort_values('dayofweek'), y='Price', x='Material', color='dayofweek')
fig.show()

fig = px.line(df_longer.sort_values('Date'), x='Date', y='Price', line_group='Material', color='Material',
              height=800)
fig.show()

fig = px.line(df_longer.sort_values('Date'), x='Date', y='Price', line_group='Material', color='Material',
              facet_row='year', height=1400)
fig.update_xaxes(matches=None)
fig.show()
# -

# Plywood and glass have an upward trend over the years, but the other materials look more stable with cyclic patterns


# ## Time series decomposition

# ### Format for time series

# +
df = df.set_index('Date')  # Set the datetimeindex
df = df.resample('B').pad()  # Set sample frequency to every business day
display(df.info())
# -

# ### Fill in NA values

# I need to do this because the time series functions don't like missing values.
# Use forward fill approach because it makes more sense than backward filling here, and using average imputation instead is not recommended for time series.

# +
df_filled = df.copy()
for m in price_cols:
    df_filled[m] = df_filled[m].fillna(method='ffill')
df_filled
# -

# ### Time series decomposition

# +
# Find the number of weeks in the dataset and use this as the period for decomposition
decomposition_period = df_filled[['week', 'year']].drop_duplicates().shape[0]

# m = 'price_rebar'
for m in price_cols:
    result_mul = sm.tsa.seasonal_decompose(df_filled[m],
                                           period=decomposition_period,
                                           model='multiplicative',
                                           extrapolate_trend='freq')
    result_add = sm.tsa.seasonal_decompose(df_filled[m],
                                           period=decomposition_period,
                                           model='additive',
                                           extrapolate_trend='freq')
    plt.rcParams.update({'figure.figsize': (10, 10)})
    result_mul.plot().suptitle('Multiplicative decomposition', fontsize=22)
    result_add.plot().suptitle('Additive decomposition', fontsize=22)
    plt.show()
# -

# There don't appear to be strong trends, but seasonal/cyclic patterns look interesting.

# ## Test for stationarity

# I'm testing for stationarity here, because if the series are not stationary I'll have to make them stationary before using the ARIMA model for forecasting.

# +
adf_pvals = []
for m in price_cols:
    result = sm.tsa.stattools.adfuller(df_filled[m], autolag='AIC')
    adf_pvals.append(result[1])
    print('-' * 25)
    print(m)
    print(f'\tADF Statistic: {result[0]}')
    print(f'\tp-value: {result[1]}')
    for key, value in result[4].items():
        print('\tCritial Values:')
        print(f'\t\t{key}, {value}')

display([(m, p < 0.05) for m, p in zip(price_cols, adf_pvals)])
# -

# The null hypothesis of the ADF test is that the time series is non-stationary, so with a significant p value that means it's stationary. All five materials come out as stationary, so there's not much trend to speak of.
# This means I can go ahead and use ARIMA without stationarizing the data.


# ## Forecast using ARIMA

# I'll be using univariate forecasting with ARIMA

# p: number of lags to use as predictors (order of AR autoregressive term)
# d: probably 0 because the prices are already stationary
# q: number of lagged forecast errors to include (order of MA moving average term)

# ### Determine order of AR term (p)

# p is the number of lags that cross the significance threshold in the partial autocorrelation plot

# +
plt.rcParams.update({'figure.figsize': (10, 5)})
for m in price_cols:
    sm.graphics.tsa.plot_pacf(df_filled[m], lags=100).suptitle(m, fontsize=18)
    plt.show()
# -

# steel: 1
# rebar: 1
# glass: 1
# concrete: 1
# plywood: 1

# ### Determine order of the MA term (q)

# How many MA terms are required to remove autocorrelation?
# The number of lags the cross the significance threshold in the autocorrelation plot

# +
plt.rcParams.update({'figure.figsize': (10, 5)})
for m in price_cols:
    sm.graphics.tsa.plot_acf(df_filled[m], lags=500).suptitle(m, fontsize=18)
    plt.show()
# -

# steel: 30 - 43
# rebar: 27 - 36
# glass: ~200
# concrete: 60 - 80
# plywood: 21

# ### Set parameters for ARIMA

# +
arima_params = {
'price_steel': {'p': 1, 'q': 30},
'price_rebar': {'p': 1, 'q': 27},
'price_glass': {'p': 1, 'q': 200},
'price_concrete': {'p': 1, 'q': 60},
'price_plywood': {'p': 1, 'q': 21}
}
display(arima_params)
# -

# These numbers are pretty high, and it was actually difficult to fit a model with 200 terms in it. So I went more conservative and set q to 10 for everything. Some of the 10 terms weren't significant anyway in the models I ended up fitting, so this seems reasonable.


# ### Get training and test sets

# Split the data into a training set (the first 80%) and a test set (the last 20%)

# +
train_max_idx = round(df_filled.shape[0] * 0.8) - 1
train = df_filled.iloc[:train_max_idx]
train.shape
test = df_filled.iloc[train_max_idx: ]
test.shape

display(train.index[-1] - train.index[0])
display(test.index[-1] - test.index[0])
# -

# Trained on 2900 days and tested on 735 days


# ### Out of time cross validation

# Use the out of time test set to evaluate the forecast values from the fit models.

# +
train_model_fits = []
forecasts = []
plt.rcParams.update({'figure.figsize': (10, 5)})
for m in price_cols:
    # m_order = (arima_params[m]['p'], 0, arima_params[m]['q'])
    m_order = (arima_params[m]['p'], 0, 10)
    model = sm.tsa.arima.ARIMA(train[m], order=m_order)
    model_fit = model.fit()
    train_model_fits.append(model_fit)
    print('-' * 25)
    print(m)
    print(model_fit.summary())
    fc = model_fit.get_prediction(start=test.index[0], end=test.index[-1])
    fc_df = fc.summary_frame()
    forecasts.append(fc)
    # plt.figure(figsize(12, 5), dpi=200)
    plt.plot(train[m], label='training')
    plt.plot(test[m], label='actual')
    plt.plot(fc_df['mean'], label='forecast')
    plt.fill_between(fc_df.index, fc_df['mean_ci_lower'], fc_df['mean_ci_upper'],
                     color='k', alpha=0.15)
    plt.title('Forecast vs. Actual {}'.format(m))
    plt.legend(loc='upper left', fontsize=8)
    plt.show()
# -


# ## Future iterations
# - Add seasonality term
# - Investigate weekly and monthly aggregated data
# - Use grid search and AIC to determine better values for p, d, and q


# ## Appendix

# ### Look for seasonality in autocorrelation plots

# +
plt.rcParams.update({'figure.figsize': (9, 5), 'figure.dpi': 120})
for m in price_cols:
    ax = pd.plotting.autocorrelation_plot(df_filled[m].tolist())

plt.show()
# -

# Not sure how to add labels to this graph, so I'll try the `statmodels` plotting method instead.

# +
for m in price_cols:
    sm.graphics.tsa.plot_acf(df_filled[m], lags=2500).suptitle(m, fontsize=18)
    plt.show()
# -

# There is some strong evidence for seasonality here, so it would be good to go back and use a SARIMA model with a term for seasonality.
