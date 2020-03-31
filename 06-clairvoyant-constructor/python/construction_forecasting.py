# # Planning

# ## Challenge
# The challenge is to forecast prices for core construction materials up to 6 months in the future. The manager of the construction company wants to be able to anticipate upcoming increases in material unit prices.

# ## Approach
# - Plot the data to view annual, monthly, weekly, and daily patterns
# - Time series decomposition to identify trends and cyclic/seasonal patterns
# - Fit an ARIMA model to predict prices for 6 months

# ## Results
#

# ## Takeaways
#

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
from statsmodels.tsa.seasonal import seasonal_decompose

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

# ### Reshape the data

# +
# df2 = pd.wide_to_long(df, stubnames='price', i='Date', j='material')
price_cols = df.columns[df.columns.str.startswith('price_')]
df_longer = df.melt(var_name='Material', value_name='Price', id_vars=['Date'], value_vars=price_cols)
df_longer['Material'] = df_longer['Material'].str.replace('price_', '')
display(df_longer)
# -


# ### Remove NaN rows

# +
# old_shape = df.shape
# df = df.dropna(subset=price_cols, how='all')
# new_shape = df.shape
# display(old_shape[0] - new_shape[0])
# -

# +
# old_shape = df_longer.shape
# df_longer = df_longer.dropna(subset=['Price'], how='all')
# new_shape = df_longer.shape
# display(old_shape[0] - new_shape[0])
# -

# ### Remove initial NaN rows

# Many of the first rows in the data are NaN

# +
first_valid_idx = df[price_cols].isna().any(axis=1).idxmin()
df = df.loc[first_valid_idx:]
# -

# Leave the NaNs because I'll need them for the time series aspect

# ## Add variables for date components

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

# price faceted by material, in boxplots by year
# +
# fig = px.box(df_longer, y='Price', x='year', color='Material')
# fig.show()
# 
# fig = px.box(df_longer, y='Price', x='year', facet_col='Material', facet_col_wrap=3)
# fig.show()
# 
fig = px.box(df_longer, y='Price', x='Material', color='year')
fig.show()

fig = px.box(df_longer.sort_values('month'), y='Price', x='Material', color='month')
fig.show()

fig = px.box(df_longer.sort_values('dayofweek'), y='Price', x='Material', color='dayofweek')
fig.show()

fig = px.line(df_longer.sort_values('Date'), x='Date', y='Price', line_group='Material', color='Material',
              height=800)
fig.show()

# Looks like plywood and glass have an upward trend over the years, but the other materials look more stable with cyclic patterns

fig = px.line(df_longer.sort_values('Date'), x='Date', y='Price', line_group='Material', color='Material',
              facet_row='year', height=1400)
fig.update_xaxes(matches=None)
fig.show()

# g = sns.catplot(x='year', y='Price', hue='year', row='Material', data=df_longer, kind='box')
# plt.show()
# 
# for m in df_longer['Material'].unique():
#     g = sns.catplot(x='year', y='Price', hue='year', row='Material', data=df_longer[],
#                     kind='box', facet_kws={'sharey': False})
#     plt.show()
# 
# -

# ## Time series decomposition

# ### Format for time series

# +
df = df.set_index('Date')  # Set the datetimeindex
df = df.resample('B').pad()  # Set sample frequency to every business day
display(df.info())

# Don't make df_longer datetimeindexed because it doesn't work properly with repeated index values
# # Re-create the reshaped data with the new index
# df_longer = df.melt(var_name='Material', value_name='Price',
#                     id_vars=df.drop(price_cols, axis=1).columns,
#                     value_vars=price_cols)
# df_longer['Material'] = df_longer['Material'].str.replace('price_', '')
# display(df_longer)
# display(df_longer.info())
# df_longer.index
# -

# ### Fill in NA values

# Because the time series functions don't like missing values

# +
df_filled = df.copy()
for m in price_cols:
    df_filled[m] = df_filled[m].fillna(method='ffill')
df_filled
# -



# +
# Find the number of weeks in the dataset
decomposition_period = df_filled[['week', 'year']].drop_duplicates().shape[0]
# I'm not quite sure why this makes any sense, but it looks ok?

# m = 'price_rebar'
for m in price_cols:
    result_mul = seasonal_decompose(df_filled[m],
                                    period=decomposition_period,
                                    model='multiplicative',
                                    extrapolate_trend='freq')
    result_add = seasonal_decompose(df_filled[m],
                                    period=decomposition_period,
                                    model='additive',
                                    extrapolate_trend='freq')
    plt.rcParams.update({'figure.figsize': (10, 10)})
    result_mul.plot().suptitle('Multiplicative decomposition', fontsize=22)
    result_add.plot().suptitle('Additive decomposition', fontsize=22)
    plt.show()
# -

# This doesn't look right, but I'm not sure why

# ## Test for stationarity

# +
from statsmodels.tsa.stattools import adfuller

adf_pvals = []
for m in price_cols:
    result = adfuller(df_filled[m], autolag='AIC')
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

# Null hypothesis of the ADF test is that the time series is non-stationary, so with a significant p value that means it's stationary. All five materials come out as stationary, so there's not much trend to speak of.


# ## Look for seasonality in autocorrelation plots

# +
from pandas.plotting import autocorrelation_plot

plt.rcParams.update({'figure.figsize': (9, 5), 'figure.dpi': 120})
for m in price_cols:
    ax = autocorrelation_plot(df_filled[m].tolist())

plt.show()

# Try the statsmodels plotting instead
for m in price_cols:
    sm.graphics.tsa.plot_acf(df_filled[m], lags=2500).suptitle(m, fontsize=18)
    plt.show()
# -


# ## Forecast using ARIMA

# I'll be using univariate forecasting

# p: number of lags to use as predictors (order of AR autoregressive term)
# d: probably 0 because the prices are already stationary
# q: number of lagged forecast errors to include (order of MA moving average term)

# Add S if there is seasonality for SARIMA

# ### Determine order of AR term (p)

# p is the number of lags that cross the significance threshold in the partial autocorrelation plot

# +
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

# How many MA terms required to remove autocorrelation
# The number of lags the cross the significance threshold in the autocorrelation plot

# +
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

# ### Fit initial ARIMA models

# +
for m in price_cols:
    # m_order = (arima_params[m]['p'], 0, arima_params[m]['q'])
    m_order = (arima_params[m]['p'], 0, 10)
    model = sm.tsa.arima.ARIMA(df_filled[m], order=m_order)
    model_fit = model.fit()
    print('-' * 25)
    print(m)
    print(model_fit.summary())

# -

# ### Get training and test sets for at least 6 months

# +
train_max_idx = round(df_filled.shape[0] * 0.8) - 1
train = df_filled.iloc[:train_max_idx]
train.shape
test = df_filled.iloc[train_max_idx: ]
test.shape

df_filled.index[-1] - test.index[0]
# -

# ### Out of time cross validation

# +
train_model_fits = []
forecasts = []
for m in price_cols:
    # m_order = (arima_params[m]['p'], 0, arima_params[m]['q'])
    m_order = (arima_params[m]['p'], 0, 10)
    model = sm.tsa.arima.ARIMA(df_filled[m], order=m_order)
    model_fit = model.fit()
    train_model_fits.append(model_fit)
    print('-' * 25)
    print(m)
    print(model_fit.summary())
    fc = model_fit.get_prediction(train.shape[0])
    fc_df = fc.summary_frame()
    forecasts.append(fc)
    # plt.figure(figsize(12, 5), dpi=200)
    plt.plot(train[m], label='training')
    plt.plot(test[m], label='actual')
    plt.plot(fc_df['mean'], label='forecast')
    plt.fill_between(fc_df.index, fc_df['mean_ci_lower'], fc_df['mean_ci_upper'],
                     color='k', alpha=0.15)
    plt.title('Forecast vs. Actuals')
    plt.legend(loc='upper left', fontsize=8)
    plt.show()
# -












# ### Make data monthly averages

# +
df_monthly = df_filled.copy()
for m in price_cols:
    df_monthly[m] = df_monthly[m].resample('MS').mean()

display(df_monthly)
df_monthly.shape
df_monthly.iloc[50:100,]
df_monthly.iloc[100:150,]
df_monthly.iloc[150:200,]
# looks like there are only valid entries for the first of the month now
df_monthly = df_monthly.dropna()
# -

# Not going to use this...

# ### Grid search over parameters

# +
import itertools

p = (1, )
d = (0, )
q = range(1, 10)
pdq = itertools.product(p, d, q)

m = 'price_plywood'
for param in pdq:
    mod = sm.tsa.arima.ARIMA(df_filled[m])
    results = mod.fit()

# Come back and do this later
# -


# References:

# - https://www.machinelearningplus.com/time-series/time-series-analysis-python/
# - https://docs.google.com/presentation/d/1d39_DpzI7LbKwhQglCm4W_UgJxh2OOwaB-S8s3LDEL4/edit#slide=id.g7e53a4bd5d_0_0
# - [altexsoft](https://www.altexsoft.com/blog/business/time-series-analysis-and-forecasting-novel-business-perspectives/)
# - [towardsdatascience](https://towardsdatascience.com/an-end-to-end-project-on-time-series-analysis-and-forecasting-with-python-4835e6bf050b)
# - https://www.digitalocean.com/community/tutorials/a-guide-to-time-series-visualization-with-python-3
# - https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/

# +
# This code is used to run the .py script from beginning to end in the python interpreter
# with open('python/construction_forecasting.py', 'r') as f:
#     exec(f.read())

# plt.close('all')
