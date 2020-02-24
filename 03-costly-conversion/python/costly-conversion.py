# # Costly conversion data challenge

# 2020-02-24

# Leslie Emery

# ## Summary

# ### The problem:
# - Goal is to evaluate a pricing A/B test
# - Focus on user segmentation and provide insights about user segments that behave differently
# - Condition A: old price $39 (66%)
# - Condition B: new price $59 (33%)
# - Does it make sense to increase the price?
# 
# ### Questions to answer:
# - Should the price be $39 or $59?
# - What are my main findings looking at the data?
# - How long should the test have been run to find significant results?
# 
# ### My approach:
# - Overall data exploration
#     - Plot all of the data
#     - Segment users (by source, by device, by OS, by user age)
#        - Could segment by day of week or time of day!
#     - Consider the customer funnel
#        - Don't really have data for this, or for geographic segmentation
# - Statistical test for A/B difference (chisq?)
# - Cost analysis of the results
# - Power analysis for determining how long to run test
# 

# - My key metric is conversion rate

# ### My conclusions:
# - 
# - 
# - 
# - 

# References:
# - https://www.priceintelligently.com/blog/bid/180676/why-you-should-never-a-b-test-your-pricing-strategy
# - https://envisionitagency.com/blog/2015/02/pricing-optimization-with-ab-and-multivariate-testing/
# - https://help.optimizely.com/Analyze_Results/How_long_to_run_an_experiment#baseline
# - https://www.invespcro.com/blog/calculating-sample-size-for-an-ab-test/
# - https://datascience.stackexchange.com/questions/11469/how-would-i-chi-squared-test-these-simple-results-from-a-b-experiment
# - https://www.mikulskibartosz.name/how-to-perform-an-ab-test-correctly-in-python/

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import plotly.express as px

# from pygeocoder import Geocoder
from scipy.stats import chi2_contingency, ttest_ind


data_dir = '~/devel/insight-data-challenges/03-costly-conversion/data/Pricing_Test_data'
# Read in and clean the tests data
tests = pd.read_csv(
    os.path.join(os.path.expanduser(data_dir), 'test_results.csv'),
    parse_dates=['timestamp']
)

tests.head()
tests.info()

# Figure out why timestamp column can't be parsed to datetime
tests['timestamp'].head(20)
tests['timestamp'].describe()
dt = pd.to_datetime(tests['timestamp'], errors='coerce')
dt.describe()
tests.loc[dt.isna()].head(50)
tests.loc[dt.isna()].tail(50)
# How many are affected?
sum(dt.isna())
# 10271
sum(dt.isna()) / tests.shape[0]
# ~3%
# This fails
# datetime.strptime(tests['timestamp'].iloc[1053], "%Y-%m-%d %H:%M:%S")

# In the problematic strings, replace all "60" in minutes or seconds with "59"
minutes_60 = re.compile(r'(?P<hour>\d+):(?P<minute>60):(?P<second>\d{2})')
minutes_replace = r'\g<hour>:59:\g<second>'
seconds_60 = re.compile(r'(?P<hour>\d+):(?P<minute>\d{2}):(?P<second>60)')
seconds_replace = r'\g<hour>:\g<minute>:59'

tests['timestamp'] = pd.to_datetime(tests['timestamp'].str.replace(
    minutes_60, minutes_replace).str.replace(
    seconds_60, seconds_replace), errors='raise')

tests.head()
tests.info()
tests.columns

category_columns = [v for v in tests.columns if tests[v].nunique() < 20]
for v in category_columns:
    print(tests[v].value_counts())


# Read in and clean the users data
users = pd.read_csv(
    os.path.join(os.path.expanduser(data_dir), 'user_table.csv')
)

users.head()
users.info()

category_columns = [v for v in users.columns if users[v].nunique() < 100]
for v in category_columns:
    print(users[v].value_counts())

# All users are from the USA and there isn't the data to determine user age.

# # Get the US state name based on lat/long - DOES NOT WORK
# def get_state(row):
#     return Geocoder.reverse_geocode(row['lat'], row['long']).administrative_area_level_1
#
#
# users_row = users.iloc[0]
# Geocoder.reverse_geocode(users_row['lat'], users_row['long']).administrative_area_level_1
# users['us_state'] = users[['lat', 'long']].apply(get_state, axis=1)

# from geopy.geocoders import Nominatim
# geolocator = Nominatim(user_agent="data-challenge")
# users_row = users.iloc[0]
# location = geolocator.reverse('{}, {}'.format(users_row['lat'], users_row['long']))
# dir(location)
# print(location.address)
# location.raw['address']['postcode']
# location.raw['address']['state']
# 
# def get_geo_data(row):
#     location = geolocator.reverse('{}, {}'.format(row['lat'], row['long']))

# This works but has a max rate limit of 1 query per second, which would take about 76 hours to do

# # From https://public.opendatasoft.com/explore/dataset/us-zip-code-latitude-and-longitude/table/
# cities = pd.read_csv(
#     os.path.join(os.path.expanduser(data_dir), 'us-zip-code-latitude-and-longitude.csv'),
#     sep=';'
# )
# 
# cities.info()
# cities.head()
# 
# cities['latitude_round'] = cities['Latitude'].round(2)
# cities['longitude_round'] = cities['Longitude'].round(2)
# 
# users_row
# cities.loc[(cities['latitude_round'] == users_row['lat']) & (cities['longitude_round'] == users_row['long'])]

# Also not very fruitful

# What is the overall conversion rate for each group?
conversions = tests.groupby('price').aggregate(
    conversion_rate=('converted', lambda x: sum(x) / len(x)),
    conversion_count=('converted', 'sum'),
    nonconversion_count=('converted', lambda x: len(x) - sum(x)),
    visitor_count=('user_id', 'count')
)
conversions = conversions.reset_index()
conversions['revenue_per_visitor'] = conversions['conversion_count'] * conversions['price'] / conversions['visitor_count']
print(conversions)
# Even with the decrease in conversion rate, the revenue earned per visitor is up by $0.14

# Is the difference in conversion rate significant?
chi2, pvalue, dof, ex = chi2_contingency(conversions[['conversion_count', 'nonconversion_count']].transpose())
print('The decreased conversion rate of {:.3f} is statististically significant with p={:.3f}'.format(
    conversions['conversion_rate'].diff().max(),
    pvalue
))

# Is the difference in revenue significant?
# this doesn't seem like a valid question to ask here because it's just another version of "are these two numbers different?"

# Plot all of the data!

# Boxplots of each variable by conversion
tests_melted = tests.melt(id_vars=['user_id', 'timestamp', 'converted', 'test', 'price'])
tests_melted_conversion_rate = tests_melted.groupby(['variable', 'value', 'test']).agg(
    conversion_rate=('converted', lambda x: sum(x) / len(x)),
    converted_count=('converted', lambda x: sum(x)),
    unconverted_count=('converted', lambda x: len(x) - sum(x))
)
tests_melted_conversion_rate = tests_melted_conversion_rate.reset_index()
tests_melted_conversion_rate['test_condition'] = tests_melted_conversion_rate['test'].replace({0: 'Old price (A)',
                                                                                               1: 'New price (B)'})

v = 'device'
for v in tests_melted_conversion_rate['variable'].unique():
    fig = px.bar(
        tests_melted_conversion_rate.loc[tests_melted_conversion_rate['variable'] == v].sort_values(
            'conversion_rate', ascending=False),
        x='value', y='conversion_rate', color='test_condition',
        title="Conversion rate by {}".format(v.title()),
        template='plotly_white',
        barmode='group'
    )
    fig.update_xaxes(matches=None)
    fig.update_traces(marker_line_width=3)
    fig.update_layout(bargap=0.1, xaxis_title=v.title())
    fig.show()

for v in tests_melted_conversion_rate['variable'].unique():
    fig = px.bar(
        tests_melted_conversion_rate.loc[tests_melted_conversion_rate['variable'] == v].sort_values(
            'converted_count', ascending=False),
        x='value', y='converted_count', color='test_condition',
        title="Conversion count by {}".format(v.title()),
        template='plotly_white',
        barmode='group'
    )
    fig.update_xaxes(matches=None)
    fig.update_traces(marker_line_width=3)
    fig.update_layout(bargap=0.1, xaxis_title=v.title())
    fig.show()

# What time were these sales happening over?
tests.groupby(['test', 'converted']).agg(Max=('timestamp', 'max'), Min=('timestamp', 'min'))

# Plot conversions over time
tests_sorted = tests.copy()
tests_sorted = tests_sorted.sort_values('timestamp', ascending=True)
tests_1_cumulative_conversions = tests_sorted.loc[tests_sorted['test'] == 1]
tests_1_cumulative_conversions['cumulative_conversions'] = tests_1_cumulative_conversions['converted'].cumsum()
tests_0_cumulative_conversions = tests_sorted.loc[tests_sorted['test'] == 0]
tests_0_cumulative_conversions['cumulative_conversions'] = tests_0_cumulative_conversions['converted'].cumsum()

tidy_cumulative_conversions = tests_1_cumulative_conversions[['timestamp', 'cumulative_conversions', 'test']].append(
    tests_0_cumulative_conversions[['timestamp', 'cumulative_conversions', 'test']])
tidy_cumulative_conversions['test_condition'] = tidy_cumulative_conversions['test'].replace({0: 'Old price (A)',
                                                                                             1: 'New price (B)'})

fig = px.line(
    tidy_cumulative_conversions,
    x='timestamp', y='cumulative_conversions', color='test_condition',
    template='plotly_white'
)
fig.show()





# Power calculations
sample_conversion_rate = tests.loc[tests['test'] == 0, 'converted'].sum() / tests.loc[tests['test'] == 0, 'converted'].shape[0]
alpha = 0.05
mde = 0.2
# Also called "desired lift"
power_target = 0.9  # min improvement for B condition to be worthwhile


from statsmodels.stats.power import GofChisquarePower

analysis = GofChisquarePower()

