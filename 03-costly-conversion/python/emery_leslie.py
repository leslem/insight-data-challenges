# # Costly conversion data challenge

# 2020-02-24

# Leslie Emery

# ## Summary

# ### The problem:
# The VP of product for company XYZ has asked me to evaluate the results from a pricing test on their software product. There are a few main questions to answer:
# 1. Should XYZ sell their software for $39 or $59?
# 2. Are there any other actionable insights that might increase conversion rate?
# 3. How long should the test be run to obtain statistically significant results?

# ### My approach
# The goal is to evaluate the results of an A/B test, where the A condition is the old price ($39) and the B condition is the new, higher price ($59). To determine which price XYZ should sell their software at, I investigated the conversion rate and the revenue earned per visitor to the site for the two different prices. I observed that conversion rate dropped from 2.00% to 1.56% with the higher price, and this drop was significant by a Chi-squared test. However, the revenue earned per visitor increased from $0.78 to $0.92.
# I also investigated different customer segments based on device, traffic source, and OS. I observed some interesting differences to follow up on, including a higher decrease in conversion rate for Web than Mobile devices, Linux compared to other OSes, and for Bing SEO, Yahoo SEO, and Facebook Ads traffic sources.
# Investigating regional customer segments would also be interesting, but rate limitations for reverse geoencoding services prevented me from doing so.

# ### Recommendations
# If revenue earned is the most important metric for XYZ, then I recommend that they increase their software price to $59. While there was a statistically significant drop in conversion rate for the higher price, the revenue earned per visitor to the site is still high enough to make up for the decrease in purchases and increase overall revenue.
# However, because drastic price increases risk alienating customers and significantly decrease conversion rate, I recommend investigating other metrics further before committing to the price change. In particular, I would investigate customer satisfaction, likelihood of customers to recommend the product to others, and customer retention rates. Especially because the highest conversion rates (and amounts) come from friend referral, it would be concerning to jeopardize this important revenue source and it's difficult to know the long term effects of the price change from this short time period.
# This particular experiment did have the power to detect a significant change in conversion rate. However, a more robust way to determine when to stop the experiment is to do a power calculation before running the experiment to determine the desired sample size for the A and B groups. Once the desired sample size is reached, the experiment could be stopped. I did not finish the power calculation.


# +
import os
import pandas as pd
import re
import plotly.express as px

from scipy.stats import chi2_contingency
# -


data_dir = '~/devel/insight-data-challenges/03-costly-conversion/data/Pricing_Test_data'

# ## Read in and clean the tests data

# +
tests = pd.read_csv(
    os.path.join(os.path.expanduser(data_dir), 'test_results.csv'),
    parse_dates=['timestamp']
)

print(tests.head())
print(tests.info())
# -

# A subset of the timestamp field has invalid values for minutes or seconds (60s or 60m).
# I chose to replace these invalid values with 59 instead, which has the effect of decrementing this subset of timestamps by 1 minute or 1 second.

# +
# In the problematic strings, replace all "60" in minutes or seconds with "59"
minutes_60 = re.compile(r'(?P<hour>\d+):(?P<minute>60):(?P<second>\d{2})')
minutes_replace = r'\g<hour>:59:\g<second>'
seconds_60 = re.compile(r'(?P<hour>\d+):(?P<minute>\d{2}):(?P<second>60)')
seconds_replace = r'\g<hour>:\g<minute>:59'

tests['timestamp'] = pd.to_datetime(tests['timestamp'].str.replace(
    minutes_60, minutes_replace).str.replace(
    seconds_60, seconds_replace), errors='raise')

print(tests.head())
print(tests.info())
# -

# What are the unique values of the categorical columns?

# +
category_columns = [v for v in tests.columns if tests[v].nunique() < 20]
for v in category_columns:
    print(tests[v].value_counts())
# -

# ## Read in and clean the users data

# +
users = pd.read_csv(
    os.path.join(os.path.expanduser(data_dir), 'user_table.csv')
)

print(users.head())
print(users.info())
# -

# What are the unique values of the categorical columns?

# +
category_columns = [v for v in users.columns if users[v].nunique() < 100]
for v in category_columns:
    print(users[v].value_counts())
# -

# ## Conversion rate

# +
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
# -

# Even with the decrease in conversion rate, the revenue earned per visitor is up by $0.14


# ## Is the difference in conversion rate significant?

# Run a chi-squared test to detect differences in conversion rate
# -
chi2, pvalue, dof, ex = chi2_contingency(conversions[['conversion_count', 'nonconversion_count']].transpose())
print('The decreased conversion rate of {:.3f} is statististically significant with p={:.3f}'.format(
    conversions['conversion_rate'].diff().max(),
    pvalue
))

# ## Plot all of the data!

# Boxplots of each variable by conversion

# +
tests_melted = tests.melt(id_vars=['user_id', 'timestamp', 'converted', 'test', 'price'])
tests_melted_conversion_rate = tests_melted.groupby(['variable', 'value', 'test']).agg(
    conversion_rate=('converted', lambda x: sum(x) / len(x)),
    converted_count=('converted', lambda x: sum(x)),
    unconverted_count=('converted', lambda x: len(x) - sum(x))
)
tests_melted_conversion_rate = tests_melted_conversion_rate.reset_index()
tests_melted_conversion_rate['test_condition'] = tests_melted_conversion_rate['test'].replace({0: 'Old price (A)',
                                                                                               1: 'New price (B)'})
# -


# Plot conversion rate by segment

# +
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
# -

# Plot conversion counts by segment

# +
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
# -


# Plot conversions over time

# +
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
# -


# ## Power calculation

# +
sample_conversion_rate = tests.loc[tests['test'] == 0, 'converted'].sum() / tests.loc[tests['test'] == 0, 'converted'].shape[0]
alpha = 0.05
mde = 0.2
# Also called "desired lift"
power_target = 0.9  # min improvement for B condition to be worthwhile
# -

# +
from statsmodels.stats.power import GofChisquarePower

analysis = GofChisquarePower()
# -
