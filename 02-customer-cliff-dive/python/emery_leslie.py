# # Customer cliff dive data challenge

# 2020-02-17

# Leslie Emery

# ## Summary

# ### The problem
# The head of the Yammer product team has noticed a precipitous drop in weekly active users, which is one of the main KPIs for customer engagement. What has caused this drop?

# ### My approach and results

# I began by coming up with several questions to investigate:
# - Was there any change in the way that weekly active users is calculated?
#    - This does not appear to be the case. To investigate this, I began by replicating the figure from the dashboard. I calculated a rolling 7-day count of engaged users, making sure to use the same method across the entire time frame covered by the dataset, and it still showed the same drop in engagement.
# - Was there a change in any one particular type of "engagement"?
#     - I looked at a rolling 7-day count of each individual type of engagement action. From plotting all of these subplots, it looks to me like home_page, like_message, login, send_message, and view_inbox are all exhibiting a similar drop around the same time, so it's these underlying events that are driving the drop.
# - Could a change in the user interface be making it more difficult or less pleasant for users?
#     - I couldn't find information in the available datasets to address this question. The `yammer_experiments` data set has information about experiments going on, presumably in the user interface. All of the listed experiments happened in June of 2014, though, which I think is too early to have caused the August drop in engagement.
# - Is this drop a seasonal change that happens around this time every year?
#     - Because the data is only available for the period of time shown in the original dashboard, I can't investigate this question. I'd be very interested to see if there is a pattern of reduced engagement at the end of the summer, perhaps related to vacation or school schedules.
# - Are users visiting the site less because they're getting more content via email?
#     - I calculated 7-day rolling counts of each type of email event, and all email events together. Email events overall went up during the time period immediately before the drop in user engagement. All four types of email events increased during the same period, indicating higher clickthroughs on emails, higher numbers of email open events, and more reengagement and weekly digest emails sent. It could be that the higher number of weekly digests sent out mean that users don't have to visit the site directly as much.
# - Are users disengaging from the site due to too many emails/notifications?
#     - I calculated a rolling 7-day count of emails sent to each user and found that the number of emails sent to each user per 7-day period has increased from 5.4 emails (July 20) to 7.75 emails (August 11). This suggests that an increasing volume of emails sent to individual users could have driven them away from using the site. To investigate this further I would want to look into email unsubscribe rates. If unsubscribe rates have also gone up, then it seems that Yammer is sending too many emails to its users.
#     - To investigate whether the number of emails sent per user is correlated with the number of engaged users, I used a Granger causality test to see if "emails sent per user" could be used to predict "number of engaged users". With a high enough lag, the test statistics might be starting to become significant, but I would want to investigate these test results further before making any recommendations based on them.
# - Is the drop in engagement due to a decrease in new activated users? e.g. they are reaching the end of potential customer base?
#    - I calculated the cumulative number of newly activated users over time, using the activation time for each user in the users table. I wanted to see if customer growth had leveled off. However, I saw that customer growth was still increasing in the same pattern. This was true when using creating date rather than activation date as well.

# What is my recommendation to Yammer?
# I have a few recommendations to Yammer:
#    - Try decreasing the number of emails sent to each individual user to see if this increases engagement. They could try this for a subset of users first.
#    - Investigate email unsubscribe rates to see if they are going up. This would indicate that increased email volume might be making users unhappy.
#    - Compare this data to a wider time range to see if the drop shown here is seasonal.


# +
import matplotlib.pyplot as plt
import numpy as np
import os
import plotly.express as px
import pandas as pd

from scipy import stats
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import grangercausalitytests
# -

data_dir = '/Users/leslie/devel/insight-data-challenges/02-customer-cliff-dive/data'
benn_normal = pd.read_csv(os.path.join(data_dir, 'benn.normal_distribution - benn.normal_distribution.csv.tsv'), sep='\t')
rollup_periods = pd.read_csv(os.path.join(data_dir, 'dimension_rollup_periods - dimension_rollup_periods.csv.tsv'), sep='\t',
                             parse_dates=['time_id', 'pst_start', 'pst_end', 'utc_start', 'utc_end'])
yammer_emails = pd.read_csv(os.path.join(data_dir, 'yammer_emails - yammer_emails.csv.tsv'), sep='\t',
                            parse_dates=['occurred_at'])
yammer_events = pd.read_csv(os.path.join(data_dir, 'yammer_events - yammer_events.csv.tsv'), sep='\t',
                            parse_dates=['occurred_at'])
yammer_experiments = pd.read_csv(os.path.join(data_dir, 'yammer_experiments - yammer_experiments.csv.tsv'), sep='\t',
                                 parse_dates=['occurred_at'])
yammer_users = pd.read_csv(os.path.join(data_dir, 'yammer_users - yammer_users.csv.tsv'), sep='\t',
                           parse_dates=['created_at', 'activated_at'])

# +
benn_normal.info()
benn_normal.head()
benn_normal.describe()

rollup_periods.info()
rollup_periods.head()
rollup_periods.describe()

yammer_emails.info()
yammer_emails.head()
yammer_emails.describe()
yammer_emails['action'].value_counts(dropna=False)
yammer_emails['user_type'].value_counts(dropna=False)

yammer_events.info()
yammer_events.head()
yammer_events.describe()
yammer_events['occurred_at']
yammer_events['event_type'].value_counts(dropna=False)
yammer_events['event_name'].value_counts(dropna=False)
yammer_events['location'].value_counts(dropna=False)
yammer_events['device'].value_counts(dropna=False)
yammer_events['user_type'].value_counts(dropna=False)
yammer_events['user_type'].dtype
# user_type should be an int, but has many missing values, and NaN is a float.
# So convert it to the Pandas Int64 dtype which can accommodate NaNs and ints.
yammer_events = yammer_events.astype({'user_type': 'Int64'})

yammer_experiments.info()
yammer_experiments.head()
yammer_experiments.describe()
yammer_experiments['experiment'].value_counts(dropna=False)
yammer_experiments['experiment_group'].value_counts(dropna=False)
yammer_experiments['location'].value_counts(dropna=False)
yammer_experiments['device'].value_counts(dropna=False)

yammer_users.info()
yammer_users.head()
yammer_users.describe()
yammer_users['language'].value_counts(dropna=False)
yammer_users['state'].value_counts(dropna=False)
yammer_users['company_id'].value_counts(dropna=False)
# -

# ## Initial data investigation
# +
# How many days in the dataset?
yammer_events['occurred_at'].max() - yammer_events['occurred_at'].min()
# 122 days!
rollup_periods['pst_start'].max() - rollup_periods['pst_end'].min()
# 1094 days - way more intervals than needed to tile this events data!

yammer_events = yammer_events.sort_values(by='occurred_at', ascending=True)
small_events = yammer_events.head(int(yammer_events.shape[0]/10)).sample(n=40)
small_events = small_events.sort_values(by='occurred_at', ascending=True)
small_events['occurred_at'].max() - small_events['occurred_at'].min()

weekly_rollup_periods = rollup_periods.loc[rollup_periods['period_id'] == 1007]
# -

# +

small_rolling_engagement = small_events.loc[small_events['event_type'] == 'engagement'].rolling(
    '7D', on='occurred_at').count()

# I'm not sure whether rollup_periods are closed on right, left, or both...
# Calculate counts of engagement events in a 7-day rolling window
rolling_engagement_counts = yammer_events.loc[yammer_events['event_type'] == 'engagement'].sort_values(
    by='occurred_at', ascending=True  # Have to sort by "on" column to use rolling()
).rolling('7D', on='occurred_at', min_periods=1).count()

# +
# Use a loop to aggregate on rollup periods
yammer_events['event_name'].unique()
event_range = [min(yammer_events['occurred_at']), max(yammer_events['occurred_at'])]
covered_weekly_rollup_periods = weekly_rollup_periods.loc[(weekly_rollup_periods['pst_end'] <= event_range[1])
                                                          & (weekly_rollup_periods['pst_start'] >= event_range[0])]
# in interval --> start < occurred_at <= end

counts_by_type = None
for (ridx, row) in covered_weekly_rollup_periods.iterrows():
    # row = covered_weekly_rollup_periods.iloc[0]
    # Get egagement events within the period
    df = yammer_events.loc[(yammer_events['occurred_at'] > row['pst_start'])
                           & (yammer_events['occurred_at'] <= row['pst_end'])
                           & (yammer_events['event_type'] == 'engagement')]
    # Count user engagement events
    cbt = df.groupby('event_name').aggregate(event_count=('user_id', 'count')).transpose()
    cbt['pst_start'] = row['pst_start']
    cbt['pst_end'] = row['pst_end']
    cbt['engaged_users'] = df['user_id'].nunique()
    cbt['engagement_event_count'] = df.shape[0]
    if counts_by_type is None:
        counts_by_type = cbt
    else:
        counts_by_type = counts_by_type.append(cbt)

counts_by_type

# +
# Plot engaged users over time
fig = px.scatter(counts_by_type, x='pst_end', y='engaged_users', template='plotly_white')
fig.update_yaxes(range=[0, 1500])
fig.show()

# Plot count of engagement_events over time
fig = px.scatter(counts_by_type, x='pst_end', y='engagement_event_count', template='plotly_white')
fig.show()

# Plot count of individual event types over time
counts_melted = counts_by_type.melt(id_vars=['pst_start', 'pst_end', 'engaged_users', 'engagement_event_count'])
fig = px.scatter(counts_melted, x='pst_end', y='value', template='plotly_white',
                 facet_col='event_name', facet_col_wrap=3, height=1200)
fig.update_yaxes(matches=None)
fig.show()
# -

# Are there any "experiments" messing things up?
yammer_experiments['occurred_at'].describe()

# No, these are all before the issue shows up

# +
# Investigate the sending of emails to user in the same rollup periods
email_counts_by_type = None
for (ridx, row) in covered_weekly_rollup_periods.iterrows():
    # row = covered_weekly_rollup_periods.iloc[0]
    # Get egagement events within the period
    df = yammer_emails.loc[(yammer_events['occurred_at'] > row['pst_start'])
                           & (yammer_events['occurred_at'] <= row['pst_end'])]
    # Count user engagement events
    cbt = df.groupby('action').aggregate(action_count=('user_id', 'count')).transpose()
    cbt['pst_start'] = row['pst_start']
    cbt['pst_end'] = row['pst_end']
    cbt['emailed_users'] = df['user_id'].nunique()
    cbt['email_event_count'] = df.shape[0]
    cbt['emails_sent_per_user'] = df.loc[df['action'].str.startswith('sent_')].groupby(
        'user_id').count().mean()['user_type']
    if email_counts_by_type is None:
        email_counts_by_type = cbt
    else:
        email_counts_by_type = email_counts_by_type.append(cbt)

email_counts_by_type

# +
# Plot emailed users over time
fig = px.scatter(email_counts_by_type, x='pst_end', y='emailed_users', template='plotly_white')
fig.update_yaxes(range=[0, 1500])
fig.show()

# Plot count of email events over time
fig = px.scatter(email_counts_by_type, x='pst_end', y='email_event_count', template='plotly_white')
fig.show()

# Plot count of individual email types over time
email_counts_melted = email_counts_by_type.melt(id_vars=[
    'pst_start', 'pst_end', 'emailed_users', 'email_event_count', 'emails_sent_per_user'])
fig = px.scatter(email_counts_melted, x='pst_end', y='value', template='plotly_white',
                 facet_col='action', facet_col_wrap=2)
fig.update_yaxes(matches=None)
fig.show()
# -

# +
# What is email engagement event count per user? Did that increase?
# +
fig = px.scatter(email_counts_by_type, x='pst_start', y='emails_sent_per_user', template='plotly_white')
fig.show()

p, r = stats.pearsonr(email_counts_by_type['emails_sent_per_user'].to_numpy(),
                      counts_by_type['engaged_users'].to_numpy())
# They do look moderately correlated, but how do I test that one has an effect on the other?
# -


acf_50 = acf(counts_by_type['engaged_users'], nlags=50, fft=True)
pacf_50 = pacf(counts_by_type['engaged_users'], nlags=50)

fig, axes = plt.subplots(1, 2, figsize=(16, 3), dpi=200)
plot_acf(counts_by_type['engaged_users'].tolist(), lags=50, ax=axes[0])
plot_pacf(counts_by_type['engaged_users'].tolist(), lags=50, ax=axes[1])
plt.show()


test_df = pd.DataFrame({'emails_sent_per_user': email_counts_by_type['emails_sent_per_user'].to_numpy(),
                        'engaged_users': counts_by_type['engaged_users'].to_numpy()})
lags = range(20)
caus_test = grangercausalitytests(test_df, maxlag=lags)


# Has there been a dropoff in new users?

# +
yammer_users = yammer_users.sort_values(by='created_at', ascending=True)
yammer_users['cumulative_users'] = pd.Series(np.ones(yammer_users.shape[0]).cumsum())

fig = px.scatter(yammer_users, x='created_at', y='cumulative_users', template='plotly_white')
fig.show()
# Nope, growth is still practicially exponenital

yammer_users['cumulative_activated_users'] = pd.Series(np.ones(yammer_users.shape[0]).cumsum())
fig = px.scatter(yammer_users, x='created_at', y='cumulative_activated_users', template='plotly_white')
fig.show()

yammer_users['company_id'].nunique()
# -
