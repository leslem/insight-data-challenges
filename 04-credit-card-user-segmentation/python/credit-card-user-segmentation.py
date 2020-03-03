# # Planning

# ## Challenge
# As a data scientist at a credit card company, I'm charged with helping a senior VP on a major new product offering project.
# The main task is to customer segments. The immediate goal is to identify signup incentives targeted to different user segments, with the longer term goals of attracting new cardholders and reducing signup incentive costs.

# ## Approach
# - Use an unsupervised clustering technique (start with k-means) to identify distinct user segments.
#    - Segments should be based on the card usage characteristics of each cardholder.
# - Identify signup incenctives that are appropriate for each segment (but *no more*)
# - Calculate revenvue if the targeted cards were used by each related segment

# +
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

from sklearn import cluster, metrics, preprocessing
from sklearn.model_selection import train_test_split


pio.templates.default = "plotly_white"
# -


data_dir = '~/devel/insight-data-challenges/04-credit-card-user-segmentation/data'

# ## Read in and clean the user data

# +
users = pd.read_csv(
    os.path.join(os.path.expanduser(data_dir), 'cc_info - cc_info.csv.tsv'),
    sep='\t'
    # parse_dates=['timestamp']
)

print(users.head())
print(users.info())
# -

# ```
# CUST_ID : Credit card holder ID
# BALANCE : Monthly average balance (based on daily balance averages)
# BALANCE_FREQUENCY : Ratio of last 12 months with balance
#     e.g. 3 / 12 months with balance
# PURCHASES : Total purchase amount spent during last 12 months
# ONEOFF_PURCHASES : Total amount of one-off purchases
# INSTALLMENTS_PURCHASES : Total amount of installment purchases
# CASH_ADVANCE : Total cash-advance amount
# PURCHASES_ FREQUENCY : Frequency of purchases (percentage of months with at least one purchase)
#     e.g. 4 / 12 months with at least one purchase (but is it just out of 12 months? or more?)
# ONEOFF_PURCHASES_FREQUENCY : Frequency of one-off-purchases
#     percentage of months with at least one one-off purchase
# PURCHASES_INSTALLMENTS_FREQUENCY : Frequency of installment purchases
#     percentage of months with at least one installment purchase
# CASH_ADVANCE_FREQUENCY : Cash-Advance frequency
#     percentage of months with at least one cash advance
# AVERAGE_PURCHASE_TRX : Average amount per purchase transaction
# CASH_ADVANCE_TRX : Average amount per cash-advance transaction
# PURCHASES_TRX : Average amount per purchase transaction
# CREDIT_LIMIT : Credit limit
# PAYMENTS : Total payments (due amount paid by the customer to decrease their statement balance) in the period
# MINIMUM_PAYMENTS : Total minimum payments due in the period.
# PRC_FULL_PAYMENT : Percentage of months with full payment of the due statement balance
# TENURE : Number of months as a customer
# ```

# Why is CUST_ID not numeric?
# +
users['CUST_ID'].head()
users['CUST_ID'].tail()
# -
# Ok, it's an integer with a C at the beginning, so that's why it's a string. Not sensible, but ok, I can deal.

# Are there any duplicate user ids?
# +
users['CUST_ID'].duplicated().sum()
# -
# Nope, no duplicates here.

# How do the numeric columns look?
# +
with pd.option_context('display.max_columns', 100):
    print(users.describe())
# -

# Are there any missing values?
# +
column_missing_totals = users.apply(lambda x: sum(x.isna()), axis=0)
print(column_missing_totals)

for v in users.columns:
    if column_missing_totals[v] > 0:
        print(users.loc[users[v].isna(), ['CUST_ID', v, 'TENURE']])
# -

# Remove one value where CREDIT_LIMIT is missing
# +
with pd.option_context('display.max_columns', 100):
    print(users.loc[users['CREDIT_LIMIT'].isna(), ])
# Nothing looks particularly strange about the rest of the values for this user. Can't figure out what's going on, so remove.
users = users.loc[~users['CREDIT_LIMIT'].isna(), ]
# -

# For users missing values for MINIMUM_PAYMENTS, replace NaN with 0
# +
print(users['MINIMUM_PAYMENTS'].min())  # There are no users with $0 min payment due, which doesn't seem right.
with pd.option_context('display.max_columns', 100):
    print(users.loc[users['MINIMUM_PAYMENTS'].isna(), ])

users.loc[users['MINIMUM_PAYMENTS'].isna(), 'MINIMUM_PAYMENTS'] = 0
# -

# Ok, now plot everything to get a look at it.
# +
users_melted = users.melt(id_vars=['CUST_ID'])
fig = px.histogram(users_melted, x='value', facet_col='variable', facet_col_wrap=4, nbins=20)
fig.update_xaxes(matches=None)
fig.update_yaxes(matches=None)
fig.show()
# -

# Plotly is stupid about labeling axes on different scales in a faceted plot. Also there are some variables that look like they have categorical or discrete values. I think it's using the same bins for every variable. Stupid!

# +
values_per_variable = users.apply('nunique', 0)
for v in users.columns:
    if v == 'CUST_ID':
        continue
    # Baplots for variables with fewer variables.
    # v = 'BALANCE_FREQUENCY'
    if values_per_variable[v] <= 100:
        tmp = users[v].value_counts().to_frame().reset_index()
        fig = px.bar(tmp, x='index', y=v)
        fig.show()
# -
# But `TENURE` is the only variable that is actually a discrete number with few values (best as barplot). The rest just have < 100 unique values because of common rounding / division patterns.

# +
barplot_vars = ['TENURE']
id_vars = ['CUST_ID']
histogram_vars = set(users.columns) - set(barplot_vars) - set(id_vars)

for v in barplot_vars:
    tmp = users[v].value_counts().to_frame().reset_index()
    fig = px.bar(tmp, x='index', y=v)
    fig.update_layout(xaxis_title=v, yaxis_title='Frequency')
    fig.show()

for v in histogram_vars:
    fig = px.histogram(users, x=v, nbins=50)
    fig.show()
# -

# Variables that have weird histograms
# PURCHASES_FREQUENCY
# BALANCE_FREQUENCY
# PURCHASES_INSTALLMENTS_FREQUENCY
# PRC_FULL_PAYMENT
# ONEOFF_PURCHASES_FREQUENCY
# CASH_ADVANCE_FREQUENCY

# +
frequency_vars = [v for v in users.columns if v.endswith('FREQUENCY')]
for v in frequency_vars + ['PRC_FULL_PAYMENT']:
    sns.distplot(users[v], rug=True, hist=True)
    plt.show()

values_per_variable[frequency_vars]
# -

# I can't really figure out the exact way that the frequency variables were calculated from these variable descriptions.
# Is a density plot better?
# It's hard to say and I don't think Plotly has great options here.

# Are all of the "frequency" variables divided by 12?
# No, they're not.
# Is the TENURE variable the divisor in the FREQUENCY variables?
# +
for v in frequency_vars:
    print('-' * 50)
    print(v)
    print((users[v] * users['TENURE']).round(0).value_counts())
    print(users[v].value_counts())
# -
# Yes, it looks like it is!


# Ok, here's the best visualization of the frequency variables.
# +
for v in frequency_vars + ['PRC_FULL_PAYMENT']:
    fig = px.histogram(users, x=v, nbins=20)
    fig.show()
# -

# And try a pair plot just for fun.
# +
fig = px.scatter_matrix(users)
fig.show()
# -
# That's really not useful.


# Standardize variables for k-means clustering

# Many variables are skewed, and will behave better if they are log-transformed
# Are there any that ARE NOT skewed? No, they are all skewed.
# +
users_log_transformed = users.copy()
vars_to_log_transform = users.columns.to_list()
vars_to_log_transform.remove('CUST_ID')

np.log10(users_log_transformed[vars_to_log_transform])
# -
# Log transform doesn't work because there are many 0 values in some of these variables

# +
# Find variables to standardize
users_summary = users.drop('CUST_ID', axis=1).apply(['std', 'mean', 'median', 'min', 'max'], axis=0)
with pd.option_context("display.max_columns", 100):
    print(users_summary)

users_summary.loc['min'] < 0
users_summary.loc['max'] > 1.0
non_std_min = users_summary.loc['min'] < 0
non_std_max = users_summary.loc['max'] > 1.0

vars_to_standardize = non_std_max | non_std_min
vars_to_standardize = vars_to_standardize.index[vars_to_standardize].to_list()
# -

# +
users_standardized = users.copy()

# This is not a fast way
for v in vars_to_standardize:
    users_standardized[v] = (users_standardized[v] - users_standardized[v].mean()) / users_standardized[v].std()

users_standardized_summary = users_standardized.drop('CUST_ID', axis=1).apply(
    ['std', 'mean', 'median', 'min', 'max'], axis=0)

with pd.option_context("display.max_columns", 100):
    print(users_standardized_summary)
# -

# Try using sklearn scaling
# +
users_standardized = users.copy()
vars_scaled = preprocessing.minmax_scale(users_standardized[vars_to_standardize])

scaler = preprocessing.MinMaxScaler()
users_standardized[vars_to_standardize] = scaler.fit_transform(users_standardized[vars_to_standardize])
# -

# +
non_id_vars = users.columns.to_list()
non_id_vars.remove('CUST_ID')

for v in non_id_vars:
    fig = px.histogram(users_standardized, x=v, nbins=20)
    fig.show()
# -
# The min max scaling looks good, but I will need to redo it for train/test splits (can't transform the training data using information from the test data or that's peeking!)

# Feature selection
# in the future, this would be a good approach to take: https://www.stat.berkeley.edu/~mmahoney/pubs/NIPS09.pdf

# Examine correlation between features
# +
feature_correlations = users_standardized[non_id_vars].corr()
sns.heatmap(feature_correlations.abs(), annot=True)
plt.show()
# -
# A few of the similar features (e.g. purchases and one-off purchases) are highly correlated
# Concerning correlations:
# ONEOFF_PURCHASES, PURCHASES (0.92)
# CASH_ADVANCE_TRX, CASH_ADVANCE_FREQUENCY (0.8)
# PURCHASES, PURCHASES_TRX (0.69)
# CASH_ADVANCE_TRX, CASH_ADVANCE (0.66)
# PURCHASES_TRX, INSTALLMENTS_PURCHASES (0.63)
# PAYMENTS, PURCHASES (0.6)

# +
corrs = feature_correlations.abs().unstack()
corrs = corrs.sort_values()
corrs = corrs.reset_index()
corrs = corrs.rename(columns={'level_0': 'var1', 'level_1': 'var2', 0: 'correlation'})

fig = px.histogram(corrs, x='correlation', nbins=20)
fig.show()
print(corrs.loc[(corrs['var1'] != corrs['var2']) & (corrs['correlation'] > 0.5), ])
# -

# ok, so there's some correlation here, but this post suggests that's fine and that PCA is a better solution than dropping variables.
# https://stats.stackexchange.com/questions/62253/do-i-need-to-drop-variables-that-are-correlated-collinear-before-running-kmeans


# K-means clustering on all of the variables.

# +
# k-means cluster single run on one train-test split
X_train, X_test = train_test_split(users, test_size=0.2)

# Just fit the model on the whole data, seems like CV is not commonly used for clustering
users_standardized_np = users_standardized[non_id_vars].to_numpy()
users_standardized_np.shape
k_means = cluster.KMeans(n_clusters=2)
k_means.fit(users_standardized_np)
cluster_assignments = k_means.predict(users_standardized_np).shape

k_means.cluster_centers_
k_means.labels_

# Plot clusters on first two variables
plt.scatter(users_standardized_np[:, 0], users_standardized_np[:, 1])
plt.scatter(k_means.cluster_centers_[:, 0], k_means.cluster_centers_[:, 1], c='red', marker='x')
plt.show()
# Not very informative...


# Fit a k-means model with increasing values of K
# Elbow plot of silhouette score and MSSE to choose best value of K clusters

k_search = range(2, 50)
silhouette_scores = []  # (higher is better)
sses = []  # Inertia = sum of squared error (lower is better)

for k in k_search:
    model = cluster.KMeans(n_clusters=k)
    model.fit(users_standardized_np)
    cluster_assignments = model.predict(users_standardized_np)
    silhouette_scores.append(metrics.silhouette_score(users_standardized_np, cluster_assignments))
    sses.append(model.inertia_)

fig = go.Figure()
fig.add_trace(go.Scatter(x=list(k_search), y=silhouette_scores,
                         mode='lines+markers'))
fig.update_layout(title='Silhouette score with increasing K',
                  xaxis_title='K',
                  yaxis_title='Silhouette score')
fig.show()

fig = go.Figure()
fig.add_trace(go.Scatter(x=list(k_search), y=sses,
                         mode='lines+markers'))
fig.update_layout(title='Inertia (SSE) with increasing K',
                  xaxis_title='K',
                  yaxis_title='Inertia (SSE)')
fig.show()
# -
# My choice of K: 11!

# +
# Fit model on all data to get the clusters fit on all users
final_k = 11
final_model = cluster.KMeans(n_clusters=final_k)
final_model.fit(users_standardized_np)
final_cluster_assignments = final_model.predict(users_standardized_np)
users['segment'] = final_cluster_assignments
# -

# Identify characteristics of the user segments
# Plot boxplots of all features by cluster
# +
# TODO: need to melt this to plot it
fig = px.box(users[non_id_vars + ['segment']], x='')
fig.show()

# -

# +
# Identify incentives for targeting each segment

# -


# Are there any featurs I could engineer here? I think these are already engineered...

# References:
# - [americanexpress](https://www.americanexpress.com/en-us/business/trends-and-insights/articles/using-customer-segmentation-find-high-value-leads/)
# - [github](https://inseaddataanalytics.github.io/INSEADAnalytics/CourseSessions/Sessions45/ClusterAnalysisReading.html)
# - [mckinsey](https://www.mckinsey.com/~/media/mckinsey/dotcom/client_service/financial%20services/latest%20thinking/payments/mop19_new%20frontiers%20in%20credit%20card%20segmentation.ashx)
#    - See slide on page 46 for CC customer segmentation ideas
# - [medium](https://medium.com/@jeffrisandy/investigating-starbucks-customers-segmentation-using-unsupervised-machine-learning-10b2ac0cfd3b)
# - [mixpanel](https://mixpanel.com/blog/2018/05/09/user-segmentation-guide-understanding-customers/)
# - [sas](https://www.sas.com/en_us/customers/citic.html)
# - [sptf](https://sptf.info/images/IND-SPTF-2018-Segmentation.pdf)
# - [towardsdatascience](https://towardsdatascience.com/clustering-algorithms-for-customer-segmentation-af637c6830ac)
# - [uxdesign](https://uxdesign.cc/how-to-think-segmentation-from-day-1-f714df093ccb)
# - https://blog.floydhub.com/introduction-to-k-means-clustering-in-python-with-scikit-learn/