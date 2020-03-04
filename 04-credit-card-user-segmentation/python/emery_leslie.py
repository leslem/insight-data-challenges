# # Credit card user segmentation data challenge

# 2020-03-03

# Leslie Emery

# ## Summary

# ## Challenge
# As a data scientist at a credit card company, I'm charged with helping a senior VP on a major new product offering project.
# The main task is to customer segments. The immediate goal is to identify signup incentives targeted to different user segments, with the longer term goals of attracting new cardholders and reducing signup incentive costs.

# ## Approach
# - Use an unsupervised clustering technique (start with k-means) to identify distinct user segments.
#    - Segments should be based on the card usage characteristics of each cardholder.
# - Identify signup incenctives that are appropriate for each segment (but *no more*)
# - Calculate revenvue if the targeted cards were used by each related segment

# # Takeaways
# - I identified 11 user segments using k-means clustering
# - Some patterns were visible from plotting each feature by segment label, but describing these patterns systematically was difficult. e.g.:
#     - Segment 5 has high cash advance frequency
#     - Segment 5 and 8 have very low tenure
#     - Segment 1 and 9 have high credit limits and higher average purchase amounts

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

# ### Investigate missing values

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
# Nothing looks particularly strange about the rest of the values for this user.
# Can't figure out what's going on, so remove.
users = users.loc[~users['CREDIT_LIMIT'].isna(), ]
# -

# For users missing values for MINIMUM_PAYMENTS, replace NaN with 0

# +
print(users['MINIMUM_PAYMENTS'].min())  # There are no users with $0 min payment due, which doesn't seem right.
with pd.option_context('display.max_columns', 100):
    print(users.loc[users['MINIMUM_PAYMENTS'].isna(), ])

users.loc[users['MINIMUM_PAYMENTS'].isna(), 'MINIMUM_PAYMENTS'] = 0
# -

# Ok, here's the best visualization of the frequency variables.

# +
non_id_vars = users.columns.to_list()
non_id_vars.remove('CUST_ID')

for v in non_id_vars:
    fig = px.histogram(users, x=v, nbins=20)
    fig.show()
# -

# ## Standardize variables for k-means clustering

# Rescale all of the non-frequency features so that they're all on a 0 - 1.0 scale.
# This prevents k-means clustering from being unduly influenced by a few larger-scale features.


# Identify variables that need standardization/rescaling as those outside of the 0 - 1.0 scale.

# +
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
vars_scaled = preprocessing.minmax_scale(users_standardized[vars_to_standardize])

scaler = preprocessing.MinMaxScaler()
users_standardized[vars_to_standardize] = scaler.fit_transform(users_standardized[vars_to_standardize])
# -

# Plot the rescaled data

# +
for v in non_id_vars:
    fig = px.histogram(users_standardized, x=v, nbins=20)
    fig.show()
# -

# ## Feature selection
# in the future, this would be a good approach to take: https://www.stat.berkeley.edu/~mmahoney/pubs/NIPS09.pdf

# Examine correlation between features

# +
feature_correlations = users_standardized[non_id_vars].corr()
# sns.heatmap(feature_correlations.abs(), annot=True)
# plt.show()
# -
# A few of the similar features (e.g. purchases and one-off purchases) are highly correlated

# +
corrs = feature_correlations.abs().unstack()
corrs = corrs.sort_values()
corrs = corrs.reset_index()
corrs = corrs.rename(columns={'level_0': 'var1', 'level_1': 'var2', 0: 'correlation'})

fig = px.histogram(corrs, x='correlation', nbins=20)
fig.show()
print(corrs.loc[(corrs['var1'] != corrs['var2']) & (corrs['correlation'] > 0.5), ])
# -

# OK, so there's some correlation here, but this post suggests that's fine and that PCA is a better solution than dropping variables.

# https://stats.stackexchange.com/questions/62253/do-i-need-to-drop-variables-that-are-correlated-collinear-before-running-kmeans


# ## K-means clustering on all of the variables.

# Get the data in a numpy array for sklearn use.

# +
users_standardized_np = users_standardized[non_id_vars].to_numpy()
users_standardized_np.shape
# -

# Fit a k-means model with increasing values of K
# Elbow plot of silhouette score and MSSE to choose best value of K clusters

# +
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
fig.add_shape(
    dict(
        type="line",
        x0=11,
        y0=min(silhouette_scores) * 0.95,
        x1=11,
        y1=max(silhouette_scores) * 1.05,
        line=dict(
            color="Red",
            width=1
        )
    )
)
fig.update_layout(title='Silhouette score with increasing K',
                  xaxis_title='K',
                  yaxis_title='Silhouette score')
fig.show()

fig = go.Figure()
fig.add_trace(go.Scatter(x=list(k_search), y=sses,
                         mode='lines+markers'))
fig.add_shape(
    dict(
        type="line",
        x0=11,
        y0=min(sses) * 0.95,
        x1=11,
        y1=max(sses) * 1.05,
        line=dict(
            color="Red",
            width=1
        )
    )
)
fig.update_layout(title='Inertia (MSSE) with increasing K',
                  xaxis_title='K',
                  yaxis_title='Inertia (MSSE)')
fig.show()
# -

# I chose K = 11 because this is the point at which the MSSE is starting to plateau, but before the Silhouette score drops drastically.

# Fit model on all data to get the clusters fit on all users

# +
final_k = 11
final_model = cluster.KMeans(n_clusters=final_k)
final_model.fit(users_standardized_np)
final_cluster_assignments = final_model.predict(users_standardized_np)
# -

# Label the original data with the clusters.

# +
users['segment'] = final_cluster_assignments
users['segment'] = users['segment'].astype('category')
users_standardized['segment'] = users['segment']
# -

# ## Identify characteristics of the user segments

# Plot the feature data by segment to look for patterns that can be used to describe each segment.

# +
users_melted = users.melt(id_vars=['CUST_ID', 'segment'])
users_standardized_melted = users_standardized.melt(id_vars=['CUST_ID', 'segment'])
fig = px.box(
    users_standardized_melted,
    x='variable', y='value', color='segment',
    facet_col='variable', facet_col_wrap=5
)
fig.update_xaxes(matches=None)
fig.show()
# -
