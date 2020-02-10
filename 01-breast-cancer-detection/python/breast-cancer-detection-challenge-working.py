# # Breast cancer detection data challenge
# 2020-02-08
# See the pdf in the GDrive, which I can't link to.

# Instructions:
# You belong to the data team at a local research hospital. You've been tasked with developing a means to help doctors
# diagnose breast cancer. You've been given data about biopsied breast cells; where it is benign (not harmful) or
# malignant (cancerous).

# 1. What features of a cell are the largest drivers of malignancy?
# 2. How would a physician use your product?
# 3. There is a non-zero cost in time and money to collect each feature about a given cell.
# How would you go about determining the most cost-effective method of detecting malignancy?

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import plotly.express as px

from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# ## Data preparation

# +
# Read the data and take a look at it
data_dir = '~/devel/insight-data-challenges/01-breast-cancer-detection/data'
cancer_df = pd.read_csv(
    os.path.join(os.path.expanduser(data_dir), 'breast-cancer-wisconsin.txt'),
    index_col='Index',
)
# These are all read in as str (object) dtypes because they have string missing codes
print(cancer_df.head())
print(cancer_df.info())
print(cancer_df.describe(include='all'))
# -

# +
# Are there duplicates per ID value?
print(cancer_df['ID'].value_counts())
# -
# Yes, there are 666 unique IDs and almost 16,000 samples

# ### Remove exact duplicates

# +
cancer_df = cancer_df[~cancer_df.duplicated()]
# -

# +
# Examine data values for every column
for v in cancer_df.columns:
    print(cancer_df[v].value_counts())
# -

# There are many missing value codes to account for
# ?, No idea, #

# +
# Replace non-standard missing value codes with NaN
cancer_df = cancer_df.replace(['?', 'No idea', '#'], np.NaN)

# Drop rows with any missing data
old_nrows = cancer_df.shape[0]
cancer_df = cancer_df.dropna(how='any')
new_nrows = cancer_df.shape[0]
print('{} rows with missing data removed'.format(old_nrows - new_nrows))

# Replace outcome codes with meaningful labels
cancer_df['Class'] = cancer_df['Class'].replace({'2': 0, '4': 1})
print(cancer_df['Class'].value_counts())
# It's easiest for the use of LogisticRegression later on to use 0/1 coding
# There are some extra values here - 40 and 20, which might be typos for 2 and 4, or might be valid data?

# Now fix the data types
print(cancer_df.info())
str_columns = cancer_df.select_dtypes(include='object').columns.to_list()
str_columns = [c for c in str_columns if c != 'Class']
cancer_df[str_columns] = cancer_df[str_columns].astype('int32')
print(cancer_df.info())
# -

# +
# Make a "tidy" version of the data
cancer_df_tidy = cancer_df.melt(id_vars=['ID', 'Class'])
print(cancer_df_tidy.head())
# -

# ## Plot all of the variables vs. the outcome

# +
fig = px.box(cancer_df_tidy, x='variable', y='value', color='Class', facet_col='variable', facet_col_wrap=3)
fig.update_yaxes(matches=None)
fig.update_xaxes(matches=None)
fig.show()
# -

# Samples with Class = 20 or 40 all have outlier values for the other variables, outside of the valid ranges specified
# in the pdf.

# Drop these values because I can't be sure that dividing them by ten will give me valid data.
# Make the conservative choice.

# +
old_shape = cancer_df.shape
outlier_samples = cancer_df['Class'].isin(['20', '40'])
cancer_df = cancer_df.loc[~outlier_samples]
new_shape = cancer_df.shape
print(cancer_df.head())
print('{} rows removed for outlier values'.format(old_shape[0] - new_shape[0]))
# -

# +
# Remake the tidy dataset
cancer_df_tidy = cancer_df.melt(id_vars=['ID', 'Class'])
print(cancer_df_tidy.head())
# -

# ## Plot all of the variables vs. the outcome

# +
# Boxplots
fig = px.box(cancer_df_tidy, x='variable', y='value', color='Class', facet_col='variable', facet_col_wrap=3)
fig.update_yaxes(matches=None)
fig.update_xaxes(matches=None)
fig.show()
# -

# There are some very drastic differences here between the two Class values, so logistic regression should do well

# All of the features are already in the same range (0 - 10), so I don't need to worry about feature scaling.

# +
fig = px.scatter_matrix(
    cancer_df,
    dimensions=cancer_df.columns.difference(['Class', 'ID']),
    color='Class', symbol='Class',
    title="Scatter matrix of breast cancer data set"
)
fig.update_traces(diagonal_visible=False)
fig.show()
# -

# This plot technically works, but doesn't look good because of the discrete values.
# A scatter plot is not really appropriate here.

# ## Modeling

# +
# Test and training split
X = cancer_df[cancer_df.columns.difference(['Class', 'ID'])]
y = cancer_df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=48)
# -

# +
# Fit a model on the entire dataset
classifier = LogisticRegression(random_state=48).fit(X, y)
dir(classifier)

fig = px.bar(x=classifier.coef_[0], y=X.columns, title='Coeffcients', orientation='h')
fig.show()
# -

# ### Hyperparameter tuning by grid search
# Logistic regression can be tuned over the parameter C, which controls whether the model tries to keep coefficients
# close to 0 or not. You can also tune on L1 vs. L2 regularization.

# +
# Create hyperparameter options
# All of the hyperparameters are related to regularization, which is a way to penalize non-intercept coefficients
# as a way to prevent overfitting. There is a penalty for any non-zero coefficient for the features.
param_grid = {
    # C = inverse of regularization strength; positive float; smaller C = stronger regularization
    'C': np.logspace(-5, 5, 11),  # Similar to defaults for LogisticRegressionCV
    'penalty': ['l1', 'l2']  # Regularization penalty type; L1 = Lasso, L2 = Ridge
    # L1 = penalized by absolute value of coefficient magnitude
    # L2 = penalized by squared magnitude of coefficient
}

# Do a grid search over the parameter values tried, using 5-fold CV.
# Then evaluate the scores for each set of parameter values.
# Use liblinear solver because it supports both L1 and L2 penalty, and works well on smaller data.
classifier_grid_search = GridSearchCV(
    LogisticRegression(solver='liblinear', random_state=48),
    param_grid, cv=5, scoring='f1', verbose=0
)
grid_search_models = classifier_grid_search.fit(X, y)

grid_search_results = pd.DataFrame(grid_search_models.cv_results_)
grid_search_results['params_string'] = grid_search_results['params'].apply(
    lambda x: 'C={:.3f}<br>Penalty={}'.format(x['C'], x['penalty']))
grid_search_results['mean_test_score']

fig = px.bar(grid_search_results.sort_values(by='rank_test_score', ascending=True),
             x=grid_search_results.index.to_list(), y='mean_test_score')
fig.update_layout(
    yaxis={'range': (0, 1)},
    xaxis={
        'tickmode': 'array',
        'tickvals': grid_search_results.index.to_list(),
        'ticktext': grid_search_results['params_string']
    }
)
fig.show()
# -

# +
best_classifier = grid_search_models.best_estimator_
best_classifier.get_params()

# coef_ gives the coefficients contributing to classes_[1], which is "malignant"
best_classifier.classes_

coefficients_df = pd.DataFrame(data=best_classifier.coef_,
                               columns=X.columns)
coefficients_df = coefficients_df.melt()
coefficients_df['absolute_value'] = coefficients_df['value'].abs()
coefficients_df['direction'] = coefficients_df['value'].apply(np.sign).replace({-1: 'negative', 1: 'positive'})
coefficients_df = coefficients_df.sort_values(by='absolute_value', ascending=True)
fig = px.bar(
    coefficients_df, x='absolute_value', y='variable', color='direction', orientation='h',
    title='Drivers of malignancy',
    labels={'absolute_value': 'Coefficient magnitude', 'direction': 'Coefficient direction', 'variable': ''}
)
fig.update_layout(yaxis={'categoryorder': 'total ascending'})
fig.show()
# -

# ## Explore evaluation metrics

# ### Compare to always classifying as the dominant class

# The data is not so imbalanced since I removed the duplicates
# +
cancer_df['Class'].value_counts()

dummy_majority_classifier = DummyClassifier(strategy='most_frequent')
dummy_majority_fit = dummy_majority_classifier.fit(X_train, y_train)
print('Just classifying everything as malignant gets you {:.3f} accuracy'.format(
    dummy_majority_fit.score(X_test, y_test)))
# -

# ### Confusion matrix on best model

# +
confusion_matrix(y_test, best_classifier.predict(X_test))
# -

# - Precision = proportion of positive calls that are correct; good for optimizing on low FP rate
# - Recall = proportion of all ground truth positives that are called correctly; good for optimizing on low FN rate
# - F-score = harmonic mean of precision and recall (F1-score)

# ### F1-score should be the best for optimizing precision vs. recall in this imbalanced dataset

# +
f1_score(y_test, best_classifier.predict(X_test))
# -

# ## Using the model for predictions

# +
# Make a fake patient by randomly selecting a value from each feature
fake_patient = X.apply(np.random.choice, axis=0)
fake_prediction = best_classifier.predict(np.array([fake_patient.to_numpy()]))
# -

# ## How to account for data collection cost for each feature?

# ### Try recursive feature elmination with CV

# +
rfecv = RFECV(
    estimator=LogisticRegression(solver='liblinear',
                                 C=best_classifier.get_params()['C'],
                                 penalty=best_classifier.get_params()['penalty'],
                                 random_state=48),
    step=1, cv=5, scoring='f1')
rfecv.fit(X, y)

print('F1 scores for each feature:')
print(['{:.3f}'.format(x) for x in rfecv.grid_scores_])

print('Recommended to select {} of the following features:'.format(rfecv.min_features_to_select))
print(X.columns[rfecv.support_])

print('All features except for Uniformity of Cell Size are tied because F1 is high for every one:')
print(pd.DataFrame({'rank': rfecv.ranking_, 'feature': X.columns.to_list()}))
# -

# ### Any individual feature will get comparable accuracy to a model with all features.

# +
cv_classifier = LogisticRegressionCV(solver='liblinear',
                                     Cs=[best_classifier.get_params()['C']],
                                     penalty=best_classifier.get_params()['penalty'],
                                     cv=5, random_state=48, scoring='f1')

single_feature_scores = []
for v in X.columns:
    feature = X[[v]]
    single_feature_model = cv_classifier.fit(feature, y)
    single_feature_scores.append(np.mean(single_feature_model.scores_[1]))

single_feature_scores = pd.DataFrame({'feature': X.columns.to_list(), 'score': single_feature_scores})
single_feature_scores = single_feature_scores.append({'feature': 'All features', 'score': grid_search_models.best_score_},
                                                     ignore_index=True)
single_feature_scores = single_feature_scores.sort_values(by='score', ascending=True)
fig = px.bar(single_feature_scores, x='feature', y='score',
             title='5-fold CV F1 scores for single-feature models compared to a model with all features',
             labels={'feature': 'Feature', 'score': 'F1'})
# fig.update_layout(yaxis={'range': (0.9, 1.0)})
fig.show()
# -

# +
# For executing all code from interpreter:
# with open('breast-cancer-detection-challenge-working.py', 'r') as f:
#     exec(f.read())
# -