# # Breast cancer detection data challenge

# 2020-02-08

# Leslie Emery

# ## Executive summary
# The dataset provided describes important characteristics of biopsied breast cells that are either malignant or benign. These data can be used to determine which cell features are the most important drivers of malignancy and to develop a diagnostic test to determine whether future biopsy samples are malignant or not.

# After removing duplicated samples and samples with missing data, I was left with 437 benign and 235 malignant samples. The challenge is a supervised binary classification problem, so I started with a binary logistic regression model. I used a grid search with 5-fold cross-validation optimizing for F1-score to tune the model hyperparameters. The final model has an accuracy of 0.976 and an F1 score of 0.955 (calculated with 5-fold cross-validation). For comparison, a naive model that classifies every sample as benign has an accuracy of 0.631 but an F1 score of 0. To determine how best to reduce the features needed for the model, I used recursive feature elimination with cross-validation.

# Based on this model, the strongest indicator of malignancy in a breast cancer cell is high clump thickness, though most of the other features are also good indicators of malignancy with the exception of single epithelial cell size and uniformity of cell size. The classification model I have produced here can be used by a physician as a diagnostic test. After measuring the same features on future biopsies, a physician could obtain a diagnosis for any future patient.

# Not all of the features are necessary to produce good results. A model with only 8 of the features ('Bare Nuclei', 'Bland Chromatin', 'Clump Thickness', 'Marginal Adhesion', 'Mitoses', 'Normal Nucleoli', 'Single Epithelial Cell Size', 'Uniformity of Cell Shape') has an accuracy of 0.970 and an F1 score of 0.959. This is one way to reduce the cost of feature collection.

# ## Relevant analysis

# +
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
# -

# +
import plotly.io as pio
print(pio.renderers.default)
pio.renderers.default = 'png'
print(pio.renderers.default)
# -


# ### Data preparation

# +
# Read the data and take a look at it
data_dir = '~/devel/insight-data-challenges/01-breast-cancer-detection/data'
cancer_df = pd.read_csv(
    os.path.join(os.path.expanduser(data_dir), 'breast-cancer-wisconsin.txt'),
    index_col='Index',
)
# These are all read in as str (object) dtypes because they have string missing codes
# -

# +
# Remove exact duplicates
old_nrows = cancer_df.shape[0]
cancer_df = cancer_df[~cancer_df.duplicated()]
new_nrows = cancer_df.shape[0]
print('{} rows with duplicated data removed'.format(old_nrows - new_nrows))
# Replace non-standard missing value codes with NaN
cancer_df = cancer_df.replace(['?', 'No idea', '#'], np.NaN)
# Drop rows with any missing data
old_nrows = cancer_df.shape[0]
cancer_df = cancer_df.dropna(how='any')
new_nrows = cancer_df.shape[0]
print('{} rows with missing data removed'.format(old_nrows - new_nrows))

# Replace outcome codes with meaningful labels
cancer_df['Class'] = cancer_df['Class'].replace({'2': 0, '4': 1})
# It's easiest for the use of LogisticRegression later on to use 0/1 coding
# There are some extra values here - 40 and 20, which might be typos for 2 and 4, or might be valid data?

# Now fix the data types
str_columns = cancer_df.select_dtypes(include='object').columns.to_list()
str_columns = [c for c in str_columns if c != 'Class']
cancer_df[str_columns] = cancer_df[str_columns].astype('int32')
# -

# +
# Make a "tidy" version of the data for plotting
cancer_df_tidy = cancer_df.melt(id_vars=['ID', 'Class'])
# -

# ### Data exploration

# +
fig = px.box(cancer_df_tidy, x='variable', y='value', color='Class', facet_col='variable', facet_col_wrap=3,
             labels={'variable': 'Feature', '': '', 'Class': 'Cancer status'},
             title='Some samples are outside of the stated data range, suggesting invalid values')
fig.update_yaxes(matches=None)
fig.update_xaxes(matches=None, showticklabels=False)
fig.show()
# -

# Samples with Class = 20 or 40 all have outlier values for the other variables, outside of the valid ranges specified in the challenge instructions pdf.

# Here I decided to make the conservative choice and remove the samples.

# +
old_shape = cancer_df.shape
outlier_samples = cancer_df['Class'].isin(['20', '40'])
cancer_df = cancer_df.loc[~outlier_samples]
new_shape = cancer_df.shape
print('{} rows removed for outlier values'.format(old_shape[0] - new_shape[0]))

# Remake the tidy dataset
cancer_df_tidy = cancer_df.melt(id_vars=['ID', 'Class'])
print(cancer_df_tidy.head())

# Updated boxplots
fig = px.box(cancer_df_tidy, x='variable', y='value', color='Class', facet_col='variable', facet_col_wrap=3,
             labels={'variable': 'Feature', '': '', 'Class': 'Cancer status'},
             title='After removing invalid value samples')
fig.update_yaxes(matches=None)
fig.update_xaxes(matches=None, showticklabels=False)
fig.show()
# -

# There are some very drastic differences here between the two Class values, so logistic regression should do well. All of the features are already in the same range (0 - 10), so I don't need to worry about feature scaling.

# ### Logistic regression

# I started with a logistic regression classifier, because it's a straightforward choice for a binary classification problem.

# +
# Reformat data for scikit-learn
X = cancer_df[cancer_df.columns.difference(['Class', 'ID'])]
y = cancer_df['Class']
# -

# #### Hyperparameter tuning via grid search with 5-fold CV

# I used a grid search with 5-fold cross-validation to tune hyperparameters for a logistic regression model. Logistic regression can be tuned over the parameter C, which controls whether the model tries to keep coefficients close to 0 or not. You can also tune on L1 vs. L2 regularization ('penalty'). Regularization can prevent overfitting. I'm optimizing for the F1 score in order to balance both precision and recall.

# +
# Create hyperparameter options
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
             x=grid_search_results.index.to_list(), y='mean_test_score',
             labels={'mean_test_score': 'F1 score', 'x': 'Hyperparameter values'},
             title='Choose the parameters that produce the best F1 score')
fig.update_layout(
    yaxis={'range': (0, 1)},
    xaxis={
        'tickmode': 'array',
        'tickvals': grid_search_results.index.to_list(),
        'ticktext': grid_search_results['params_string'],
        'tickangle': 90
    }
)
fig.show()
# -

# #### Exploring the tuned model

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
    title='Magnitue of model coefficients identifies indicators of malignancy',
    labels={'absolute_value': 'Coefficient magnitude', 'direction': 'Coefficient direction', 'variable': 'Feature'}
)
fig.update_layout(yaxis={'categoryorder': 'total ascending'})
fig.show()
# -

# ### Model evaluation

# +
# Split data for testing and training
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=48)
# -

# #### Compare to always classifying as the dominant class

# The trained model should be better than a naive model of always choosing the more frequent class.

# +
dummy_majority_classifier = DummyClassifier(strategy='most_frequent')
dummy_majority_fit = dummy_majority_classifier.fit(X_train, y_train)
dummy_accuracy = dummy_majority_fit.score(X_test, y_test)
dummy_f1 = f1_score(dummy_majority_fit.predict(X_test), y_test)
print('Just classifying everything as malignant gets you {:.3f} accuracy'.format(dummy_accuracy))
print('Just classifying everything as malignant gets you {:.3f} F1 score'.format(dummy_f1))

actual_accuracy = f1_score(y_test, best_classifier.predict(X_test))
actual_f1 = best_classifier.score(X_test, y_test)
print('Tuned logistic regression has {:.3f} accuracy'.format(actual_accuracy))
print('Tuned logistic regression has {:.3f} F1 score'.format(actual_f1))
# -

# #### Confusion matrix on best model

# The confusion matrix compares True positive, False positive, False negative, and True negative classifications.

# +
print('Naive model confusion matrix:')
print(confusion_matrix(y_test, dummy_majority_fit.predict(X_test)))
print('Tuned logistic regression model confusion matrix:')
print(confusion_matrix(y_test, best_classifier.predict(X_test)))
# -

# In comparison, the naive model has a much higher false negative rate, which would be the worst kind of error to make for cancer diagnosis.

# ### Using the model for predictions

# The code below shows how this model could be used as a diagnostic test on future patients.

# +
# Make a fake patient by randomly selecting a value from each feature
fake_patient = X.apply(np.random.choice, axis=0)
fake_prediction = best_classifier.predict(np.array([fake_patient.to_numpy()]))
# -

# ### How to account for data collection cost for each feature?

# #### Try recursive feature elmination with CV

# RFECV attempts to select the best combination of features by fitting models (with 5-fold CV) recursively eliminating a feature at a time.

# +
rfecv = RFECV(
    estimator=LogisticRegression(solver='liblinear',
                                 C=best_classifier.get_params()['C'],
                                 penalty=best_classifier.get_params()['penalty'],
                                 random_state=48),
    step=1, cv=5, scoring='f1')
rfecv.fit(X, y)

print('Recommended to select the following {} features:'.format(rfecv.n_features_))
print(X.columns[rfecv.support_])

best_feature_subset_classifier = LogisticRegression(
    solver='liblinear',
    C=best_classifier.get_params()['C'],
    penalty=best_classifier.get_params()['penalty'],
    random_state=48
)
best_feature_subset_classifier.fit(X_train, y_train)
minimal_accuracy = f1_score(y_test, best_feature_subset_classifier.predict(X_test))
minimal_f1 = best_feature_subset_classifier.score(X_test, y_test)
print('Minimal tuned logistic regression has {:.3f} accuracy'.format(minimal_accuracy))
print('Minimal tuned logistic regression has {:.3f} F1 score'.format(minimal_f1))
# -
