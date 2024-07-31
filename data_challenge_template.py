# # Planning

# ## Challenge

# ## Approach
#

# ## Results
#

# ## Takeaways
#

# +
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from IPython.display import display
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split, StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder



sns.set(style="whitegrid", font_scale=1.25)
plt.figure(figsize=(12.8, 9.6), dpi=400)
# -

# +
data_dir = '~/devel/insight-data-challenges/05-nyc-restaurant-inspections/data'
output_dir = '~/devel/insight-data-challenges/05-nyc-restaurant-inspections/output'
# -

# ## Read in and clean the data

# +
df = pd.read_csv(
    os.path.join(os.path.expanduser(data_dir), 'file.csv'),
    parse_dates=[]
)

display(df.info())

with pd.option_context('display.max_columns', 100):
    display(df.head(15))
# -

# ### Fix data types

# Find the categorical variables

# +
# Are there any that look categorical based on number of unique values?
values_per_variable = df.apply('nunique', 0)
variable_dtypes = df.dtypes.apply(lambda x: x.name)
variable_info = pd.DataFrame({'n_categories': values_per_variable,
                              'dtype': variable_dtypes,
                              'variable': values_per_variable.index}).reset_index(drop=True)
display(variable_info)

# Convert columns to categorical
cat_threshold = 110  # If n unique values is below this, it's probably categorical
known_cat_cols = [
]

variable_info['to_category'] = (variable_info['n_categories'] < cat_threshold)\
                               & (~variable_info['dtype'].isin(('datetime64[ns]', )))
display(variable_info)
# Are there any known categorical variables missing? Or vice versa?
set(variable_info['variable'].loc[variable_info['to_category']].to_list()) - set(known_cat_cols)
set(known_cat_cols) - set(variable_info['variable'].loc[variable_info['to_category']].to_list())

for v in variable_info['variable'].loc[variable_info['to_category']]:
    df[v] = df[v].astype('category')

display(df.info())
variable_info['dtype'] = df.dtypes.apply(lambda x: x.name).to_numpy()
# -

# Fix missing value codes

# +
df[colname] = df[colname].replace('missing code', np.NaN)

for v in df.select_dtypes(include='category').columns:
    display('_' * 20)
    display(v)
    with pd.option_context('display.max_rows', cat_threshold):
        display(df[v].value_counts(dropna=False))

for v in df.select_dtypes(include='datetime').columns:
    display('_' * 20)
    display(v)
    with pd.option_context('display.max_rows', cat_threshold):
        display(df[v].value_counts(dropna=False))

with pd.option_context('display.max_columns', 100):
    display(df.select_dtypes(include='number').describe())

variable_info['n_missing'] = df.apply(lambda x: x.isna().sum()).to_numpy()
# -


# ## Add some derived variables

# +
# -

# ### Add variables for date components

# +
df['inspection_year'] = df['INSPECTION DATE'].dt.year.astype('category')
df['inspection_month'] = df['INSPECTION DATE'].dt.month.astype('category')
df['inspection_day'] = df['INSPECTION DATE'].dt.day
df['inspection_dayofyear'] = df['INSPECTION DATE'].dt.dayofyear
df['inspection_dayofweek'] = df['INSPECTION DATE'].dt.dayofweek.astype('category')
df['inspection_isweekday'] = df['inspection_dayofweek'].isin(range(5))
df['inspection_week'] = df['INSPECTION DATE'].dt.week.astype('category')
display(df.info())
# -

# ## Plot everything

# +
df.select_dtypes(exclude='bool').hist()
plt.show()
# -


# ### Histograms of the numeric variables

g = sns.FacetGrid(
    df.select_dtypes(include='number').melt(), col='variable', col_wrap=4,
    sharex=False, sharey=False
)
g.map(plt.hist, 'value', color='steelblue', bins=20)
plt.show()

# ### Barplots of the categorical & boolean variables

# Individual plots for variables with too many categories

# +
cat_col_n_values = df.select_dtypes(include='category').apply('nunique', 0)
many_values_cat_vars = cat_col_n_values.loc[cat_col_n_values > 20].index
other_cat_vars = cat_col_n_values.loc[cat_col_n_values <= 20].index

# for v in many_values_cat_vars:
#     g = sns.countplot(data=df, x=v)
#     g.set_xticklabels(g.get_xticklabels(), rotation=60, horizontalalignment='right')
#     plt.tight_layout()
#     plt.show()

# The best is really just a sorted table of value counts.
for v in many_values_cat_vars:
    display('_' * 20)
    display(v)
    with pd.option_context('display.max_rows', cat_threshold):
        display(df[v].value_counts(dropna=False))
# -

# A facet grid for those with fewer categories

# +
# tmp = df[other_cat_vars].melt()
# tmp['value_trunc'] = tmp['value'].str.slice(stop=25)
# g = sns.catplot(
#     data=tmp, col='variable', col_wrap=3,
#     x='value_trunc', kind='count',
#     facet_kws={'sharex': False, 'sharey': False},
#     margin_titles=False
# )
# for ax in g.axes.flat:
#     for label in ax.get_xticklabels():
#         label.set_rotation(70)
# plt.show()
# I can't get the sharex/sharey arguments to work properly.

for v in other_cat_vars:
    g = sns.countplot(data=df, x=v)
    g.set_xticklabels(g.get_xticklabels(), rotation=60, horizontalalignment='right')
    plt.tight_layout()
    plt.show()

# -

# ### Histogram of the datetime variables

g = sns.FacetGrid(
    df.select_dtypes(include='datetime').melt(), col='variable', col_wrap=2,
    sharex=False, sharey=False
)
g.map(plt.hist, 'value', color='steelblue', bins=20)
plt.show()

# ### Head and tail of the object variables

# +
for v in df.select_dtypes(include='object').columns:
    display('_' * 20)
    display(v)
    display(df[v].head(15))
    display(df[v].tail(15))
# -


# ## Classification

# ### Train test split

# +
id_vars = []
label_var = []
feature_vars = list(set(df.columns) - set(id_vars) - set(label_var))
feature_vars.sort()

X = df[feature_vars].to_numpy()
y = df[label_var]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=48)
# -

# ### Choose evaluation metrics

# +
# Look at https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
chosen_metrics = ['accuracy', 'balanced_accuracy', 'f1', 'f1_weighted', 'recall', 'roc_auc']
# -

# ### Logistic regression

# +
# Fit one model
logreg = LogisticRegression(random_state=48, max_iter=1000)
logreg.fit(X_train, y_train)
logreg.predict(X_test)
logreg.score(X_test, y_test)

# Do cross-validation
logreg = LogisticRegression(random_state=48, max_iter=1000)
cv_results = cross_validate(logreg, X, y,
                            cv=StratifiedKFold(n_splits=5),
                            return_train_score=False,
                            scoring=chosen_metrics
                            )
cv_means = {k: np.mean(v) for k, v in cv_results.items()}
display(cv_means)
# -

# ### Grid search to tune over regularization hyperparameters

# +
param_grid = {
    # C = inverse of regularization strength; positive float; smaller C = stronger regularization
    'C': (10.0, 4.0, 2.0, 1.0, 0.5, 0.1, 0.01, 0.001),  # Similar to defaults for LogisticRegressionCV
    'penalty': ['l1', 'l2']  # Regularization penalty type; L1 = Lasso, L2 = Ridge
    # L1 = penalized by absolute value of coefficient magnitude
    # L2 = penalized by squared magnitude of coefficient
}

logreg_gridsearch_outfile = os.path.join(os.path.expanduser(output_dir), 'logreg_gridsearch_results.pickle')

# The higher the value of C, the longer the fit takes and the higher the max_iter needed.
# Use saga solver because it is faster for large data and supports both L1 and L2 regularization
if not os.path.exists(logreg_gridsearch_outfile):
    classifier_grid_search = GridSearchCV(
        estimator=LogisticRegression(solver='saga', random_state=48, max_iter=5000),
        param_grid=param_grid,
        cv=5, scoring='roc_auc', verbose=2, n_jobs=3
    )
    grid_search_models = classifier_grid_search.fit(X, y)
    with open(logreg_gridsearch_outfile, 'wb') as f:
        pickle.dump(grid_search_models, f, pickle.HIGHEST_PROTOCOL)
else:
    with open(logreg_gridsearch_outfile, 'rb') as f:
        grid_search_models = pickle.load(f)

grid_search_results = pd.DataFrame(grid_search_models.cv_results_)
grid_search_results['params_string'] = grid_search_results['params'].apply(
    lambda x: 'C={:.3f}\nPenalty={}'.format(x['C'], x['penalty']))
grid_search_results = grid_search_results.sort_values(by='rank_test_score', ascending=True)
with pd.option_context('display.max_columns', 50):
    display(grid_search_results)

plt.figure(figsize=(7, 10), dpi=400)
g = sns.barplot(x='mean_test_score', y='params_string', data=grid_search_results)
g.set(xlabel='Mean AUC ROC', ylabel='Hyperparameter values')
plt.tight_layout()
plt.show()
plt.figure(figsize=(12.8, 9.6), dpi=400)
# -

# +
# Fit the final model
final_logreg = LogisticRegression(random_state=48, max_iter=1000)
final_logreg.fit(X_train, y_train)
final_logreg_predictions = final_logreg.predict(X_test)
final_logreg.coef_
# -

# ### Gather coefficients & odds ratios

# +
final_coefficients = pd.DataFrame({'feature': feature_vars, 'coefficient': final_logreg.coef_[0]})
# coef_ gives the coefficients contributing to classes_[1], which is passed="True"
final_coefficients['magnitude'] = final_coefficients['coefficient'].abs()
final_coefficients['direction'] = np.sign(final_coefficients['coefficient'])
final_coefficients['direction'] = final_coefficients['direction'].replace({-1.0: 'negative', 1.0: 'positive', 0: 'NA'})
final_coefficients = final_coefficients.sort_values('magnitude', ascending=False)
final_coefficients['odds_ratio'] = np.exp(final_coefficients['coefficient'])
# "The odds ratio is defined as the ratio of the odds of A in the presence of B and the odds of A in the absence of B,
# or equivalently (due to symmetry), the ratio of the odds of B in the presence of A and the odds of B in the
# absence of A."
# Here it is the odds of passing the initial inspection in the presence of the feature.

with pd.option_context('display.max_rows', 200):
    display(final_coefficients)

g = sns.barplot(x='magnitude', y='feature', hue='direction', data=final_coefficients.head(40))
g.set(xlabel='Coefficient magnitude', ylabel='Feature')
plt.tight_layout()
plt.show()
# -


final_feature_importances = pd.DataFrame({'feature': feature_vars, 'importance': final_forest.feature_importances_})
final_feature_importances = final_feature_importances.sort_values('importance', ascending=False)

g = sns.barplot(x='importance', y='feature', data=final_feature_importances.head(40))
plt.tight_layout()
plt.show()

# Plot confusion matrix
plot_confusion_matrix(final_logreg, X_test, y_test)
plt.show()
# -


# ### Random forest

# +
# Fit one model
forest = RandomForestClassifier(n_estimators=20, class_weight='balanced', oob_score=True, random_state=48)
forest.fit(X_train, y_train)
forest.predict(X_test)
forest.score(X_test, y_test)
forest.feature_importances_
forest.oob_score_

# Do cross-validation
forest = RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=48)
cv_results = cross_validate(forest, X, y,
                            cv=StratifiedKFold(n_splits=5),
                            return_train_score=False,
                            scoring=chosen_metrics
                            )
cv_means = {k: np.mean(v) for k, v in cv_results.items()}
display(cv_means)

# Fit the final model
final_forest = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=48)
final_forest.fit(X_train, y_train)
final_predictions = final_forest.predict(X_test)
final_forest.feature_importances_

# Plot feature importances
final_feature_importances = pd.DataFrame({'feature': feature_vars, 'importance': final_forest.feature_importances_})
final_feature_importances = final_feature_importances.sort_values('importance', ascending=False)

g = sns.barplot(x='importance', y='feature', data=final_feature_importances.head(40))
plt.tight_layout()
plt.show()

# Plot confusion matrix
plot_confusion_matrix(final_forest, X_test, y_test)
plt.show()
# -


# ## Regression

# ### Train test split

# +
id_vars = []
outcome_var = []
feature_vars = list(set(df.columns) - set(id_vars) - set(outcome_var))
feature_vars.sort()

X = df[feature_vars].to_numpy()
y = df[outcome_var]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=48)
# -

# ### Choose evaluation metrics

# +
# Look at https://scikit-learn.org/stable/modules/model_evaluation.html
chosen_metrics = ['explained_variance', 'r2', 'neg_mean_squared_error']
# -

# ### Linear regression

# +
# Fit one model
linreg = LinearRegression(random_state=48, max_iter=1000)
linreg.fit(X_train, y_train)
linreg.predict(X_test)
linreg.score(X_test, y_test)

# Do cross-validation
linreg = LinearRegression(random_state=48, max_iter=1000)
cv_results = cross_validate(linreg, X, y,
                            return_train_score=False,
                            scoring=chosen_metrics
                            )
cv_means = {k: np.mean(v) for k, v in cv_results.items()}
display(cv_means)

# Fit the final model
final_linreg = LinearRegression(random_state=48, max_iter=1000)
final_linreg.fit(X_train, y_train)
final_linreg_predictions = final_linreg.predict(X_test)
final_linreg.coef_
final_linreg.intercept_
# -

# Gather coefficients

# +
final_coefficients = pd.DataFrame({'feature': feature_vars, 'coefficient': final_logreg.coef_[0]})
# coef_ gives the coefficients contributing to classes_[1], which is passed="True"
final_coefficients['magnitude'] = final_coefficients['coefficient'].abs()
final_coefficients['direction'] = np.sign(final_coefficients['coefficient'])
final_coefficients['direction'] = final_coefficients['direction'].replace({-1.0: 'negative', 1.0: 'positive', 0: 'NA'})
final_coefficients = final_coefficients.sort_values('magnitude', ascending=False)

with pd.option_context('display.max_rows', 200):
    display(final_coefficients)

g = sns.barplot(x='magnitude', y='feature', hue='direction', data=final_coefficients.head(40))
g.set(xlabel='Coefficient magnitude', ylabel='Feature')
plt.tight_layout()
plt.show()
# -


# References:


# +
# This code is used to run the .py script from beginning to end in the python interpreter
# with open('python/nyc-restaurant-df.py', 'r') as f:
#     exec(f.read())

# plt.close('all')
