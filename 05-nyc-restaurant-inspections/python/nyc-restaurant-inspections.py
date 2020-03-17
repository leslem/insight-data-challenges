# # Planning

# ## Challenge
# This is an open-ended challenge to find something interesting and useful (with a business case!) from a dataset of New York City's restaurant health inspections. The inspections are performed by the Department of Health and Mental Hygiene (DOHMH). Some suggestions include identifying trends or actionable insights, or providing recommendations. The audience could be restaurant customers, inspectors, or restauranteurs.
# I came up with some questions I was interested in answering:
# 1. What factors contribute to inspection failures?
# 2. Is there any evidence of geographic bias in inspections?
# 3. Is there any evidence of cuisine bias in inspections?
# 4. Is there any evidence of inspection timing affecting results?

# ## Approach
# I cleaned, plotted, and examined the data. Documentation describing the inspection process suggested two possible outcome variables to look into: 1) initial inspection failure and 2) closure after reinspection. I wanted to investigate both, but started with initial inspection failure.
# I investigated both logistic regression and random forest classification models. I chose to focus on the logistic regression results because I wanted to be able to interpret the coefficients and odds ratios. I tuned hyperparameters and evaluated the model using AUC ROC, because it is a good overall summary of model performance, considering all cells of the confusion matrix. A logistic regression model with L2 (ridge) regression and a penalty of 0.1 classifies initial inspection failures with an AUC of 0.932.

# ## Results

# ### 1. What factors contribute to inspection failures?

# Looking at the odds ratios for each of the features in the logistic regression model, here are some of the most important factors affecting initial inspection failure.

# - Features associated with lower odds of passing initial inspection:
#     - Violation codes related to the presence of mice, rats, cockroaches, or flies
#     - Violation codes related to lack of washing facilities, lack of food safety plan, improper food storage temperature, and lack of a required certificate
#     - The borough Queens
#     - Many kinds of cuisine, including Bangladeshi, Indian, Moroccan, Asian, Malaysian, Spanish, African, Turkish, Latin, Chinese, Mediterranean, Hawaiian, Egyptian, Thai, etc.
#     - The number of violations cited

# - Features associated with higher odds of passing initial inspection:
#     - Violation codes with lower stakes issues, such as violation of a recently-introduced ban on styrofoam, improper lighting or ventilation, or reuse of single use items
#     - The borough Staten Island
#     - Many kinds of cuisine including ice cream, hot dogs, donuts, soups/sandwiches, hamburgers, Continental, cafe/coffee/tea shops, juices/smoothies, Ethiopian, steak, sandwiches, bakeries, bagel/pretzel shops, etc. Many of these seem to be shops that would have less food prep and smaller facilities to maintain, so they make sense.
#     - Increasing day of the week

# ### 2. Is there any evidence of geographic bias in inspections?
# Yes, there is some evidence for Queens establishments having lower odds of passing the initial inspection and for Staten Island establishments having higher odds of passing. It's difficult to answer this question without a more sophisticated version of logistic regression to use.

# ### 3. Is there any evidence of cuisine bias in inspections?
# Yes, the cuisine types with the lowest odds of passing the initial inspection include many of the "ethnic" cuisines. Other information is needed to determine if this is a cause or an effect.

# ### 4. Is there any evidence of inspection timing affecting results?
# There might be a slight increase in odds of passing the initial inspection for inspections happening later in the week, but it was slight and of unknown significance. There is no evidence of any effect of the time of year (month) on the odds of passing inspection.

# ## Takeaways
# - Restauranteurs in Queens or those running establishments serving at-risk cuisines (e.g. Bangladeshi, Indian, Moroccan, Malaysian, etc.) should be extra vigilant before inspections.
# - Restauranteurs should pay special attention to the violations most associated with lower odds of passing the inspection, such as presence of vermin, lack of washing facilities, improper food storage temperature, and lack of required certficiations or food safety plans.
# - NYC food inspectors should carefully examine their inspection process to see if it is being affected by bias against certain cuisines.
# - Aspiring restauranteurs could open an ice cream, hot dog, donut, soup & sandwich, or coffee & tea shop to start out with lower odds of failing the initial food saftey inspection.

# +
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns

from datetime import datetime
from IPython.display import display
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve
from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split, StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from treeinterpreter import treeinterpreter as ti


sns.set(style="whitegrid", font_scale=1.25)
plt.figure(figsize=(12.8, 9.6), dpi=400)
# -

# +
data_dir = '~/devel/insight-data-challenges/05-nyc-restaurant-inspections/data'
output_dir = '~/devel/insight-data-challenges/05-nyc-restaurant-inspections/output'
# -

# ## Read in and clean the user data

# +
inspections = pd.read_csv(
    os.path.join(os.path.expanduser(data_dir), 'DOHMH_New_York_City_Restaurant_Inspection_Results.csv'),
    parse_dates=['INSPECTION DATE', 'GRADE DATE', 'RECORD DATE']
)

display(inspections.info())
display(inspections.head(15))
# -

# ### Fix data types

# Find the categorical variables

# +
# Are there any that look categorical based on number of unique values?
values_per_variable = inspections.apply('nunique', 0)
variable_dtypes = inspections.dtypes.apply(lambda x: x.name)
variable_info = pd.DataFrame({'n_categories': values_per_variable,
                              'dtype': variable_dtypes,
                              'variable': values_per_variable.index}).reset_index(drop=True)
display(variable_info)

# Convert columns to categorical
cat_threshold = 110  # If n unique values is below this, it's probably categorical
known_cat_cols = [
    'ACTION', 'BORO', 'GRADE', 'INSPECTION TYPE', 'CRITICAL FLAG', 'CUISINE DESCRIPTION',
    'VIOLATION CODE', 'VIOLATION DESCRIPTION', 'Community Board', 'Council District'
]

variable_info['to_category'] = (variable_info['n_categories'] < cat_threshold)\
                               & (~variable_info['dtype'].isin(('datetime64[ns]', )))
display(variable_info)
# Are there any known categorical variables missing? Or vice versa?
set(variable_info['variable'].loc[variable_info['to_category']].to_list()) - set(known_cat_cols)
set(known_cat_cols) - set(variable_info['variable'].loc[variable_info['to_category']].to_list())

for v in variable_info['variable'].loc[variable_info['to_category']]:
    inspections[v] = inspections[v].astype('category')

display(inspections.info())
variable_info['dtype'] = inspections.dtypes.apply(lambda x: x.name).to_numpy()
# -

# ### Convert zipcode to an int

# +
display(inspections['ZIPCODE'].describe())
display(inspections['ZIPCODE'].isna().sum())  # 5500 NaN values, which is why it's not an int. Leave it for now.
# -

# ### Fix missing value codes

# +
inspections['BORO'] = inspections['BORO'].replace('0', np.NaN)

for v in inspections.select_dtypes(include='category').columns:
    print('_' * 20)
    print(v)
    display(inspections[v].value_counts(dropna=False))

new_establishment_inspection_date = datetime(1900, 1, 1)
inspections['INSPECTION DATE'] = inspections['INSPECTION DATE'].replace(new_establishment_inspection_date, pd.NaT)

for v in inspections.select_dtypes(include='datetime').columns:
    print('_' * 20)
    print(v)
    display(inspections[v].value_counts(dropna=False))

display(inspections.select_dtypes(include='number').describe())

variable_info['n_missing'] = inspections.apply(lambda x: x.isna().sum()).to_numpy()
# -

# ### Make a map from violation code to violation description

# +
# Check if there's more than one description per violation code, to see if it will work to select the first one
display(
    inspections[['VIOLATION CODE', 'VIOLATION DESCRIPTION']].groupby(
        'VIOLATION CODE').aggregate('nunique')['VIOLATION DESCRIPTION'].value_counts()
)
# -

# There are 15 violation codes without any matching description.

# +
inspections['VIOLATION CODE'].nunique()
violation_descriptions = inspections[['VIOLATION CODE', 'VIOLATION DESCRIPTION']].groupby(
    'VIOLATION CODE').aggregate('first')

with pd.option_context('display.max_rows', 200):
    display(violation_descriptions)
# -


# ## Add some derived variables

# ### Use documentation instructions to label gradeable/ungradeable inspections

# +
gradeable_inspection_types = (
    'Cycle Inspection / Initial Inspection',
    'Cycle Inspection / Re-Inspection',
    'Pre-Permit (Operational) / Initial Inspection',
    'Pre-Permit (Operational)/Re-Inspection',
)
gradeable_actions = (
    'Violations were cited in the following area(s).',
    'No violations were recorded at the time of this inspection.',
    'Establishment Closed by DOHMH.',
)
gradeable_inspection_date_min = datetime(2010, 7, 27)

inspections['INSPECTION TYPE'].isin(gradeable_inspection_types).sum()
inspections['ACTION'].isin(gradeable_actions).sum()
np.sum(inspections['INSPECTION DATE'] >= gradeable_inspection_date_min)

inspections['is_gradeable'] = ((inspections['INSPECTION TYPE'].isin(gradeable_inspection_types))
                               & (inspections['ACTION'].isin(gradeable_actions))
                               & (inspections['INSPECTION DATE'] >= gradeable_inspection_date_min)
                               )
display(inspections['is_gradeable'].value_counts(dropna=False))
# -

# ### Add variables for what kind of inspection it was

# +
inspections['INSPECTION TYPE'].value_counts()
inspections['is_cycle_inspection'] = inspections['INSPECTION TYPE'].str.contains('Cycle')
inspections['is_opening_inspection'] = inspections['INSPECTION TYPE'].str.contains(
    'Pre-permit (Operational)', regex=False)
inspections['is_initial_inspection'] = inspections['INSPECTION TYPE'].str.contains('Initial')
inspections['is_reinspection'] = inspections['INSPECTION TYPE'].str.contains('Re-inspection')
inspections['is_compliance_inspection'] = inspections['INSPECTION TYPE'].str.contains('Compliance')
# -

# ### Add variables for date components

# +
inspections['inspection_year'] = inspections['INSPECTION DATE'].dt.year.astype('category')
inspections['inspection_month'] = inspections['INSPECTION DATE'].dt.month.astype('category')
inspections['inspection_day'] = inspections['INSPECTION DATE'].dt.day
inspections['inspection_dayofyear'] = inspections['INSPECTION DATE'].dt.dayofyear
inspections['inspection_dayofweek'] = inspections['INSPECTION DATE'].dt.dayofweek.astype('category')
inspections['inspection_isweekday'] = inspections['inspection_dayofweek'].isin(range(5))
inspections['inspection_week'] = inspections['INSPECTION DATE'].dt.week.astype('category')
display(inspections.info())
# -

# ## Plot everything

# +
# Try the Pandas built in histogram function, even though it's mediocre
inspections.select_dtypes(exclude='bool').hist(figsize=(20, 15))
plt.show()
# And it fails on boolean columns!
# -


# ### Histograms of the numeric variables

# +
g = sns.FacetGrid(
    inspections.select_dtypes(include='number').melt(), col='variable', col_wrap=4,
    sharex=False, sharey=False, height=4
)
g.map(plt.hist, 'value', color='steelblue', bins=20)
plt.show()
# -

# ### Barplots of the categorical & boolean variables

# Individual plots for variables with too many categories

# +
cat_col_n_values = inspections.select_dtypes(include='category').apply('nunique', 0)
many_values_cat_vars = cat_col_n_values.loc[cat_col_n_values > 20].index
other_cat_vars = cat_col_n_values.loc[cat_col_n_values <= 20].index

# for v in many_values_cat_vars:
#     g = sns.countplot(data=inspections, x=v)
#     g.set_xticklabels(g.get_xticklabels(), rotation=60, horizontalalignment='right')
#     plt.tight_layout()
#     plt.show()

# The best is really just a sorted table of value counts.
for v in many_values_cat_vars:
    print('_' * 20)
    print(v)
    with pd.option_context('display.max_rows', cat_threshold):
        display(inspections[v].value_counts(dropna=False))
# -

# A facet grid for those with fewer categories

# +
# tmp = inspections[other_cat_vars].melt()
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
# I can't get the sharex/sharey arguments to work properly. God do I miss ggplot!

for v in other_cat_vars:
    g = sns.countplot(data=inspections, x=v)
    g.set_xticklabels(g.get_xticklabels(), rotation=60, horizontalalignment='right')
    plt.tight_layout()
    plt.show()
# -

# ### Histograms of the datetime variables

# +
g = sns.FacetGrid(
    inspections.select_dtypes(include='datetime').melt(), col='variable', col_wrap=3,
    sharex=False, sharey=False, height=4
)
g.map(plt.hist, 'value', color='steelblue', bins=20)
plt.show()
# -

# ### Head and tail of the object variables

# +
for v in inspections.select_dtypes(include='object').columns:
    print('_' * 20)
    print(v)
    display(inspections[v].head(15))
    display(inspections[v].tail(15))
# -

# ## Filter to most important core inspection types

# +
core_inspections = inspections.loc[(inspections['is_cycle_inspection'] | inspections['is_opening_inspection'])
                                   & (inspections['is_initial_inspection'] | inspections['is_reinspection']), ]
# Make sure it's sorted by ascending inspection date
core_inspections = core_inspections.sort_values('INSPECTION DATE', ascending=True)
# -

# ## Summary of inspections

# ### Summary by business

# +
business_summary = core_inspections.groupby('CAMIS').aggregate(
    n_rows=('CAMIS', 'count'),
    n_inspections=('INSPECTION DATE', 'nunique'),
    avg_inspection_frequency=('INSPECTION DATE', lambda x: np.mean(np.diff(x.unique())).astype('timedelta64[D]'))
)
business_summary['avg_inspection_frequency'] = business_summary['avg_inspection_frequency'].dt.days
display(business_summary.info())

g = sns.FacetGrid(
    business_summary.melt(), col='variable',
    sharex=False, sharey=False, height=4
)
g.map(plt.hist, 'value', color='steelblue', bins=20)
plt.show()
# -

# ### Summary of initial inspection failures

# +
passing_grades = ('A', )
nonpassing_grades = ('B', 'C', )
pending_grades = ('N', 'Z', 'P', )

# Since there are NaNs in both gradeable and ungradeable, I'm going to infer that GRADE of NaN means non-passing
core_inspections.loc[core_inspections['is_gradeable'], 'GRADE'].value_counts(dropna=False)
core_inspections.loc[~core_inspections['is_gradeable'], 'GRADE'].value_counts(dropna=False)

# When using categorical variables in a groupby, Pandas will by default plan to have NaN values for each empty
# group as well, and that led to an array allocation error here. Using observed=True fixed it.
initial_inspections = core_inspections.loc[core_inspections['is_initial_inspection'], ].groupby(
    ['CAMIS', 'BORO', 'INSPECTION DATE', 'inspection_month', 'inspection_dayofweek',
     'CUISINE DESCRIPTION', 'INSPECTION TYPE'], observed=True).aggregate(
    passed=('GRADE', lambda x: x.iloc[0] == 'A'),
    grade=('GRADE', 'first'),
    has_critical_flag=('CRITICAL FLAG', lambda x: np.any(x == 'Y')),
    n_violations=('VIOLATION CODE', lambda x: x.loc[~x.isna()].nunique()),
    violation_codes=('VIOLATION CODE', lambda x: x.loc[~x.isna()].to_list())
).reset_index()

for v in ['passed', 'grade', 'has_critical_flag', 'n_violations']:
    g = sns.countplot(data=initial_inspections, x=v)
    g.set_xticklabels(g.get_xticklabels(), rotation=60, horizontalalignment='right')
    plt.tight_layout()
    plt.show()

# Add one-hot encoding for each violation code, BORO, and CUISINE DESCRIPTION
initial_inspections['violation_codes']
mlb = MultiLabelBinarizer()
expanded_violation_codes = mlb.fit_transform(initial_inspections['violation_codes'])
initial_inspections_violation_code_vars = 'violation_' + mlb.classes_
expanded_violation_codes = pd.DataFrame(expanded_violation_codes, columns=initial_inspections_violation_code_vars)
initial_inspections = pd.concat([initial_inspections, expanded_violation_codes], axis=1)

ohe = OneHotEncoder(sparse=False)
boro_encoding = ohe.fit_transform(initial_inspections['BORO'].to_numpy().reshape(-1, 1))
initial_inspections_boro_vars = 'BORO_' + ohe.categories_[0]
boro_encoding = pd.DataFrame(boro_encoding, columns=initial_inspections_boro_vars)
initial_inspections = pd.concat([initial_inspections, boro_encoding], axis=1)

ohe = OneHotEncoder(sparse=False)
cuisine_encoding = ohe.fit_transform(initial_inspections['CUISINE DESCRIPTION'].to_numpy().reshape(-1, 1))
initial_inspections_cuisine_vars = 'cuisine_' + ohe.categories_[0]
cuisine_encoding = pd.DataFrame(cuisine_encoding, columns=initial_inspections_cuisine_vars)
initial_inspections = pd.concat([initial_inspections, cuisine_encoding], axis=1)

display(initial_inspections.info(max_cols=500))
# -


# +
closed_actions = (
    'Establishment Closed by DOHMH.  Violations were cited in the following area(s) and those requiring immediate action were addressed.',
    'Establishment re-closed by DOHMH',
)

reinspections = core_inspections.loc[core_inspections['is_reinspection'], ].groupby(
    ['CAMIS', 'BORO', 'INSPECTION DATE', 'inspection_month', 'inspection_dayofweek',
     'CUISINE DESCRIPTION', 'INSPECTION TYPE'], observed=True).aggregate(
    passed=('GRADE', lambda x: x.iloc[0] == 'A'),
    grade=('GRADE', 'first'),
    closed=('ACTION', lambda x: x.isin(closed_actions).any()),
    has_critical_flag=('CRITICAL FLAG', lambda x: np.any(x == 'Y')),
    n_violations=('VIOLATION CODE', lambda x: x.loc[~x.isna()].nunique()),
    violation_codes=('VIOLATION CODE', lambda x: x.loc[~x.isna()].to_list())
).reset_index()

# Put some plotting in here
for v in ['passed', 'grade', 'closed', 'has_critical_flag', 'n_violations']:
    g = sns.countplot(data=reinspections, x=v)
    g.set_xticklabels(g.get_xticklabels(), rotation=60, horizontalalignment='right')
    plt.tight_layout()
    plt.show()

reinspections['violation_codes']
mlb = MultiLabelBinarizer()
expanded_violation_codes = mlb.fit_transform(reinspections['violation_codes'])
expanded_violation_codes = pd.DataFrame(expanded_violation_codes, columns='violation_' + mlb.classes_)
reinspections = pd.concat([reinspections, expanded_violation_codes], axis=1)

ohe = OneHotEncoder(sparse=False)
boro_encoding = ohe.fit_transform(reinspections['BORO'].to_numpy().reshape(-1, 1))
reinspections_boro_vars = 'BORO_' + ohe.categories_[0]
boro_encoding = pd.DataFrame(boro_encoding, columns=reinspections_boro_vars)
reinspections = pd.concat([reinspections, boro_encoding], axis=1)

ohe = OneHotEncoder(sparse=False)
cuisine_encoding = ohe.fit_transform(reinspections['CUISINE DESCRIPTION'].to_numpy().reshape(-1, 1))
reinspections_cuisine_vars = 'cuisine_' + ohe.categories_[0]
cuisine_encoding = pd.DataFrame(cuisine_encoding, columns=reinspections_cuisine_vars)
reinspections = pd.concat([reinspections, cuisine_encoding], axis=1)

display(reinspections.info(max_cols=500))
# -

# ## Find important features for classification of failed initial inspections using RandomForest

# ### Prepare data for random forest

# Are there low-variance features that should be removed?

# +
initial_inspections_variances = initial_inspections.var(axis=0)
with pd.option_context('display.max_rows', 200):
    display(initial_inspections_variances.sort_values())

g = sns.distplot(initial_inspections_variances, rug=False)
plt.show()
# -

# I'm not sure how meaningful variance is for categorical variables

# ### Just try running random forest to get it working

# +
id_vars = ['CAMIS', 'INSPECTION DATE', 'INSPECTION TYPE', 'violation_codes', 'BORO', 'CUISINE DESCRIPTION']
label_vars = ['passed', 'grade']
feature_vars = list(set(initial_inspections.columns) - set(id_vars) - set(label_vars))
feature_vars.sort()

X = initial_inspections[feature_vars].to_numpy()
y = initial_inspections[label_vars[0]]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=48)

forest = RandomForestClassifier(n_estimators=20, class_weight='balanced', oob_score=True, random_state=48)
forest.fit(X_train, y_train)
forest.predict(X_test)
forest.score(X_test, y_test)
print(forest.feature_importances_)
print(forest.oob_score_)
# g = sns.barplot(x=forest.feature_importances_, y=feature_vars)
# plt.show()
# -

# This model looks fine, just based on accuracy (89%). Now to do cross-validation...

# ### Cross validation

# +
chosen_metrics = ['accuracy', 'balanced_accuracy', 'f1', 'f1_weighted', 'recall', 'roc_auc']
forest = RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=48)
cv_results = cross_validate(forest, X, y,
                            cv=StratifiedKFold(n_splits=5),
                            return_train_score=False,
                            scoring=chosen_metrics
                            )
cv_means = {k: np.mean(v) for k, v in cv_results.items()}
display(cv_means)

# n_estimators_grid = np.concatenate((np.arange(25, 175, 25), np.arange(200, 600, 100)))
# n_estimators_grid = np.arange(25, 175, 25)
# n_estimators_search = GridSearchCV(
#     estimator=RandomForestClassifier(random_state=48),
#     param_grid={'n_estimators': n_estimators_grid},
#     scoring=chosen_metrics,
#     n_jobs=3,
#     cv=StratifiedKFold(n_splits=5),
#     refit=False
# )
# n_estimators_search.fit(X, y)
# -

# ### Fit the final model

# +
final_forest = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=48)
final_forest.fit(X_train, y_train)
final_predictions = final_forest.predict(X_test)
final_forest.feature_importances_

final_feature_importances = pd.DataFrame({'feature': feature_vars, 'importance': final_forest.feature_importances_})
final_feature_importances = final_feature_importances.sort_values('importance', ascending=False)

plt.figure(figsize=(6, 10), dpi=400)
g = sns.barplot(x='importance', y='feature', data=final_feature_importances.head(40))
plt.tight_layout()
plt.show()

plt.figure(figsize=(12.8, 9.6), dpi=400)
plot_confusion_matrix(final_forest, X_test, y_test, values_format=',.0f')
plt.show()

plot_roc_curve(final_forest, X_test, y_test)
plt.plot([0, 1], [0, 1], 'k-', color='r', label='Chance', linestyle='--')
plt.show()
# -

# ### Interpret the feature contributions

# +
prediction, bias, contributions = ti.predict(final_forest, X_train[1, ].reshape(1, -1))
print("Prediction"), prediction
print("Bias (trainset prior)"), bias
print("Feature contributions:")
for c, feature in zip(contributions[0], feature_vars):
    print(feature, c)
# -

# I don't really find this very helpful. I would need to do this for every sample in the dataset to get something informative.
# Go back to logistic regression so you can actually explain it!

# ## Find important features for classification of failed initial inspections using logistic regression

# ### Fit an initial logistic regression model

# +
logreg = LogisticRegression(random_state=48, max_iter=1000, solver='saga')
logreg.fit(X_train, y_train)
logreg.predict(X_test)
print(logreg.score(X_test, y_test))
# -


# ### Cross validation of a basic logistic regression model

# +
logreg = LogisticRegression(random_state=48, max_iter=1000, solver='saga')
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
display(grid_search_results)

plt.figure(figsize=(8, 8), dpi=400)
g = sns.barplot(x='mean_test_score', y='params_string', data=grid_search_results)
g.set(xlabel='Mean AUC ROC', ylabel='Hyperparameter values')
plt.tight_layout()
plt.show()
plt.figure(figsize=(12.8, 9.6), dpi=400)
# -

# L1 and L2 regularization are bascially both good. And any C higher than 0.1 is good.
# I chose C = 0.1 and penalty = l2 because they're fastest of the ones with good AUC ROC.

# ### Fit the final logistic regression model

# +
final_C = 0.1
final_penalty = 'l2'
final_logreg = LogisticRegression(random_state=48, max_iter=5000, C=final_C, penalty=final_penalty, solver='saga')
final_logreg.fit(X_train, y_train)
final_logreg_predictions = final_logreg.predict(X_test)

plot_confusion_matrix(final_logreg, X_test, y_test, values_format=',.0f')
plt.show()

plot_roc_curve(final_logreg, X_test, y_test)
plt.plot([0, 1], [0, 1], 'k-', color='r', label='Chance', linestyle='--')
plt.show()
# -

# ### Gather coefficients & odds ratios

# coef_ gives the coefficients contributing to classes_[1], which is "True" for passing the initial inspection.

# So coefficients quantify contribution to probability of passing initial inspection.

# Odds ratios greater than 1.0 indicating a higher probability of passing the initial inspection and ORs less than 1.0 indicate a lower probability of passing the initial inspection.

# +
final_coefficients = pd.DataFrame({'feature': feature_vars, 'coefficient': final_logreg.coef_[0]})
# coef_ gives the coefficients contributing to classes_[1], which is passed="True"
final_coefficients['magnitude'] = final_coefficients['coefficient'].abs()
final_coefficients['direction'] = np.sign(final_coefficients['coefficient'])
final_coefficients['direction'] = final_coefficients['direction'].replace({-1.0: 'negative', 1.0: 'positive', 0: 'NA'})
final_coefficients = final_coefficients.sort_values('magnitude', ascending=False)
final_coefficients['odds_ratio'] = np.exp(final_coefficients['coefficient'])
# The odds ratio is the ratio of the odds of passing inspection to the odds of failing inspection. For a given feature, it is the OR when other features are held constant.

with pd.option_context('display.max_rows', 200):
    display(final_coefficients.sort_values('odds_ratio', ascending=False))

# g = sns.barplot(x='magnitude', y='feature', hue='direction', data=final_coefficients.head(40))
# g.set(xlabel='Coefficient magnitude', ylabel='Feature')
# plt.tight_layout()
# plt.show()
# -

# ### Investigate model results

# #### What are odds ratios for the various violation codes?

# +
plt.figure(figsize=(9, 12), dpi=400)
g = sns.barplot(x='odds_ratio', y='feature', hue='direction',
                data=final_coefficients[final_coefficients['feature'].isin(
                    initial_inspections_violation_code_vars)].sort_values('odds_ratio', ascending=True))
g.axvline(1.0)
g.set(xlabel='Odds ratio', ylabel='Feature')
plt.tight_layout()
plt.show()

top_violation_codes = final_coefficients.loc[final_coefficients['feature'].isin(
    initial_inspections_violation_code_vars),
].sort_values('odds_ratio', ascending=False).head(10)['feature'].str.replace('violation_', '')

bottom_violation_codes = final_coefficients.loc[final_coefficients['feature'].isin(
    initial_inspections_violation_code_vars),
].sort_values('odds_ratio', ascending=True).head(10)['feature'].str.replace('violation_', '')

with pd.option_context('display.max_colwidth', 150):
    print('HIGHEST ODDS RATIO VIOLATION CODES - higher odds of passing initial inspection')
    display(violation_descriptions.loc[violation_descriptions.index.isin(top_violation_codes), ])
    print('LOWEST ODDS RATIO VIOLATION CODES - lower odds of passing initial inspection')
    display(violation_descriptions.loc[violation_descriptions.index.isin(bottom_violation_codes), ])
# -

# Investigating the violation code 22G with a missing description:

# https://rules.cityofnewyork.us/tags/sanitary-inspection

# "In the list of unscored violations, a new violation code 22G containing a penalty for violations of Administrative Code §16-329 (c) which prohibits use of expanded polystyrene single service articles, is being added.   "
# https://www1.nyc.gov/office-of-the-mayor/news/295-18/mayor-de-blasio-ban-single-use-styrofoam-products-new-york-city-will-be-effect
# From a recent ban on styrofoam products!

# What is an HACCP plan? 

# "Hazard Analysis Critical Control Points (HACCP) is an internationally recognized method of identifying and managing food safety related risk and, when central to an active food safety program, can provide your customers, the public, and regulatory agencies assurance that a food safety program is well managed." (Source)[https://safefoodalliance.com/food-safety-resources/haccp-overview/]

# #### What are odds ratios for the boroughs?

# +
plt.figure(figsize=(8, 4), dpi=400)
g = sns.barplot(x='odds_ratio', y='feature',
                data=final_coefficients[final_coefficients['feature'].isin(
                    initial_inspections_boro_vars)].sort_values('odds_ratio', ascending=True))
g.set(xlabel='Odds ratio', ylabel='Feature')
g.axvline(1.0)
plt.tight_layout()
plt.show()
# -

# The boroughs are all pretty close to having the same odds of passing initial inspection, though Queens and Staten Island are perhaps a bit different. It's hard to say without p values for the coefficients, which I would need to use a different package or do bootstrapping for.

# #### What are odds ratios for the various cuisines?

# +
plt.figure(figsize=(10, 13), dpi=400)
g = sns.barplot(x='odds_ratio', y='feature', hue='direction',
                data=final_coefficients[final_coefficients['feature'].isin(
                    initial_inspections_cuisine_vars)].sort_values('odds_ratio', ascending=True))
g.axvline(1.0)
g.set(xlabel='Odds ratio', ylabel='Feature')
plt.tight_layout()
plt.show()

top_cuisines = final_coefficients.loc[
    final_coefficients['feature'].isin(initial_inspections_cuisine_vars) & (final_coefficients['odds_ratio'] > 1.0),
].sort_values('odds_ratio', ascending=False)

bottom_cuisines = final_coefficients.loc[
    final_coefficients['feature'].isin(initial_inspections_cuisine_vars) & (final_coefficients['odds_ratio'] < 1.0),
].sort_values('odds_ratio', ascending=True)

with pd.option_context('display.max_rows', 100):
    print('HIGHEST ODDS RATIO CUISINES')
    display(top_cuisines)
    print('LOWEST ODDS RATIO CUISINES')
    display(bottom_cuisines)
# -

# Just by eye, it does appear that many of the "ethnic" food categories are in the lower OR range.
# If cuisine type is the effect, then this could indicate a concerning bias in the inspections. If it is the cause, then it would just mean there are (potentially systemic) reasons for these particular cuisine types to be less likely to pass inspections.

# All of the high OR cuisines make a lot of sense, like ice cream shops, donut shops, cafes, etc. where there is less food prep and less equipment and facilities to maintain.

# #### What about the other features?

# +
other_vars = (set(feature_vars) - set(initial_inspections_boro_vars) - set(('has_critical_flag', ))
              - set(initial_inspections_cuisine_vars) - set(initial_inspections_violation_code_vars))

plt.figure(figsize=(8, 4), dpi=400)
g = sns.barplot(x='odds_ratio', y='feature',
                data=final_coefficients[final_coefficients['feature'].isin(
                    other_vars)].sort_values('odds_ratio', ascending=True))
g.axvline(1.0)
g.set(xlabel='Odds ratio', ylabel='Feature')
plt.tight_layout()
plt.show()

inspections['CRITICAL FLAG'].value_counts(dropna=False)
with pd.option_context('display.max_rows', 100):
    display(inspections.loc[(inspections['CRITICAL FLAG'] == 'Y'), ['VIOLATION DESCRIPTION', 'ACTION']].head(100))
# -

# It looks like something is wrong with 'has_critical_flag' because way too many inspections have this. This is true of the original 'CRITICAL FLAG' variable as well, and the values don't match what's in the DD, so I'm concerned about it.
# It's also concerning that `has_critical_flag` appears to be associated with higher odds of **passing** the initial inspection, by quite a bit (OR=7.3).


# References:
# - https://medium.com/@sam.weinger/looking-for-borough-bias-in-nyc-restaurant-inspection-results-e15640cd3f97
# - https://www.foodsafetynews.com/2018/05/harvard-researchers-say-fixing-food-safety-inspectors-schedules-could-end-many-violations/


# - https://www.researchgate.net/post/Im_trying_to_apply_random_forests_in_a_sparse_data_set_Unfortunately_there_is_more_than_40_error_in_my_result_Can_anyone_suggest_where_to_refine
# - https://medium.com/all-things-ai/in-depth-parameter-tuning-for-random-forest-d67bb7e920d

# - https://towardsdatascience.com/explaining-feature-importance-by-example-of-a-random-forest-d9166011959e
# - https://blog.datadive.net/random-forest-interpretation-with-scikit-learn/
# - https://github.com/andosa/treeinterpreter

# - https://stats.idre.ucla.edu/other/mult-pkg/faq/general/faq-how-do-i-interpret-odds-ratios-in-logistic-regression/
# - https://stackoverflow.com/questions/39626401/how-to-get-odds-ratios-and-other-related-features-with-scikit-learn




# +
# -

# +
# This code is used to run the .py script from beginning to end in the python interpreter
# with open('python/nyc-restaurant-inspections.py', 'r') as f:
#     exec(f.read())

# plt.close('all')
