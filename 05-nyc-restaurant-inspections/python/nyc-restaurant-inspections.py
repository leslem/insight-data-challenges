# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Planning

# ## Challenge
# This is an open-ended challenge to basically come up with something interesting and useful (with a business case!) from the given dataset. Some suggestions include identifying trends or actionable insights, or providing recommendations. The audience could be restaurant customers, inspectors, or restauranteurs.

# ## Approach
# - Read data dictionary and accompanying documentation
# - Read in and clean the data
#     - Fill in missing grade where relevant
# - Plot everything
# - Add derived variables
#     - Gradeable/non-gradeable inspection
#     - Get month, year, day components of inspection date
# - Aggregate data
#    - By establishment
#    - By borough

# - Make a plan for what I will do
# - What factors are most important for determining whether a business will fail inspection or not?
# - What factors are most important for determining whether a business will be closed after inspection failure or not?
# - Need to engineer many features here to get the data needed on failed inspections and business closures/nonclosures
# - Use a random forest classifier to learn more about it

# ## Results

# ## Takeaways

# +
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from datetime import datetime, timedelta
from sklearn .preprocessing import MultiLabelBinarizer

sns.set_style("whitegrid")
# -

# +
data_dir = '~/devel/insight-data-challenges/05-nyc-restaurant-inspections/data'
# -

# ## Read in and clean the user data

# +
inspections = pd.read_csv(
    os.path.join(os.path.expanduser(data_dir), 'DOHMH_New_York_City_Restaurant_Inspection_Results.csv'),
    parse_dates=['INSPECTION DATE', 'GRADE DATE', 'RECORD DATE']
)

print(inspections.info())

with pd.option_context('display.max_columns', 100):
    print(inspections.head(15))
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
print(variable_info)

# Convert columns to categorical
cat_threshold = 110  # If n unique values is below this, it's probably categorical
known_cat_cols = [
    'ACTION', 'BORO', 'GRADE', 'INSPECTION TYPE', 'CRITICAL FLAG', 'CUISINE DESCRIPTION',
    'VIOLATION CODE', 'VIOLATION DESCRIPTION', 'Community Board', 'Council District'
]

variable_info['to_category'] = (variable_info['n_categories'] < cat_threshold)\
                               & (~variable_info['dtype'].isin(('datetime64[ns]', )))
print(variable_info)
# Are there any known categorical variables missing? Or vice versa?
set(variable_info['variable'].loc[variable_info['to_category']].to_list()) - set(known_cat_cols)
set(known_cat_cols) - set(variable_info['variable'].loc[variable_info['to_category']].to_list())

for v in variable_info['variable'].loc[variable_info['to_category']]:
    inspections[v] = inspections[v].astype('category')

print(inspections.info())
variable_info['dtype'] = inspections.dtypes.apply(lambda x: x.name).to_numpy()
# -

# ### Convert zipcode to an int

# +
inspections['ZIPCODE'].describe()
inspections['ZIPCODE'].isna().sum()  # 5500 NaN values, which is why it's not an int. Leave it for now.
# -

# ### Fix missing value codes

# +
inspections['BORO'] = inspections['BORO'].replace('0', np.NaN)

for v in inspections.select_dtypes(include='category').columns:
    print('_' * 20)
    print(v)
    with pd.option_context('display.max_rows', cat_threshold):
        print(inspections[v].value_counts(dropna=False))

new_establishment_inspection_date = datetime(1900, 1, 1)
inspections['INSPECTION DATE'] = inspections['INSPECTION DATE'].replace(new_establishment_inspection_date, pd.NaT)

for v in inspections.select_dtypes(include='datetime').columns:
    print('_' * 20)
    print(v)
    with pd.option_context('display.max_rows', cat_threshold):
        print(inspections[v].value_counts(dropna=False))

with pd.option_context('display.max_columns', 100):
    print(inspections.select_dtypes(include='number').describe())

variable_info['n_missing'] = inspections.apply(lambda x: x.isna().sum()).to_numpy()
# -

# ### Make a map from violation code to violation description

# +
# Check if there's more than one description per violation code, to see if it will work to select the first one
print(
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
    print(violation_descriptions)
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
inspections['is_gradeable'].value_counts(dropna=False)
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
print(inspections.info())
# -

# ## Plot everything

# +
# Try the Pandas built in histogram function, even though it's mediocre
inspections.select_dtypes(exclude='bool').hist()
plt.show()
# And it fails on boolean columns!
# -


# ### Histograms of the numeric variables

# +
g = sns.FacetGrid(
    inspections.select_dtypes(include='number').melt(), col='variable', col_wrap=4,
    sharex=False, sharey=False
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
        print(inspections[v].value_counts(dropna=False))
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

# ### Scatter plot by index of the datetime variables

# +
g = sns.FacetGrid(
    inspections.select_dtypes(include='datetime').melt(), col='variable', col_wrap=2,
    sharex=False, sharey=False
)
g.map(plt.hist, 'value', color='steelblue', bins=20)
plt.show()
# -

# ### Head and tail of the object variables

# +
for v in inspections.select_dtypes(include='object').columns:
    print('_' * 20)
    print(v)
    print(inspections[v].head(15))
    print(inspections[v].tail(15))
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
business_summary.info()

g = sns.FacetGrid(
    business_summary.melt(), col='variable',
    sharex=False, sharey=False
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
# group as well, and that led to an array allocation here. Using observed=True fixed it
initial_inspections = core_inspections.loc[core_inspections['is_initial_inspection'], ].groupby(
    ['CAMIS', 'BORO', 'Census Tract', 'INSPECTION DATE', 'inspection_month', 'inspection_dayofweek',
     'CUISINE DESCRIPTION', 'INSPECTION TYPE'], observed=True).aggregate(
    passed=('GRADE', lambda x: x.iloc[0] == 'A'),
    grade=('GRADE', 'first'),
    has_critical_flag=('CRITICAL FLAG', lambda x: np.any(x == 'Y')),
    n_violations=('VIOLATION CODE', lambda x: x.loc[~x.isna()].nunique()),
    violation_codes=('VIOLATION CODE', lambda x: x.loc[~x.isna()].to_list())
).reset_index()

# Put some plotting in here


# Add one-hot encoding for each violation code
initial_inspections['violation_codes']
mlb = MultiLabelBinarizer()
expanded_violation_codes = mlb.fit_transform(initial_inspections['violation_codes'])
expanded_violation_codes = pd.DataFrame(expanded_violation_codes, columns='violation_' + mlb.classes_)
initial_inspections = pd.concat([initial_inspections, expanded_violation_codes], axis=1)

initial_inspections.info()
# -


# +
closed_actions = (
    'Establishment Closed by DOHMH.  Violations were cited in the following area(s) and those requiring immediate action were addressed.',
    'Establishment re-closed by DOHMH',
)

reinspections = core_inspections.loc[core_inspections['is_reinspection'], ].groupby(
    ['CAMIS', 'BORO', 'Census Tract', 'INSPECTION DATE', 'inspection_month', 'inspection_dayofweek',
     'CUISINE DESCRIPTION', 'INSPECTION TYPE'], observed=True).aggregate(
    passed=('GRADE', lambda x: x.iloc[0] == 'A'),
    grade=('GRADE', 'first'),
    closed=('ACTION', lambda x: x.isin(closed_actions).any()),
    has_critical_flag=('CRITICAL FLAG', lambda x: np.any(x == 'Y')),
    n_violations=('VIOLATION CODE', lambda x: x.loc[~x.isna()].nunique()),
    violation_codes=('VIOLATION CODE', lambda x: x.loc[~x.isna()].to_list())
).reset_index()

# Put some plotting in here



reinspections['violation_codes']
mlb = MultiLabelBinarizer()
expanded_violation_codes = mlb.fit_transform(reinspections['violation_codes'])
expanded_violation_codes = pd.DataFrame(expanded_violation_codes, columns='violation_' + mlb.classes_)
reinspections = pd.concat([reinspections, expanded_violation_codes], axis=1)

reinspections.info()
# -

# ## Find important features for classification of failed inspections

# +

# -



# References:
# - https://medium.com/@sam.weinger/looking-for-borough-bias-in-nyc-restaurant-inspection-results-e15640cd3f97
# - https://www.foodsafetynews.com/2018/05/harvard-researchers-say-fixing-food-safety-inspectors-schedules-could-end-many-violations/


# +
# -

# +
# This code is used to run the .py script from beginning to end in the python interpreter
# with open('python/nyc-restaurant-inspections.py', 'r') as f:
#     exec(f.read())

# plt.close('all')
