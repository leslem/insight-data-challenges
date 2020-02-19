
# +
import plotly.express as px
import pandas as pd
# -

# +
benn_normal = pd.read_csv('data/benn.normal_distribution - benn.normal_distribution.csv.tsv', sep='\t')
rollup_periods = pd.read_csv('data/dimension_rollup_periods - dimension_rollup_periods.csv.tsv', sep='\t',
                             parse_dates=['time_id', 'pst_start', 'pst_end', 'utc_start', 'utc_end'])
yammer_emails = pd.read_csv('data/yammer_emails - yammer_emails.csv.tsv', sep='\t',
                            parse_dates=['occurred_at'])
yammer_events = pd.read_csv('data/yammer_events - yammer_events.csv.tsv', sep='\t',
                            parse_dates=['occurred_at'])
yammer_experiments = pd.read_csv('data/yammer_experiments - yammer_experiments.csv.tsv', sep='\t',
                                 parse_dates=['occurred_at'])
yammer_users = pd.read_csv('data/yammer_users - yammer_users.csv.tsv', sep='\t',
                           parse_dates=['created_at', 'activated_at'])
# -

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

# ## 1. Replicate the graph in the example
# +

# -