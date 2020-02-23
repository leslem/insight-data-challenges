# # Costly conversion data challenge

# 2020-02-24

# Leslie Emery

# ## Summary

# ### The problem:
# - Goal is to evaluate a pricing A/B test
# - Focus on user segmentation and provide insights about user segments that behave differently
# - Condition A: old price $39 (66%)
# - Condition B: new price $59 (33%)
# - Does it make sense to increase the price?
# 
# ### Questions to answer:
# - Should the price be $39 or $59?
# - What are my main findings looking at the data?
# - How long should the test have been run to find significant results?
# 
# ### My approach:
# - Overall data exploration
#     - Plot all of the data
#     - Segment users
#     - Consider the customer funnel
# - Statistical test for A/B difference
# - Cost analysis of the results
# - Power analysis for determining how long to run test
# 
# ### My conclusions:
# - 
# - 
# - 
# - 

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import plotly.express as px


# Read in the data
data_dir = '~/devel/insight-data-challenges/03-costly-conversion/data/Pricing_Test_data'
tests = pd.read_csv(
    os.path.join(os.path.expanduser(data_dir), 'test_results.csv')
)
users = pd.read_csv(
    os.path.join(os.path.expanduser(data_dir), 'user_table.csv')
)

tests.head()
tests.info()

users.head()
users.info()