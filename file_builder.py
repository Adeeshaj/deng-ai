from __future__ import print_function
from __future__ import division

import pandas as pd
import numpy as np

# from matplotlib import pyplot as plt
# import seaborn as sns

# from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# just for the sake of this blog post!

from warnings import filterwarnings
filterwarnings('ignore')

#In 2
# load the provided data
train_features = pd.read_csv('data/train_iq.csv',
                             index_col=[0,1,2])

#In 3
# Seperate data for San Juan
sj_train_features = train_features.loc['iq']

#In 6
# Remove `week_start_date` string.
sj_train_features.drop('week_start_date', axis=1, inplace=True)

#In 7
# Null check
pd.isnull(sj_train_features).any()

#In 9
sj_train_features.fillna(method='ffill', inplace=True)

#In 13
sj_train_features.to_csv("data/missing_filled_train_iq.csv")
