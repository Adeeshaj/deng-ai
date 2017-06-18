#%matplotlib inline

from __future__ import print_function
from __future__ import division

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# just for the sake of this blog post!
from warnings import filterwarnings
filterwarnings('ignore')

# load the provided data
train_features = pd.read_csv('data/dengue_features_train.csv',
                             index_col=[0,1,2])

train_labels = pd.read_csv('data/dengue_labels_train.csv',
                           index_col=[0,1,2])

# Seperate data for San Juan
sj_train_features = train_features.loc['sj']
sj_train_labels = train_labels.loc['sj']

print('San Juan')
print('features: ', sj_train_features.shape)
print('labels  : ', sj_train_labels.shape)

sj_train_features.head()

# Remove `week_start_date` string.
sj_train_features.drop('week_start_date', axis=1, inplace=True)

# Null check
pd.isnull(sj_train_features).any()

(sj_train_features
     .ndvi_ne
     .plot
     .line(lw=0.8))

plt.title('Vegetation Index over Time')
plt.xlabel('Time')

sj_train_features.fillna(method='ffill', inplace=True)

print('San Juan')
print('mean: ', sj_train_labels.mean()[0])
print('var :', sj_train_labels.var()[0])

sj_train_labels.hist()

sj_train_features['total_cases'] = sj_train_labels.total_cases

# compute the correlations
sj_correlations = sj_train_features.corr()

# plot san juan
sj_corr_heat = sns.heatmap(sj_correlations)
plt.title('San Juan Variable Correlations')

# San Juan
(sj_correlations
     .total_cases
     .drop('total_cases') # don't compare with myself
     .sort_values(ascending=False)
     .plot
     .barh())
