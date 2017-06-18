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

# Separate data for Iquitos
iq_train_features = train_features.loc['iq']
iq_train_labels = train_labels.loc['iq']

print('\nIquitos')
print('features: ', iq_train_features.shape)
print('labels  : ', iq_train_labels.shape)

# Remove `week_start_date` string.
iq_train_features.drop('week_start_date', axis=1, inplace=True)

iq_train_features.fillna(method='ffill', inplace=True)

print('\nIquitos')
print('mean: ', iq_train_labels.mean()[0])
print('var :', iq_train_labels.var()[0])

iq_train_labels.hist()

iq_train_features['total_cases'] = iq_train_labels.total_cases

# compute the correlations
iq_correlations = iq_train_features.corr()

# plot iquitos
iq_corr_heat = sns.heatmap(iq_correlations)
plt.title('Iquitos Variable Correlations')

# Iquitos
(iq_correlations
     .total_cases
     .drop('total_cases') # don't compare with myself
     .sort_values(ascending=False)
     .plot
     .barh())