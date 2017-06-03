#In 1

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
train_features = pd.read_csv('data/dengue_features_train.csv',
                             index_col=[0,1,2])

train_labels = pd.read_csv('data/dengue_labels_train.csv',
                           index_col=[0,1,2])

# Separate data for Iquitos
iq_train_features = train_features.loc['iq']
iq_train_labels = train_labels.loc['iq']
iq_train_features.drop('week_start_date', axis=1, inplace=True)

#In 7
# Null check
iq_train_features.fillna(method='ffill', inplace=True)

#In 13
iq_train_features['total_cases'] = iq_train_labels.total_cases

#In 14
# compute the correlations
iq_correlations = iq_train_features.corr()

#In 19
def preprocess_data(data_path, labels_path=None):
    # load data and set index to city, year, weekofyear
    df = pd.read_csv(data_path, index_col=[0, 1, 2])
    
    # select features we want
    features = ['reanalysis_specific_humidity_g_per_kg', 
             'reanalysis_dew_point_temp_k',  
             'reanalysis_min_air_temp_k']

    df = df[features]
    
    # fill missing values
    df.fillna(method='ffill', inplace=True)

    # add labels to dataframe
    if labels_path:
        labels = pd.read_csv(labels_path, index_col=[0, 1, 2])
        df = df.join(labels)
    
    # separate san juan and iquitos
    iq = df.loc['iq']
    
    return iq

#In 20
iq_train = preprocess_data('data/dengue_features_train.csv',
                                    labels_path="data/dengue_labels_train.csv")

iq_train_subtrain = iq_train.head(400)
iq_train_subtest = iq_train.tail(iq_train.shape[0] - 400)

#In 24
from statsmodels.tools import eval_measures
import statsmodels.formula.api as smf

def get_best_model(train, test):
    # Step 1: specify the form of the model
    model_formula = "total_cases ~ 1 + " \
                        "reanalysis_specific_humidity_g_per_kg + " \
                        "reanalysis_dew_point_temp_k + " \
                        "reanalysis_min_air_temp_k"
    
    grid = 10 ** np.arange(-8, -3, dtype=np.float64)
                    
    best_alpha = []
    best_score = 1000
        
    # Step 2: Find the best hyper parameter, alpha
    for alpha in grid:
        model = smf.glm(formula=model_formula,
                        data=train,
                        family=sm.families.NegativeBinomial(alpha=alpha))

        results = model.fit()
        predictions = results.predict(test).astype(int)
        score = eval_measures.meanabs(predictions, test.total_cases)

        if score < best_score:
            best_alpha = alpha
            best_score = score


    # Step 3: refit on entire dataset
    full_dataset = pd.concat([train, test])
    model = smf.glm(formula=model_formula,
                    data=full_dataset,
                    family=sm.families.NegativeBinomial(alpha=best_alpha))

    fitted_model = model.fit()
    return fitted_model
    
iq_best_model = get_best_model(iq_train_subtrain, iq_train_subtest)
iq_test = preprocess_data('data/dengue_features_test.csv')

iq_predictions = iq_best_model.predict(iq_test).astype(int)

submission = pd.read_csv("data/submission_format.csv",
                         index_col=[0, 1, 2])

submission.total_cases = np.concatenate([iq_predictions])
submission.to_csv("data/benchmark.csv")
