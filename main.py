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

import math

###In 2
### load the provided data
##train_features = pd.read_csv('data/dengue_features_train.csv',
##                             index_col=[0,1,2])
##
##train_labels = pd.read_csv('data/dengue_labels_train.csv',
##                           index_col=[0,1,2])
##
###In 3
### Seperate data for San Juan
##sj_train_features = train_features.loc['sj']
##sj_train_labels = train_labels.loc['sj']
##
### Separate data for Iquitos
##iq_train_features = train_features.loc['iq']
##iq_train_labels = train_labels.loc['iq']
##
###In 4
##print('San Juan')
##print('features: ', sj_train_features.shape)
##print('labels  : ', sj_train_labels.shape)
##
##print('\nIquitos')
##print('features: ', iq_train_features.shape)
##print('labels  : ', iq_train_labels.shape)
##
###In 5
##sj_train_features.head()
##
###In 6
### Remove `week_start_date` string.
##sj_train_features.drop('week_start_date', axis=1, inplace=True)
##iq_train_features.drop('week_start_date', axis=1, inplace=True)
##
###In 7
### Null check
##pd.isnull(sj_train_features).any()
##
###In 8
##(sj_train_features
##     .ndvi_ne
##     .plot
##     .line(lw=0.8))
##
##plt.title('Vegetation Index over Time')
##plt.xlabel('Time')
##plt.show()
##
###In 9
##sj_train_features.fillna(method='ffill', inplace=True)
##iq_train_features.fillna(method='ffill', inplace=True)
##
###In 10
##print('San Juan')
##print('mean: ', sj_train_labels.mean()[0])
##print('var :', sj_train_labels.var()[0])
##
##print('\nIquitos')
##print('mean: ', iq_train_labels.mean()[0])
##print('var :', iq_train_labels.var()[0])
##
###In 11
##sj_train_labels.hist()
##
###In 12
##iq_train_labels.hist()
##
###In 13
##sj_train_features['total_cases'] = sj_train_labels.total_cases
##iq_train_features['total_cases'] = iq_train_labels.total_cases
##
###In 14
### compute the correlations
##sj_correlations = sj_train_features.corr()
##iq_correlations = iq_train_features.corr()
##
###In 15
### plot san juan
##sj_corr_heat = sns.heatmap(sj_correlations)
##plt.title('San Juan Variable Correlations')
##plt.show()
##
###In 16
### plot iquitos
##iq_corr_heat = sns.heatmap(iq_correlations)
##plt.title('Iquitos Variable Correlations')
##plt.show()
##
###In 17
### San Juan
##(sj_correlations
##     .total_cases
##     .drop('total_cases') # don't compare with myself
##     .sort_values(ascending=False)
##     .plot
##     .barh())
##
###In 18
### Iquitos
##(iq_correlations
##     .total_cases
##     .drop('total_cases') # don't compare with myself
##     .sort_values(ascending=False)
##     .plot
##     .barh())

#In 19

features_sj = ['reanalysis_specific_humidity_g_per_kg', 
             'reanalysis_dew_point_temp_k', 
             'station_avg_temp_c', 
             'reanalysis_max_air_temp_k']

features_iq = ['reanalysis_specific_humidity_g_per_kg', 
             'reanalysis_dew_point_temp_k',  
             'station_min_temp_c',
             'reanalysis_min_air_temp_k']

features = ['reanalysis_specific_humidity_g_per_kg', 
                 'reanalysis_dew_point_temp_k', 
                 'station_avg_temp_c', 
                 'station_min_temp_c',
                 'reanalysis_min_air_temp_k',
                 'reanalysis_max_air_temp_k']                          

def preprocess_data(data_path, labels_path=None):
	# load data and set index to city, year, weekofyear
    df = pd.read_csv(data_path, index_col=[0, 1, 2])

    sj = df[features_sj]
    iq = df[features_iq]

    # add labels to dataframe
    if labels_path:
        labels = pd.read_csv(labels_path, index_col=[0, 1, 2])
        sj = sj.join(labels)
        iq = iq.join(labels)

    # separate san juan and iquitos
    sj = sj.loc['sj']
    iq = iq.loc['iq']

    # fill missing values
    avg_sj = sj.groupby(by='total_cases').agg({'reanalysis_specific_humidity_g_per_kg':'mean', 'reanalysis_dew_point_temp_k':'mean', 'station_avg_temp_c':'mean', 'reanalysis_max_air_temp_k':'mean'})
    avg_iq = iq.groupby(by='total_cases').agg({'reanalysis_specific_humidity_g_per_kg':'mean', 'reanalysis_dew_point_temp_k':'mean', 'station_min_temp_c':'mean', 'reanalysis_min_air_temp_k':'mean'})

    avg_t_sj = sj.mean()
    avg_t_iq = iq.mean()

    for c in sj:
        for r in range(len(sj[c])):
            if(math.isnan(sj[c][r])):
                if(math.isnan(avg_sj[c][sj['total_cases'][r]])):
                    sj[c][r] = avg_t_sj[c]
                else:
                    sj[c][r] = avg_sj[c][sj['total_cases'][r]]

    for c in iq:
            for r in range(len(iq[c])):
                if(math.isnan(iq[c][r])):
                    if(math.isnan(avg_iq[c][iq['total_cases'][r]])):
                        iq[c][r] = avg_t_iq[c]
                    else:
                        iq[c][r] = avg_iq[c][iq['total_cases'][r]]
    
    return sj, iq

def preprocess_data_test(data_path, labels_path=None):
    # load data and set index to city, year, weekofyear
    df = pd.read_csv(data_path, index_col=[0, 1, 2])
    
    df = df[features]

    # add labels to dataframe
    if labels_path:
        labels = pd.read_csv(labels_path, index_col=[0, 1, 2])
        df = df.join(labels)
    
    # separate san juan and iquitos
    sj = df.loc['sj']
    iq = df.loc['iq']
    
    # fill missing values
    avg_t_sj = sj.mean()
    avg_t_iq = iq.mean()

    for c in sj:
        for r in range(len(sj[c])):
            if(math.isnan(sj[c][r])):
                sj[c][r] = avg_t_sj[c]
               
    for c in iq:
            for r in range(len(iq[c])):
                if(math.isnan(iq[c][r])):
                    iq[c][r] = avg_t_iq[c]
    
    return sj, iq

#In 20
sj_train, iq_train = preprocess_data('data/dengue_features_train.csv',
                                    labels_path="data/dengue_labels_train.csv")

#In 21
##sj_train.describe()

#In 22
##iq_train.describe()

#In 23
sj_train_subtrain = sj_train.head(800)
sj_train_subtest = sj_train.tail(sj_train.shape[0] - 800)

iq_train_subtrain = iq_train.head(400)
iq_train_subtest = iq_train.tail(iq_train.shape[0] - 400)

#In 24
from statsmodels.tools import eval_measures
import statsmodels.formula.api as smf

def get_best_model(train, test, loc):
    # Step 1: specify the form of the model

    if(loc=="sj"):
    	model_formula = "total_cases ~ 1 + " \
	                    "reanalysis_specific_humidity_g_per_kg + " \
	                    "reanalysis_dew_point_temp_k + " \
	                    "station_avg_temp_c + " \
	                    "reanalysis_max_air_temp_k"

    elif(loc=="iq"):
    	model_formula = "total_cases ~ 1 + " \
	                    "reanalysis_specific_humidity_g_per_kg + " \
	                    "reanalysis_dew_point_temp_k + " \
	                    "station_min_temp_c + " \
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

##    print('best alpha = ', best_alpha)
##    print('best score = ', best_score)
            
    # Step 3: refit on entire dataset
    full_dataset = pd.concat([train, test])
    model = smf.glm(formula=model_formula,
                    data=full_dataset,
                    family=sm.families.NegativeBinomial(alpha=best_alpha))

    fitted_model = model.fit()
    return fitted_model
    
sj_best_model = get_best_model(sj_train_subtrain, sj_train_subtest, 'sj')
iq_best_model = get_best_model(iq_train_subtrain, iq_train_subtest, 'iq')

#In 25
# figs, axes = plt.subplots(nrows=2, ncols=1)

# plot sj
# sj_train['fitted'] = sj_best_model.fittedvalues
# sj_train.fitted.plot(ax=axes[0], label="Predictions")
# sj_train.total_cases.plot(ax=axes[0], label="Actual")

# plot iq
# iq_train['fitted'] = iq_best_model.fittedvalues
# iq_train.fitted.plot(ax=axes[1], label="Predictions")
# iq_train.total_cases.plot(ax=axes[1], label="Actual")

##plt.suptitle("Dengue Predicted Cases vs. Actual Cases")
##plt.legend()

#In 27
sj_test, iq_test = preprocess_data_test('data/dengue_features_test.csv')

sj_predictions = sj_best_model.predict(sj_test).astype(int)
iq_predictions = iq_best_model.predict(iq_test).astype(int)

submission = pd.read_csv("data/submission_format.csv",
                         index_col=[0, 1, 2])

submission.total_cases = np.concatenate([sj_predictions, iq_predictions])
submission.to_csv("data/benchmark.csv")