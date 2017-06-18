
import h2o
import os
import pandas as pd
import numpy as np


h2o.init(max_mem_size = "2G")             #specify max number of bytes. uses all cores by default.
h2o.remove_all()                          #clean slate, in case cluster was already running


from h2o.estimators.random_forest import H2ORandomForestEstimator


def preprocess_data(data_path, labels_path=None):
    # load data and set index to city, year, weekofyear
    df = pd.read_csv(data_path, index_col=[0, 1, 2])
    
    # select features we want
    features = ['reanalysis_specific_humidity_g_per_kg', 
                 'reanalysis_dew_point_temp_k', 
                 'station_avg_temp_c', 
                 'station_min_temp_c',
                 'reanalysis_min_air_temp_k',
                 'reanalysis_max_air_temp_k']
    df = df[features]
    
    # fill missing values
    df.fillna(method='ffill', inplace=True)

    # add labels to dataframe
    if labels_path:
        labels = pd.read_csv(labels_path, index_col=[0, 1, 2])
        df = df.join(labels)
    
    # separate san juan and iquitos
    sj = df.loc['sj']
    iq = df.loc['iq']
    
    return sj, iq

sj_train, iq_train = preprocess_data('data/dengue_features_train.csv',
                                    labels_path="data/dengue_labels_train.csv")
#covtype_df = h2o.import_file(os.path.realpath("E:/University/Semester 7/Data Mining/DM python/dengue_features_train.csv"))
sj_train.to_csv("data/sj_dengue_features_train.csv")
iq_train.to_csv("data/iq_dengue_features_train.csv")

sj_train_h2o = h2o.import_file(os.path.realpath("data/sj_dengue_features_train.csv"))
iq_train_h2o = h2o.import_file(os.path.realpath("data/iq_dengue_features_train.csv"))
#split the data as described above
sj_train, sj_valid, sj_test = sj_train_h2o.split_frame([0.6, 0.2], seed=1234)
iq_train, iq_valid, iq_test = iq_train_h2o.split_frame([0.6, 0.2], seed=1234)

#Prepare predictors and response columns
sj_covtype_X = sj_train_h2o.col_names[:-1]     #last column is Cover_Type, our desired response variable 
sj_covtype_y = sj_train_h2o.col_names[-1]

iq_covtype_X = iq_train_h2o.col_names[:-1]     #last column is Cover_Type, our desired response variable 
iq_covtype_y = iq_train_h2o.col_names[-1]

sj_rf_v2 = H2ORandomForestEstimator(
    model_id="sj_rf_covType_v2",
    ntrees=200,
    max_depth=30,
    stopping_rounds=2,
    stopping_tolerance=0.01,
    score_each_iteration=True,
    seed=3000000)
sj_rf_v2.train(sj_covtype_X, sj_covtype_y, training_frame=sj_train, validation_frame=sj_valid)

iq_rf_v2 = H2ORandomForestEstimator(
    model_id="iq_rf_covType_v2",
    ntrees=200,
    max_depth=30,
    stopping_rounds=2,
    stopping_tolerance=0.01,
    score_each_iteration=True,
    seed=3000000)
iq_rf_v2.train(iq_covtype_X, iq_covtype_y, training_frame=iq_train, validation_frame=iq_valid)

sj_test,iq_test = preprocess_data('data/dengue_features_test.csv')

sj_test.to_csv("data/sj_dengue_features_test.csv")
iq_test.to_csv("data/iq_dengue_features_test.csv")

sj_test_h2o = h2o.import_file(os.path.realpath("data/sj_dengue_features_test.csv"))
iq_test_h2o = h2o.import_file(os.path.realpath("data/iq_dengue_features_test.csv"))

sj_final_rf_predictions = sj_rf_v2.predict(sj_test_h2o)
iq_final_rf_predictions = iq_rf_v2.predict(iq_test_h2o)


h2o.export_file(sj_final_rf_predictions, "data/sj_benchmark.csv", force = False, parts = 1)
h2o.export_file(iq_final_rf_predictions, "data/iq_benchmark.csv", force = False, parts = 1)
#validation set accuracy
#sj_rf_v2.hit_ratio_table(valid=True)

#test set accuracy
#print((final_rf_predictions['predict']==sj_test['total_cases']).as_data_frame(use_pandas=True).mean())

h2o.cluster().shutdown()

               
