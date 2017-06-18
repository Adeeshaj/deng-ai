# deng-ai
CS3230 - Data Minning Group Project

Task is to compete in [DengAI: Predicting Disease Spread](https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/) competition

Group name is DVios-UOMCSE13

## Python modules used
* pandas
* statsmodels

## Data Preprocessing Approaches
* Filling the missing values with latest value, attribute mean, class mean
* Time shifting the data

## Prediction 
* Negative Binomial
* Random fores

## Other Approaches
* Use 2 separate models for 2 cities with 4 features with the highest coorelation. 

* First the 2/3 of the data set as used for model building and 1/3 for model testing. Using the complete set of data for model building improved the accuracy

* Updating the model again using the best set of predicted values imporved the accuracy

* Reducing the features to top 3 incresed the accuracy of predicted data.
