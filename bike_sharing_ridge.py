import pandas as pd
import seaborn as sns
import numpy as np
from ridge_util import *
from knn_utils import *
from tree_util import *
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from sklearn import linear_model

path = "./plots/bike_sharing/"
# %% MSE for testing
groundTruth = pd.read_csv("datasets/bike_sharing/groundTruth.csv")
sampleSolution = pd.read_csv("datasets/bike_sharing/bikeSharing.shuf.sampleSolution.csv")
index = sampleSolution["id"].to_numpy()
index = map(lambda x: int(x) - 1, index)
groundTruth = groundTruth.loc[index][["cnt"]]

# %%
train = pd.read_csv("datasets/bike_sharing/bikeSharing.shuf.train.csv", skipinitialspace=True)
train.columns = train.columns.str.strip()
test = pd.read_csv("datasets/bike_sharing/bikeSharing.shuf.test.csv", skipinitialspace=True)
test.columns = test.columns.str.strip()

# Remove unecessary attributes
train = train.drop(['dteday', 'yr', 'id'], axis=1)
test = test.drop(['dteday', 'yr', 'id'], axis=1)

category_features = ['season', 'holiday', 'mnth', 'hr', 'weekday', 'workingday', 'weathersit']
number_features = ['temp', 'atemp', 'hum', 'windspeed']
top4_attributes = ["hum", "hr", "temp", "season"]
all_attributes_tupels_list = [[number_features, 'Numeric attributes'], [number_features + category_features, 'Number + category'], [top4_attributes, 'Top 4 attributes']]


target = ['cnt']

fig, ax = pyplot.subplots(figsize=(10, 10))
# %% Check for missing values
train.isnull().sum()

# %% Histogram of target value
plot = sns.distplot(train['cnt'], ax=ax).get_figure()
plot.show()
plot.savefig(path + "hist_outliers.png")

# %% Heatmap to check correlation of numerical values
correlation_matrix = train[target + number_features].corr().round(2).abs()
plot = sns.heatmap(correlation_matrix, linewidths=.5).get_figure()
plot.show()
plot.savefig(path + "heatmap_numerical.png")
number_features.remove('atemp')

# %% Check impact of hours
plot = sns.boxplot(data=train, y="cnt", x="hr", orient="v").get_figure()
plot.show()
plot.savefig(path + "hour_boxplot.png")

# %% Bike Sharing Ridge Regression todo DEPRECATED
''' 
ridgeRegression(train, test, 0, 100, 10, number_features+category_features, target, groundTruth, "All attributes")
ridgeRegression(train, test, 0, 100, 10, number_features, target, groundTruth, "Only Numerical values")
ridgeRegression(train, test, 0, 100, 10, ["hum","hr","temp","season"], target, groundTruth, "Maximum correlating values")
plt.show()
# test_label = test_label.assign(cnt=bike_y_pred)
# test_label.to_csv(path + "linreg_pred.csv", index=False)
'''

# %% Bike Sharing Ridge regression Alpha comparison with Cross validation and different attributes
ridge_regression_alpha_comparison(train, target,
                                  number_features + category_features,
                                  0, 100, 10, "All attributes")
ridge_regression_alpha_comparison(train, target,
                                  number_features,
                                  0, 100, 10, "Only Numerical values")

ridge_regression_alpha_comparison(train, target,
                                  ["hum", "hr", "temp", "season"],
                                  0, 100, 10,
                                  "Maximum correlating values")
plt.show()

#%% Bike sharing decision tree regression criterion comparison

decision_tree_regression_criterion_comparison(train,target,
                                              number_features + category_features,
                                              criterion = ['mse', 'friedman_mse'],
                                              name = "Number features + category features")

decision_tree_regression_criterion_comparison(train, target,
                                              number_features,
                                              criterion=['mse', 'friedman_mse'],
                                              name = "Number features")

decision_tree_regression_criterion_comparison(train, target,
                                              ["hum", "hr", "temp", "season"],
                                              criterion=['mse', 'friedman_mse'],
                                              name = "Top 4")
plt.show()

#%% Bike sharing decision tree regression max_depth comparison
decision_tree_comparison(train, target,all_attributes_tupels_list,
                         comp_type='max_depth',
                         max_depth_from=1,
                         max_depth_to=30,       #bei 25 konstante Tiefe
                         step=1)
plt.show()
#decision_tree_crossvalidation(train, target, number_features+category_features)
print("done")