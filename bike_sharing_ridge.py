import pandas as pd
import seaborn as sns
import numpy as np
from ridge_util import *
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

# %% Bike Sharing Ridge Regression

result = ridgeRegression(train, test, 0, 200, 20, number_features+category_features, target, groundTruth, "Numerical values")
plt.show()
# test_label = test_label.assign(cnt=bike_y_pred)
# test_label.to_csv(path + "linreg_pred.csv", index=False)
