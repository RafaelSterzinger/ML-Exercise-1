import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot
from sklearn import linear_model

path = "./plots/bike_sharing/"

train = pd.read_csv("datasets/bike_sharing/bikeSharing.shuf.train.csv")
test_data = pd.read_csv("datasets/bike_sharing/bikeSharing.shuf.test.csv")
test_label = pd.read_csv("datasets/bike_sharing/bikeSharing.shuf.sampleSolution.csv")

# Check for missing values
print(train.isnull().sum())

fig, ax = pyplot.subplots(figsize=(10, 10))

# Histogram of target value
plot = sns.distplot(train['cnt'], ax=ax).get_figure()
plot.savefig(path + "hist1.png")

# Heatmap to check correlation
correlation_matrix = train.corr().round(2).abs()
cnt_corr = correlation_matrix["cnt"].nlargest(8)

# Selecting 5 largest without id, year
attr = ["cnt", "temp", "atemp", "hum", "hr", "season"]
correlation_matrix = train[attr].corr().round(2).abs()
plot = sns.heatmap(correlation_matrix, ax=ax, linewidths=.5).get_figure()
plot.savefig(path + "heatmap.png")

# Linear Regression
regr = linear_model.LinearRegression()
bike_train_X = train[["temp", "atemp", "hum", "hr", "season"]]
bike_train_y = train[["cnt"]]

regr.fit(bike_train_X, bike_train_y)
bike_y_pred = regr.predict(test_data[["temp", "atemp", "hum", "hr", "season"]])

test_label = test_label.assign(cnt=bike_y_pred)
print(test_label)
test_label.to_csv(path + "linreg_pred.csv", index=False)

