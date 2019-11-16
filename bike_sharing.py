import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from sklearn import linear_model

# %%
path = "./plots/bike_sharing/"

train = pd.read_csv("datasets/bike_sharing/bikeSharing.shuf.train.csv", skipinitialspace=True)
train.columns = train.columns.str.strip()
test = pd.read_csv("datasets/bike_sharing/bikeSharing.shuf.test.csv", skipinitialspace=True)
test.columns = test.columns.str.strip()

#Remove unecessary attributes
train = train.drop(['dteday', 'yr', 'id'], axis=1)
test = test.drop(['dteday', 'yr', 'id'], axis=1)

fig, ax = pyplot.subplots(figsize=(10, 10))
# %% Check for missing values
train.isnull().sum()

# %% Histogram of target value
plot = sns.distplot(train['cnt'], ax=ax).get_figure()
plot.show()
plot.savefig(path + "hist1.png")

# %% Heatmap to check correlation
correlation_matrix = train.corr().round(2).abs()
cnt_corr = correlation_matrix["cnt"].nlargest(8)

# %% Selecting 5 largest without id, year
attr = ["cnt", "temp", "atemp", "hum", "hr", "season"]
correlation_matrix = train[attr].corr().round(2).abs()
plot = sns.heatmap(correlation_matrix, ax=ax, linewidths=.5).get_figure()
plot.show()
plot.savefig(path + "heatmap.png")

# %% Linear Regression
regr = linear_model.LinearRegression()
bike_train_X = train[["temp", "atemp", "hum", "hr", "season"]]
bike_train_y = train[["cnt"]]

regr.fit(bike_train_X, bike_train_y)
bike_y_pred = regr.predict(test_data[["temp", "atemp", "hum", "hr", "season"]])

# test_label = test_label.assign(cnt=bike_y_pred)
# test_label.to_csv(path + "linreg_pred.csv", index=False)

# %% MSE for testing
hour = pd.read_csv("datasets/bike_sharing/groundTruth.csv")
hour.set_index("instant")

index = test_label["id"].to_numpy()
index = map(lambda x: int(x) - 1, index)
info = hour.loc[index][["cnt"]]

print(np.sqrt(mean_squared_error(info, bike_y_pred)))
# %% Linear Regression
regr = linear_model.LinearRegression()
bike_train_X = train.drop(['id', 'cnt', 'dteday'], axis=1)
bike_train_y = train['cnt']

regr.fit(bike_train_X, bike_train_y)
bike_y_pred = regr.predict(test_data.drop(['id', 'dteday'], axis=1))

print(info)
print(np.sqrt(mean_squared_error(info, bike_y_pred)))
