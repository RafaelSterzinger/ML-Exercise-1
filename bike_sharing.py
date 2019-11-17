import pandas as pd
import seaborn as sns
import numpy as np
from ridge_util import *
from knn_utils import *
from tree_util import *
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn import linear_model

path = "./plots/bike_sharing/"
plt.rcParams["patch.force_edgecolor"] = True
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
# %% Check for missing values
train.isnull().sum()

# %% Histogram of target value
sns.distplot(train['cnt']).get_figure()
plt.xlabel("Amount of shared bikes")
plt.show()
plt.savefig(path + "hist_outliers.png")

# %% Heatmap to check correlation of numerical values
correlation_matrix = train[target + number_features].corr().round(2).abs()
sns.heatmap(correlation_matrix, linewidths=.5).get_figure()
plt.show()
plt.savefig(path + "heatmap_numerical.png")
number_features.remove('atemp')

# %% Check impact of hours
sns.boxplot(data=train, y="cnt", x="hr", orient="v").get_figure()
plt.xlabel("Hours")
plt.ylabel("Amount of shared bikes")
plt.show()
plt.savefig(path + "hour_boxplot.png")

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
                         step=2)
plt.show()
#%% knn with different distances
data = pd.read_csv("datasets/bike_sharing/bikeSharing.shuf.train.csv",skipinitialspace=True)
data.columns=data.columns.str.strip()

trimmed_data = trim_data(data,['dteday'])
x_train, y_train, x_test, y_test = make_split(trimmed_data, 'cnt')
find_best_rmse('with all attributes + id and euclidean',
               x_train, y_train, x_test, y_test)

trimmed_data = trim_data(trimmed_data,['id'])

x_train, y_train, x_test, y_test = make_split(trimmed_data, 'cnt')
find_best_rmse('with all attributes and manhatten',
               x_train, y_train, x_test, y_test,metric='manhattan')

x_train, y_train, x_test, y_test = make_split(trimmed_data, 'cnt')
find_best_rmse('with all attributes and euclidean',x_train, y_train, x_test, y_test)

x_train, y_train, x_test, y_test = make_split(trimmed_data, 'cnt')
find_best_rmse('with all attributes and minkowski',x_train, y_train, x_test, y_test,metric="minkowski")

plt.savefig(path + "knn_all_attributes.png")
plt.ylabel("Root Mean Squared Error")
plt.show()
#%% Correlation not so good
trimmed_data = trim_data(trimmed_data,['yr','weekday','atemp','windspeed','weekday','mnth','holiday','workingday'])

x_train, y_train, x_test, y_test = make_split(trimmed_data, 'cnt')
find_best_rmse('with hr, temp, hum, season, weathersit and minkowski',x_train, y_train, x_test, y_test, metric="minkowski")

plt.savefig(path + "knn_selection_attributes.png")
plt.ylabel("Root Mean Squared Error")
plt.show()
'''
trimmed_data = trim_data(trimmed_data,['yr','atemp','windspeed','weekday'])

x_train, y_train, x_test, y_test = create_cross_validation(trimmed_data,'cnt')
find_best_rmse('without year, atemp, windspeed, weekday',x_train, y_train, x_test, y_test)
trimmed_data = trim_data(data,['id','dteday',
                               'yr','atemp','windspeed','weekday',
                               'season','mnth','holiday','workingday','weathersit'])

x_train, y_train, x_test, y_test = create_cross_validation(trimmed_data, 'cnt')
find_best_rmse('only temperature and humidity',x_train, y_train, x_test, y_test)

plt.savefig(path + "k_rmse_attributes.png")
plt.show()
'''
#%% make hr circular
train = pd.read_csv("datasets/bike_sharing/bikeSharing.shuf.train.csv", skipinitialspace=True)
train.columns = train.columns.str.strip()
hr = np.sin(2*np.pi*train["hr"]/24)
train.update({"hr":hr})

x_train, y_train, x_test, y_test = make_split(train[["hr", "hum", "season", "weathersit", "temp", "cnt"]], 'cnt')
find_best_rmse('with hr, temp, hum, season, weathersit and minkowski',x_train, y_train, x_test, y_test, metric="minkowski")

plt.savefig(path + "knn_selection_attributes.png")
plt.ylabel("Root Mean Squared Error")
plt.show()