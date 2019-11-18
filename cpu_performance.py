# %% Load data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from ridge_util import *
from knn_utils import *
from tree_util import *
from mlp_utils import *
from scipy import stats

path = "./plots/cpu_performance/"
plt.rcParams["patch.force_edgecolor"] = True

train_data = pd.read_csv("datasets/cpu_performance/CPUPerformance.csv", skipinitialspace=True)
train_data.columns = (["Vendor", "Name", "MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP", "EPR"])
target = ["EPR"]
category_features = ["Vendor", "Name"]
number_features = ["MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP"]

# %% Remove Vendors => some vendors only published 1 to 2 chips
sns.barplot(train_data["Vendor"].unique(), train_data["Vendor"].value_counts())
plt.xlabel('Vendors')
plt.xticks(rotation=90)
plt.ylabel('Count')
plt.title('Chips per Vendor')
plt.savefig(path + 'vendor_histogram.png')
plt.show()

# %% Describtion Check for missing values
train_data.describe()
train_data.isnull().sum()

# Remove unnecessary data like "Name", Vendor is questionable
train_data = train_data.drop(["Name", "Vendor"], axis=1)
# %% Histogram of grades, strongly skewed, outliers, scale
sns.distplot(train_data["EPR"], kde=True)
plt.xlabel('Estimated Performance')
plt.ylabel('Count')
plt.title('Distribution of Performance')
plt.savefig(path + 'performance_histogram.png')
plt.show()

# %% Heatplot
correlation_matrix = train_data.corr().round(2).abs()
sns.heatmap(correlation_matrix, square=True, linewidths=.5).get_figure()
plt.savefig(path + "heatmap_numerical.png")
plt.show()

train_data.corr()['EPR'].abs().sort_values()
number_features.remove("PRP")

# %%  Ridge regression
ridge_regression_alpha_comparison(train_data, target, number_features,
                                  0,
                                  100000,
                                  10000,
                                  "Only Numerical values")

ridge_regression_alpha_comparison(train_data, target, ["MMAX", "MMIN", "CACH"],
                                  0,
                                  100000,
                                  10000,
                                  "Top three correlating values")
plt.savefig(path + "ridge_comparision.png")
plt.show()

# %% decision tree max_depth comparison

options = [[number_features, 'Number Features'], [["MMAX", "MMIN", "CACH"], 'Top 3 attributes']]

decision_tree_comparison(train_data, target, options,
                         comp_type='max_depth',
                         p_from=1,
                         p_to=25,
                         p_step=1)
plt.savefig(path + "tree_comparision.png")
plt.show()

# %% knn comparision

x_train, y_train, x_test, y_test = make_split(train_data, 'EPR')
find_best_rmse('with all attributes and manhatten',
               x_train, y_train, x_test, y_test, metric='manhattan')

x_train, y_train, x_test, y_test = make_split(train_data, 'EPR')
find_best_rmse('with all attributes and euclidean', x_train, y_train, x_test, y_test)

x_train, y_train, x_test, y_test = make_split(train_data, 'EPR')
find_best_rmse('with all attributes and minkowski', x_train, y_train, x_test, y_test, metric="minkowski")
plt.savefig(path + "knn_comparision.png")
plt.show()

# %% mlp comparision
models = [(5, 7, 7), (7, 5, 5), (7, 7, 5, 3)]
df = mlp_regression(train_data, number_features, 'EPR', models, "logistic")

df = pd.concat([df, mlp_regression(train_data, number_features, 'EPR', models, "relu")])

df = pd.concat([df, mlp_regression(train_data, number_features, 'EPR', models, "tanh")])

sns.catplot(x='Layers', y='RMSE', hue='Activation',data = df, kind='bar')

plt.savefig(path + "mlp_regression.png")
plt.show()

# %% Outlier removal
train_data_out = train_data[(np.abs(stats.zscore(train_data)) < 3).all(axis=1)]
train_data_out.index = np.arange(0, len(train_data_out), 1)

# %%  Ridge regression
ridge_regression_alpha_comparison(train_data_out, target, number_features,
                                  0,
                                  1000,
                                  100,
                                  "Only Numerical values")

ridge_regression_alpha_comparison(train_data_out, target, ["MMAX", "MMIN", "CACH"],
                                  0,
                                  1000,
                                  100,
                                  "Top three correlating values")
plt.savefig(path + "ridge_comparision_outlier.png")
plt.show()

# %% decision tree max_depth comparison

options = [[number_features, 'Number Features'], [["MMAX", "MMIN", "CACH"], 'Top 3 attributes']]

decision_tree_comparison(train_data, target, options,
                         comp_type='max_depth',
                         p_from=1,
                         p_to=30,
                         p_step=2)

options = [[number_features, 'Number Features without outliers'],
           [["MMAX", "MMIN", "CACH"], 'Top 3 attributes without outliers']]

decision_tree_comparison(train_data_out, target, options,
                         comp_type='max_depth',
                         p_from=1,
                         p_to=30,
                         p_step=2)

plt.savefig(path + "tree_comparision_outlier.png")
plt.show()
# %% knn comparision
x_train, y_train, x_test, y_test = make_split(train_data_out, 'EPR')
find_best_rmse('with all attributes and manhatten',
               x_train, y_train, x_test, y_test, metric='manhattan')

x_train, y_train, x_test, y_test = make_split(train_data_out, 'EPR')
find_best_rmse('with all attributes and minkowski', x_train, y_train, x_test, y_test, metric="minkowski")

selection = ["MMIN", "MMAX", "CACH", "EPR"]

x_train, y_train, x_test, y_test = make_split(train_data_out[selection], 'EPR')
find_best_rmse('with top 3 attributes and manhatten',
               x_train, y_train, x_test, y_test, metric='manhattan')

x_train, y_train, x_test, y_test = make_split(train_data_out[selection], 'EPR')
find_best_rmse('with top 3 attributes and minkowski', x_train, y_train, x_test, y_test, metric="minkowski")

plt.savefig(path + "knn_comparision_outlier.png")
plt.show()

# %% mlp comparision, with outlier cleansing
models = [(5, 7, 7), (7, 5, 5), (7, 7, 5, 3)]
df = mlp_regression(train_data_out, number_features, 'EPR', models, "logistic")

df = pd.concat([df, mlp_regression(train_data_out, number_features, 'EPR', models, "relu")])

df = pd.concat([df, mlp_regression(train_data_out, number_features, 'EPR', models, "tanh")])

sns.catplot(x='Layers', y='RMSE', hue='Activation',data = df, kind='bar')

plt.savefig(path + "mlp_comparision_outlier.png")
plt.show()

# %% mlp comparision outlier cleansing, with top correlating
models = [(5, 7, 7), (7, 5, 5), (7, 7, 5, 3)]
df = mlp_regression(train_data_out, ["MMIN", "MMAX", "CACH", "EPR"], 'EPR', models, "logistic")

df = pd.concat([df, mlp_regression(train_data_out, ["MMIN", "MMAX", "CACH", "EPR"], 'EPR', models, "relu")])

df = pd.concat([df, mlp_regression(train_data_out, ["MMIN", "MMAX", "CACH", "EPR"], 'EPR', models, "tanh")])

sns.catplot(x='Layers', y='RMSE', hue='Activation',data = df, kind='bar')

plt.savefig(path + "mlp_comparision_top_correlating_outlier.png")
plt.show()

# %% mlp comparision outlier cleansing, with top correlating
models = [(5, 7, 7), (7, 5, 5), (7, 7, 5, 3)]
df = mlp_regression_layer_comparison(train_data_out, ["MMIN", "MMAX", "CACH", "EPR"], 'EPR', models, "logistic")

df = pd.concat([df, mlp_regression_layer_comparison(train_data_out, ["MMIN", "MMAX", "CACH", "EPR"], 'EPR', models, "relu")])

df = pd.concat([df, mlp_regression_layer_comparison(train_data_out, ["MMIN", "MMAX", "CACH", "EPR"], 'EPR', models, "tanh")])

sns.catplot(x='Layers', y='RMSE', hue='Activation',data = df, kind='bar')

plt.savefig(path + "mlp_comparision_top_correlating_outlier.png")
plt.show()
# TODO: mit minmax skalierung
