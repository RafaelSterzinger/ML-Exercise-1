# %% Load data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ridge_util import *
from knn_utils import *
from sklearn import linear_model
from tree_util import *
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
plt.savefig('plots/cpu_performance/vendor_histogram.png')
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
plt.savefig('plots/cpu_performance/performance_histogram.png')
plt.show()

# %% Heatplot
correlation_matrix = train_data.corr().round(2).abs()
sns.heatmap(correlation_matrix, square=True, linewidths=.5).get_figure()
plt.show()
plt.savefig(path + "heatmap_numerical.png")

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
plt.show()
plt.savefig(path + "ridge_comparision.png")

# %% decision tree max_depth comparison

options = [[number_features, 'Number Features'], [["MMAX", "MMIN", "CACH"], 'Top 3 attributes']]

decision_tree_comparison(train_data, target, options,
                         comp_type='max_depth',
                         max_depth_from=1,
                         max_depth_to=25,
                         step=1)
plt.show()
plt.savefig(path + "tree_comparision.png")

# %% Outlier removal and normed
train_data_out = train_data[(np.abs(stats.zscore(train_data))<3).all(axis=1)]
train_data_out.index = np.arange(0,len(train_data_out),1)

# %%  Ridge regression
ridge_regression_alpha_comparison(train_data_out, target, number_features,
                                  0,
                                  100000,
                                  10000,
                                  "Only Numerical values")

ridge_regression_alpha_comparison(train_data_out, target, ["MMAX", "MMIN", "CACH"],
                                  0,
                                  100000,
                                  10000,
                                  "Top three correlating values")
plt.show()
plt.savefig(path + "ridge_comparision_outlier.png")

# %% decision tree max_depth comparison

options = [[number_features, 'Number Features'], [["MMAX", "MMIN", "CACH"], 'Top 3 attributes']]

decision_tree_comparison(train_data, target, options,
                         comp_type='max_depth',
                         max_depth_from=1,
                         max_depth_to=30,
                         step=2)

options = [[number_features, 'Number Features without outliers'], [["MMAX", "MMIN", "CACH"], 'Top 3 attributes without outliers']]

decision_tree_comparison(train_data_out, target, options,
                         comp_type='max_depth',
                         max_depth_from=1,
                         max_depth_to=30,
                         step=2)

plt.show()
plt.savefig(path + "tree_comparision_outlier.png")



