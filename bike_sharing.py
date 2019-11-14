import pandas as pd
import seaborn as sns
from matplotlib import pyplot

path = "./plots/bike_sharing/"

train = pd.read_csv("datasets/bike_sharing/bikeSharing.shuf.train.csv")
test_data = pd.read_csv("datasets/bike_sharing/bikeSharing.shuf.test.csv")
test_label = pd.read_csv("datasets/bike_sharing/bikeSharing.shuf.sampleSolution.csv")

# Check for missing values
print(train.isnull().sum())

fig, ax = pyplot.subplots(figsize=(6, 6))

# Histogram of target value
plot = sns.distplot(train['cnt'], ax=ax).get_figure()
plot.savefig(path + "hist1.png")

# Heatmap to check correlation
correlation_matrix = train.corr().round(2)
plot = sns.heatmap(correlation_matrix, ax=ax, linewidths=.5).get_figure()
plot.savefig(path + "heatmap.png")

# Linear Regression

