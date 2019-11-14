import pandas as pd
import matplotlib as plt
import seaborn as sns

path = "./plots/bike_sharing/"

train = pd.read_csv("datasets/bike_sharing/bikeSharing.shuf.train.csv")
test_data = pd.read_csv("datasets/bike_sharing/bikeSharing.shuf.test.csv")
test_label = pd.read_csv("datasets/bike_sharing/bikeSharing.shuf.sampleSolution.csv")


#Check for missing values
print(train.isnull().sum())

#Histogram of target value
plot = sns.distplot(train['cnt']).get_figure()
plot.savefig(path + "hist1.png")



#Linear Regression


