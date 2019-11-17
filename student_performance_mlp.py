from mlp_utils import *
from utils import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from math import sqrt

#%%
pd.set_option('display.expand_frame_repr', False)

#%% enforce black border on histogram bins
# TODO: move in main
plt.rcParams["patch.force_edgecolor"] = True
#%% read dataset
data = pd.read_csv('datasets/student_performance/StudentPerformance.shuf.train.csv', index_col=0)

#%% use one hot encoding
data = pd.get_dummies(data)

#%%
def highest_correlated_data(df, target, size=6):
    # category_df = df.select_dtypes('object')  # One hot encode the variables
    # Find correlations with the Grade
    most_correlated = df.corr().abs()[target].sort_values(ascending=False)
    # Maintain the top 6 most correlation features with Grade
    most_correlated = most_correlated[:size]

    return df[most_correlated.index.tolist()]

print(data)
trimmed_data = highest_correlated_data(data,'Grade')

#%%
data.describe()

#%%

x_train, y_train, x_test, y_test = divide_data(trimmed_data,'cnt')
mlp(data,,'Grade')

#%%
def highest_correlated_data(df, target, size=6):
    # category_df = df.select_dtypes('object')  # One hot encode the variables
    # Find correlations with the Grade
    most_correlated = df.corr().abs()[target].sort_values(ascending=False)
    # Maintain the top 6 most correlation features with Grade
    most_correlated = most_correlated[:size]

    return df[most_correlated.index.tolist()]

data = pd.get_dummies(data)
data = highest_correlated_data(data,'Grade',10)
