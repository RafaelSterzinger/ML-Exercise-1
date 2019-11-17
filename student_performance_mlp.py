from utils import *
from mlp_utils import *
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#%%
# TODO: move in main
pd.set_option('display.expand_frame_repr', False)
plt.rcParams["patch.force_edgecolor"] = True

#%% enforce black border on histogram bins
#%% read dataset
data = pd.read_csv('datasets/student_performance/StudentPerformance.shuf.train.csv', index_col=0)

#%%
correlated_columns=highest_correlated_data_as_list(data,'Grade',10)
correlated_columns.remove('Grade')
print(correlated_columns)

train, test = train_test_split(data, test_size=0.3)
mlp(train, test, 'Grade', correlated_columns)
