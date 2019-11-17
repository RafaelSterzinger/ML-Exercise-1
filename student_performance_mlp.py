from utils import *
from mlp_utils import *
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#%%
# TODO: move in main
pd.set_option('display.expand_frame_repr', False)
plt.rcParams["patch.force_edgecolor"] = True


#%% read dataset
data = pd.read_csv('datasets/student_performance/StudentPerformance.shuf.train.csv', index_col=0)

#%%
correlated_columns=highest_correlated_data_as_list(data,'Grade',10)
correlated_columns.remove('Grade')

numeric_attributes_student = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences"]
mlp_regression_neu(data,numeric_attributes_student, 'Grade', [(7,),(5,),(1,)],"logistic")
