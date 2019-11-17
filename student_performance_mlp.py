from utils import *
from mlp_utils import *
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# %%
# TODO: move in main
pd.set_option('display.expand_frame_repr', False)
plt.rcParams["patch.force_edgecolor"] = True

# %% read dataset
data = pd.read_csv('datasets/student_performance/StudentPerformance.shuf.train.csv', index_col=0)

# %%
correlated_columns = highest_correlated_data_as_list(data, 'Grade', 10)
correlated_columns.remove('Grade')

numeric_attributes_student = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel", "freetime",
                              "goout", "Dalc", "Walc", "health", "absences"]
df = mlp_regression(data, numeric_attributes_student, 'Grade', [(5, 7, 7), (7, 5, 5), (7, 7, 5, 3)], "logistic")

df = pd.concat([df, mlp_regression(data, numeric_attributes_student, 'Grade', [(5, 7, 7), (7, 5, 5), (7, 7, 5, 3)], "relu")])

df = pd.concat([df, mlp_regression(data, numeric_attributes_student, 'Grade', [(5, 7, 7), (7, 5, 5), (7, 7, 5, 3)], "tanh")])

sns.catplot(x='Layers', y='RSME', hue='Activation',data = df, kind='bar')
#df = pd.melt(df, id_vars="Layers", var_name="Activation", value_name="RSME")
plt.ylim(3.5,5.5)
plt.show()
