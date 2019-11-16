import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%% enforce black border on histogram bins
# TODO: move in main
plt.rcParams["patch.force_edgecolor"] = True
#%% read dataset
df = pd.read_csv('datasets/student_performance/StudentPerformance.shuf.train.csv')
df.head()

#%% Histogram of grades

sns.distplot(df['Grade'], kde=True, bins=15)
plt.xlabel('Grade')
plt.ylabel('Count')
plt.title('Distribution of Final Grades')
plt.savefig('plots/student_performance/grades_histogram.png')
plt.show()

#%%
# Make one plot for each different location
# sns.kdeplot(df.ix[df['address'] == 'U', 'Grade'],
#             label = 'Urban', shade = True)
# sns.kdeplot(df.ix[df['address'] == 'R', 'Grade'],
#             label = 'Rural', shade = True)# Add labeling
# plt.xlabel('Grade')
# plt.ylabel('Density')
# plt.title('Density Plot of Final Grades by Location')
# plt.show()
#
# plt.savefig('plots/student_performance/grades_by_location.png')

#%% find correlation
df.corr()['Grade'].sort_values()

#%%
df.describe()

#%% One hot encode variables
# Select only categorical variables
category_df = df.select_dtypes('object')# One hot encode the variables
dummy_df = pd.get_dummies(category_df)# Put the grade back in the dataframe
dummy_df['Grade'] = df['Grade']# Find correlations with grade
dummy_df.corr()['Grade'].sort_values()
