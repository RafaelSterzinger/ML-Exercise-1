import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from math import sqrt

#%% enforce black border on histogram bins
# TODO: move in main
plt.rcParams["patch.force_edgecolor"] = True
#%% read dataset
df = pd.read_csv('datasets/student_performance/StudentPerformance.shuf.train.csv', index_col=0)
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


#%% refactor data
# Takes in a dataframe, finds the most correlated variables with the
# grade and returns training and testing datasets
def highest_correlated_data(df, target, size=6):
    # category_df = df.select_dtypes('object')  # One hot encode the variables
    # Find correlations with the Grade
    most_correlated = df.corr().abs()[target].sort_values(ascending=False)
    # Maintain the top 6 most correlation features with Grade
    most_correlated = most_correlated[:size]

    return df[most_correlated.index.tolist()]

data = pd.get_dummies(df)
data = highest_correlated_data(df,'Grade')

#%%
from sklearn.model_selection import train_test_split# df is features and labels are the targets
# Split by putting 25% in the testing set
x_train, x_test, y_train, y_test = train_test_split(data.drop('Grade',axis=1),
                                                    data['Grade'],test_size = 0.25,
                                                    random_state=42)

#%%
from scipy import stats

# Calculate correlation coefficient
def corrfunc(x, y, **kws):
    r, _ = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f}".format(r),
                xy=(.1, .6), xycoords=ax.transAxes,
                size=24)


cmap = sns.cubehelix_palette(light=1, dark=0.1,
                             hue=0.5, as_cmap=True)

sns.set_context(font_scale=2)

# Pair grid set up
g = sns.PairGrid(x_train)

# Scatter plot on the upper triangle
g.map_upper(plt.scatter, s=10, color='red')

# Distribution on the diagonal
g.map_diag(sns.distplot, kde=False, color='red')

# Density Plot and Correlation coefficients on the lower triangle
g.map_lower(sns.kdeplot, cmap=cmap)
g.map_lower(corrfunc)

plt.show()

#%%
x_train.append(y_train)
y_train

#%%
data.head()

#%%
# X_train is our training data, we will make a copy for plotting
X_plot = data.copy()
# Compare grades to the median
X_plot['relation_median'] = (X_plot['Grade'] >= 12)
X_plot['Grade'] = X_plot['Grade'].replace({True: 'above',
                                           False: 'below'})
# Plot all variables in a loop
plt.figure(figsize=(12, 12))
for i, col in enumerate(X_plot.columns[:-1]):
    plt.subplot(3, 2, i + 1)
    subset_above = X_plot[X_plot['relation_median'] == 'above']
    subset_below = X_plot[X_plot['relation_median'] == 'below']
    sns.kdeplot(subset_above[col], label='Above Median')
    sns.kdeplot(subset_below[col], label='Below Median')
    plt.legend()
    plt.title('Distribution of %s' % col)

plt.tight_layout()

#%%
from sklearn.linear_model import BayesianRidge

clf = BayesianRidge(compute_score=True)
clf.fit(x_train, y_train)
pred=clf.predict(x_test)

rsme = sqrt(mean_squared_error(y_test, pred))  # calculate rmse
rsme

