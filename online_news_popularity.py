from utils import *
from ridge_util import *
from knn_utils import *
from tree_util import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from mlp_utils import *
from scipy import stats

#%% set global params for plots and display
plt.rcParams["patch.force_edgecolor"] = True
pd.set_option('display.expand_frame_repr', False)

#%% read data and remove useless attributes
train_data = pd.read_csv('datasets/online_news_popularity/OnlineNewsPopularity.csv', skipinitialspace=True)
train_data = train_data.drop(['url'], axis=1)
path = 'plots/online_news_popularity/'
#%% describe data
train_data.describe()
train_data['shares']

#%% plot all shares
sns.distplot(train_data["shares"], kde=True)
plt.xlabel('shares')
plt.ylabel('count')
plt.title('Distribution of Shares')
plt.savefig(path+'shares_histogram.png')
plt.show()

#%% find inliers/outliers with threshold
threshold = 10000
inlier=train_data[train_data['shares'] <= threshold]
outlier=train_data[train_data['shares'] > threshold]
print('inlier: ',len(inlier),'outlier',len(outlier))

#%% plot shares with threshold <= threshold
sns.distplot(inlier["shares"], kde=True, bins=20)
plt.xlabel('shares')
plt.ylabel('count')
plt.title('Distribution of Shares <= '+str(threshold))
plt.savefig('plots/online_news_popularity/shares_histogram.png')
plt.show()

#%% plot shares with threshold > threshold
sns.distplot(outlier["shares"], kde=True, bins=20)
plt.xlabel('shares')
plt.ylabel('count')
plt.title('Distribution of Shares > '+str(threshold))
plt.savefig(path+'shares_histogram.png')
plt.show()

#%% find attributes with highest correlation
highest_correlated=highest_correlated_data_as_list(train_data,'shares',10)
highest_correlated.remove('shares')

#%%


#%% compare ridge regression
rmse,alpha = ridge_regression_alpha_comparison(train_data,
                                  'shares',
                                  highest_correlated,
                                  0, 600, 50,
                                  "highest correlated attributes")
plt.show()
plt.savefig(path+'ridge_regression_alpha_comparison.png')
print('best rmse: ',rmse, 'best alpha', alpha)

#%% compare knn
knn_regression_k_comparison(train_data,
                            'shares',
                            highest_correlated,
                            'highest 30 correlated attributes with euclidean',
                            k_to=25)
plt.show()

#%%
knn_regression_k_comparison(inlier,
                            'shares',
                            highest_correlated,
                            'highest 30 correlated attributes with euclidean',
                            k_to=25)

plt.show()
plt.savefig(path+'knn_k_comparison.png')

#%% compare decision trees
# %% min depth comparison
decision_tree_comparison(train_data,
                         'shares',
                         [highest_correlated],
                         comp_type='max_depth',
                         p_from=1,
                         p_to=30,
                         p_step=2)
plt.show()
