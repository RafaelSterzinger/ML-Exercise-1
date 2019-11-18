from utils import highest_correlated_data_as_list, highest_correlated_data
from ridge_util import *
from knn_utils import *
from tree_util import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from mlp_utils import *
from scipy import stats
import numpy as np

#%%
self_chosen_attributes = ['kw_avg_avg',
                          'timedelta',
                          'data_channel_is_lifestyle',
                          'data_channel_is_entertainment',
                          'data_channel_is_bus',
                          'data_channel_is_socmed',
                          'data_channel_is_tech',
                          'data_channel_is_world',
                          'num_imgs',
                          'self_reference_avg_sharess',
                          'num_hrefs',
                          'global_subjectivity',
                          'num_videos',
                          'num_keywords',
                          'LDA_03']

#%% set global params for plots and display
plt.rcParams["patch.force_edgecolor"] = True
pd.set_option('display.expand_frame_repr', False)

#%% read data and remove useless attributes
train_data = pd.read_csv('datasets/online_news_popularity/OnlineNewsPopularity.csv', skipinitialspace=True)
train_data = train_data.drop(['url'], axis=1)
path = 'plots/online_news_popularity/'

# %% plot all shares
sns.distplot(train_data["shares"], kde=True)
plt.xlabel('shares')
plt.ylabel('count')
plt.title('Distribution of Shares')
plt.savefig(path + 'original_shares_histogram.png')
plt.show()

#%%
train_data.describe()

#%% recent train data without outliers
train_data_recent = train_data[train_data['timedelta'] < 30]
train_data_recent.describe()
train_data_recent.index = np.arange(0, len(train_data_recent), 1)

#%% plot recent
sns.distplot(train_data_recent["shares"], kde=True)
plt.xlabel('shares')
plt.ylabel('count')
plt.title('Distribution of Shares')
plt.savefig(path + 'recent_shares_histogram.png')
plt.show()

#%% remove outliers
train_data_recent = train_data_recent[(np.abs(stats.zscore(train_data_recent)) < 3).all(axis=1)]
train_data_recent.index = np.arange(0, len(train_data_recent), 1)

#%%
sns.distplot(train_data_recent["shares"], kde=True)
plt.xlabel('shares')
plt.ylabel('count')
plt.title('Distribution of Shares')
plt.savefig(path + 'normalized_recent_shares_histogram.png')
plt.show()


#%% describe data

sns.distplot(train_data["timedelta"], kde=True)
plt.xlabel('timedelta')
plt.ylabel('count')
plt.title('Distribution of timedelta')
plt.savefig(path + 'timedelta.png')
plt.show()

#%% plot highest correlated
highest_correlated = highest_correlated_data_as_list(train_data_recent, 'shares', 10)
correlation_matrix = train_data_recent[highest_correlated].corr().abs()
sns.heatmap(correlation_matrix, square=True, linewidths=.5).get_figure()
plt.show()
plt.savefig(path + 'heatmap_highest_correlated.png')

#%%
sns.distplot(train_data_recent["shares"], kde=True)
plt.xlabel('shares')
plt.ylabel('count')
plt.title('Distribution of Shares')
plt.savefig(path + 'shares_histogram.png')
plt.show()

#%%
print(len(train_data_recent))

#%% plot own attributes
correlation_matrix = train_data_recent[highest_correlated].corr().abs()
sns.heatmap(correlation_matrix, square=True, linewidths=.5).get_figure()
plt.show()
plt.savefig(path + 'heatmap_self_chosen.png')


# %% plot shares with threshold <= threshold
sns.distplot(train_data_recent["shares"], kde=True, bins=20)
plt.xlabel('shares')
plt.ylabel('count')
plt.title('Distribution of Shares where zscore < 2.1')
plt.savefig('plots/online_news_popularity/shares_histogram.png')
plt.show()

# %% print highest correlated attributes
highest_correlated = highest_correlated_data(train_data, 'shares', 60)
print(highest_correlated)

# %% find attributes with highest correlation
highest_correlated = highest_correlated_data_as_list(train_data, 'shares', 11)
highest_correlated.remove('shares')


# %% compare ridge regression
rmse, alpha = ridge_regression_alpha_comparison(train_data_recent,
                                                'shares',
                                                highest_correlated,
                                                0, 600, 25,
                                                "highest correlated attributes")


rmse, alpha = ridge_regression_alpha_comparison(train_data_recent,
                                                'shares',
                                                highest_correlated,
                                                0, 600, 25,
                                                "train data recent highest correlated")


'''
rmse, alpha = ridge_regression_alpha_comparison(inlier,
                                                'shares',
                                                self_chosen_attributes,
                                                0, 600, 25,
                                                "self chosen attributes")
'''
plt.show()
plt.savefig(path + 'ridge_regression_alpha_comparison.png')
print('best rmse: ', rmse, 'best alpha', alpha)

# %% compare knn
knn_regression_k_comparison(train_data_recent,
                            'shares',
                            highest_correlated,
                            'highest ' + str(len(highest_correlated)) + ' correlated attributes with euclidean',
                            k_to=50)

knn_regression_k_comparison(train_data_recent,
                            'shares',
                            self_chosen_attributes,
                            'self chosen attributes',
                            k_to=50)
plt.show()

# %%
# knn_regression_k_comparison(inlier,
#                             'shares',
#                             highest_correlated,
#                             'highest 30 correlated attributes with euclidean',
#                             k_to=25)
#
# knn_regression_k_comparison(inlier,
#                             'shares',
#                             self_chosen_attributes,
#                             '15 self chosen attributes',
#                             k_to=25)
#
# plt.show()
# plt.savefig(path + 'knn_k_comparison.png')

#%% compare decision trees


#%% min depth comparison
decision_tree_comparison(train_data_recent,
                         'shares',
                         [[highest_correlated, 'highest correlated without inliers']],
                         comp_type='max_depth',
                         p_from=1,
                         p_to=30,
                         p_step=2)

decision_tree_comparison(train_data_recent,
                         'shares',
                         [[highest_correlated, 'highest correlated with inliers']],
                         comp_type='max_depth',
                         p_from=1,
                         p_to=30,
                         p_step=2)


decision_tree_comparison(train_data_recent,
                         'shares',
                         [[highest_correlated, 'REC highest correlated without inliers']],
                         comp_type='max_depth',
                         p_from=1,
                         p_to=30,
                         p_step=2)

decision_tree_comparison(train_data_recent,
                         'shares',
                         [[highest_correlated, 'REC highest correlated with inliers']],
                         comp_type='max_depth',
                         p_from=1,
                         p_to=30,
                         p_step=2)
plt.show()

#%%
print(len(train_data_recent))

#%%
highest_correlated.append('shares')

#%%
print(len(highest_correlated))

#%%
models = [(5, 7, 5)]
df = mlp_regression(train_data_recent, highest_correlated, 'shares', models, "logistic")
df = pd.concat([df, mlp_regression(train_data_recent, highest_correlated, 'shares', models, "relu")])
df = pd.concat([df, mlp_regression(train_data_recent, highest_correlated, 'shares', models, "tanh")])

sns.catplot(x='Layers', y='RMSE', hue='Activation',data = df, kind='bar')

plt.savefig(path + "mlp_comparision_top_correlating_outlier.png")
plt.show()

