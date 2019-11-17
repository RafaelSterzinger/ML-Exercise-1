import utils
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold


def find_best_k(x_train, y_train, ks):
    """ Grid search, for efficient searching for k """
    params = {'n_neighbors': ks}
    knn = neighbors.KNeighborsRegressor()
    model = GridSearchCV(knn, params, cv=5)
    model.fit(x_train, y_train)
    best_k = model.best_params_
    return best_k


def get_ratio_missing(data, debug=False):
    """output ratio of missing values:"""
    data_missing = data.isna()
    data_num_missing = data_missing.sum()
    data_ratio_missing = data_num_missing / len(data)
    if debug:
        print("Ratio of missing values:", data_ratio_missing)
    return data_ratio_missing


def trim_data(data, attributes):
    """ remove categorical and identifier columns """
    return data.drop(attributes, axis=1)


def make_split(data, target, test_size=0.3):
    """ create train and test set """
    train, test = train_test_split(data, test_size=test_size)
    x_train = train.drop(target, axis=1)
    y_train = train[target]
    x_test = test.drop(target, axis=1)
    y_test = test[target]
    return x_train, y_train, x_test, y_test


def scale_min_max(data):
    """ scaling features with MinMax """
    minmax_scaler = MinMaxScaler(feature_range=(0, 1))
    x_train_minmax_scaled = pd.DataFrame(minmax_scaler.fit_transform(data), index=data.index,
                                         columns=data.columns)
    return x_train_minmax_scaled

def scale_min_max_without_target(data, target):
    """ scaling features with MinMax """
    data_wo_target = data.drop(target, axis = 1)
    minmax_scaler = MinMaxScaler(feature_range=(0, 1))
    x_train_minmax_scaled = pd.DataFrame(minmax_scaler.fit_transform(data_wo_target), index=data.index,
                                         columns=data_wo_target.columns)
    x_train_minmax_scaled[target] = data[target]
    return x_train_minmax_scaled


def scale_standard(data):
    """ scaling features with zscore """
    standard_scalar = StandardScaler()
    x_train_zscore_scaled = pd.DataFrame(standard_scalar.fit_transform(data), index=data.index,
                                         columns=data.colums)
    return x_train_zscore_scaled


def find_best_rmse(name, x_train, y_train, x_test, y_test, k_max=23, metric='euclidean', plot=True, debug=False):
    """ plot k vs rms and calculate best k """
    """
    :param k_max: maximum number of k to test for
    :param plot: True if matplotlib plot should be created
    :param debug: print additional info about accuracy test results
    :return: best k and best accuracy for given training and test set with features (k < k_max)
    """

    # Go through all k between k=1 and k=k_max-1 and find best_k and best_a
    # rsmes = np.zeros(k_max)  # Write rsmes for each k into here for plot to work...

    rmse_val = []  # to store rmse values for different k
    for k in range(1, k_max):
        model = neighbors.KNeighborsRegressor(n_neighbors=k, metric=metric)
        model.fit(x_train, y_train)  # fit the model
        pred = model.predict(x_test)  # make prediction on test set
        rsme = sqrt(mean_squared_error(y_test, pred))  # calculate rmse

        if k == 1:
            best_rmse = rsme
            best_k = k
        elif rsme < best_rmse:
            best_rmse = rsme
            best_k = k

        rmse_val.append(rsme)  # store rmse values
        if debug:
            print('RMSE value for k=', k, 'is:', rsme)

    if plot:
        t = range(1, k_max)
        plt.plot(t, rmse_val[0:k_max - 1], '--', label=name)
        plt.xticks(t)
        plt.xlabel('# neighbours (k)')
        plt.ylabel('rsme')
        plt.scatter(best_k, best_rmse)
        plt.legend()
    return best_rmse, best_k


def kNN(train_data, test_data, y_target, x_attributes,
        k = 23,
        metric = 'euclidean'):
    train_x = train_data[x_attributes]
    train_y = train_data[y_target]

    test_x = test_data[x_attributes]
    test_y = test_data[y_target]

    model = neighbors.KNeighborsRegressor(n_neighbors=k, metric=metric)
    model.fit(train_x, train_y)

    predict_test = model.predict(test_x)
    root_mean_squared_error = np.sqrt(mean_squared_error(test_y, predict_test))

    return root_mean_squared_error #, predict_test

def kNN_crossvalidation(train_data, ytarget, x_attributes,
                        k=23,
                        metric='euclidean',
                        splits=10):
    kf = KFold(n_splits=splits)
    kf.get_n_splits(train_data) # returns the number of splitting iterations in the cross-validatorprint(kf) KFold(n_splits=10, random_state=None, shuffle=False)

    count = 0
    sum_RMSE = 0
    for train_index, test_index in kf.split(train_data):
        sum_RMSE = sum_RMSE + kNN(train_data=train_data.drop(test_index),test_data=train_data.drop(train_index),y_target=ytarget,x_attributes=x_attributes,
                                  k=k,
                                  metric=metric)
        count = count + 1

    mean_root_mean_squared_error = sum_RMSE/count
    #print("Crossvalidation mean_root_mean_squared_error : ", mean_root_mean_squared_error)
    return  mean_root_mean_squared_error


def kNN_regression_k_comparison(train_data, ytarget, x_attributes, name,
                                k_from=1,
                                k_to=23,
                                step=2,
                                metric='euclidean',
                                plot=True, splits=10):
    mrmse_val = []
    best_mrmse = np.infty
    x_steps = np.arange(k_from, k_to, step)
    for x in x_steps:
        mrmse = kNN_crossvalidation(train_data=train_data, ytarget=ytarget, x_attributes=x_attributes,metric=metric,splits=splits,
                                    k=x)
        if x == k_from:
            best_mrmse = mrmse
            best_k = x
        elif mrmse < best_mrmse:
            best_mrmse = mrmse
            best_k = x

        mrmse_val.append(mrmse)  # store rmse values

    if plot:
        plt.plot(x_steps, mrmse_val[0:len(mrmse_val)] , '--', label=name)
        plt.xticks(x_steps)
        plt.xlabel('k')
        plt.ylabel('rmse')
        plt.scatter(best_k, best_mrmse)
        plt.legend()

    return best_mrmse, best_k