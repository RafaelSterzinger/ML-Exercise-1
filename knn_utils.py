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

def find_best_k(x_train, y_train, ks):
    """ Grid search, for efficient searching for k """
    params = {'n_neighbors':ks}
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
        print("Ratio of missing values:",data_ratio_missing)
    return data_ratio_missing


def trim_data(data, attributes):
    """ remove categorical and identifier columns """
    return data.drop(attributes, axis=1)


def create_cross_validation(data, target, test_size=0.3):
    """ create train and test set """
    train , test = train_test_split(data, test_size = test_size)
    x_train = train.drop(target, axis=1)
    y_train = train[target]
    x_test = test.drop(target, axis = 1)
    y_test = test[target]
    return x_train, y_train, x_test, y_test


def scale_min_max(x_train, x_test):
    """ scaling features with MinMax """
    minmax_scaler = MinMaxScaler(feature_range=(0, 1))
    x_train_minmax_scaled = pd.DataFrame(minmax_scaler.fit_transform(x_train))
    x_test_minmax_scaled = pd.DataFrame(minmax_scaler.fit_transform(x_test))
    return x_train_minmax_scaled, x_test_minmax_scaled


def scale_standard(x_train,x_test):
    """ scaling features with zscore """
    standard_scalar = StandardScaler()
    x_train_zscore_scaled = pd.DataFrame(standard_scalar.fit_transform(x_train))
    x_test_zscore_scaled = pd.DataFrame(standard_scalar.fit_transform(x_test))
    return x_train_zscore_scaled, x_test_zscore_scaled



def find_best_rmse(name,x_train, y_train, x_test, y_test, k_max=23, metric='minkowski', plot=True, debug=False):
    """ plot k vs rms and calculate best k """
    """
    :param k_max: maximum number of k to test for
    :param plot: True if matplotlib plot should be created
    :param debug: print additional info about accuracy test results
    :return: best k and best accuracy for given training and test set with features (k < k_max)
    """

    # Go through all k between k=1 and k=k_max-1 and find best_k and best_a
    # rsmes = np.zeros(k_max)  # Write rsmes for each k into here for plot to work...

    rmse_val = [] # to store rmse values for different k
    for k in range(1, k_max):
        model = neighbors.KNeighborsRegressor(n_neighbors = k,metric=metric)
        model.fit(x_train, y_train)  #fit the model
        pred=model.predict(x_test) #make prediction on test set
        rsme = sqrt(mean_squared_error(y_test, pred)) #calculate rmse

        if k == 1:
            best_rmse=rsme
            best_k=k
        elif rsme < best_rmse:
            best_rmse = rsme
            best_k = k

        rmse_val.append(rsme) #store rmse values
        if debug:
            print('RMSE value for k=', k, 'is:', rsme)

    if plot:
        t = np.arange(1, k_max - 1)
        plt.plot(t, rmse_val[1:k_max - 1], '--', label=name)
        plt.xticks(t)
        plt.xlabel('# neighbours (k)')
        plt.ylabel('rsme')
        plt.scatter(best_k-1, best_rmse)
        plt.legend()
    return best_rmse,best_k


