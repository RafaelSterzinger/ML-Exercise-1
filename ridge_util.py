import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

#todo DEPRECATED
'''
def ridgeRegression(train, test, alphaFrom, alphaTo, step, attr, target, groundTruth, name, plot=True):
    train_X = train[attr]
    train_y = train[target]

    rmse_val = []
    best_rmse = np.infty
    for x in np.arange(alphaFrom, alphaTo, step):
        regr = linear_model.Ridge(alpha=x)
        regr.fit(train_X, train_y)
        pred = regr.predict(test[attr])
        rmse = np.sqrt(mean_squared_error(groundTruth, pred))
        if x == alphaFrom:
            best_rmse = rmse
            best_alpha = x
        elif rmse < best_rmse:
            best_rmse = rmse
            best_alpha = x

        rmse_val.append(rmse)  # store rmse values

    if plot:
        t = np.arange(0,len(rmse_val))
        plt.plot(t, rmse_val[0:len(rmse_val)], '--', label=name)
        plt.xticks(t)
        plt.xlabel('alpha')
        plt.ylabel('rsme')
        plt.scatter(best_alpha, best_rmse)
        plt.legend()
    return best_rmse, best_alpha
'''

def ridge_regression(train_data, test_data, x_attributes, y_target, alpha=1):
    train_x = train_data[x_attributes]
    train_y = train_data[y_target]

    test_x = test_data[x_attributes]
    test_y = test_data[y_target]

    regr = linear_model.Ridge(alpha=alpha)
    regr.fit(train_x, train_y)
    pred = regr.predict(test_x)
    rmse = np.sqrt(mean_squared_error(test_y, pred))

    return rmse

def ridge_regression_crossvalidation(train, target, attr, alpha=1, splits=10):
    kf = KFold(n_splits=splits)
    kf.get_n_splits(train) # returns the number of splitting iterations in the cross-validatorprint(kf) KFold(n_splits=10, random_state=None, shuffle=False)

    count = 0
    sum_RMSE = 0
    for train_index, test_index in kf.split(train):
        #print("TRAIN:", train_index, "TEST:", test_index)
        #print("______________________")

        sum_RMSE = sum_RMSE + ridge_regression(train.drop(test_index),
                                               train.drop(train_index),
                                               attr,
                                               target,
                                               alpha)
        count = count + 1

    mean_root_mean_squared_error = sum_RMSE/count
    #print("Crossvalidation mean_root_mean_squared_error : ", mean_root_mean_squared_error)
    return  mean_root_mean_squared_error


def ridge_regression_alpha_comparison(train, target, attr, alpha_from, alpha_to, step, name, plot=True):
    mrmse_val = []
    best_mrmse = np.infty
    for x in np.arange(alpha_from, alpha_to, step):
        mrmse = ridge_regression_crossvalidation(train,
                                                 target, attr,
                                                 x)
        if x == alpha_from:
            best_mrmse = mrmse
            best_alpha = x
        elif mrmse < best_mrmse:
            best_mrmse = mrmse
            best_alpha = x

        mrmse_val.append(mrmse)  # store rmse values

    if plot:
        t = np.arange(0, len(mrmse_val))
        plt.plot(t * step, mrmse_val[0:len(mrmse_val)] , '--', label=name)
        plt.xticks(t * step)
        plt.xlabel('Alpha')
        plt.ylabel('Root Mean Squared Error')
        plt.scatter(best_alpha, best_mrmse)
        plt.legend()

    return best_mrmse, best_alpha