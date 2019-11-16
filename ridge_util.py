import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


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
