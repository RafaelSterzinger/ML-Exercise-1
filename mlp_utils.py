import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


def mlp(train_data, test_data, target, attributes=None):
    if attributes is None:
        train_x = train_data.drop[target]
        train_y = train_data[target]
        test_x = test_data.drop[target]
        test_y = test_data[target]
    else:
        train_x = train_data[attributes]
        train_y = train_data[target]
        test_x = test_data[attributes]
        test_y = test_data[target]

    model = MLPRegressor(hidden_layer_sizes=(20, 4),
                         activation='logistic',
                         solver='adam',
                         learning_rate_init=0.001,
                         learning_rate='adaptive',
                         max_iter=10000)

    model.fit(train_x, train_y)

    predict_test = model.predict(test_x)
    root_mean_squared_error = np.sqrt(mean_squared_error(test_y, predict_test))

    return root_mean_squared_error


def mlp_cross_validation(train_data, target, x_attributes=None, splits=10):
    kf = KFold(n_splits=splits)
    kf.get_n_splits(train_data) # returns the number of splitting iterations in the cross-validatorprint(kf) KFold(n_splits=10, random_state=None, shuffle=False)

    count = 0
    sum_RMSE = 0
    for train_index, test_index in kf.split(train_data):
        sum_RMSE = sum_RMSE + mlp(train_data=train_data.drop(test_index),
                                  test_data=train_data.drop(train_index),
                                  target=target,
                                  attributes=x_attributes)
        count = count + 1

    mean_root_mean_squared_error = sum_RMSE/count
    return  mean_root_mean_squared_error



