import utils
from sklearn.model_selection import train_test_split
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
import seaborn as sns
from sklearn.model_selection import KFold


from knn_utils import make_split


def convertTuple(tup):
    t = '('
    t +=  ','.join(str(x) for x in tup)
    t += ')'
    return t


def mlp_regression(train_data, x_attributes, y_target, layers, activation='relu'):
    rsme_layers = []
    values = x_attributes[:]
    values.append(y_target)

    for layer in layers:
        model = MLPRegressor(hidden_layer_sizes=layer,
                             activation=activation,
                             solver='adam',
                             learning_rate_init=0.01,
                             learning_rate='adaptive',
                             max_iter=10000)
        train_x, train_y, test_x, test_y = make_split(train_data[values],y_target)
        model.fit(train_x, train_y)
        predict_test = model.predict(test_x)
        root_mean_squared_error = np.sqrt(mean_squared_error(test_y, predict_test))
        rsme_layers.append(root_mean_squared_error)

    layers = map(convertTuple, layers)
    activation = [activation]
    return pd.DataFrame(list(zip(activation*len(rsme_layers),layers,rsme_layers)),columns=['Activation','Layers','RSME'])

def plot_loss(model):
    # Plot the 'loss_curve_' protery on model to see how well we are learning over the iterations
    # Use Pandas built in plot method on DataFrame to creat plot in one line of code
    pd.DataFrame(model.loss_curve_).plot()
    plt.show()

def mlp_regression_only(train_data, test_data, x_attributes, y_target, layer, activation='relu'):
    train_x = train_data[x_attributes]
    train_y = train_data[y_target]

    test_x = test_data[x_attributes]
    test_y = test_data[y_target]

    model = MLPRegressor(hidden_layer_sizes=layer,
                         activation=activation,
                         solver='adam',
                         learning_rate_init=0.01,
                         learning_rate='adaptive',
                         max_iter=10000)
    model.fit(train_x, train_y)
    predict_test = model.predict(test_x)
    root_mean_squared_error = np.sqrt(mean_squared_error(test_y, predict_test))

    return root_mean_squared_error


def mlp_regression_cross_validation(train_data, ytarget, x_attributes,
                                    layer,
                                    activation='relu',
                                    splits=10):
    kf = KFold(n_splits=splits)
    kf.get_n_splits(train_data) # returns the number of splitting iterations in the cross-validatorprint(kf) KFold(n_splits=10, random_state=None, shuffle=False)

    count = 0
    sum_RMSE = 0
    for train_index, test_index in kf.split(train_data):
        sum_RMSE = sum_RMSE + mlp_regression_only(train_data=train_data.drop(test_index),test_data=train_data.drop(train_index),y_target=ytarget,x_attributes=x_attributes,
                                  layer=layer,
                                  activation=activation)
        count = count + 1

    mean_root_mean_squared_error = sum_RMSE/count
    #print("Crossvalidation mean_root_mean_squared_error : ", mean_root_mean_squared_error)
    return  mean_root_mean_squared_error


def mlp_regression_layer_comparison(train_data, x_attributes, y_target, layers, activation='relu'):
    rsme_layers = []
    values = x_attributes[:]
    values.append(y_target)

    for layer in layers:
        rsme_layers.append(mlp_regression_cross_validation(train_data, y_target,x_attributes,layer,activation))

    layers = map(convertTuple, layers)
    activation = [activation]
    return pd.DataFrame(list(zip(activation*len(rsme_layers),layers,rsme_layers)),columns=['Activation','Layers','RSME'])


