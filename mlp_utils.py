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

from knn_utils import make_split


def convertTuple(tup):
    t = '('
    t +=  ','.join(str(x) for x in tup)
    t += ')'
    return t

def mlp_regression(train_data, x_attributes, y_target, layers, activation='relu'):
    rsme_layers = []
    for layer in layers:
        model = MLPRegressor(hidden_layer_sizes=layer,
                             activation=activation,
                             solver='adam',
                             learning_rate_init=0.01,
                             learning_rate='adaptive',
                             max_iter=10000)
        splits = 10
        kf = KFold(n_splits=splits)
        kf.get_n_splits(train_data)
        sum = 0
        train_data.index = range(0,len(train_data["Grade"]))
        for train_index, test_index in kf.split(train_data):
            train = train_data.drop(test_index)
            model.fit(train[x_attributes], train[y_target])
            test = train_data.drop(train_index)
            predict_test = model.predict(test[x_attributes])
            root_mean_squared_error = np.sqrt(mean_squared_error(test[y_target], predict_test))
            sum += root_mean_squared_error
        rsme_layers.append(sum/splits)

    layers = map(convertTuple, layers)
    df = pd.DataFrame(list(zip(layers,rsme_layers)),columns=['Layers','RSME'])
    sns.barplot(df["Layers"],df["RSME"])
    plt.show()


def mlp_regression_neu(train_data, x_attributes, y_target, layers, activation='relu'):
    rsme_layers = []
    for layer in layers:
        model = MLPRegressor(hidden_layer_sizes=layer,
                             activation=activation,
                             solver='adam',
                             learning_rate_init=0.01,
                             learning_rate='adaptive',
                             max_iter=10000)
        train_x, train_y, test_x, test_y = make_split(train_data,y_target)
        model.fit(train_x, train_y)
        predict_test = model.predict(test_x[x_attributes])
        root_mean_squared_error = np.sqrt(mean_squared_error(test_y[x_attributes], predict_test))
        rsme_layers.append(root_mean_squared_error)

    layers = map(convertTuple, layers)
    df = pd.DataFrame(list(zip(layers,rsme_layers)),columns=['Layers','RSME'])
    sns.barplot(df["Layers"],df["RSME"])
    plt.show()

def plot_loss(model):
    # Plot the 'loss_curve_' protery on model to see how well we are learning over the iterations
    # Use Pandas built in plot method on DataFrame to creat plot in one line of code
    pd.DataFrame(model.loss_curve_).plot()
    plt.show()
