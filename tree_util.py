from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


def decision_tree(train_data, test_data, y_target, x_attributes,
                  min_samples_leaf=1, #evtl max_features
                  max_depth=None,
                  criterion="mse",
                  min_samples_split=2):
    train_x = train_data[x_attributes]
    train_y = train_data[y_target]

    test_x = test_data[x_attributes]
    test_y = test_data[y_target]

    model = DecisionTreeRegressor(min_samples_leaf=min_samples_leaf, max_depth=max_depth, criterion=criterion, min_samples_split=min_samples_split)
    model.fit(train_x, train_y)

    predict_test = model.predict(test_x)
                                                                                               #todo call the function with the data unprocessed, but use processing here to get MSE from
    root_mean_squared_error = np.sqrt(mean_squared_error(test_y, predict_test))

    return root_mean_squared_error #, predict_test



def decision_tree_crossvalidation(train_data, ytarget, x_attributes,
                                  criterion = 'mse',
                                  min_samples_leaf=1,
                                  splits=10,
                                  max_depth=None,
                                  min_samples_split=10):
    kf = KFold(n_splits=splits)
    kf.get_n_splits(train_data) # returns the number of splitting iterations in the cross-validatorprint(kf) KFold(n_splits=10, random_state=None, shuffle=False)

    count = 0
    sum_RMSE = 0
    for train_index, test_index in kf.split(train_data):
        #print("TRAIN:", train_index, "TEST:", test_index)
        #print("______________________")
        sum_RMSE = sum_RMSE + decision_tree(train_data=train_data.drop(test_index),test_data=train_data.drop(train_index),y_target=ytarget,x_attributes=x_attributes,
                                            min_samples_leaf=min_samples_leaf,
                                            max_depth=max_depth,
                                            criterion=criterion,
                                            min_samples_split=min_samples_split)
        count = count + 1

    mean_root_mean_squared_error = sum_RMSE/count
    #print("Crossvalidation mean_root_mean_squared_error : ", mean_root_mean_squared_error)
    return  mean_root_mean_squared_error


def decision_tree_regression_criterion_comparison(train_data, ytarget, x_attributes, name,
                                  criterion,# = ['mse', 'friedman_mse'],
                                  plot=True, splits=10, min_samples_leaf=1,max_depth=None):
    mrmse_val = []
    best_mrmse = np.infty
    for x in criterion:
        mrmse = decision_tree_crossvalidation(train_data=train_data, ytarget=ytarget, x_attributes=x_attributes,
                                              min_samples_leaf=min_samples_leaf,splits=splits,                                              max_depth=max_depth,
                                              criterion = x)
        if x == 'mse':
            best_mrmse = mrmse
            best_criterion = x
        elif mrmse < best_mrmse:
            best_mrmse = mrmse
            best_criterion = x

        mrmse_val.append(mrmse)  # store rmse values

    if plot:
        t = np.arange(0, len(mrmse_val))
        plt.plot(criterion, mrmse_val[0:len(mrmse_val)] , '--', label=name)
        plt.xticks(t)
        plt.xlabel('criterion')
        plt.ylabel('rmse')
        plt.scatter(best_criterion, best_mrmse)
        plt.legend()

    return best_mrmse, best_criterion

def decision_tree_regression_max_depth_comparison(train_data, ytarget, x_attributes, name,
                                                  max_depth_from,
                                                  max_depth_to,
                                                  step,
                                                  plot=True, splits=10,criterion='mse',min_samples_leaf=1):
    mrmse_val = []
    best_mrmse = np.infty
    for x in np.arange(max_depth_from, max_depth_to, step):
        mrmse = decision_tree_crossvalidation(train_data=train_data, ytarget=ytarget, x_attributes=x_attributes,
                                              criterion=criterion, min_samples_leaf=min_samples_leaf,splits=splits,
                                              max_depth=x)
        if x == max_depth_from:
            best_mrmse = mrmse
            best_max_depth = x
        elif mrmse < best_mrmse:
            best_mrmse = mrmse
            best_max_depth = x

        mrmse_val.append(mrmse)  # store rmse values

    if plot:
        t = np.arange(0, len(mrmse_val))
        plt.plot(t * step + 1, mrmse_val[0:len(mrmse_val)] , '--', label=name)
        plt.xticks(t * step)
        plt.xlabel('Max Depth')
        plt.ylabel('Root Mean Squared Error')
        plt.scatter(best_max_depth, best_mrmse)
        plt.legend()

    return best_mrmse, best_max_depth

def decision_tree_comparison(train_data, ytarget, x_attributes_tupels_list,
                             comp_type,
                             #max_depth_from, max_depth_to, max_depth_step,
                             #min_samples_leaf_from,min_samples_leaf_to, min_samples_leaf_step,
                             p_from, p_to, p_step,
                             criterion='mse',
                             plot=True, splits=10, min_samples_leaf=1):
    if (comp_type == 'criterion'):
        for x_attributes in x_attributes_tupels_list:
            decision_tree_regression_criterion_comparison(train_data,ytarget, x_attributes[0], x_attributes[1],
                                                          criterion,)
    elif(comp_type == 'max_depth'):
        for x_attributes in x_attributes_tupels_list:
            decision_tree_regression_max_depth_comparison(train_data, ytarget, x_attributes[0], x_attributes[1],
                                                          p_from, p_to, p_step)
                                                          #max_depth_from,
                                                          #max_depth_to,
                                                          #max_depth_step)
    elif(comp_type == 'min_samples_leaf'):
        for x_attributes in x_attributes_tupels_list:
            decision_tree_regression_min_samples_leaf_comparison(train_data, ytarget, x_attributes[0], x_attributes[1],
                                                                 p_from, p_to, p_step)
                                                                 #min_samples_leaf_from,
                                                                 #min_samples_leaf_to,

    elif (comp_type == 'min_samples_split'):
        for x_attributes in x_attributes_tupels_list:
            decision_tree_regression_min_samples_split_comparison(train_data, ytarget, x_attributes[0],
                                                                  x_attributes[1],
                                                                  p_from, p_to, p_step)
                #min_samples_leaf_step)


#todo verschieden Hyperparameter ineinander verschachteln und das beste zurück geben
def decision_tree_regression_min_samples_leaf_comparison(train_data, ytarget, x_attributes, name,
                                                  min_samples_leaf_from,
                                                  min_samples_leaf_to,
                                                  step,
                                                  plot=True, splits=10,criterion='mse',max_depth=None):
    mrmse_val = []
    best_mrmse = np.infty
    x_steps = np.arange(min_samples_leaf_from, min_samples_leaf_to, step)
    for x in x_steps:
        mrmse = decision_tree_crossvalidation(train_data=train_data, ytarget=ytarget, x_attributes=x_attributes,
                                              criterion=criterion,splits=splits, max_depth=max_depth,
                                              min_samples_leaf=x)
        if x == min_samples_leaf_from:
            best_mrmse = mrmse
            best_min_samples_leaf = x
        elif mrmse < best_mrmse:
            best_mrmse = mrmse
            best_min_samples_leaf = x

        mrmse_val.append(mrmse)  # store rmse values

    if plot:
        plt.plot(x_steps, mrmse_val[0:len(mrmse_val)] , '--', label=name)
        plt.xticks(x_steps)
        plt.xlabel('Minimum samples split')
        plt.ylabel('rmse')
        plt.scatter(best_min_samples_leaf, best_mrmse)
        plt.legend()

    return best_mrmse, best_min_samples_leaf

def decision_tree_regression_min_samples_split_comparison(train_data, ytarget, x_attributes, name,
                                                          min_samples_split_from,
                                                          min_samples_split_to,
                                                          step,
                                                          plot=True, splits=10, criterion='mse', max_depth=None):
    mrmse_val = []
    best_mrmse = np.infty
    x_steps = np.arange(min_samples_split_from, min_samples_split_to, step)
    for x in x_steps:
        mrmse = decision_tree_crossvalidation(train_data=train_data, ytarget=ytarget, x_attributes=x_attributes,
                                              criterion=criterion,splits=splits, max_depth=max_depth,
                                              min_samples_split=x)
        if x == min_samples_split_from:
            best_mrmse = mrmse
            best_min_samples_split = x
        elif mrmse < best_mrmse:
            best_mrmse = mrmse
            best_min_samples_split = x

        mrmse_val.append(mrmse)  # store rmse values

    if plot:
        plt.plot(x_steps, mrmse_val[0:len(mrmse_val)] , '--', label=name)
        plt.xticks(x_steps)
        plt.xlabel('Minimum samples split')
        plt.ylabel('rmse')
        plt.scatter(best_min_samples_split, best_mrmse)
        plt.legend()

    return best_mrmse, best_min_samples_split