import pandas as pd
import seaborn as sns
from matplotlib import pyplot
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


path = "./plots/student_performance/"
data_initials = "sp_"
target_attribute = "Grade"
numeric_attributes = ["Grade", "age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences"]

train_data = pd.read_csv("datasets/student_performance/StudentPerformance.shuf.train.csv")
test_data = pd.read_csv("datasets/student_performance/StudentPerformance.shuf.test.csv")
test_label = pd.read_csv("datasets/student_performance/StudentPerformance.shuf.sampleSolution.csv")

#%% Vizualisation Check for missing values
print(train_data.isnull().sum())

fig, ax = pyplot.subplots(figsize=(6, 6))

# Histogram of target value
plot = sns.distplot(train_data[target_attribute], ax=ax).get_figure()
plot.savefig(path + data_initials + "hist1.png")

# Heatmap to check correlation
correlation_matrix = train_data.corr().round(2)
plot = sns.heatmap(correlation_matrix, ax=ax, linewidths=.5).get_figure()
plot.savefig(path + data_initials + "heatmap.png")

#%% Gradient Boosting Classifier
def gradient_boosting_classifier(train_data, test_data, test_label):
    # shape of the dataset
        #print('Shape of training data :',train_data.shape)
        #print('Shape of testing data :',test_data.shape)

    # Now, we need to predict the missing target variable in the test data
    # target variable - Survived

    # seperate the independent and target variable on training data
    train_x = train_data.drop(columns=[target_attribute],axis=1)
    train_y = train_data[target_attribute]

    # seperate the independent and target variable on testing data
    test_x = test_data.drop(columns=[target_attribute],axis=1)
    test_y = test_data[target_attribute]


    '''
    Create the object of the GradientBoosting Classifier model
    You can also add other parameters and test your code here
    Some parameters are : learning_rate, n_estimators
    Documentation of sklearn GradientBoosting Classifier:
    
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
    '''
    model = GradientBoostingClassifier(n_estimators=100,max_depth=5)

    # fit the model with the training data
    model.fit(train_x,train_y)

    # predict the target on the train dataset
    predict_train = model.predict(train_x)
    print('\nTarget on train data',predict_train)

    # Accuray Score on train dataset
    accuracy_train = accuracy_score(train_y,predict_train)
    print('\naccuracy_score on train dataset : ', accuracy_train)

    # predict the target on the test dataset
    predict_test = model.predict(test_x)
    print('\nTarget on test data',predict_test)

    # Accuracy Score on test dataset
    accuracy_test = accuracy_score(test_y,predict_test)
    print('\naccuracy_score on test dataset : ', accuracy_test)

#%% Decision Tree
def decision_tree(train_data, test_data, test_label, bool_accuracy_score):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import mean_squared_error
    from math import sqrt

    root_mean_squared_error = -1                         #only changed if bool_accuracy_score = true

    # shape of the dataset
    # print('Shape of training data :',train_data.shape)
    # print('Shape of testing data :',test_data.shape)

    # Now, we need to predict the missing target variable in the test data
    # target variable - Survived

    # seperate the independent and target variable on training data
    train_x = train_data.drop(columns=[target_attribute], axis=1)
    train_y = train_data[target_attribute]

    # seperate the independent and target variable on testing data
    test_x = test_data.drop(columns=[target_attribute], axis=1)

    if (bool_accuracy_score):
        test_y = test_data[target_attribute]

    '''
    Create the object of the Decision Tree model
    You can also add other parameters and test your code here
    Some parameters are : max_depth and max_features
    Documentation of sklearn DecisionTreeClassifier: 

    https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

     '''
    model = DecisionTreeClassifier()

    # fit the model with the training data
    model.fit(train_x, train_y)

    # depth of the decision tree
    print('Depth of the Decision Tree :', model.get_depth())

    '''
    # predict the target on the train dataset
    predict_train = model.predict(train_x)
    print('Target on train data', predict_train)

    # Accuray Score on train dataset
    accuracy_train = accuracy_score(train_y, predict_train)
    print('accuracy_score on train dataset : ', accuracy_train)
    '''

    # predict the target on the test dataset
    predict_test = model.predict(test_x)
    print('Target on test data', predict_test)

    # Accuracy Score on test dataset
    if (bool_accuracy_score):
        accuracy_test = accuracy_score(test_y, predict_test)
        print('accuracy_score on test dataset : ', accuracy_test)

    # Mean Squared Error on test dataset
    if (bool_accuracy_score):
        mean_squared_error = mean_squared_error(test_y, predict_test)
        root_mean_squared_error = sqrt(mean_squared_error)
        print('root mean_squared_error on test dataset : ', root_mean_squared_error)


    return root_mean_squared_error, predict_test

#%% Kfold Test Student Performance
from sklearn.model_selection import KFold
kf = KFold(n_splits=10)
kf.get_n_splits(train_data) # returns the number of splitting iterations in the cross-validatorprint(kf) KFold(n_splits=10, random_state=None, shuffle=False)

count = 1
sum_RMSE = 0
for train_index, test_index in kf.split(train_data):
    #print("TRAIN:", train_index, "TEST:", test_index)
    print("______________________")
    '''gradient_boosting_classifier(train_data.drop(test_index)[numeric_attributes],
                                 train_data.drop(train_index)[numeric_attributes],
                                 test_label)
    '''
    sum_RMSE = sum_RMSE + decision_tree(               train_data.drop(test_index)[numeric_attributes],
                                 train_data.drop(train_index)[numeric_attributes],
                                 test_label, True)[0]
    count = count + 1

mean_root_mean_squared_error = sum_RMSE/count
print("Crossvalidation mean_root_mean_squared_error : ", mean_root_mean_squared_error)

#%% Prediction Student Performance
student_y_pred = decision_tree(train_data,test_data, train_data, False)[1]

student_submission_tree = test_label.assign(Grade=student_y_pred)
print(student_submission_tree)
student_submission_tree.to_csv(path + "tree_pred.csv", index=False)


#%%