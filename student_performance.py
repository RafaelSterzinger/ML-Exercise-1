#%% import
import pandas as pd
import seaborn as sns
from matplotlib import pyplot
import numpy as np
import matplotlib.pyplot as plt
from ridge_util import *
from tree_util import *

#%% Student performance init

path_student = "./plots/student_performance/"
data_initials_student = "sp_"
target_attribute_student = "Grade"
numeric_attributes_student = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences"]
other_attributes_student = ["failures", "Medu","studytime", "goout", "age", "freetime",   "Fedu", "absences"]           #schneidet am besten ab bis jetzt
top7_attributes_student =  ["failures", "Medu","studytime", "goout", "age", "traveltime", "Fedu"]
all_attributes_tupels_list = [[numeric_attributes_student, 'Numeric attributes'], [other_attributes_student, 'Other attributes'], [top7_attributes_student, 'Top 7 attributes']]


#1, 2, 3, 6, 7, 8, 13, 14, 15, 21, 22, 30 , 31, 32, 33
#categorical data that is missing above: "Pstatus","higher" (corr:0.21, wants higher education), "internet", "absences"
# comparing with means we see that school GP is  but Pstatus is not correlated https://rstudio-pubs-static.s3.amazonaws.com/108835_65a73467d96f4c79a5f808f5b8833922.html
# Walc, Dalc should also be related since the more you drink the worse your grades
binary_attributes_and_target = ["Grade", "school", "sex", "address", "famsize", "Pstatus", "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic"]
binary_tupels = [["school", "GP"], ["sex", "F"], ["address", "U"], ["famsize", "LE3"], ["Pstatus", "T"], ["schoolsup", "yes"], ["famsup", "yes"], ["paid", "yes"], ["activities", "yes"], ["nursery", "yes"], ["higher", "yes"], ["internet", "yes"], ["romantic", "yes"]]

nominal_attributes = ["Mjob", "Fjob", "reason", "guardian"]

train_data_student = pd.read_csv("datasets/student_performance/StudentPerformance.shuf.train.csv")
test_data_student = pd.read_csv("datasets/student_performance/StudentPerformance.shuf.test.csv")
test_label_student = pd.read_csv("datasets/student_performance/StudentPerformance.shuf.sampleSolution.csv")

#%% Bike Sharing init
path_bike = "./plots/bike_sharing/"

train_data_bike = pd.read_csv("datasets/bike_sharing/bikeSharing.shuf.train.csv")
test_data_bike = pd.read_csv("datasets/bike_sharing/bikeSharing.shuf.test.csv")
test_label_bike = pd.read_csv("datasets/bike_sharing/bikeSharing.shuf.sampleSolution.csv")

target_attribute_bike = 'cnt'
attr_bike = ["temp", "atemp", "hum", "hr", "season"]


#%% Vizualisation Check for missing values
print(train_data_student.isnull().sum())

fig, ax = pyplot.subplots(figsize=(6, 6))

# Histogram of target value
plot = sns.distplot(train_data_student[target_attribute_student], ax=ax).get_figure()
plot.savefig(path_student + data_initials_student + "hist1.png")

# Heatmap to check correlation
correlation_matrix = train_data_student.corr().round(2).abs()
cnt_corr = correlation_matrix["Grade"]#.nlargest(10)
print("8 largest correlations", cnt_corr)

plot = sns.heatmap(correlation_matrix, ax=ax, linewidths=.5, annot=True).get_figure()
plot.savefig(path_student + data_initials_student + "heatmap.png")


#%% Data analysis
print("Mean by school", train_data_student.groupby('school')[target_attribute_student].mean())
pairplot_student = sns.pairplot(data = train_data_student[numeric_attributes_student])              #todo is missing Grade to see visualisation
pairplot_student.savefig(path_student + data_initials_student + "pairplot_numeric_values.png")

#plt.figure(figsize=(10,6))
# positive correlation if parent is teacher, negative if stay at home
boxplot_student_father = sns.boxplot(x = train_data_student["Fjob"], y=train_data_student[target_attribute_student])
boxplot_student_father.savefig(path_student + data_initials_student + "boxplot_father.png")
boxplot_student_mother = sns.boxplot(x = train_data_student["Mjob"], y=train_data_student[target_attribute_student])
boxplot_student_mother.savefig(path_student + data_initials_student + "boxplot_mother.png")

# females get better grades
boxplot_student_gender = sns.boxplot(x = train_data_student["sex"], y = train_data_student[target_attribute_student])
boxplot_student_gender.savefig(path_student + data_initials_student + "boxplot_gender.png")
'''
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


   
    Create the object of the GradientBoosting Classifier model
    You can also add other parameters and test your code here
    Some parameters are : learning_rate, n_estimators
    Documentation of sklearn GradientBoosting Classifier:
    
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
   
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
'''
#%% transform categorical into numerical/dummy variables
# only True or False -> Point Biserial Correlation
train_data_student_binary_encoded = train_data_student#[[target_attribute_student, 'higher']]
#higher_education['higher'] = (higher_education['higher'] == 'yes').astype(int)
#higher_education.assign(Grade = train_data_student[target_attribute_student])
for i in range(len(binary_tupels)):
    train_data_student_binary_encoded[binary_tupels[i][0]] = (train_data_student_binary_encoded[binary_tupels[i][0]] == binary_tupels[i][1] ).astype(int)
# multiple variables -> ANOVA (analysis of variance) the higher the F score the higher the correlation

correlation_matrix_binary = train_data_student_binary_encoded[binary_attributes_and_target].corr().round(2).abs()
bin_corr = correlation_matrix_binary[target_attribute_student].nlargest(13)

print("higher education corr", bin_corr)

# create categories


# one hot encoding -> each value becomes a new column
#pd.get_dummies(train_data_student, columns=["reason"]).head()


#%% Decision Tree
'''def decision_tree(train_data, test_data, y_target, x_attributes, test_label, bool_accuracy_score, min_samples_leaf=1,  max_depth=None, criterion="mse"):#'gini'):            only for Classification
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
    train_x = train_data[x_attributes]
    train_y = train_data[y_target]                                                                                      #todo double brackets if single value changes type? train_data[[y_target]]

    # seperate the independent and target variable on testing data
    test_x = test_data[x_attributes]
    #if (bool_accuracy_score):
    #    test_x = test_data.drop(columns=[target_attribute], axis=1)

    if (bool_accuracy_score):
        test_y = test_data[y_target]

    
    Create the object of the Decision Tree model
    You can also add other parameters and test your code here
    Some parameters are : max_depth and max_features
    Documentation of sklearn DecisionTreeClassifier: 

    https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

    #model = DecisionTreeClassifier(criterion=criterion, min_samples_leaf=min_samples_leaf, max_depth=max_depth)
    model = DecisionTreeRegressor(min_samples_leaf=min_samples_leaf, max_depth=max_depth, criterion=criterion)


    # fit the model with the training data
    model.fit(train_x, train_y)

    # depth of the decision tree
    print('Depth of the Decision Tree :', model.get_depth())

    # predict the target on the train dataset
    predict_train = model.predict(train_x)
    print('Target on train data', predict_train)

    # Accuray Score on train dataset
    accuracy_train = accuracy_score(train_y, predict_train)
    print('accuracy_score on train dataset : ', accuracy_train)

    # predict the target on the test dataset
    predict_test = model.predict(test_x)
    print('Target on test data', predict_test)
    if (bool_accuracy_score):
        print('Actual test y values', test_y)

    # Accuracy Score on test dataset
    if (bool_accuracy_score):
        accuracy_test = accuracy_score(test_y, predict_test)
        print('accuracy_score on test dataset : ', accuracy_test)

    # Mean Squared Error on test dataset
    if (bool_accuracy_score):                                                                                           #todo call the function with the data unprocessed, but use processing here to get MSE from
        mean_squared_error = mean_squared_error(test_y, predict_test)
        root_mean_squared_error = sqrt(mean_squared_error)
        print('root mean_squared_error on test dataset : ', root_mean_squared_error)

    return root_mean_squared_error, predict_test
'''
#%% Kfold Test Student Performance
'''
def decision_tree_crossvalidation(train_data, target_attribute, numeric_attributes, test_label, criterion, min_samples_leaf=1, splits=10, max_depth=None):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=splits)
    kf.get_n_splits(train_data) # returns the number of splitting iterations in the cross-validatorprint(kf) KFold(n_splits=10, random_state=None, shuffle=False)

    count = 1
    sum_RMSE = 0
    for train_index, test_index in kf.split(train_data):
        print("TRAIN:", train_index, "TEST:", test_index)
        print("______________________")
gradient_boosting_classifier(train_data.drop(test_index)[numeric_attributes],
                                     train_data.drop(train_index)[numeric_attributes],
                                     test_label)

        sum_RMSE = sum_RMSE + decision_tree(train_data.drop(test_index),
                                            train_data.drop(train_index),
                                            target_attribute,
                                            numeric_attributes,
                                            test_label, True,
                                            min_samples_leaf,
                                            max_depth=max_depth,
                                            criterion=criterion)[0]
        count = count + 1

    mean_root_mean_squared_error = sum_RMSE/count
    print("Crossvalidation mean_root_mean_squared_error : ", mean_root_mean_squared_error)
    return  mean_root_mean_squared_error
'''
#%% Prediction Student Performance
print("prediction start")

student_y_pred = decision_tree(train_data_student, test_data_student, target_attribute_student, numeric_attributes_student, test_label_student, False, max_depth=6)[1]

print("prediction done")

student_submission_tree = test_label_student.assign(Grade=student_y_pred)
print(student_submission_tree)
student_submission_tree.to_csv(path_student + "tree_pred.csv", index=False)


#%% Kfold Test Student Performance with
list_MRMSE_student_mse = []
list_MRMSE_student_friedman_mse = []
min_samples_leaf_student = 100                      #bei ~100 am besten mit cross validation gini 128 & min_samples_leaf ~100 (ohne zu setzten für den Datensatz explizit, noch besser)
max_depth_student = 1
while max_depth_student < 15: #min_samples_leaf <= 200:
    MRMSE_student_mse = decision_tree_crossvalidation(train_data_student, target_attribute_student, numeric_attributes_student, test_label_student, criterion='mse', max_depth=max_depth_student)#, min_samples_leaf)
    MRMSE_student_friedman_mse = decision_tree_crossvalidation(train_data_student, target_attribute_student, numeric_attributes_student, test_label_student, criterion='friedman_mse', max_depth=max_depth_student)

    list_MRMSE_student_mse.append(MRMSE_student_mse)
    list_MRMSE_student_friedman_mse.append(MRMSE_student_friedman_mse)
    #min_samples_leaf = min_samples_leaf + 10
    max_depth_student = max_depth_student + 1

print("max depth", max_depth_student)
print("mse", list_MRMSE_student_mse)
print("friedman_mse", list_MRMSE_student_friedman_mse)


#%% Student performance ridge regression alpha comparison
ridge_regression_alpha_comparison(train_data_student, target_attribute_student,
                                  numeric_attributes_student,
                                  0, 50, 5,
                                  "Numeric")
ridge_regression_alpha_comparison(train_data_student, target_attribute_student,
                                  other_attributes_student,
                                  0, 50, 5,
                                  "Other attributes")

ridge_regression_alpha_comparison(train_data_student, target_attribute_student,
                                  top7_attributes_student,
                                  0, 50, 5,
                                  "Top 7 attributes")
plt.show()

#%% Student performance decision tree regression criterion comparison
decision_tree_regression_criterion_comparison(train_data_student,target_attribute_student,
                                              numeric_attributes_student,
                                              criterion = ['mse', 'friedman_mse'],
                                              name = "Numeric")

decision_tree_regression_criterion_comparison(train_data_student,target_attribute_student,
                                              other_attributes_student,
                                              criterion=['mse', 'friedman_mse'],
                                              name = "Other attributes")

decision_tree_regression_criterion_comparison(train_data_student,target_attribute_student,
                                              top7_attributes_student,
                                              criterion=['mse', 'friedman_mse'],
                                              name = "Top 7 attributes")
plt.show()

#%% Student performance decision tree regression max_depth comparison

decision_tree_comparison(train_data_student, target_attribute_student,all_attributes_tupels_list,
                         comp_type='max_depth',
                         max_depth_from=1,
                         max_depth_to=20,
                         step=1)
plt.show()
print("done")