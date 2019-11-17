#%% import
import pandas as pd
import seaborn as sns
from matplotlib import pyplot
import numpy as np
import matplotlib.pyplot as plt
from ridge_util import *
from tree_util import *
from knn_utils import *

#%% Student performance init

path_student = "./plots/student_performance/"
data_initials_student = "sp_"
target_attribute_student = "Grade"
numeric_attributes_student = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences"]
other_attributes_student = ["failures", "Medu","studytime", "goout", "age", "freetime",   "Fedu", "absences"]           #schneidet am besten ab bis jetzt
top7_attributes_student =  ["failures", "Medu","studytime", "goout", "age", "traveltime", "Fedu"]
all_attributes_tupels_list = [[numeric_attributes_student, 'Numeric attributes'], [other_attributes_student, 'Other attributes'], [top7_attributes_student, 'Top 7 attributes']]

#one hot encoding & Ordinal Encoding

# lots of categories -> Binary encoding

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

decision_tree_comparison(train_data_student, target_attribute_student, all_attributes_tupels_list,
                         comp_type='max_depth',
                         p_from=1,
                         p_to=20,
                         p_step=1)
plt.show()
print("done")

#%%
decision_tree_crossvalidation(train_data_student, test_data_student, target_attribute_student, nominal_attributes)

#%% Bike sharing decision tree regression max_depth comparison
decision_tree_comparison(train_data_student, target_attribute_student,all_attributes_tupels_list,
                         comp_type='max_depth',
                         p_from=1,
                         p_to=10,       #bei 25 konstante Tiefe
                         p_step=1)
plt.show()

#%% knn with different distances
data = train_data_student

trimmed_data = trim_data(data,['dteday'])
x_train, y_train, x_test, y_test = create_cross_validation(trimmed_data,'cnt')
find_best_rmse('with all attributes + id and euclidean',
               x_train, y_train, x_test, y_test)

trimmed_data = trim_data(trimmed_data,['id'])

x_train, y_train, x_test, y_test = create_cross_validation(trimmed_data,'cnt')
find_best_rmse('with all attributes and manhatten',
               x_train, y_train, x_test, y_test,metric='manhattan')

x_train, y_train, x_test, y_test = create_cross_validation(trimmed_data,'cnt')
find_best_rmse('with all attributes and euclidean',x_train, y_train, x_test, y_test)

x_train, y_train, x_test, y_test = create_cross_validation(trimmed_data,'cnt')
find_best_rmse('with all attributes and minkowski',x_train, y_train, x_test, y_test,metric="minkowski")

plt.savefig(path_student + "knn_all_attributes.png")
plt.ylabel("Root Mean Squared Error")
plt.show()

#%% knn with different distances
workaroudn = numeric_attributes_student + [target_attribute_student]
trimmed_data = train_data_student[workaroudn]
x_train, y_train, x_test, y_test = create_cross_validation(trimmed_data,'Grade')
find_best_rmse('with all numeric euclidean',
               x_train, y_train, x_test, y_test)


x_train, y_train, x_test, y_test = create_cross_validation(trimmed_data,'Grade')
find_best_rmse('with all numeric manhatten',
               x_train, y_train, x_test, y_test,metric='manhattan')

x_train, y_train, x_test, y_test = create_cross_validation(trimmed_data,'Grade')
find_best_rmse('with all attributes and minkowski',x_train, y_train, x_test, y_test,metric="minkowski")

plt.savefig(path_student + "knn_all_attributes.png")
plt.ylabel("Root Mean Squared Error")
plt.show()