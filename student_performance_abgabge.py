import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from ridge_util import *
from tree_util import *
from knn_utils import *
from mlp_utils import *
from utils import *
from scipy import stats

# %% init
path = "plots/student_performance/"

orig_data = pd.read_csv("datasets/student_performance/StudentPerformance.shuf.train.csv")

train_data = pd.read_csv("datasets/student_performance/StudentPerformance.shuf.train.csv")
test_data = pd.read_csv("datasets/student_performance/StudentPerformance.shuf.test.csv")
test_label = pd.read_csv("datasets/student_performance/StudentPerformance.shuf.sampleSolution.csv")

target = ['Grade']
# all numeric
numeric_attributes_student = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel", "freetime",
                              "goout", "Dalc", "Walc", "health", "absences"]
# top 7 highest correlated ones
top7_attributes_student = ["failures", "Medu", "studytime", "goout", "age", "traveltime", "Fedu"]

# found through testing and analysis
attr_own = ["failures", "Medu", "studytime", "goout", "age", "freetime", "traveltime", "Fedu", "absences"]
attr_own_cat_before_preprocss = ["Mjob",
                                 "Fjob"]  # since Medu and Fedu were correlated we created some barplots to see if there could be a correlation job wise as well
attr_own_cat_after_preprocess = ['Mjob_teacher', 'Mjob_health', 'Mjob_services', 'Mjob_at_home', 'Mjob_other'] + [
    'Fjob_teacher', 'Fjob_health', 'Fjob_services', 'Fjob_at_home', 'Fjob_other']
attr_own_bin = ["higher", "internet"]

attr_own_all = attr_own + attr_own_cat_after_preprocess + attr_own_bin + ['school', 'sex']

attributes_names_tupels = [[numeric_attributes_student, 'All numeric attributes'],
                           [top7_attributes_student, 'Top 7 attributes by correlation'],
                           [attr_own_all, 'Attributes found through trial and error (OHE + BE)']]

all_attributes = target + numeric_attributes_student + top7_attributes_student + attr_own + attr_own_cat_before_preprocss + attr_own_bin + [
    'sex', 'school']

train_data = train_data[list(set(all_attributes))]

# One Hot Encoding with
train_data_encoded = pd.get_dummies(train_data, columns=attr_own_cat_before_preprocss)
for bin_attr in attr_own_bin:
    train_data_encoded[bin_attr] = (train_data_encoded[bin_attr] == 'Yes').astype(
        int)  # replace Yes with other binomical value

train_data_encoded['sex'] = (train_data_encoded['sex'] == 'F').astype(int)
train_data_encoded['school'] = (train_data_encoded['school'] == 'GP').astype(int)

# Outlier removal
# train_data_encoded_out = train_data_encoded[(np.abs(stats.zscore(train_data_encoded)) < 3).all(axis=1)]
# train_data_encoded_out.index = np.arange(0, len(train_data_encoded_out), 1)


# Normalizing
# train_data_encoded_out_normalized = scale_min_max_without_target(train_data_encoded_out, target)
train_data_encoded_normalized = scale_min_max_without_target(train_data_encoded, target)

# %% Data Analysis


# Histogramm
sns.distplot(train_data[target], kde=True, bins=15)
plt.xlabel('Grade')
plt.ylabel('Count')
plt.title('Distribution of Final Grades')
plt.savefig('plots/student_performance/grades_histogram.png')
plt.show()

highest_7_correlated = highest_correlated_data_as_list(train_data, 'Grade', 8)

correlation_matrix = train_data[highest_7_correlated].corr().abs()

sns.heatmap(correlation_matrix, square=True, linewidths=.5).get_figure()
plt.savefig(path + "heatmap_7_highest_numerical.png")
plt.show()

# positive correlation if parent is teacher, negative if stay at home
boxplot_student_father = sns.boxplot(x=train_data["Fjob"], y=train_data['Grade'])
plt.show()
boxplot_student_father.figure.savefig(path + "boxplot_father.png")

boxplot_student_mother = sns.boxplot(x=train_data["Mjob"], y=train_data['Grade'])
plt.show()
boxplot_student_mother.figure.savefig(path + "boxplot_mother.png")

# females get better grades
boxplot_student_gender = sns.boxplot(x=train_data["sex"], y=train_data['Grade'])
plt.show()
boxplot_student_gender.figure.savefig(path + "boxplot_gender.png")

# %% Ridge regression comparison original vs pre processed data

ridge_regression_alpha_comparison(train_data_encoded, target,
                                  numeric_attributes_student,
                                  0, 50, 5,
                                  "All numerical attributes")

ridge_regression_alpha_comparison(train_data_encoded, target,
                                  top7_attributes_student,
                                  0, 50, 5,
                                  "Top 7 attributes by correlation")

ridge_regression_alpha_comparison(train_data_encoded, target,
                                  attr_own_all,
                                  0, 50, 5,
                                  "Attributes found through trial and error (OHE + BE)")
plt.ylim(4, 4.3)
plt.savefig(path + "ridge_alpha_comparison.png")
plt.show()

ridge_regression_alpha_comparison(train_data_encoded_normalized, target,
                                  numeric_attributes_student,
                                  0, 50, 5,
                                  "Min-max: All numerical attributes")

ridge_regression_alpha_comparison(train_data_encoded_normalized, target,
                                  top7_attributes_student,
                                  0, 50, 5,
                                  "Min-max: Top 7 attributes by correlation")

ridge_regression_alpha_comparison(train_data_encoded_normalized, target,
                                  attr_own_all,
                                  0, 50, 5,
                                  "Min-max: Attributes found through trial and error (OHE + BE)")
plt.savefig(path + "ridge_alpha_comparison_normalized.png")
plt.ylim(4, 4.3)
plt.show()

# %% max depth comparison
decision_tree_comparison(train_data_encoded, target, attributes_names_tupels,
                         comp_type='max_depth',
                         p_from=1,
                         p_to=30,
                         p_step=2)
plt.savefig(path + "tree_max_depth_comparison.png")
plt.show()

# %% criterion comparison -> we will keep using mse
decision_tree_regression_criterion_comparison(train_data_encoded, target,
                                              numeric_attributes_student,
                                              criterion=['mse', 'friedman_mse'],
                                              name="All numerical attributes")

decision_tree_regression_criterion_comparison(train_data_encoded, target,
                                              top7_attributes_student,
                                              criterion=['mse', 'friedman_mse'],
                                              name="Top 7 attributes by correlation")

decision_tree_regression_criterion_comparison(train_data_encoded, target,
                                              attr_own_all,
                                              criterion=['mse', 'friedman_mse'],
                                              name="Attributes found through trial and error (OHE + BE)")
plt.savefig(path + "tree_criterion_comparison.png")
plt.show()

# %% min samples leaf comparison normalizing the data makes no difference
decision_tree_regression_min_samples_leaf_comparison(train_data_encoded, target, numeric_attributes_student,
                                                     "All numerical attributes",
                                                     5, 50, 5)

decision_tree_regression_min_samples_leaf_comparison(train_data_encoded, target, top7_attributes_student,
                                                     "Top 7 attributes by correlation",
                                                     5, 50, 5)

decision_tree_regression_min_samples_leaf_comparison(train_data_encoded, target, attr_own_all,
                                                     "Attributes found through trial and error (OHE + BE)",

                                                     5, 50, 5)
plt.ylim(4, 5)
plt.savefig(path + "tree_min_samples_leaf_comparison.png")
plt.show()

decision_tree_regression_min_samples_leaf_comparison(train_data_encoded_normalized, target, numeric_attributes_student,
                                                     "Min-Max: All numerical attributes",
                                                     5, 50, 5)

decision_tree_regression_min_samples_leaf_comparison(train_data_encoded_normalized, target, top7_attributes_student,
                                                     "Min-Max: Top 7 attributes by correlation",
                                                     5, 50, 5)

decision_tree_regression_min_samples_leaf_comparison(train_data_encoded_normalized, target,
                                                     attr_own_all,
                                                     "Min-Max: Attributes found through trial and error (OHE + BE)",
                                                     5, 50, 5)
plt.ylim(4, 5)
plt.savefig(path + "tree_min_samples_leaf_comparison_normalized.png")
plt.show()

# %% min samples split comparison
decision_tree_comparison(train_data_encoded, target, attributes_names_tupels,
                         comp_type='min_samples_split',
                         p_from=20,
                         p_to=50,  # bei 25 konstante Tiefe
                         p_step=2)
plt.ylim(4, 5)
plt.savefig(path + "tree_min_samples_split_comparison.png")
plt.show()
# %% k Nearest Neighbour k comparison
knn_regression_k_comparison(train_data_encoded, target, numeric_attributes_student,
                            "Euclidean: All numerical attributes",
                            metric='euclidean')
knn_regression_k_comparison(train_data_encoded, target, numeric_attributes_student,
                            "Minkowski: All numerical attributes",
                            metric='minkowski')
knn_regression_k_comparison(train_data_encoded, target, numeric_attributes_student,
                            "Manhattan: All numerical attributes",
                            metric='manhattan')

knn_regression_k_comparison(train_data_encoded, target, top7_attributes_student,
                            "Euclidean: Top 7 attributes by correlation",
                            metric='euclidean')
knn_regression_k_comparison(train_data_encoded, target, top7_attributes_student,
                            "Minkowski: Top 7 attributes by correlation",
                            metric='minkowski')
knn_regression_k_comparison(train_data_encoded, target, top7_attributes_student,
                            "Manhattan: Top 7 attributes by correlation",
                            metric='manhattan')

knn_regression_k_comparison(train_data_encoded, target,
                            attr_own_all,
                            "Attributes found through trial and error (OHE + BE)",
                            metric='euclidean')
knn_regression_k_comparison(train_data_encoded, target,
                            attr_own_all,
                            "Attributes found through trial and error (OHE + BE)",
                            metric='minkowski')
knn_regression_k_comparison(train_data_encoded, target,
                            attr_own_all,
                            "Attributes found through trial and error (OHE + BE)",
                            metric='manhattan')
plt.savefig(path + "knn_metric_comparison.png")
plt.show()

knn_regression_k_comparison(train_data_encoded_normalized, target, numeric_attributes_student,
                            "Min-max|Euclidean: All numerical attributes",
                            metric='euclidean')
knn_regression_k_comparison(train_data_encoded_normalized, target, numeric_attributes_student,
                            "Min-max|Minkowski: All numerical attributes",
                            metric='minkowski')
knn_regression_k_comparison(train_data_encoded_normalized, target, numeric_attributes_student,
                            "Min-max|Manhattan: All numerical attributes",
                            metric='manhattan')

knn_regression_k_comparison(train_data_encoded_normalized, target, top7_attributes_student,
                            "Min-max|Euclidean: Top 7 attributes by correlation",
                            metric='euclidean')
knn_regression_k_comparison(train_data_encoded_normalized, target, top7_attributes_student,
                            "Min-max|Minkowski: Top 7 attributes by correlation",
                            metric='minkowski')
knn_regression_k_comparison(train_data_encoded_normalized, target, top7_attributes_student,
                            "Min-max|Manhattan: Top 7 attributes by correlation",
                            metric='manhattan')

knn_regression_k_comparison(train_data_encoded_normalized, target,
                            attr_own_all,
                            "Min-max|Euclidean: Attributes found through trial and error (OHE + BE)",
                            metric='euclidean')
knn_regression_k_comparison(train_data_encoded_normalized, target,
                            attr_own_all,
                            "Min-max|Minkowski: Attributes found through trial and error (OHE + BE)",
                            metric='minkowski')
knn_regression_k_comparison(train_data_encoded_normalized, target,
                            attr_own_all,
                            "Min-max|Manhattan: Attributes found through trial and error (OHE + BE)",
                            metric='manhattan')
plt.savefig(path + "tree_min_samples_leaf_comparison_normalized.png")
plt.show()

# %% k Nearest Neighbour k comparison
knn_regression_k_comparison(train_data_encoded, target, numeric_attributes_student, "All numerical attributes",
                            metric='euclidean')

knn_regression_k_comparison(train_data_encoded, target, top7_attributes_student, "Top 7 attributes by correlation",
                            metric='euclidean')
knn_regression_k_comparison(train_data_encoded, target,
                            attr_own_all,
                            "Attributes found through trial and error (OHE + BE)",
                            metric='euclidean')
plt.ylim(3.8, 6)
plt.savefig(path + 'knn_k_comparison.png')
plt.show()

knn_regression_k_comparison(train_data_encoded_normalized, target, numeric_attributes_student,
                            "Min-max: All numerical attributes",
                            metric='euclidean')
knn_regression_k_comparison(train_data_encoded_normalized, target, top7_attributes_student,
                            "Min-max: Top 7 attributes by correlation",
                            metric='euclidean')
knn_regression_k_comparison(train_data_encoded_normalized, target,
                            attr_own_all,
                            "Min-max: Attributes found through trial and error (OHE + BE)",
                            metric='euclidean')
plt.ylim(3.8, 6)
plt.savefig(path + "knn_k_normalized_comparison.png")
plt.show()

# %% knn with different distances
workaround = attr_own_all + target
trimmed_data = train_data_encoded[workaround]
x_train, y_train, x_test, y_test = make_split(trimmed_data, 'Grade')
find_best_rmse('with all numeric and euclidean',
               x_train, y_train, x_test, y_test)

find_best_rmse('with all numeric manhatten',
               x_train, y_train, x_test, y_test, metric='manhattan')

find_best_rmse('with all attributes and minkowski', x_train, y_train, x_test, y_test, metric="minkowski")

# plt.savefig(path_student + "knn_all_attributes.png")
# plt.ylabel("Root Mean Squared Error")
# plt.show()

trimmed_data = train_data_encoded_normalized[workaround]
x_train, y_train, x_test, y_test = make_split(trimmed_data, 'Grade')
find_best_rmse('Min:max with all numeric and euclidean',
               x_train, y_train, x_test, y_test)

x_train, y_train, x_test, y_test = make_split(trimmed_data, 'Grade')
find_best_rmse('Min:max with all numeric manhatten',
               x_train, y_train, x_test, y_test, metric='manhattan')

x_train, y_train, x_test, y_test = make_split(trimmed_data, 'Grade')
find_best_rmse('Min:max with all attributes and minkowski', x_train, y_train, x_test, y_test, metric="minkowski")

plt.savefig(path + "knn_wo_crossvalidation_normalized_comparison.png")
# plt.savefig(path_student + "knn_all_attributes.png")
# plt.ylabel("Root Mean Squared Error")
plt.show()

# %% MLP
# correlated_columns = highest_correlated_data_as_list(data, 'Grade', 10)
# correlated_columns.remove('Grade')

numeric_attributes_student = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel", "freetime",
                              "goout", "Dalc", "Walc", "health", "absences"]
df = mlp_regression(train_data_encoded, attr_own_all, 'Grade', [(5, 7, 7), (7, 5, 5), (7, 7, 5, 3)], "logistic")

df = pd.concat(
    [df, mlp_regression(train_data_encoded, attr_own_all, 'Grade', [(5, 7, 7), (7, 5, 5), (7, 7, 5, 3)], "relu")])

df = pd.concat(
    [df, mlp_regression(train_data_encoded, attr_own_all, 'Grade', [(5, 7, 7), (7, 5, 5), (7, 7, 5, 3)], "tanh")])

sns.catplot(x='Layers', y='RMSE', hue='Activation', data=df, kind='bar')
# df = pd.melt(df, id_vars="Layers", var_name="Activation", value_name="RMSE")
plt.ylim(2.5, 5.5)
plt.savefig(path + "mlp_wo_crossvalidation_layer_activation_function_comparison.png")
plt.show()

# %% MLP Encoded data
df = mlp_regression_layer_comparison(train_data_encoded, attr_own_all, 'Grade', [(5, 7, 7), (7, 5, 5), (7, 7, 5, 3)],
                                     "logistic")

df = pd.concat([df, mlp_regression_layer_comparison(train_data_encoded, attr_own_all, 'Grade',
                                                    [(5, 7, 7), (7, 5, 5), (7, 7, 5, 3)], "relu")])

df = pd.concat([df, mlp_regression_layer_comparison(train_data_encoded, attr_own_all, 'Grade',
                                                    [(5, 7, 7), (7, 5, 5), (7, 7, 5, 3)], "tanh")])

sns.catplot(x='Layers', y='RMSE', hue='Activation', data=df, kind='bar')
# df = pd.melt(df, id_vars="Layers", var_name="Activation", value_name="RMSE")
plt.ylim(4, 5.5)
plt.savefig(path + "mlp_crossvalidation_layer_activation_function_comparison.png")
plt.show()

# %% MLP Encoded data Normalized -> Logistik unbeeinflusst -> andere schlechter!
df = mlp_regression_layer_comparison(train_data_encoded_normalized, attr_own_all, 'Grade',
                                     [(5, 7, 7), (7, 5, 5), (7, 7, 5, 3)], "logistic")

df = pd.concat([df, mlp_regression_layer_comparison(train_data_encoded_normalized, attr_own_all, 'Grade',
                                                    [(5, 7, 7), (7, 5, 5), (7, 7, 5, 3)], "relu")])

df = pd.concat([df, mlp_regression_layer_comparison(train_data_encoded_normalized, attr_own_all, 'Grade',
                                                    [(5, 7, 7), (7, 5, 5), (7, 7, 5, 3)], "tanh")])

sns.catplot(x='Layers', y='RMSE', hue='Activation', data=df, kind='bar')
# df = pd.melt(df, id_vars="Layers", var_name="Activation", value_name="RMSE")
plt.ylim(4, 5.5)
plt.savefig(path + "mlp_crossvalidation_layer_activation_function_comparison_normalized.png")
plt.show()
