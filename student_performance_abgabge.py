import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from ridge_util import *
from tree_util import *
from knn_utils import *
from scipy import stats


#%% init
path = "./plots/student_performance/"

train_data = pd.read_csv("datasets/student_performance/StudentPerformance.shuf.train.csv")
test_data = pd.read_csv("datasets/student_performance/StudentPerformance.shuf.test.csv")
test_label = pd.read_csv("datasets/student_performance/StudentPerformance.shuf.sampleSolution.csv")

target = ['Grade']
#all numeric
numeric_attributes_student = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences"]
#top 7 highest correlated ones
top7_attributes_student =  ["failures", "Medu","studytime", "goout", "age", "traveltime", "Fedu"]


# found through testing and analysis
attr_own = ["failures", "Medu", "studytime", "goout", "age", "freetime", "traveltime", "Fedu", "absences"]
attr_own_cat_before_preprocss = ["Mjob", "Fjob"]                           # since Medu and Fedu were correlated we created some barplots to see if there could be a correlation job wise as well
attr_own_cat_after_preprocess = ['Mjob_teacher', 'Mjob_health', 'Mjob_services', 'Mjob_at_home', 'Mjob_other'] + ['Fjob_teacher', 'Fjob_health', 'Fjob_services', 'Fjob_at_home', 'Fjob_other']
attr_own_bin = ["higher", "internet"]

attributes_names_tupels = [[numeric_attributes_student, 'All numeric attributes'],
                           [top7_attributes_student, '"Top 7 attributes by correlation"'],
                           [attr_own + attr_own_cat_after_preprocess + attr_own_bin, 'Attributes found through trial and error (OHE + BE)']]


all_attributes = target + numeric_attributes_student + top7_attributes_student + attr_own + attr_own_cat_before_preprocss + attr_own_bin

train_data = train_data[all_attributes]

# One Hot Encoding with
train_data_encoded = pd.get_dummies(train_data, columns=attr_own_cat_before_preprocss)
for bin_attr in attr_own_bin:
    train_data_encoded[bin_attr] = (train_data_encoded[bin_attr] == 'Yes' ).astype(int)     #replace Yes with other binomical value

# Outlier removal
# train_data_encoded_out = train_data_encoded[(np.abs(stats.zscore(train_data_encoded)) < 3).all(axis=1)]
# train_data_encoded_out.index = np.arange(0, len(train_data_encoded_out), 1)


# Normalizing
# train_data_encoded_out_normalized = scale_min_max_without_target(train_data_encoded_out, target)
train_data_encoded_normalized = scale_min_max_without_target(train_data_encoded, target)

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
                                  attr_own + attr_own_cat_after_preprocess + attr_own_bin,
                                  0, 50, 5,
                                  "Attributes found through trial and error (OHE + BE)")
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
                                  attr_own + attr_own_cat_after_preprocess + attr_own_bin,
                                  0, 50, 5,
                                  "Min-max: Attributes found through trial and error (OHE + BE)")

plt.show()

# %% min samples depth
# %% criterion comparison -> we will keep using mse
decision_tree_regression_criterion_comparison(train_data_encoded, target,
                                  numeric_attributes_student,
                                              criterion = ['mse', 'friedman_mse'],
                                              name = "All numerical attributes")


decision_tree_regression_criterion_comparison(train_data_encoded, target,
                                  top7_attributes_student,
                                              criterion=['mse', 'friedman_mse'],
                                              name = "Top 7 attributes by correlation")

decision_tree_regression_criterion_comparison(train_data_encoded, target,
                                  attr_own + attr_own_cat_after_preprocess + attr_own_bin,
                                              criterion=['mse', 'friedman_mse'],
                                              name = "Attributes found through trial and error (OHE + BE)")

plt.show()

# %% min samples leaf comparison normalizing the data makes no difference
decision_tree_regression_min_samples_leaf_comparison(train_data_encoded, target, numeric_attributes_student,"All numerical attributes",
                                  1, 50, 5)

decision_tree_regression_min_samples_leaf_comparison(train_data_encoded, target, top7_attributes_student, "Top 7 attributes by correlation",
                                  1, 50, 5)

decision_tree_regression_min_samples_leaf_comparison(train_data_encoded, target, attr_own + attr_own_cat_after_preprocess + attr_own_bin,"Attributes found through trial and error (OHE + BE)",

                                  1, 50, 5)
plt.show()


decision_tree_regression_min_samples_leaf_comparison(train_data_encoded_normalized, target, numeric_attributes_student,
                                                     "All numerical attributes",
                                                     1, 50, 5)

decision_tree_regression_min_samples_leaf_comparison(train_data_encoded_normalized, target, top7_attributes_student,
                                                     "Top 7 attributes by correlation",
                                                     1, 50, 5)

decision_tree_regression_min_samples_leaf_comparison(train_data_encoded_normalized, target,
                                                     attr_own + attr_own_cat_after_preprocess + attr_own_bin,
                                                     "Attributes found through trial and error (OHE + BE)",
                                                     1, 50, 5)
plt.show()

# %% min samples split comparison
decision_tree_comparison(train_data_encoded, target, attributes_names_tupels,
                         comp_type='min_samples_split',
                         p_from=2,
                         p_to=30,       #bei 25 konstante Tiefe
                         p_step=1)
plt.show()
