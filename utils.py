from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler



def divide_data(data, target, test_size=0.3):
    """ create train and test set """
    train , test = train_test_split(data, test_size = test_size)
    x_train = train.drop(target, axis=1)
    y_train = train[target]
    x_test = test.drop(target, axis = 1)
    y_test = test[target]
    return x_train, y_train, x_test, y_test


def highest_correlated_data(df, target, size=6):
    # Find correlations with the target
    most_correlated = df.corr().abs()[target].sort_values(ascending=False)
    # Maintain the top 6 most correlation features with Grade
    most_correlated = most_correlated[:size]

    return most_correlated


def highest_correlated_data_as_list(df, target, size=6):
    """ parses correlated data to a string list """
    return highest_correlated_data(df,target,size).index.tolist()


def scale_min_max(data):
    """ scaling features with MinMax """
    minmax_scaler = MinMaxScaler(feature_range=(0, 1))
    x_train_minmax_scaled = pd.DataFrame(minmax_scaler.fit_transform(data), index=data.index,
                                         columns=data.columns)
    return x_train_minmax_scaled


def scale_min_max_without_target(data, target):
    """ scaling features with MinMax """
    data_wo_target = data.drop(target, axis = 1)
    minmax_scaler = MinMaxScaler(feature_range=(0, 1))
    x_train_minmax_scaled = pd.DataFrame(minmax_scaler.fit_transform(data_wo_target), index=data.index,
                                         columns=data_wo_target.columns)
    x_train_minmax_scaled[target] = data[target]
    return x_train_minmax_scaled


def scale_standard(data):
    """ scaling features with zscore """
    standard_scalar = StandardScaler()
    x_train_zscore_scaled = pd.DataFrame(standard_scalar.fit_transform(data), index=data.index,
                                         columns=data.colums)
    return x_train_zscore_scaled
