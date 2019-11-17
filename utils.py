from sklearn.model_selection import train_test_split



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
