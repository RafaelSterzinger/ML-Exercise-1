from sklearn.model_selection import train_test_split


def divide_data(data, target, test_size=0.3):
    """ create train and test set """
    train , test = train_test_split(data, test_size = test_size)
    x_train = train.drop(target, axis=1)
    y_train = train[target]
    x_test = test.drop(target, axis = 1)
    y_test = test[target]
    return x_train, y_train, x_test, y_test