#%% import libs
from knn_utils import *

#%% read data, define path
data = pd.read_csv("datasets/bike_sharing/bikeSharing.shuf.train.csv",skipinitialspace=True)
data.columns=data.columns.str.strip()
path = 'plots/bike_sharing/'

#%% try different attributes and save plot
plt.figure (figsize=(16,8))

trimmed_data = trim_data(data,['dteday'])
x_train, y_train, x_test, y_test = create_cross_validation(trimmed_data,'cnt')
find_best_rmse('with all attributes (including id) and manhatten',
               x_train, y_train, x_test, y_test,metric='manhattan')

trimmed_data = trim_data(trimmed_data,['id'])

x_train, y_train, x_test, y_test = create_cross_validation(trimmed_data,'cnt')
find_best_rmse('with all attributes and manhatten',
               x_train, y_train, x_test, y_test,metric='manhattan')

x_train, y_train, x_test, y_test = create_cross_validation(trimmed_data,'cnt')
find_best_rmse('with all attributes and euclidean',x_train, y_train, x_test, y_test)

x_train, y_train, x_test, y_test = create_cross_validation(trimmed_data,'cnt')
find_best_rmse('with all attributes and mahalanobis',x_train, y_train, x_test, y_test)

trimmed_data = trim_data(trimmed_data,['yr','atemp','windspeed','weekday'])

x_train, y_train, x_test, y_test = create_cross_validation(trimmed_data,'cnt')
find_best_rmse('without year, atemp, windspeed, weekday',x_train, y_train, x_test, y_test)
trimmed_data = trim_data(data,['id','dteday',
                               'yr','atemp','windspeed','weekday',
                               'season','mnth','holiday','workingday','weathersit'])

x_train, y_train, x_test, y_test = create_cross_validation(trimmed_data, 'cnt')
find_best_rmse('only temperature and humidity',x_train, y_train, x_test, y_test)

plt.savefig(path + "k_rmse_attributes.png")
plt.show()

#%% Test dataset
#reading test and submission files
test_data = pd.read_csv("datasets/bike_sharing/bikeSharing.shuf.test.csv")
sample_solution = pd.read_csv("datasets/bike_sharing/bikeSharing.shuf.sampleSolution.csv")
test_data.columns=test_data.columns.str.strip()
sample_solution.columns=sample_solution.columns.str.strip()

#preprocessing test dataset
trimmed_data = trim_data(data,['id','dteday'])
trimmed_test = trim_data(test_data,['id', 'dteday'])
model = neighbors.KNeighborsRegressor(n_neighbors = 5,metric='manhattan')
x_train=trimmed_data.drop('cnt',axis=1)
y_train=trimmed_data['cnt']

model.fit(x_train, y_train)  #fit the model
pred=model.predict(trimmed_test) #make prediction on test set
solution=sample_solution.assign(cnt=pred)
solution.to_csv('solution/bike_sharing.csv',index=False)

#%%
# print('-----------------------------------------')
# print('not normalized data')
# find_best_rsme(x_train, y_train, x_test, y_test)
# print('-----------------------------------------')
# print('minmax normalized data')
# find_best_rsme(x_train_minmax_scaled, y_train, x_test_minmax_scaled, y_test)
# print('-----------------------------------------')
# print('zscore normalized data')
# find_best_rsme(x_train_zscore_scaled, y_train, x_test_zscore_scaled, y_test)
# print('-----------------------------------------')
