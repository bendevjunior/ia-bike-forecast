import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt

bike = pd.read_csv('bike-sharing-daily.csv')

# show the first 5 rows of the dataset
#bike.head()
# show the last 5 rows of the dataset
#bike.info()
# show the last 5 rows of the dataset
#bike.describe()

# check for missing values
#sns.heatmap(bike.isnull())
#print(bike.head())

# drop the columns if null

bike = bike.drop(['instant', 'casual', 'registered'], axis=1)
bike.dteday = pd.to_datetime(bike.dteday, format='%m/%d/%Y')
bike.index = pd.DatetimeIndex(bike.dteday)
bike = bike.drop(labels=['dteday'], axis=1)
#print(bike.head())


x_numerical = bike[['temp', 'hum', 'windspeed', 'cnt']]
x_cat = bike[['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']]

# convert categorical variables to dummy variables
onehotencoder = OneHotEncoder()
x_cat = onehotencoder.fit_transform(x_cat).toarray()
x_cat = pd.DataFrame(x_cat)

x_numerical = x_numerical.reset_index()
#print(x_numerical.head())


x_all = pd.concat([x_cat, x_numerical], axis=1)

# verifica se existe algum com valor NAN
#print(x_all.head())
x_all = x_all.drop(labels=['dteday'], axis=1)

x = x_all.iloc[:, :-1].values
y = x_all.iloc[:, -1:].values
#print(y)

scaler = MinMaxScaler()
y = scaler.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=100, activation='relu', input_shape=(x_train.shape[1],)))
model.add(tf.keras.layers.Dense(units=100, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='linear'))

print(model.summary())

model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mean_squared_error')

epochs_hist = model.fit(x_train, y_train, epochs=25, batch_size=50, validation_split=0.2)

# graph of the loss function
plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('Model Loss Progress During Training')
plt.ylabel('Training and Validation Loss')
plt.xlabel('Epoch Number')
plt.legend(['Training Loss', 'Validation Loss'])

# prediction on test data
y_predict = model.predict(x_test)
y_predict_original = scaler.inverse_transform(y_predict)
y_test_original = scaler.inverse_transform(y_test)

k = x_test.shape[1]
n = len(x_test)

# calculate the R2 score
mae = mean_absolute_error(y_test_original, y_predict_original)
mse = mean_squared_error(y_test_original, y_predict_original)
rmse = sqrt(mse)
r2_square = r2_score(y_test_original, y_predict_original)
adj_r2 = 1 - (1 - r2_square) * (n - 1) / (n - k - 1)

print('MAE: ', mae, '\nMSE: ', mse, '\nRMSE: ', rmse, '\nR2 Square: ', r2_square, '\nAdjusted R2: ', adj_r2)

