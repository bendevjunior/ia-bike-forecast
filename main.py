import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

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