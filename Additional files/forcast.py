# to create typed array, required for the LSTM
import numpy as np
# pandas to hold data-set
import pandas as pd
# to generate date sequence
from pandas.tseries.offsets import DateOffset
# to generate visualization
import matplotlib.pyplot as plot
# to measure RMS (root mean square) error
from statsmodels.tools.eval_measures import rmse
# for the normalization of the data
from sklearn.preprocessing import MinMaxScaler
# keras library is used for the prediction/ forecasting future value
from keras.preprocessing.sequence import TimeseriesGenerator
# other supportig modules form the keras
from keras.layers import Dense
# LSTM is the actual model form the keras which has been used for the time serise prediction
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import Sequential

# to hide any warning messages in the code
import warnings
warnings.filterwarnings("ignore")

# read the data-set in pandas data frame
#data_frame = pd.read_csv('AirPassengers.csv')

import yfinance as yf

df=yf.download("ITC.NS").reset_index()
print(df.head(5))

df["Date"]=pd.to_datetime(df.Date,format="%Y-%m-%d")
df.index=df['Date']

data_frame= df [["Date","Close"]]

# get the month column
data_frame.Date = pd.to_datetime(data_frame.Date)
# set the month value as the index
data_frame = data_frame.set_index("Date")
print(data_frame.head)

train=data_frame.tail(365)

# scale date in 0 to 1 range
scl = MinMaxScaler()
scl.fit(train)
# get the scaled train
train = scl.transform(train)
input_length = 12
feature_count = 1
generator = TimeseriesGenerator(train, train, length=input_length, batch_size=6)

# its a sequential model
predection_model = Sequential()
# prediction based on LSTM
predection_model.add(LSTM(200, activation='relu', input_shape=(input_length, feature_count)))
# drouut 0.15
predection_model.add(Dropout(0.15))
# dense model 1
predection_model.add(Dense(1))
# compile the model
predection_model.compile(optimizer='adam', loss='mse')
# number of epochs 90
predection_model.fit_generator(generator, epochs=90)

# set the input length and feature count
input_length = 12
feature_count = 1

# create TimeseriesGenerator
generator = TimeseriesGenerator(train, train, length=input_length, batch_size=6)

# make prediction with the generator
predection_model.fit_generator(generator, epochs=90)
# list to hold prediction values
prediction_list = []

# reshape the training data
batch = train[-input_length:].reshape((1, input_length, feature_count))

# populate the prediction_list
for i in range(input_length):
    prediction_list.append(predection_model.predict(batch)[0])
    batch = np.append(batch[:,1:,:], [[prediction_list[i]]], axis=1)

# generates future dates
add_dates = [data_frame.index[-1] + DateOffset(months=x) for x in range(0, 13)]
date_future = pd.DataFrame(index=add_dates[1:], columns=data_frame.columns)

# generate the prediction data
data_predict = pd.DataFrame(scl.inverse_transform(prediction_list),
                            index=date_future[-input_length:].index, columns=['Prediction'])

# define the project data
df_projection = pd.concat([data_frame, data_predict], axis=1)

# plot the visualization
plot.figure(figsize=(20, 6))
plot.plot(df_projection.index, df_projection['Close'])
plot.plot(df_projection.index, df_projection['Prediction'], color='r')
plot.legend(loc='best', fontsize='xx-large')
plot.xticks(fontsize=17)
plot.yticks(fontsize=17)
# show the visualization
plot.show()


