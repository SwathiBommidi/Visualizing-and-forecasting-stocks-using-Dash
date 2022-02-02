'''import dash
import dash_core_components as dcc
import dash_html_components as html'''
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt


df=yf.download("ITC.NS").reset_index()

print(df.head())

model=load_model("saved_lstm_model.h5")

data=df.sort_index(ascending=True,axis=0)
print(data.head())
new_data=pd.DataFrame(index=range(0,len(df)),columns=['Date','Close'])
print(new_data.head())

for i in range(0,len(data)):
    new_data["Date"][i]=data['Date'][i]
    new_data["Close"][i]=data["Close"][i]
print(new_data.head())
new_data.index=new_data.Date
new_data.drop("Date",axis=1,inplace=True)
print(new_data.tail())

dataset=new_data.values

train=dataset[:,:]

x_train,y_train=[],[]

scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)

for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])

x_train,y_train=np.array(x_train),np.array(y_train)

x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

inputs=new_data[len(new_data)-365-60:].values
inputs=inputs.reshape(-1,1)
inputs=scaler.transform(inputs)

X_test=[]
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test=np.array(X_test)

print(X_test.shape)


X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
closing_price=model.predict(X_test)
closing_price=scaler.inverse_transform(closing_price)

train=new_data[:]

import datetime
date_list=[]
for i in range(365):
    date_list.append(df.iloc[-1,0] + datetime.timedelta(days=i+1))

df1 = pd.DataFrame(list(zip(date_list, closing_price)),
               columns =['Date', 'Close'])

#print(closing_price)

#print(df[-1])

plt.plot(new_data['close'])
plt.plot(df1)

plt.show()

