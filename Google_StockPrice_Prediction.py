#Importing Data and Preprocessing it
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data_train=pd.read_csv("Google_Stock_Price_Train.csv")
train_set=data_train.iloc[:,1:2].values
#Scaling Data
from sklearn.preprocessing import MinMaxScaler
Scaler=MinMaxScaler(feature_range=(0,1))
train_set_scaled=Scaler.fit_transform(train_set)
#Splitting Data 
X_train=[]
y_train=[]
for i in range(120,len(train_set_scaled)):
    X_train.append(train_set_scaled[i-120:i,0])
    y_train.append(train_set_scaled[i,0])
X_train,y_train=np.array(X_train),np.array(y_train)
#Reshaping 
X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
#Making RNNs!!!
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor=Sequential()
#First Layer
regressor.add(LSTM(units =50,return_sequences="True",input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(rate=0.2))
#Second Layer
regressor.add(LSTM(units =50,return_sequences="True"))
regressor.add(Dropout(rate=0.2))
#Third Layer
regressor.add(LSTM(units =50,return_sequences="True"))
regressor.add(Dropout(rate=0.2))
#Fourth Layer
regressor.add(LSTM(units =50))
regressor.add(Dropout(rate=0.2))
# Final Layer
regressor.add(Dense(units=1))
#Compiling it
regressor.compile(optimizer="adam",loss="mean_squared_error",metrics=['accuracy'])
#Fitting the trainig values
regressor.fit(X_train,y_train,epochs=100,batch_size=32)
#Testing the model for real stock price
data_test=pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price=data_test.iloc[:,1:2].values
dataset=pd.concat((data_train['Open'],data_test['Open']),axis=0)
dataset=dataset[train_set_scaled.shape[0]-120:].values
dataset=dataset.reshape(-1,1)
dataset=Scaler.transform(dataset)
#Testing the model
X_test=[]
for i in range(120,140):
    X_test.append(dataset[i-120:i,0])
    
X_test=np.array(X_test)
X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
predicted=regressor.predict(X_test)
predicted=Scaler.inverse_transform(predicted)
#Visualizing the results
plt.plot(real_stock_price,color="blue")
plt.plot(predicted,color="red")
plt.show()























