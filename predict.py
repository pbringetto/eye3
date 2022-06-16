import api.ftx as f
import cfg_load
import os
dir = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(dir, 'alpha.yaml')
alpha = cfg_load.load(path)
import pandas as pd
import helpers.util as u
import math
import numpy as np 
import ta_signals
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from datetime import datetime, timedelta
import twitter as t

class Predict:
    def __init__(self):
        self.go()

    def go(self):
        ftx = f.FtxClient(alpha["ftx_key"], alpha["ftx_secret"])
        for pair in alpha["pairs"]:
            single_market = ftx.get_single_market(pair['pair'])
            price = single_market['price']
            for tf in alpha["timeframes"]:
                import matplotlib.pyplot as plt
                if tf['enabled']:
                    print(tf['label'])
                    df = pd.DataFrame(ftx.get_historical_prices(pair['pair'], tf['seconds']))
                    data, df = ta_signals.go(df, 'close', 2)
                    df.index = pd.to_datetime(df['startTime'], errors='coerce', utc=True)
                    data=df.filter(['close'])
                    dataset=data.values
                    training_data_len= math.ceil(len(dataset) * .8)
                    scaler=MinMaxScaler(feature_range=(0,1))
                    scaled_data=scaler.fit_transform(dataset)
                    train_data=scaled_data[0:training_data_len,:]
                    x_train =[]
                    y_train=[]
                    for i in range(60,len(train_data)):
                        x_train.append(train_data[i-60:i,0])
                        y_train.append(train_data[i,0])
              
                    x_train,y_train=np.array(x_train),np.array(y_train)

                    x_train.shape

                    model=Sequential()
                    model.add(LSTM(55,return_sequences=True,input_shape=(x_train.shape[1],1)))
                    model.add(LSTM(55,return_sequences=False))
                    model.add(Dense(34))
                    model.add(Dense(1))
                    model.compile(optimizer='adam',loss='mean_squared_error')

                    model.fit(x_train,y_train,batch_size=1,epochs=1)
                    test_data=scaled_data[training_data_len - 60:,:]
                    x_test=[]
                    y_test=dataset[training_data_len:,:]
                    for i in range(60,len(test_data)):
                        x_test.append(test_data[i-60:i,0])
                    x_test=np.array(x_test)
                    x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
                    predictions=model.predict(x_test)
                    predictions=scaler.inverse_transform(predictions)
                    rmse=np.sqrt( np.mean((predictions - y_test)**2))
                    train=data[:training_data_len]
                    valid=data[training_data_len:]
                    valid['Predictions']=predictions
                
                    last_index = df.index[-1]
                    new_df=df.filter(['close'])
                    count =1
                    data = []
                    while count < 14 :
                        last_index = last_index + timedelta(seconds=tf['seconds'])
                        last_200_days=new_df[-200:].values
                        last_200_days_scaled=scaler.transform(last_200_days)
                        x_test=[]
                        x_test.append(last_200_days_scaled)
                        x_test=np.array(x_test)
                        x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
                        pred_price=model.predict(x_test)
                        pred_price=scaler.inverse_transform(pred_price)
                        data.append([last_index, pred_price])
                        new_row = pd.Series(data={'close':pred_price}, name=last_index)
                        new_df = new_df.append(new_row, ignore_index=False)
                        count+=1
                    
                    plt.figure(figsize=(16,8))
                    plt.title(pair['pair'] + ' - ' + tf['label'])
                    plt.xlabel('Date',fontsize=18)
                    plt.ylabel('Close Price USD($)',fontsize=18)
                    plt.plot(df['close'])
                    plt.plot(valid[['close','Predictions']])
                    plt.plot(new_df[['close']][-14:])
                    plt.legend(['Training Data','Validation','Training Prediction','Prediction'],loc='lower left')
                
                    file = os.path.join(dir, 'img/predict-' + str(datetime.now()) + '.png')
                    plt.savefig(file)

                    twitter = t.Twitter()
                    data = 'Predicting the future price of #bitcoin with Long Short Term Memory(LSTM), an artificial neural network' + '\r\n'
                    #twitter.tweet(data, file)

p = Predict()


