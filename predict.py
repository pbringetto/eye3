from sklearn.metrics import accuracy_score
import api.ftx as f
import cfg_load
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
dir = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(dir, 'alpha.yaml')
alpha = cfg_load.load(path)
import ta_signals
import twitter as t
import sys
import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from datetime import datetime, timedelta
'''
from pandas_datareader import data


import datetime as dt
import urllib.request, json
import os

import tensorflow as tf 
from sklearn.preprocessing import MinMaxScaler
'''

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
np.set_printoptions(threshold=sys.maxsize)



class Predict:

    def __init__(self):
        self.ftx = f.FtxClient(alpha["ftx_key"], alpha["ftx_secret"])
        self.go()

    def data(self):
        df = pd.DataFrame( self.ftx.get_historical_prices( str(sys.argv[2]), str(sys.argv[1]) ) )
        df['date'] = pd.to_datetime(df['startTime'], errors='coerce', utc=True)
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        return df

    def train(self, df, prediction_column):
        # Extracting the closing prices of each day
        FullData=df[[prediction_column]].values
        #print(FullData[0:5])
        
        # Feature Scaling for fast training of neural networks
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        
        # Choosing between Standardization or normalization
        #sc = StandardScaler()
        sc=MinMaxScaler()
        
        DataScaler = sc.fit(FullData)
        X=DataScaler.transform(FullData)
        #X=FullData
        
        #print('### After Normalization ###')
        #print(X[0:5])

        # split into samples
        X_samples = list()
        y_samples = list()

        NumerOfRows = len(X)
        TimeSteps=10  # next day's Price Prediction is based on last how many past day's prices

        # Iterate thru the values to create combinations
        for i in range(TimeSteps , NumerOfRows , 1):
            x_sample = X[i-TimeSteps:i]
            y_sample = X[i]
            X_samples.append(x_sample)
            y_samples.append(y_sample)

        ################################################
        # Reshape the Input as a 3D (number of samples, Time Steps, Features)
        X_data=np.array(X_samples)
        X_data=X_data.reshape(X_data.shape[0],X_data.shape[1], 1)
        #print('\n#### Input Data shape ####')
        #print(X_data.shape)

        # We do not reshape y as a 3D data  as it is supposed to be a single column only
        y_data=np.array(y_samples)
        y_data=y_data.reshape(y_data.shape[0], 1)
        #print('\n#### Output Data shape ####')
        #print(y_data.shape)

        # Choosing the number of testing data records
        TestingRecords=5

        # Splitting the data into train and test
        X_train=X_data[:-TestingRecords]
        X_test=X_data[-TestingRecords:]
        y_train=y_data[:-TestingRecords]
        y_test=y_data[-TestingRecords:]

        ############################################

        # Printing the shape of training and testing
        #print('\n#### Training Data shape ####')
        #print(X_train.shape)
        #print(y_train.shape)
        #print('\n#### Testing Data shape ####')
        #print(X_test.shape)
        #print(y_test.shape)
            
        # Visualizing the input and output being sent to the LSTM model
        #for inp, out in zip(X_train[0:2], y_train[0:2]):
        #    print(inp,'--', out)
            
        # Defining Input shapes for LSTM
        TimeSteps=X_train.shape[1]
        TotalFeatures=X_train.shape[2]
        #print("Number of TimeSteps:", TimeSteps)
        #print("Number of Features:", TotalFeatures)
        return X, DataScaler

    def predict_many(self, df, X, DataScaler, prediction_column, epochs, batch_size):
        # Considering the Full Data again which we extracted above
        # Printing the last 10 values
        #print('Original Prices')
        #print(FullData[-10:])
        
        #print('###################')
        
        # Printing last 10 values of the scaled data which we have created above for the last model
        # Here I am changing the shape of the data to one dimensional array because
        # for Multi step data preparation we need to X input in this fashion
        X=X.reshape(X.shape[0],)
        #print('Scaled Prices')
        #print(X[-10:])

        # Multi step data preparation

        # split into samples
        X_samples = list()
        y_samples = list()

        NumerOfRows = len(X)
        TimeSteps=10  # next few day's Price Prediction is based on last how many past day's prices
        FutureTimeSteps=5 # How many days in future you want to predict the prices

        # Iterate thru the values to create combinations
        for i in range(TimeSteps , NumerOfRows-FutureTimeSteps , 1):
            x_sample = X[i-TimeSteps:i]
            y_sample = X[i:i+FutureTimeSteps]
            X_samples.append(x_sample)
            y_samples.append(y_sample)

        ################################################

        # Reshape the Input as a 3D (samples, Time Steps, Features)
        X_data=np.array(X_samples)
        X_data=X_data.reshape(X_data.shape[0],X_data.shape[1], 1)
        #print('### Input Data Shape ###') 
        #print(X_data.shape)

        # We do not reshape y as a 3D data  as it is supposed to be a single column only
        y_data=np.array(y_samples)
        #print('### Output Data Shape ###') 
        #print(y_data.shape)

        # Choosing the number of testing data records
        TestingRecords=5

        # Splitting the data into train and test
        X_train=X_data[:-TestingRecords]
        X_test=X_data[-TestingRecords:]
        y_train=y_data[:-TestingRecords]
        y_test=y_data[-TestingRecords:]
        '''
        #############################################
        # Printing the shape of training and testing
        print('\n#### Training Data shape ####')
        print(X_train.shape)
        print(y_train.shape)

        print('\n#### Testing Data shape ####')
        print(X_test.shape)
        print(y_test.shape)
        '''
        # Visualizing the input and output being sent to the LSTM model
        # Based on last 10 days prices we are learning the next 5 days of prices

        '''
        for inp, out in zip(X_train[0:2], y_train[0:2]):
            print(inp)
            print('====>')
            print(out)
            print('#'*20)
        '''
        # Defining Input shapes for LSTM
        TimeSteps=X_train.shape[1]
        TotalFeatures=X_train.shape[2]
        #print("Number of TimeSteps:", TimeSteps)
        #print("Number of Features:", TotalFeatures)

        # Importing the Keras libraries and packages
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import LSTM

        # Initialising the RNN
        regressor = Sequential()

        # Adding the First input hidden layer and the LSTM layer
        # return_sequences = True, means the output of every time step to be shared with hidden next layer
        regressor.add(LSTM(units = 10, activation = 'relu', input_shape = (TimeSteps, TotalFeatures), return_sequences=True))


        # Adding the Second hidden layer and the LSTM layer
        regressor.add(LSTM(units = 5, activation = 'relu', input_shape = (TimeSteps, TotalFeatures), return_sequences=True))

        # Adding the Third hidden layer and the LSTM layer
        regressor.add(LSTM(units = 5, activation = 'relu', return_sequences=False ))


        # Adding the output layer
        # Notice the number of neurons in the dense layer is now the number of future time steps 
        # Based on the number of future days we want to predict
        regressor.add(Dense(units = FutureTimeSteps))

        # Compiling the RNN
        regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

        ###################################################################

        import time
        # Measuring the time taken by the model to train
        StartTime=time.time()

        # Fitting the RNN to the Training set
        regressor.fit(X_train, y_train, batch_size = batch_size, epochs = epochs)

        EndTime=time.time()
        #print("############### Total Time Taken: ", round((EndTime-StartTime)/60), 'Minutes #############')

        predicted_Price = regressor.predict(X_test)
        predicted_Price = DataScaler.inverse_transform(predicted_Price)
        #print('#### Predicted Prices ####')
        #print(predicted_Price)
        
        # Getting the original price values for testing data
        orig=y_test
        orig=DataScaler.inverse_transform(y_test)
        #print('\n#### Original Prices ####')
        #print(orig)

        #import matplotlib.pyplot as plt
        
        accuracy = []
        for i in range(len(orig)):
            Prediction=predicted_Price[i]
            Original=orig[i]
            accuracy.append(100 - (100*(abs(Original-Prediction)/Original)).mean().round(2))

            #print('### Accuracy of the predictions:'+ str(100 - (100*(abs(Original-Prediction)/Original)).mean().round(2))+'% ###')

            # Visualising the results
            #plt.plot(Prediction, color = 'blue', label = 'Predicted Volume')
            #plt.plot(Original, color = 'lightblue', label = 'Original Volume')

            #plt.title('### Accuracy of the predictions:'+ str(100 - (100*(abs(Original-Prediction)/Original)).mean().round(2))+'% ###')
            #plt.xlabel('Trading Date')
            
            startDateIndex=(FutureTimeSteps*TestingRecords)-FutureTimeSteps*(i+1)
            endDateIndex=(FutureTimeSteps*TestingRecords)-FutureTimeSteps*(i+1) + FutureTimeSteps
            TotalRows=df.shape[0]

            #plt.xticks(range(FutureTimeSteps), df.iloc[TotalRows-endDateIndex : TotalRows-(startDateIndex) , :]['TradeDate'])
            #plt.ylabel('Stock Price')

            #plt.legend()
            #fig=plt.gcf()
            #fig.set_figwidth(16)
            #fig.set_figheight(3)
            #plt.show()


        # Making predictions on test data
        Last10DaysPrices=np.array([1376.2, 1371.75,1387.15,1370.5 ,1344.95, 
                        1312.05, 1316.65, 1339.45, 1339.7 ,1340.85])
        
        Last10DaysPrices = np.array(df[prediction_column][-10:].tolist())

        # Reshaping the data to (-1,1 )because its a single entry
        Last10DaysPrices=Last10DaysPrices.reshape(-1, 1)
        
        # Scaling the data on the same level on which model was trained
        X_test=DataScaler.transform(Last10DaysPrices)
        
        NumberofSamples=1
        TimeSteps=X_test.shape[0]
        NumberofFeatures=X_test.shape[1]
        # Reshaping the data as 3D input
        X_test=X_test.reshape(NumberofSamples,TimeSteps,NumberofFeatures)
        
        # Generating the predictions for next 5 days
        Next5DaysPrice = regressor.predict(X_test)
        
        # Generating the prices in original scale
        Next5DaysPrice = DataScaler.inverse_transform(Next5DaysPrice)

        return Next5DaysPrice, accuracy

    def predict_one(self, epochs):


        '''
        # Importing the Keras libraries and packages
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import LSTM

        # Initialising the RNN
        regressor = Sequential()

        # Adding the First input hidden layer and the LSTM layer
        # return_sequences = True, means the output of every time step to be shared with hidden next layer
        regressor.add(LSTM(units = 10, activation = 'relu', input_shape = (TimeSteps, TotalFeatures), return_sequences=True))

        # Adding the Second Second hidden layer and the LSTM layer
        regressor.add(LSTM(units = 5, activation = 'relu', input_shape = (TimeSteps, TotalFeatures), return_sequences=True))

        # Adding the Second Third hidden layer and the LSTM layer
        regressor.add(LSTM(units = 5, activation = 'relu', return_sequences=False ))


        # Adding the output layer
        regressor.add(Dense(units = 1))

        # Compiling the RNN
        regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

        ##################################################

        import time
        # Measuring the time taken by the model to train
        StartTime=time.time()

        # Fitting the RNN to the Training set
        regressor.fit(X_train, y_train, batch_size = 5, epochs = 10)

        EndTime=time.time()
        print("## Total Time Taken: ", round((EndTime-StartTime)/60), 'Minutes ##')


        # Making predictions on test data
        predicted_Price = regressor.predict(X_test)
        predicted_Price = DataScaler.inverse_transform(predicted_Price)

        # Getting the original price values for testing data
        orig=y_test
        orig=DataScaler.inverse_transform(y_test)

        # Accuracy of the predictions
        print('Accuracy:', 100 - (100*(abs(orig-predicted_Price)/orig)).mean())

        # Visualising the results
        #import matplotlib.pyplot as plt

        plt.plot(predicted_Price, color = 'blue', label = 'Predicted Volume')
        plt.plot(orig, color = 'lightblue', label = 'Original Volume')

        plt.title('Stock Price Predictions')
        plt.xlabel('Trading Date')
        plt.xticks(range(TestingRecords), df.tail(TestingRecords)['TradeDate'])
        plt.ylabel('Stock Price')

        plt.legend()
        fig=plt.gcf()
        fig.set_figwidth(20)
        fig.set_figheight(6)
        plt.show()


        # Generating predictions on full data
        TrainPredictions=DataScaler.inverse_transform(regressor.predict(X_train))
        TestPredictions=DataScaler.inverse_transform(regressor.predict(X_test))

        FullDataPredictions=np.append(TrainPredictions, TestPredictions)
        FullDataOrig=FullData[TimeSteps:]

        # plotting the full data
        plt.plot(FullDataPredictions, color = 'blue', label = 'Predicted Price')
        plt.plot(FullDataOrig , color = 'lightblue', label = 'Original Price')


        plt.title('Stock Price Predictions')
        plt.xlabel('Trading Date')
        plt.ylabel('Stock Price')
        plt.legend()
        fig=plt.gcf()
        fig.set_figwidth(20)
        fig.set_figheight(8)
        plt.show()
        '''







        '''
        #get next price
        # Last 10 days prices
        Last10Days=np.array([1002.15, 1009.9, 1007.5, 1019.75, 975.4,
                    1011.45, 1010.4, 1009,1008.25, 1017.65])

        Last10Days = np.array(df[prediction_column][-10:].tolist())

        # Normalizing the data just like we did for training the model
        Last10Days=DataScaler.transform(Last10Days.reshape(-1,1))

        # Changing the shape of the data to 3D
        # Choosing TimeSteps as 10 because we have used the same for training
        NumSamples=1
        TimeSteps=10
        NumFeatures=1
        Last10Days=Last10Days.reshape(NumSamples,TimeSteps,NumFeatures)

        #############################

        # Making predictions on data
        predicted_Price = regressor.predict(Last10Days)
        predicted_Price = DataScaler.inverse_transform(predicted_Price)
        print(predicted_Price)

        '''



    def go(self):
        prediction_column = 'close'
        df = self.data()

        mean_accuracy = 0
        '''
        while mean_accuracy < 70:
            X, DataScaler = self.train(df, prediction_column)
            Next5DaysPrice, accuracy = self.predict_many(df, X, DataScaler, prediction_column, 3)
            mean_accuracy = statistics.mean(accuracy)
            print(mean_accuracy)
        '''

        X, DataScaler = self.train(df, prediction_column)
        Next5DaysPrice, accuracy = self.predict_many(df, X, DataScaler, prediction_column, 100, 5)
        mean_accuracy = statistics.mean(accuracy)
        
        print(accuracy)
        print(mean_accuracy)

        Next5DaysPrice = pd.DataFrame(Next5DaysPrice[0].tolist(), columns=['close'])
        base = datetime.today() #+ timedelta(days=1)
        date_list = [base + timedelta(days=x) for x in range(5)]
        date_list = [o.strftime('%Y-%m-%d') for o in date_list]

        Next5DaysPrice['date'] = date_list
        Next5DaysPrice['accuracy'] = accuracy
  
        files = []

        plt.figure(figsize=(16,8))
        plt.title('Deep learning LSTM prediction model for ' + str(sys.argv[2]) + ' on ' + str(sys.argv[1]) + ' Timeframe')
        plt.xlabel('Date',fontsize=18)
        plt.ylabel('Close Price USD($)',fontsize=18)
        plt.plot(df['date'][-55:], df['close'][-55:])
        plt.plot(Next5DaysPrice['date'], Next5DaysPrice['close'])
        plt.xticks(rotation="vertical")
        file = os.path.join(dir, 'img/predict-' + str(datetime.now()) + '.png')
        plt.savefig(file)
        files.append(file)
    
        plt.figure(figsize=(12,6))
        plt.title('5 Day close price prediction')
        plt.xlabel('Date',fontsize=18)
        plt.ylabel('Close Price USD($)',fontsize=18)  
        plt.plot(Next5DaysPrice['date'], Next5DaysPrice['close'])
        file = os.path.join(dir, 'img/5DayPredict-' + str(datetime.now()) + '.png')
        plt.savefig(file)
        files.append(file)

        plt.figure(figsize=(12,6))
        plt.title('Prediction accuracy')
        plt.xlabel('Date',fontsize=18)
        plt.ylabel('Prediction Accuracy',fontsize=18)  
        plt.plot(Next5DaysPrice['date'], Next5DaysPrice['accuracy'])
        file = os.path.join(dir, 'img/5DayPredictAccuracy-' + str(datetime.now()) + '.png')
        plt.savefig(file)
        files.append(file)

        twitter = t.Twitter()
        utc_datetime = datetime.utcnow()
        data = ''
        for h in ['BTCUSD','BTC','Bitcoin','DeepLearning','NeuralNetworks']:
            data = data + '#' + h+ ' '
        data = data + '\r\n'
        data = data + str(utc_datetime.strftime("%Y-%m-%d %H:%M:%S")) + '\r\n'
        data = data + 'Timeframe: ' + '1 Day' + '\r\n\r\n'
        print(data)
        twitter.tweet(data, files)
p = Predict()
