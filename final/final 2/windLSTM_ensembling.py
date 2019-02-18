import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import SVR
import pandas as pd
import os
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
from scipy import spatial
from math import *
import sys
import time
import math
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import scipy.stats as st
import datetime
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Average,LSTM,Dropout,CuDNNLSTM, Input
from sklearn.preprocessing import MinMaxScaler
from keras import optimizers
from keras import Model

def singleTurbineData(turbine_iter):
	df = pd.read_csv('la-haute-borne-data-2013-2016.csv', sep=';')
	df['Date_time'] = df['Date_time'].astype(str).str[:-6] #remove timezone (caused me an hour of pain)
	df.Date_time=pd.to_datetime(df['Date_time'])
	df=df.fillna(method='ffill')
	turbines=df.Wind_turbine_name.unique()
	df['sin']=np.sin(df['Wa_c_avg']/360*2*math.pi)
	df['cos']=np.cos(df['Wa_c_avg']/360*2*math.pi)
	df=df[df['Wind_turbine_name']==turbines[turbine_iter]]
	df=df.sort_values(by='Date_time')
	df = df.reset_index()
	return df,turbines[turbine_iter]


def updateDataset(df, test_date,train_start_date,end_date,trainSet,testSet,recordsBack):
	'''
	this function cleans up assumed datasets lengths.  Because some values are missing from the dataset, I update counts
	for test and train dataset variables based on the length of the value I receive from "currentTurbine"
	'''	
	currentTurbine=df[(df['Date_time']>= train_start_date) & (df['Date_time']<end_date)]
	
	if(len(currentTurbine.Date_time.values)==(trainSet+testSet+recordsBack)):
		return currentTurbine
	else:
		print("Adjusting dataset, value(s) missing from time series.")
		s = currentTurbine.Date_time.eq(test_date)
		location=s.index[s][-1]
		currentTurbine=df.loc[location-trainSet-recordsBack:location+testSet-1]
		if(len(currentTurbine.Date_time.values)==(trainSet+testSet+recordsBack)):
			return currentTurbine
		else:
			print("Exiting...")                    
			sys.exit("Error Retrieving data")



def createGraph(weighted,actual, rmse):
	X = np.arange(0,len(actual))
	figure = plt.figure()
	tick_plot = figure.add_subplot(1, 1, 1)
	tick_plot.plot(X, actual,  color='green', linestyle='-', marker='*', label='Actual')
	tick_plot.plot(X, weighted,  color='blue',linestyle='-', marker='*', label='Predictions')
	plt.xlabel('Time (ten minute increments for a day)')
	plt.ylabel('Angle')
	plt.legend(loc='upper left')
	plt.title('Wind Angles and SVR Predictions\nError:  '+str(rmse))
	plt.show()

def setupTrainTestSets(train_test_data,total,recordsBack, trainSet,cos=False):
	from sklearn.preprocessing import normalize
	i=0
	training_data = []
	wind_direction_actual = []
	wind_speed=[]
	temperature=[]
	actual=[]

	while i <total:
		wind_speed.append(train_test_data.Ws_avg.values[i:recordsBack+i])
		temperature.append(train_test_data.Ot_avg.values[i:recordsBack+i])		
		if(cos):
			training_data.append(train_test_data.cos.values[i:recordsBack+i])
			wind_direction_actual.append(train_test_data.cos.values[recordsBack+i])
		else:
			training_data.append(train_test_data.sin.values[i:recordsBack+i])
			wind_direction_actual.append(train_test_data.sin.values[recordsBack+i])
		
		actual.append(train_test_data['Wa_c_avg'].values[recordsBack+i])
		i+=1

	training_data=np.array(training_data)
	wind_direction_actual=np.array(wind_direction_actual)
	
	wind_speed=np.array(wind_speed)
	wind_speed=normalize(wind_speed)
	
	temperature=np.array(temperature)
	temperature=normalize(temperature)

	
	training_data=np.reshape(training_data, (training_data.shape[0], training_data.shape[1],1))
	wind_speed=np.reshape(wind_speed, (wind_speed.shape[0], wind_speed.shape[1],1))
	temperature=np.reshape(temperature, (temperature.shape[0], temperature.shape[1],1))

	
	training_data=np.concatenate((training_data, wind_speed), axis=2)
	training_data=np.concatenate((training_data, temperature), axis=2)

	trainX_initial=training_data[:trainSet-144]
	trainY_initial=wind_direction_actual[:trainSet-144]

	validationX=training_data[trainSet-144:trainSet]
	validationY=wind_direction_actual[trainSet-144:trainSet]


	testX=training_data[trainSet:]
	testY=wind_direction_actual[trainSet:]
	actual=np.array(actual[trainSet:])


	return trainX_initial, trainY_initial, validationX, validationY, testX,testY, actual


'''
initialize variables
'''
def dataSetup(test_date):
        testSet=24*6 #test 1 day of values

        previousDays_rows=75
        trainSet=previousDays_rows*24*6 

        total=trainSet+testSet

        previousDays_columns=6
        recordsBack=previousDays_columns*24*6

        test_date=test_date+datetime.timedelta(days=0)
        train_start_date=test_date+datetime.timedelta(days=-(previousDays_rows+previousDays_columns))
        end_date=test_date+datetime.timedelta(minutes = 10*testSet)

        df, turbine_name=singleTurbineData(0)	

        currentTurbine=updateDataset(df,test_date,train_start_date,end_date,trainSet,testSet,recordsBack)
        return currentTurbine,total,recordsBack, trainSet

def convertToDegrees(sin_prediction,cos_prediction):
	'''
	Converting sine and cosine back to its circular angle depends on finding which of the the 4 circular quadrants the 
	prediction will fall into. If sin and cos are both GT 0, degrees will fall in 0-90.  If sin>0 cos<0, degrees will fall into 90-180, etc. 
	'''
	inverseSin=np.degrees(np.arcsin(sin_prediction))
	inverseCos=np.degrees(np.arccos(cos_prediction))
	radians_sin=[]
	radians_cos=[]
	for a,b,c,d in zip(sin_prediction, cos_prediction, inverseSin, inverseCos):
		if(a>0 and b>0):
			radians_sin.append(c)
			radians_cos.append(d)	
		elif(a>0 and b<0):
			radians_sin.append(180-c)
			radians_cos.append(d)	
		elif(a<0 and b<0):
			radians_sin.append(180-c)
			radians_cos.append(360-d)	
		elif(a<0 and b>0):
			radians_sin.append(360+c)
			radians_cos.append(360-d)
	radians_sin=np.array(radians_sin)
	radians_cos=np.array(radians_cos)
	return radians_sin, radians_cos



def calcWeightedDegreePredictions(sin_error,cos_error,radians_sin,radians_cos):
	errorTotal=cos_error+sin_error
	sinWeight=(errorTotal-sin_error)/errorTotal
	cosWeight=(errorTotal-cos_error)/errorTotal
	weighted=np.add(sinWeight*radians_sin, cosWeight*radians_cos)
	return weighted






def model_2():
        adam=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model = Sequential()
        model.add(LSTM(4, input_shape=(testX.shape[1], testX.shape[2])))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(loss='mean_absolute_error', optimizer=adam)
        return model


def model_3():
        return model_1()
def model_4():
        return model_1()

def model_1():
        adam=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model = Sequential()
        model.add(LSTM(8, input_shape=(testX.shape[1], testX.shape[2])))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(loss='mean_absolute_error', optimizer=adam)
        return model



def ensemble(models, model_input):
    
    outputs = [model(model_input) for model in models]
    y = Average()(outputs)
    model = Model(model_input, y, name='ensemble')
    
    return model

def train_predict(model_name, cos=False):
        if(cos):
                weightFile='weights'+str(model_name)+'_cos.h5'                
        else:
                weightFile='weights'+str(model_name)+'_sin.h5'
        checkpointer=ModelCheckpoint(weightFile, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
        
        if(model_name==1):
                model=model_1()
                reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.9, patience=2, min_lr=0.000001, verbose=1)
                checkpointer=ModelCheckpoint(weightFile, monitor='loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
                earlystopper=EarlyStopping(monitor='loss', min_delta=0, patience=5, verbose=0, mode='auto')
                model.fit(trainX_initial, trainY_initial, validation_data=(validationX, validationY),epochs=25, batch_size=testX.shape[0], verbose=0, shuffle=True,callbacks=[checkpointer,reduce_lr,earlystopper])

        elif(model_name==2):
                '''model=model_2()
                for i in range(5):
                    model.fit(trainX_initial, trainY_initial, validation_data=(validationX, validationY),epochs=8, batch_size=testX.shape[0], verbose=2, shuffle=False,callbacks=[checkpointer])
                    model.reset_states()'''
                model=model_1()
                reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.9, patience=2, min_lr=0.000001, verbose=1)
                checkpointer=ModelCheckpoint(weightFile, monitor='loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
                earlystopper=EarlyStopping(monitor='loss', min_delta=0, patience=5, verbose=0, mode='auto')
                model.fit(trainX_initial, trainY_initial, validation_data=(validationX, validationY),epochs=25, batch_size=testX.shape[0], verbose=0, shuffle=True,callbacks=[checkpointer,reduce_lr,earlystopper])
                    
        elif(model_name==3):
                '''model=model_3()
                reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=2, min_lr=0.000001, verbose=1)
                checkpointer=ModelCheckpoint(weightFile, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
                earlystopper=EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
                model.fit(trainX_initial, trainY_initial, validation_data=(validationX, validationY),epochs=25, batch_size=testX.shape[0], verbose=0, shuffle=False,callbacks=[checkpointer,reduce_lr,earlystopper])'''
                model=model_1()
                reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.9, patience=2, min_lr=0.000001, verbose=1)
                checkpointer=ModelCheckpoint(weightFile, monitor='loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
                earlystopper=EarlyStopping(monitor='loss', min_delta=0, patience=5, verbose=0, mode='auto')
                model.fit(trainX_initial, trainY_initial, validation_data=(validationX, validationY),epochs=25, batch_size=testX.shape[0], verbose=0, shuffle=True,callbacks=[checkpointer,reduce_lr,earlystopper])
               
        elif(model_name==4):
                '''model=model_4()
                for i in range(5):
                    model.fit(trainX_initial, trainY_initial, validation_data=(validationX, validationY),epochs=8, batch_size=testX.shape[0], verbose=2, shuffle=False,callbacks=[checkpointer])
                    model.reset_states()'''
                model=model_1()
                reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.9, patience=2, min_lr=0.000001, verbose=1)
                checkpointer=ModelCheckpoint(weightFile, monitor='loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
                earlystopper=EarlyStopping(monitor='loss', min_delta=0, patience=5, verbose=0, mode='auto')
                model.fit(trainX_initial, trainY_initial, validation_data=(validationX, validationY),epochs=25, batch_size=testX.shape[0], verbose=0, shuffle=True,callbacks=[checkpointer,reduce_lr,earlystopper])

        validationPredict=model.predict(validationX,batch_size=testX.shape[0])
        testPredict = model.predict(testX, batch_size=testX.shape[0])                        
        testPredict[testPredict > 1] = 1
        testPredict[testPredict <-1] = -1
        
        return testPredict, validationPredict




def errors(actual,predicted):
        mae=mean_absolute_error(actual, predicted)
        rmse = sqrt(mean_squared_error(actual, predicted))
        return mae, rmse

def weightedPredictions(predictions,errors):
        inverseErrorTotal=0
        for e in errors:
                inverseErrorTotal+=(sum(errors)-e)
        weighted= np.zeros(predictions[0].shape)
        for p,e in zip(predictions,errors):
                prediction_weight=(sum(errors)-e)/inverseErrorTotal
                weighted+=(p*prediction_weight)
        return weighted

results = pd.DataFrame(columns=['test_date','degrees_mae','rmse'])

for i in range(1,4):
                date_to_test=datetime.datetime(2016, 1, i)
                currentTurbine,total,recordsBack, trainSet=dataSetup(date_to_test)
                
                trainX_initial, trainY_initial, validationX, validationY, testX,testY,actual=setupTrainTestSets(currentTurbine,total,recordsBack, trainSet)
                testPredict_sin_1,validation_sin1=train_predict(1,False)
                testPredict_sin_2,validation_sin2=train_predict(2,False)
                testPredict_sin_3,validation_sin3=train_predict(3,False)
                testPredict_sin_4,validation_sin4=train_predict(4,False)

                val_mae_sin_1,val_rmse_sin_1=errors(validationY,validation_sin1)
                val_mae_sin_2,val_rmse_sin_2=errors(validationY,validation_sin2)
                val_mae_sin_3,val_rmse_sin_3=errors(validationY,validation_sin3)
                val_mae_sin_4,val_rmse_sin_4=errors(validationY,validation_sin4)

                val_ensemble_sin_mae=weightedPredictions([validation_sin1,validation_sin2,validation_sin3,validation_sin4],[val_mae_sin_1,val_mae_sin_2,val_mae_sin_3,val_mae_sin_4])
                val_ensemble_sin_rmse=weightedPredictions([validation_sin1,validation_sin2,validation_sin3,validation_sin4],[val_rmse_sin_1,val_rmse_sin_2,val_rmse_sin_3,val_rmse_sin_4])

                '''m1=model_1()
                m1.load_weights('weights1_sin.h5')                
                m2=model_2()
                m2.load_weights('weights2_sin.h5')
                m3=model_3()
                m3.load_weights('weights3_sin.h5')
                models=[m1,m2,m3]
                ensemble=ensemble(models,Input(shape=(testX.shape[1], testX.shape[2])))
                testPredict = ensemble.predict(testX, batch_size=testX.shape[0])
                print(mean_squared_error(testY, testPredict))'''

                test_ensemble_sin_mae=weightedPredictions([testPredict_sin_1,testPredict_sin_2,testPredict_sin_3,testPredict_sin_4],[val_mae_sin_1,val_mae_sin_2,val_mae_sin_3,val_mae_sin_4])
                test_ensemble_sin_rmse=weightedPredictions([testPredict_sin_1,testPredict_sin_2,testPredict_sin_3,testPredict_sin_4],[val_rmse_sin_1,val_rmse_sin_2,val_rmse_sin_3,val_rmse_sin_4])            


                val_mae_sin_en1,val_rmse_sin_en1=errors(validationY,val_ensemble_sin_mae)
                val_mae_sin_en2,val_rmse_sin_en2=errors(validationY,val_ensemble_sin_rmse)



                trainX_initial, trainY_initial, validationX, validationY,  testX,testY,actual=setupTrainTestSets(currentTurbine,total,recordsBack, trainSet,cos=True)
                testPredict_cos_1,validation_cos1=train_predict(1,True)
                testPredict_cos_2,validation_cos2=train_predict(2,True)
                testPredict_cos_3,validation_cos3=train_predict(3,True)
                testPredict_cos_4,validation_cos4=train_predict(4,True)

                val_mae_cos_1,val_rmse_cos_1=errors(validationY,validation_cos1)
                val_mae_cos_2,val_rmse_cos_2=errors(validationY,validation_cos2)
                val_mae_cos_3,val_rmse_cos_3=errors(validationY,validation_cos3)
                val_mae_cos_4,val_rmse_cos_4=errors(validationY,validation_cos4)

                val_ensemble_cos_mae=weightedPredictions([validation_cos1,validation_sin2,validation_cos3,validation_cos4],[val_mae_cos_1,val_mae_cos_2,val_mae_cos_3,val_mae_cos_4])
                val_ensemble_cos_rmse=weightedPredictions([validation_cos1,validation_cos2,validation_cos3,validation_cos4],[val_rmse_cos_1,val_rmse_cos_2,val_rmse_cos_3,val_rmse_cos_4])

                test_ensemble_cos_mae=weightedPredictions([testPredict_cos_1,testPredict_cos_2,testPredict_cos_3,testPredict_cos_4],[val_mae_cos_1,val_mae_cos_2,val_mae_cos_3,val_mae_cos_4])
                test_ensemble_cos_rmse=weightedPredictions([testPredict_cos_1,testPredict_cos_2,testPredict_cos_3,testPredict_cos_4],[val_rmse_cos_1,val_rmse_cos_2,val_rmse_cos_3,val_rmse_cos_4])            


                val_mae_cos_en1,val_rmse_cos_en1=errors(validationY,val_ensemble_cos_mae)
                val_mae_cos_en2,val_rmse_cos_en2=errors(validationY,val_ensemble_cos_rmse)

                #model 1
                print("Model 1")
                radians_sin, radians_cos=convertToDegrees(testPredict_sin_1,testPredict_cos_1)
                predicted=calcWeightedDegreePredictions(val_mae_sin_1,val_mae_cos_1,radians_sin,radians_cos)
                mae,rmse=errors(actual,predicted)
                print("MAE weighted:"+str(mae)+" rmse: "+str(rmse))
                predicted=calcWeightedDegreePredictions(val_rmse_sin_1,val_rmse_cos_1,radians_sin,radians_cos)
                mae,rmse=errors(actual,predicted)
                print("RMSE weighted mae:"+str(mae)+" rmse: "+str(rmse))


                print("\nModel 2")
                radians_sin, radians_cos=convertToDegrees(testPredict_sin_2,testPredict_cos_2)
                predicted=calcWeightedDegreePredictions(val_mae_sin_2,val_mae_cos_2,radians_sin,radians_cos)
                mae,rmse=errors(actual,predicted)
                print("MAE weighted:"+str(mae)+" rmse: "+str(rmse))
                predicted=calcWeightedDegreePredictions(val_rmse_sin_2,val_rmse_cos_2,radians_sin,radians_cos)
                mae,rmse=errors(actual,predicted)
                print("RMSE weighted mae:"+str(mae)+" rmse: "+str(rmse))                



                print("\nModel 3")
                radians_sin, radians_cos=convertToDegrees(testPredict_sin_3,testPredict_cos_3)
                predicted=calcWeightedDegreePredictions(val_mae_sin_3,val_mae_cos_3,radians_sin,radians_cos)
                mae,rmse=errors(actual,predicted)
                print("MAE weighted:"+str(mae)+" rmse: "+str(rmse))
                predicted=calcWeightedDegreePredictions(val_rmse_sin_3,val_rmse_cos_3,radians_sin,radians_cos)
                mae,rmse=errors(actual,predicted)
                print("RMSE weighted mae:"+str(mae)+" rmse: "+str(rmse))


                print("\nModel 4")
                radians_sin, radians_cos=convertToDegrees(testPredict_sin_4,testPredict_cos_4)
                predicted=calcWeightedDegreePredictions(val_mae_sin_4,val_mae_cos_4,radians_sin,radians_cos)
                mae,rmse=errors(actual,predicted)
                print("MAE weighted:"+str(mae)+" rmse: "+str(rmse))
                predicted=calcWeightedDegreePredictions(val_rmse_sin_4,val_rmse_cos_4,radians_sin,radians_cos)
                mae,rmse=errors(actual,predicted)
                print("RMSE weighted mae:"+str(mae)+" rmse: "+str(rmse))       


                print("\nEnsembled")
                radians_sin, radians_cos=convertToDegrees(test_ensemble_sin_mae,test_ensemble_cos_mae)
                predicted=calcWeightedDegreePredictions(val_mae_sin_en1,val_mae_cos_en1,radians_sin,radians_cos)
                mae,rmse=errors(actual,predicted)
                print("MAE weighted:"+str(mae)+" rmse: "+str(rmse))
                
                predicted=calcWeightedDegreePredictions(val_rmse_sin_en2,val_rmse_cos_en2,radians_sin,radians_cos)
                mae,rmse=errors(actual,predicted)
                print("RMSE weighted mae:"+str(mae)+" rmse: "+str(rmse))       

                results = results.append({'test_date':str(date_to_test)[:10],'mae': mae, 'rmse': rmse}, ignore_index=True)
                print(results)

                #createGraph(weighted,actual,rmse)
                guesses_pandas = pd.DataFrame(predicted)
                actual_pandas = pd.DataFrame(actual)
                guesses_file="ensemble/"+str(date_to_test)[:10]+"guesses.csv"
                actual_file="ensemble/"+str(date_to_test)[:10]+"actual.csv"
                guesses_pandas.to_csv(guesses_file)
                actual_pandas.to_csv(actual_file)
                results.to_csv("ensemble/year_results.csv")


