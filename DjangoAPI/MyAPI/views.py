from django.shortcuts import render
from rest_framework import viewsets
from rest_framework.decorators import api_view
from django.core import serializers
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from rest_framework.parsers import JSONParser
import pickle
import json
import joblib
import numpy as np
from sklearn import preprocessing
import pandas as pd
from datetime import date
from sklearn.preprocessing import MinMaxScaler
from numpy import array
import keras
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import yfinance as yf







@api_view(["POST"])
def predict(request):
    try:
        mdl=tf.keras.models.load_model("btc__model.h5")
        mdl_open=tf.keras.models.load_model("btc__model_open.h5")
        mdl_low=tf.keras.models.load_model("btc__model_low.h5")
        mdl_high=tf.keras.models.load_model("btc__model_high.h5")
        mydata=request.data
        print(mydata)
       
        Date=str(mydata['date'])
        day=int(Date[:2])
        month=int(Date[3:5])
        year=int(Date[6:10])
        
        df=yf.download(tickers='BTC-USD', period = '10d', interval = '1d')
        df.reset_index(inplace=True)
        y_date=df['Date'].tolist()
        y_date_str=str(y_date[-1])
        y=int(y_date_str[:4])
        m=int(y_date_str[5:7])
        d=int(y_date_str[8:10])
        ndays=(date(year,month,day)-date(y,m,d)).days
        df1=df.reset_index()['Close']
        df1_low=df.reset_index()['Low']                      
        df1_high=df.reset_index()['High']                          
        df1_open=df.reset_index()['Open']                          
        
        
        
        
        scaler_low=joblib.load('scaler_low.save')
        df1_low=scaler_low.transform(np.array(df1_low).reshape(-1,1))   
        test_data_low=df1_low[len(df1_low)-10:]                  
        x_input_low=test_data_low.reshape(1,-1)                   
        temp_input_low=list(x_input_low)                       
        temp_input_low=temp_input_low[0].tolist()                     

        
        scaler=joblib.load('scaler.save')
        df1=scaler.transform(np.array(df1).reshape(-1,1))
        test_data=df1[len(df1)-10:]
        x_input=test_data.reshape(1,-1)
        temp_input=list(x_input)
        temp_input=temp_input[0].tolist()

            
        
        scaler_high=joblib.load('scaler_high.save')             
        df1_high=scaler_high.transform(np.array(df1_high).reshape(-1,1))    
        test_data_high=df1_high[len(df1_high)-10:]                  
        x_input_high=test_data_high.reshape(1,-1)                   
        temp_input_high=list(x_input_high)                          
        temp_input_high=temp_input_high[0].tolist() 
        
        scaler_open=joblib.load('scaler_open.save')            
        df1_open=scaler_open.transform(np.array(df1_open).reshape(-1,1))    
        test_data_open=df1_open[len(df1_open)-10:]                 
        x_input_open=test_data_open.reshape(1,-1)                   
        temp_input_open=list(x_input_open)                          
        temp_input_open=temp_input_open[0].tolist() 
        
        lst_output=[]
        lst_open=[]                                                 
        lst_low=[]
        lst_high=[]
        
        
        n_steps=10
        i=0
        print("-------------->no of days :",ndays)
        print("--------------> input going into model of close price prediction :",x_input)
        while(i<ndays):
            
            if(len(temp_input)>10):
                #print(temp_input)
                
                x_input=np.array(temp_input[1:])
                x_input_low=np.array(temp_input_low[1:])
                x_input_high=np.array(temp_input_high[1:])
                x_input_open=np.array(temp_input_open[1:])


                # print("{} day input {}".format(i,x_input))
                x_input=x_input.reshape(1,-1)
                x_input = x_input.reshape((1, n_steps, 1))
                
                x_input_low=x_input_low.reshape(1,-1)
                x_input_low = x_input_low.reshape((1, n_steps, 1))


                x_input_high=x_input_high.reshape(1,-1)
                x_input_high = x_input_high.reshape((1, n_steps, 1))
                
                x_input_open=x_input_open.reshape(1,-1)
                x_input_open = x_input_open.reshape((1, n_steps, 1))
                
                #print(x_input)
                yhat = mdl.predict(x_input)
                yhat_low=mdl_low.predict(x_input_low)
                yhat_high=mdl_high.predict(x_input_high)
                yhat_open=mdl_open.predict(x_input_open)
                
                # print("{} day output {}".format(i,yhat))
                temp_input.extend(yhat[0].tolist())
                temp_input=temp_input[1:]
                
                temp_input_low.extend(yhat_low[0].tolist())
                temp_input_low=temp_input_low[1:]

                temp_input_high.extend(yhat_high[0].tolist())
                temp_input_high=temp_input_high[1:]

                temp_input_open.extend(yhat_open[0].tolist())
                temp_input_open=temp_input_open[1:]

                #print(temp_input)
                lst_output.extend(yhat.tolist())
                lst_low.extend(yhat_low.tolist())
                lst_high.extend(yhat_high.tolist())
                lst_open.extend(yhat_open.tolist())
                i=i+1
            else:
                x_input = x_input.reshape((1, n_steps,1))
                x_input_low=x_input_low.reshape((1,n_steps,1))
                x_input_high=x_input_high.reshape((1,n_steps,1))
                x_input_open=x_input_open.reshape((1,n_steps,1))
               
                yhat = mdl.predict(x_input)
                yhat_low=mdl_low.predict(x_input_low)    
                yhat_high=mdl_high.predict(x_input_high)
                yhat_open=mdl_open.predict(x_input_open)                    

                print("---------------> predicted 1st value :",yhat[0])
                
                temp_input.extend(yhat[0].tolist())
                temp_input_low.extend(yhat_low[0].tolist())          
                temp_input_high.extend(yhat_high[0].tolist())
                temp_input_open.extend(yhat_open[0].tolist())          

                
                print("---------------> length of input list",len(temp_input))
                lst_output.extend(yhat.tolist())
                lst_low.extend(yhat_low.tolist())                            
                lst_high.extend(yhat_high.tolist())
                lst_open.extend(yhat_open.tolist())                            
                
                i=i+1
        print("-----------> before rescaling: ",lst_output[-1])
        lst_output=scaler.inverse_transform(lst_output)
        lst_low=scaler_low.inverse_transform(lst_low)
        lst_high=scaler_high.inverse_transform(lst_high)
        lst_open=scaler_open.inverse_transform(lst_open)
        prediction=lst_output[-1]
        prediction_low=lst_low[-1]
        prediction_high=lst_high[-1]
        prediction_open=lst_open[-1]
        print("----------> After rescaling:",prediction)
        predict_obj={
                
                "open_price":prediction_open[0],
                "high_price":prediction_high[0],
                "low_price":prediction_low[0],
                "close_price":prediction[0]
        }
        return JsonResponse(predict_obj)

    except ValueError as e:
        return Response(e.args[0], status.HTTP_400_BAD_REQUEST)


@api_view(["POST"])
def predict_from_sample(request):
    try:
        mdl=tf.keras.models.load_model("btc__model.h5")
        mdl_low=tf.keras.models.load_model("btc__model_low.h5")
        mdl_high=tf.keras.models.load_model("btc__model_high.h5")
        mdl_open=tf.keras.models.load_model("btc__model_open.h5")
        
        mydata=request.data
        
        df=pd.DataFrame({'Close':mydata['close']})
        df_low=pd.DataFrame({'Low':mydata['low']})
        df_high=pd.DataFrame({'High':mydata['high']})
        df_open=pd.DataFrame({'Open':mydata['open']})
        
        
        
        df1=df.reset_index()['Close']
        scaler=joblib.load('scaler.save')
        df1=scaler.transform(np.array(df1).reshape(-1,1))
        test_data=df1[len(df1)-10:]
        x_input=test_data.reshape(1,-1)
        
        
        
        df1_low=df_low.reset_index()['Low']
        scaler_low=joblib.load('scaler_low.save')
        df1_low=scaler_low.transform(np.array(df1_low).reshape(-1,1))   
        test_data_low=df1_low[len(df1_low)-10:]                  
        x_input_low=test_data_low.reshape(1,-1)                   
                              
        
        
        
        
        
        
        df1_high=df_high.reset_index()['High']
        scaler_high=joblib.load('scaler_high.save')           
        df1_high=scaler_high.transform(np.array(df1_high).reshape(-1,1))    
        test_data_high=df1_high[len(df1_high)-10:]                 
        x_input_high=test_data_high.reshape(1,-1)                 
       
        
        df1_open=df_open.reset_index()['Open']
        scaler_open=joblib.load('scaler_open.save')             
        df1_open=scaler_open.transform(np.array(df1_open).reshape(-1,1))    
        test_data_open=df1_open[len(df1_open)-10:]                 
        x_input_open=test_data_open.reshape(1,-1)
        
        
        lst_output=[]
        lst_low=[]
        lst_high=[]
        lst_open=[]
        
        
        n_steps=10
        i=0
        
        yhat = mdl.predict(x_input)
        yhat_low=mdl_low.predict(x_input_low)
        yhat_high=mdl_high.predict(x_input_high)
        yhat_open=mdl_open.predict(x_input_open)

        lst_output.extend(yhat.tolist())
        lst_low.extend(yhat_low.tolist())
        lst_high.extend(yhat_high.tolist())
        lst_open.extend(yhat_open.tolist())
        
        lst_output=scaler.inverse_transform(lst_output)
        lst_low=scaler_low.inverse_transform(lst_low)
        lst_high=scaler_high.inverse_transform(lst_high)
        lst_open=scaler_open.inverse_transform(lst_open)
        
        prediction=lst_output[-1]
        prediction_low=lst_low[-1]
        prediction_high=lst_high[-1]
        prediction_open=lst_open[-1]
        
        predict_obj={
                "open_price":prediction_open[0],
                "high_price":prediction_high[0],
                "low_price":prediction_low[0],
                "close_price":prediction[0]
        }
        # predict_json=json.dumps(predict_obj)
        return JsonResponse(predict_obj)

    except ValueError as e:
        return Response(e.args[0], status.HTTP_400_BAD_REQUEST)






@api_view(["POST"])
def predict_range(request):
    try:
        mdl=tf.keras.models.load_model("btc__model.h5")
        mdl_low=tf.keras.models.load_model("btc__model_low.h5")
        mdl_high=tf.keras.models.load_model("btc__model_high.h5")
        mdl_open=tf.keras.models.load_model("btc__model_open.h5")
        
        mydata=request.data
        df=yf.download(tickers='BTC-USD', period = '10d', interval = '1d')
        df.reset_index(inplace=True)
        y_date=df['Date'].tolist()
        y_date_str=str(y_date[-1])
        y=int(y_date_str[:4])
        m=int(y_date_str[5:7])
        d=int(y_date_str[8:10])
       
        date1=str(mydata['from date'])
        d1=int(date1[:2])
        m1=int(date1[3:5])
        y1=int(date1[6:10])
        date2=str(mydata['to date'])
        d2=int(date2[:2])
        m2=int(date2[3:5])
        y2=int(date2[6:10])

        negday=(date(y1,m1,d1)-date(y,m,d)).days
        ndays=(date(y2,m2,d2)-date(y,m,d)).days
       
        
        
        df1=df.reset_index()['Close']
        df1_low=df.reset_index()['Low']                      
        df1_high=df.reset_index()['High']                       
        df1_open=df.reset_index()['Open'] 
        
        
        scaler=joblib.load('scaler.save')
        df1=scaler.transform(np.array(df1).reshape(-1,1))
        test_data=df1[len(df1)-10:]
        x_input=test_data.reshape(1,-1)
        temp_input=list(x_input)
        temp_input=temp_input[0].tolist()
        
        scaler_low=joblib.load('scaler_low.save')
        df1_low=scaler_low.transform(np.array(df1_low).reshape(-1,1))    
        test_data_low=df1_low[len(df1_low)-10:]                  
        x_input_low=test_data_low.reshape(1,-1)                 
        temp_input_low=list(x_input_low)                         
        temp_input_low=temp_input_low[0].tolist()

        scaler_high=joblib.load('scaler_high.save')             
        df1_high=scaler_high.transform(np.array(df1_high).reshape(-1,1))   
        test_data_high=df1_high[len(df1_high)-10:]                  
        x_input_high=test_data_high.reshape(1,-1)                  
        temp_input_high=list(x_input_high)                         
        temp_input_high=temp_input_high[0].tolist() 

        scaler_open=joblib.load('scaler_open.save')             
        df1_open=scaler_open.transform(np.array(df1_open).reshape(-1,1))   
        test_data_open=df1_open[len(df1_open)-10:]                 
        x_input_open=test_data_open.reshape(1,-1)               
        temp_input_open=list(x_input_open)                         
        temp_input_open=temp_input_open[0].tolist()
        
        
        lst_output=[]
        lst_low=[]
        lst_high=[]
        lst_open=[]
       
        n_steps=10
        i=0
        print("-------------->no of days :",ndays)
        print("-------------->negate days:",negday)
        print("--------------> input going into model :",x_input)
        while(i<ndays):
            
            if(len(temp_input)>10):
                
                
                x_input=np.array(temp_input[1:])
                x_input_low=np.array(temp_input_low[1:])
                x_input_high=np.array(temp_input_high[1:])
                x_input_open=np.array(temp_input_open[1:])


                # print("{} day input {}".format(i,x_input))
                x_input=x_input.reshape(1,-1)
                x_input = x_input.reshape((1, n_steps, 1))
                
                x_input_low=x_input_low.reshape(1,-1)
                x_input_low = x_input_low.reshape((1, n_steps, 1))


                x_input_high=x_input_high.reshape(1,-1)
                x_input_high = x_input_high.reshape((1, n_steps, 1))
                
                x_input_open=x_input_open.reshape(1,-1)
                x_input_open = x_input_open.reshape((1, n_steps, 1))
                
                #print(x_input)
                yhat = mdl.predict(x_input)
                yhat_low=mdl_low.predict(x_input_low)
                yhat_high=mdl_high.predict(x_input_high)
                yhat_open=mdl_open.predict(x_input_open)
                
                # print("{} day output {}".format(i,yhat))
                temp_input.extend(yhat[0].tolist())
                temp_input=temp_input[1:]
                
                temp_input_low.extend(yhat_low[0].tolist())
                temp_input_low=temp_input_low[1:]

                temp_input_high.extend(yhat_high[0].tolist())
                temp_input_high=temp_input_high[1:]

                temp_input_open.extend(yhat_open[0].tolist())
                temp_input_open=temp_input_open[1:]

                #print(temp_input)
                lst_output.extend(yhat.tolist())
                lst_low.extend(yhat_low.tolist())
                lst_high.extend(yhat_high.tolist())
                lst_open.extend(yhat_open.tolist())
                i=i+1
            else:
                
                x_input = x_input.reshape((1, n_steps,1))
                x_input_low=x_input_low.reshape((1,n_steps,1))
                x_input_high=x_input_high.reshape((1,n_steps,1))
                x_input_open=x_input_open.reshape((1,n_steps,1))
               
                yhat = mdl.predict(x_input)
                yhat_low=mdl_low.predict(x_input_low)    
                yhat_high=mdl_high.predict(x_input_high)
                yhat_open=mdl_open.predict(x_input_open)                   

                print("---------------> predicted 1st value :",yhat[0])
                
                temp_input.extend(yhat[0].tolist())
                temp_input_low.extend(yhat_low[0].tolist())        
                temp_input_high.extend(yhat_high[0].tolist())
                temp_input_open.extend(yhat_open[0].tolist())         

                
                print("---------------> length of input list",len(temp_input))
                lst_output.extend(yhat.tolist())
                lst_low.extend(yhat_low.tolist())                           
                lst_high.extend(yhat_high.tolist())
                lst_open.extend(yhat_open.tolist())                          
                
                i=i+1
        print("--------->Before rescaling:",lst_output[negday-1:])
        lst_output=scaler.inverse_transform(lst_output)
        lst_low=scaler_low.inverse_transform(lst_low)
        lst_high=scaler_high.inverse_transform(lst_high)
        lst_open=scaler_open.inverse_transform(lst_open)
    
        prediction=lst_output[negday-1:]
        prediction_low=lst_low[negday-1:]
        prediction_high=lst_high[negday-1:]
        prediction_open=lst_open[negday-1:]
        print("----------> After rescaling close price:",prediction)
        arr=prediction[:len(prediction)]
        arr_low=prediction_low[:len(prediction)]
        arr_high=prediction_high[:len(prediction)]
        arr_open=prediction_open[:len(prediction)]
        predict_obj={
                "open_price":arr_open.tolist(),
                "high_price":arr_high.tolist(),
                "low_price":arr_low.tolist(),                
                "close_price":arr.tolist()
        }
        return JsonResponse(predict_obj)
    
    except ValueError as e:
        return Response(e.args[0], status.HTTP_400_BAD_REQUEST)        