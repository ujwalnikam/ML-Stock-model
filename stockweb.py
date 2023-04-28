from cProfile import label
from tkinter import E, NE
from matplotlib.style import use
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import pandas_datareader as data
import pylab
import numpy as np
from pandas_datareader import data as wb


st.set_page_config(
    page_title="Stockweb"
)

st.sidebar.success("Select a page")

html_temp = """
<div style="background-color:tomato;padding:10px">
<h2 style="color:white;text-align:center;">Stock Price Prediction</h2>
</div>
<br>
<br>
"""
st.markdown(html_temp,unsafe_allow_html=True)

user_input = st.text_input('Enter Stock Ticker','AAPL')
st.text('Find Stock Ticker from yahoo finance Eg.TSLA,GOOG')

st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #1888d9;
    color:balck;
}
div.stButton > button:hover {
    background-color: #2ad44c;
    color:white;
    }
</style>""", unsafe_allow_html=True)

if st.button("Predict"):
    #st.title('Stock trend')

    start  = '2010-01-01'
    end =  '2021-02-11'

    

    #df = data.DataReader(user_input,'yahoo',start,end)
    #df = pd.read_csv('TSLA2010.csv')
    #df=wb.DataReader(user_input,data_source='yahoo',start='2015-1-1',end='2023-1-1')
    import yfinance as yahooFinance
    GetFacebookInformation = yahooFinance.Ticker(user_input)
    df = GetFacebookInformation.history(period="10y")
    #df.drop(['Dividends', 'Stock Splits'], axis=1)

    st.subheader('Data from 2010 - 2022')
    st.write(df.head(3))
    st.write(df.tail(3))

    st.subheader('Describing data')
    st.write(df.describe())

    st.subheader('Closing Price vs Time chart')
    fig = plt.figure(figsize=(12,6))
    plt.plot(df.Close,label='Closing price')
    plt.xlabel('Time')
    plt.ylabel('Closing Price')
    plt.legend()
    st.pyplot(fig)

    st.subheader('Closing Price vs Time chart with 100 days Moving Avg')
    ma100 = df.Close.rolling(100).mean() 
    fig = plt.figure(figsize=(12,6))
    plt.plot(ma100,'r',label='MA100days')
    plt.plot(df.Close,'b',label='Real Closing Price')
    plt.xlabel('Time')
    plt.ylabel('Closing Price')
    plt.legend()
    st.pyplot(fig)

    st.subheader('Closing Price vs Time chart with 100MA & 200MA')
    st.text('When the red line is above the green line then it is Uptrend')
    st.text('When the red line is below the green line then it is Downtrend')
    ma100 = df.Close.rolling(100).mean() 
    ma200 = df.Close.rolling(200).mean() 
    fig = plt.figure(figsize=(12,6))
    plt.plot(ma100,'r',label='MA100')
    plt.plot(ma200,'g',label='MA200')
    plt.plot(df.Close,'b',label='Real Closing Price')
    plt.xlabel('Time')
    plt.ylabel('Closing Price')
    plt.legend()
    st.pyplot(fig)

    #################### split data train test 70:30
    data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])


    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))

    data_training_array = scaler.fit_transform(data_training)


    ##### load already trained model
    model = load_model('keras_model.h5')

    ### testing part
    past_100_days = data_training.tail(100)
    final_df = past_100_days.append(data_testing,ignore_index=True)
    input_data = scaler.fit_transform(final_df)

    x_test = []
    y_test = []

    for i in range(100,input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i,0])

    x_test,y_test = np.array(x_test),np.array(y_test)

    ### predictions
    y_predicted = model.predict(x_test)
    #scaleup all values
    scaler = scaler.scale_    

    scalar_factor = 1/scaler[0]
    y_predicted = y_predicted * scalar_factor
    y_test = y_test * scalar_factor

    ########### final visualization
    st.subheader('Prediction vs Original')
    lstmfig = plt.figure(figsize=(12,6))
    plt.plot(y_test,'b',label='Original Price')
    plt.plot(y_predicted,'r',label='Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(lstmfig)

    ################ model 2 #########
    regressor = load_model('keras_model2.h5')
    FullData=df[['Close']].values
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    # Choosing between Standardization or normalization
    #sc = StandardScaler()
    sc=MinMaxScaler()
    DataScaler = sc.fit(FullData)
    X=DataScaler.transform(FullData)

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
   

    # Defining Input shapes for LSTM
    TimeSteps=X_train.shape[1]
    TotalFeatures=X_train.shape[2]
    #print("Number of TimeSteps:", TimeSteps)
    #print("Number of Features:", TotalFeatures)

    # Making predictions on test data
    predicted_Price = regressor.predict(X_test)
    predicted_Price = DataScaler.inverse_transform(predicted_Price)
    #print('#### Predicted Prices ####')
    #print(predicted_Price)

    # Getting the original price values for testing data
    original=y_test
    original=DataScaler.inverse_transform(y_test)
    #print('\n#### Original Prices ####')
    #print(original)
    Accuracy = str(100 - (100*(abs(original-predicted_Price)/original)).mean().round(2))
    #print("Accuracy "+Accuracy)
    st.text("Accuracy : "+Accuracy)

        # Making predictions on test data
    Last10DaysPrices= FullData[-10:]

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
    #print('Next 5 days',Next5DaysPrice)
    
    Next5DaysPrice.shape = (5,1)
    newdata = list(FullData[-5:])+list(Next5DaysPrice)
    ########### next 5 days visualization
    st.subheader('Prediction for next 5 days')
    predfig = plt.figure(figsize=(12,6))
    plt.plot(newdata)
    plt.plot(5,newdata[5],marker="o", markersize=8,color='red',label='After next 5 days')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(predfig)

    st.subheader('Stock Prices for next days')
    stf =""
    for i in range(len(Next5DaysPrice)):
        stf += str(Next5DaysPrice[i][0])
        stf += "   "
    st.text(stf)


    html_temp2 = """
    <div style="background-color:tomato;padding:10px">
    <h3 style="color:white;text-align:center;">Thank You !</h3>
    </div>
    <br>
    """
    st.markdown(html_temp2,unsafe_allow_html=True)

    html_temp3 = """
    <div style=padding:10px">
    <h5 style="color:white;text-align:center;">@copyright ujwal</h5>
    </div>
    <br>
    """
    st.markdown(html_temp3,unsafe_allow_html=True)

    
##### check for run successfully
#print('hello') 
