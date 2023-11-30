import pandas as pd
import numpy as np
from PIL import Image
import pickle
import streamlit as st
import sklearn
my_model = pickle.load(open("C:\\Users\\Amarachi Uzochukwu\\Desktop\\RFR_model.pkl", 'rb'))
st.title('Ola Bike Delivery Prediction')
img = Image.open("C:\\Users\\Amarachi Uzochukwu\\Downloads\\pexels-norma-mortenson-4391470.jpg")
st.image(img, width=350)

def user_report():
    instant = st.sidebar.slider('instant', 1.0, 731.0, 0.1)
    season = st.sidebar.slider('season', 1.0, 4.0, 1.0)
    year = st.sidebar.slider('year', 0.1, 4.0, 1.0)
    month = st.sidebar.slider('month', 1.0, 12.0, 1.0)
    holidays = st.sidebar.slider('holidays', 0.1, 4.0, 1.0)
    weekday = st.sidebar.slider('weekday', 0.1, 7.0, 1.0)
    workingday = st.sidebar.slider('workingday', 0.1, 3.0, 1.0)
    weather = st.sidebar.slider('weather', 1.0, 4.0, 1.0)
    temp = st.sidebar.slider('temperature', 0.1, 4.0, 0.1)
    atemp = st.sidebar.slider('atemp', 0.1, 4.0, 0.1 )
    humidity = st.sidebar.slider('humidity', 0.0, 4.0, 0.1)
    windspeed = st.sidebar.slider('windspeed', 0.0, 4.0, 3.0)
    casual = st.sidebar.slider('casual', 1.0, 3500.0, 3.0)
    registered = st.sidebar.slider('registered', 1.0, 7000.0, 5.0)

    input_data = {
        'instant': instant,
        'season': season,
        'year': year,
        'month': month,
        'holidays': holidays,
        'weekday': weekday,
        'workingday': workingday,
        'weather': weather,
        'temp': temp,
        'atemp': atemp,
        'humidity': humidity,
        'windspeed': windspeed,
        'casual': casual,
        'registered': registered
    }
    data = pd.DataFrame(input_data, index=[0])
    return data
user_data = user_report()
st.write(user_data)
prediction = my_model.predict(user_data)
st.subheader('Predicted Number of Deliveries')
st.subheader(np.round(prediction[0]))
