import streamlit as st
import numpy as np
import pandas as pd
from streamlit_option_menu import option_menu
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import model_from_json
from joblib import dump, load

st.title('CROP MODELLING YIELD PREDICTION')
with st.sidebar:
    selected = option_menu("Model Selection", ['CNN', 'LSTM', 'BiLSTM'])

if selected == 'CNN':
    st.write("CNN_Model is currently selected")
    json_file = open('cnn_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("cnn_model.h5")
    
elif selected == 'LSTM':
    st.write("LSTM_Model is currently selected")
    json_file = open('lstm_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("lstm_model.h5")
    
elif selected == 'BiLSTM':
    st.write("BiLSTM model")
    json_file = open('bidirectional_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("bidirectional_model.h5")
    
    
result1 = st.number_input("bdod_mean_0-5cm", format="%f")
result2 = st.number_input("bdod_mean_5-15cm", format="%f")
result3 = st.number_input("cec_mean_0-5cm", format="%f")
result4 = st.number_input("cfvo_mean_0-5cm", format="%f")
result5 = st.number_input("clay_mean_0-5cm", format="%f")
result6 = st.number_input("nitrogen_mean_0-5cm", format="%f")
result7 = st.number_input("nitrogen_mean_5-15cm", format="%f")
result8 = st.number_input("ocd_mean_0-5cm", format="%f")
result9 = st.number_input("phh2o_mean_0-5cm", format="%f")
result10 = st.number_input("Plantation_1", format="%f")
result11 = st.number_input("Winter_2", format="%f")
result12 = st.number_input("Winter_4", format="%f")
result13 = st.number_input("Spring_1", format="%f")
result14 = st.number_input("Spring_2", format="%f")
result15 = st.number_input("Spring_4", format="%f")
result16 = st.number_input("Summer_1", format="%f")
result17 = st.number_input("Summer_2", format="%f")
result18 = st.number_input("Summer_4", format="%f")
result19 = st.number_input("Summer_5", format="%f")
result20 = st.number_input("Fall_2", format="%f")
result21 = st.number_input("Fall_4", format="%f")
    
scaler = load('std_scaler.bin')
    
test_data = pd.DataFrame([[result1, result2, result3, result4, result5, result6, result7, result8, result9, result10, result11, result12, result13, result14, result15, result16, result17, result18, result19, result20, result21]], columns=['bdod_mean_0-5cm', 'bdod_mean_5-15cm', 'cec_mean_0-5cm', 'cfvo_mean_0-5cm', 'clay_mean_0-5cm', 'nitrogen_mean_0-5cm', 'nitrogen_mean_5-15cm', 'ocd_mean_0-5cm', 'phh2o_mean_0-5cm', 'Plantation_1', 'Winter_2', 'Winter_4', 'Spring_1', 'Spring_2', 'Spring_4', 'Summer_1', 'Summer_2', 'Summer_4', 'Summer_5', 'Fall_2', 'Fall_4'])
    
test_data_scaled = scaler.transform(test_data)
prediction = loaded_model.predict(test_data_scaled)
st.write(prediction)
    
    
#if __name__ == "__main__":
#    st.write([result1, result2, result3, result4, result5, result6, result7, result8, result9, result10, result11, result12, result13, result14, result15, result16, result17, result18, result19, result20, result21])
