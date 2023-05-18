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

def fertilizer (area):
    dap1 = 40/0.46
    urea1 = (0.18*dap1)+((37.5-(0.18*dap1))/0.46)
    mop1 = 12.5/0.6
    mop2 = mop1
    urea2 = (37.5/0.46)/2
    urea3 = urea2
    print("Basal Application:")
    print("1. Urea: ", round(urea1*area,2), "Kg")
    print("2. Di-ammonium Phosphate: ", round(dap1*area,2), "Kg")
    print("3. Muriate of Potash: ", round(mop1*area,2), "Kg")
    print("Split Application:")
    print("After 20-30 days of plantation:")
    print("1. Urea: ", round(urea2*area,2), "Kg")
    print("2. Muriate of Potash: ", round(mop2*area,2), "Kg")
    print("After 60 days of plantation:")
    print("1. Urea: ", round(urea3*area,2), "Kg")
    
import math

def EnotT(T):
    temp = 17.27 * T
    temp = temp / (T + 237.3)
    return 0.6108 * math.exp(temp)

def ETnot(Tmax, Tmin, RHmin, RHmax, z, J, latirad, n, uz, G):
    Tmean = (Tmax + Tmin) / 2
    Es = (EnotT(Tmax) + EnotT(Tmin)) / 2
    Ea = ((EnotT(Tmin)* (RHmax/100)) + (EnotT(Tmax)* (RHmin/100)))/2
    delta = (4098 * EnotT(Tmean))/ (Tmean + 237.3)
    delta = delta / (Tmean + 237.3)
    P = 101.3 * math.pow(293 - 0.0065*z, 5.26)
    P = P / math.pow(293, 5.26)
    gamma = 0.665 * math.pow(10, -3) * P
    dr = 1 + 0.033 * math.cos(0.017205479 * J)
    delta = 0.409 * math.sin((0.017205479 * J) - 1.39)
    X = 1 - (math.tan(latirad) * math.tan(latirad))* (math.tan(delta)* math.tan(delta))
    if X <= 0 :
        X = 0.00001
    temp = math.tan(latirad) * math.tan(delta)
    ws = 1.57 - math.tanh(-temp / math.sqrt(X))
    Ra = 37.586031361 * dr
    Ra = Ra * ws * math.sin(latirad) * math.sin(delta)
    Ra = Ra + (math.cos(latirad) * math.cos(delta) * math.sin(ws))
    Rso = (0.75 + (2 * math.pow(10, -5)*z)) * Ra
    N = 7.643312102 * ws
    Rs = (0.25 + ((0.5 * n)/N))
    Rns = 0.77 * Rs
    Tmaxk = Tmax + 273
    Tmink = Tmin + 273
    Rnl = (4.903 * math.pow(10,-9)) * ((math.pow(Tmaxk, 4) + math.pow(Tmink, 4))/2)
    Rnl = Rnl * (0.34 - 0.14 * math.sqrt(Ea))
    Rnl = Rnl * ((1.35 * (Rs/Rso)) - 0.35)
    Rn = Rns - Rnl
    u2 = uz * (4.87)
    if 67.8 * z - 5.42 > 0 : 
    	u2 = u2 / math.log(67.8 * z - 5.42) 
    else :
    	u2 = 0
    ET0 = (0.408 * delta * (Rn - G)) + (gamma * (900/(Tmean + 273)) * u2 * (Es - Ea))
    ET0 = ET0 / ( delta + (gamma * (1 + (0.34 * u2))))
    return ET0

def irrigationwater(Tmax, Tmin, RHmin, RHmax, z, J, latirad, n, uz, G, IrrigationInterval, Timeofseason, P) :
    et0 = ETnot(Tmax, Tmin, RHmin, RHmax, z, J, latirad, n, uz, G)
    if 0 <= Timeofseason <= 30 :
        kc = 0.6
    if 30 < Timeofseason < 70 :
        kc = (Timeofseason * 0.01375) + 0.1875
    elif 70 <= Timeofseason <= 110 :
        kc = 1.15
    elif 110 < Timeofseason <= 140 :
        kc = (-0.01 * Timeofseason) + 2.25
    Er = P * (125 - (0.2 * P))
    Er = Er / 125
    return ((IrrigationInterval * et0 * kc) - Er)

def pest_alert(tmin, tmax, humidity):
    if tmin > tmax or humidity >= 90 or humidity <= 40:
        return
    pest = []
    disease = []
    trange = 0
    if tmin >= 10:
        if tmax <= 25:
            trange = 1
        elif tmax <= 30:
            if tmin >= 25:
                trange = 4
            elif tmin >= 20:
                trange = 3
            elif tmin >= 15:
                trange = 2
    if trange == 1 and humidity >= 70:
        disease.append('Late Blight')
    elif trange == 2 and humidity <= 70:
        pest.append('Aphids')
    elif trange == 3:
        if humidity <= 50:
            disease.append('Powdery Mildew')
            pest.append('Tomato Hornworms')
        elif humidity <= 70:
            pest.append('Whiteflies')
            disease.append('Powdery Mildew')
            pest.append('Tomato Hornworms')
        elif humidity <= 75:
            pest.append('Whiteflies')
        elif humidity <= 80:
            disease.append('Early Blight')
            pest.append('Whiteflies')
        elif humidity <= 85:
            disease.append('Bacterial Spot')
            disease.append('Early Blight')
        else:
            disease.append('Bacterial Spot')
    elif trange == 4 and humidity >= 40 and humidity <= 60:
        pest.append('Spider Mites')
        if humidity >= 50:
            disease.append('Fusarium Wilt')
    l = []
    if pest:
        l.append('1')
        l.append(pest)
        #print("Your crop is uder risk for following pest infestations:", pest, sep='\n')
    if disease:
        l.append('2')
        l.append(disease)
        #print("Your crop is prone to get infected by following diseases:", disease, sep='\n')
    return l

if 'active_page' not in st.session_state:
    st.session_state.active_page = 'main_home'
st.session_state.update(st.session_state)


def cb_main_home() :
    st.session_state.active_page = 'main_home'
def cb_advisory_home():
    st.session_state.active_page = 'advisory_home'
def cb_yield_home():
    st.session_state.active_page = 'yield_home'
    

def advisoryhome() :
	st.title("CROP MODELLING")
	with st.sidebar:
	     selected = option_menu("Select Advisory", ['Pest and Disease Alert', 'Fertilizer Advisory', 'Irrigation Water Requirement'])
	     st.button("Back to Main Page", on_click = cb_main_home)
	if selected == 'Pest and Disease Alert' :	
		st.subheader("Pest and Disease Alert")
		st.info("Please enter the required details")
		tmax = st.number_input("Maximum temperature in Celsius", format="%f")
		tmin = st.number_input("Minimum temperature in Celsius", format="%f")
		humidity = st.number_input("Humidity in percent", format = "%f")
		check = st.button("Submit")
		if check : 
			st.success("Submitted")
			ans = pest_alert(tmin, tmax, humidity)
			if ans :
				if len(ans) == 4 :
					plant_pest = ans[1]
					st.info(f"Your crop is under risk for following pest infections : {plant_pest}")
					plant_disease = ans[3]
					st.info(f"Your crop is prone to get infected by following diseases : {plant_disease}")
				elif len(ans) == 2 :
					if ans[0] == '1' :
						plant_pest = ans[1]
						st.info(f"Your crop is under risk for following pest infections : {plant_pest}")
					elif ans[0] == '2' :
						plant_disease = ans[1]
						st.info(f"Your crop is prone to get infected by following diseases : {plant_disease}")
				else :
					st.info("No Pest infections or Diseases affected for the crop in the given circumstances")
			else :
				st.info("No Pest infections or Diseases affected for the crop in the given circumstances")
	if selected == 'Fertilizer Advisory' :
		st.subheader("Fertilizer Advisory")
		st.info("Please enter the required detail")
		area = st.number_input("Area of Plantation in Hectares")
		check = st.button("Submit")
		if check : 
			dap1 = 40/0.46
			urea1 = (0.18*dap1)+((37.5-(0.18*dap1))/0.46)
			mop1 = 12.5/0.6
			mop2 = mop1
			urea2 = (37.5/0.46)/2
			urea3 = urea2
			st.info("Basal Application :")
			temp = round(urea1*area,2)
			st.write(f"1. Urea: {temp} Kg")
			temp = round(dap1*area,2)
			st.write(f"2. Di-ammonium Phosphate: {temp} Kg")
			temp = round(mop1*area,2)
			st.write(f"3. Muriate of Potash: {temp} Kg")
			st.info("Split Application : ")
			st.info("After 20-30 days of plantation:")
			temp = round(urea2*area,2)
			st.write(f"1. Urea: {temp} Kg")
			temp = round(mop2*area,2)
			st.write(f"2. Muriate of Potash: {temp} Kg")
			st.info("After 60 days of plantation:")
			temp =  round(urea3*area,2)
			st.write(f"1. Urea:  {temp} Kg")
	if selected == 'Irrigation Water Requirement' :
		st.subheader("Irrigation Water Requirement")
		st.info("Please enter the required details")
		# Tmax, Tmin, RHmin, RHmax, z, J, latirad, n, uz, G, IrrigationInterval, Timeofseason, P
		tmax = st.number_input("Maximum temperature in Celsius", format="%f")
		tmin = st.number_input("Minimum temperature in Celsius", format="%f")
		RHmin = st.number_input("Minimum Relative Humidity in percent", format = "%f", min_value = 0.00, max_value = 100.00)
		RHmax = st.number_input("Maximum Relative Humidity in percent", format = "%f", min_value = 0.00, max_value = 100.00)
		z = st.number_input("Elevation above sea level in meters", format = "%f")
		J = st.number_input("Number of the day in the year between 1 (1 January) and 365 or 366 (31 December)", min_value = 1, max_value = 366, step = 1)
		latirad = st.number_input("Latitude of Plantation area in radian" , format = "%f")
		n = st.number_input("Sunshine duration in hr", format = "%f")
		uz = st.number_input("Measured Wind Speed at the plantation area height above ground surface in m/s", format = "%f")
		G = st.number_input("Ground heat flux in Plantation Area in (W m−2)")
		IrrigationInterval = st.number_input("Irrigation Interval in days", step = 1, min_value = 2, max_value = 3)
		Timeofseason = st.number_input("Time of Season in days", step = 1, min_value = 0)
		P = st.number_input("Rainfall in mm", format = "%f")
		check = st.button("Submit")
		if check :
			st.success("Submitted")
			answer = irrigationwater(tmax, tmin, RHmin, RHmax, z, J, latirad, n, uz, G, IrrigationInterval, Timeofseason, P)
			if answer >= 0 :
				st.info(f"Irrigation Water Requirement is {answer}")
			else :
				st.info("Please check the given inputs")

    		    
	
def yieldprediction() : 
	st.title("CROP MODELLING")
	st.header("Crop Yield Prediction")
	with st.sidebar:
	    selected = option_menu("Model Selection", ['CNN', 'LSTM', 'BiLSTM'])
	    st.button("Back to Main Page", on_click = cb_main_home)

	st.info("Please enter the required details")
	if selected == 'CNN':
	    json_file = open('cnn_model.json', 'r')
	    loaded_model_json = json_file.read()
	    json_file.close()
	    loaded_model = model_from_json(loaded_model_json)
	    loaded_model.load_weights("cnn_model.h5")
	    
	elif selected == 'LSTM':
	    json_file = open('lstm_model.json', 'r')
	    loaded_model_json = json_file.read()
	    json_file.close()
	    loaded_model = model_from_json(loaded_model_json)
	    loaded_model.load_weights("lstm_model.h5")
	    
	elif selected == 'BiLSTM':
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
	result10 = st.number_input("Planting week date in Plantation", format="%f")
	result11 = st.number_input("Solar radiation in Winter", format="%f")
	result12 = st.number_input("Maximum temperature in Winter", format="%f")
	result13 = st.number_input("Precipitation in Spring", format="%f")
	result14 = st.number_input("Solar radiation in Spring", format="%f")
	result15 = st.number_input("Maximum temperature in Spring", format="%f")
	result16 = st.number_input("Precipitation in Summer", format="%f")
	result17 = st.number_input("Solar radiation in Summer", format="%f")
	result18 = st.number_input("Maximum temperature in Summer", format="%f")
	result19 = st.number_input("Minimum temperature in Summer", format="%f")
	result20 = st.number_input("Solar radiation in Fall", format="%f")
	result21 = st.number_input("Maximum temperature in Fall", format="%f")

	check = st.button("Submit")
	scaler = load('std_scaler.bin')
	    
	test_data = pd.DataFrame([[result1, result2, result3, result4, result5, result6, result7, result8, result9, result10, result11, result12, result13, result14, result15, result16, result17, result18, result19, result20, result21]], columns=['bdod_mean_0-5cm', 'bdod_mean_5-15cm', 'cec_mean_0-5cm', 'cfvo_mean_0-5cm', 'clay_mean_0-5cm', 'nitrogen_mean_0-5cm', 'nitrogen_mean_5-15cm', 'ocd_mean_0-5cm', 'phh2o_mean_0-5cm', 'Plantation_1', 'Winter_2', 'Winter_4', 'Spring_1', 'Spring_2', 'Spring_4', 'Summer_1', 'Summer_2', 'Summer_4', 'Summer_5', 'Fall_2', 'Fall_4'])
	    
	test_data_scaled = scaler.transform(test_data)
	if check : 
		st.success("Submitted")
		prediction = loaded_model.predict(test_data_scaled)
		if prediction[0][0] > 0 :
			x = prediction[0][0]
			st.info(f"Predicted Yield is  {x} bushels per arce")
		else :
			x = 0
			st.info(f"Predicted Yield is  {x} bushel per arce")
	    

def mainmenu() :
	st.title("CROP MODELLING")
	st.subheader("CROP ADVISORY AND CROP YIELD PREDICTION")
	st.write("")
	c1, c2 = st.columns(2)
	with c1:
		st.write("Click the below button for Crop Advisory ")
		st.button("Crop Advisory", on_click = cb_advisory_home)
	with c2:
		st.write("Click the below button for Crop Yield Prediction")
		st.button("Crop Yield", on_click=cb_yield_home)
        		
if st.session_state.active_page == 'yield_home' :
	yieldprediction()
elif st.session_state.active_page == 'main_home' :
	mainmenu()
elif st.session_state.active_page == 'advisory_home' :
	advisoryhome()
