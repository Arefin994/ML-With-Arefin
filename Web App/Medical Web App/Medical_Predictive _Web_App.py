# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 20:39:47 2024

@author: Arefin994
"""

import pickle
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu

# Loading Models
diabetes_model = pickle.load(open(r'C:\Users\Admin\Downloads\ML\Trained Model\trained_diabetes_model.sav', 'rb')) # rb = reading in binary
heart_disease_model = pickle.load(open(r'C:\Users\Admin\Downloads\ML\Trained Model\trained_heart_disease_model.sav', 'rb')) # rb = reading in binary
calories_burn_model = pickle.load(open(r'C:\Users\Admin\Downloads\ML\Trained Model\trained_calorie_burn_model.sav', 'rb')) # rb = reading in binary

# Sidebar
with st.sidebar:
    selected = option_menu("Medical Predictive System", 
                           [ 'Heart Disease Prediction', 
                            'Calories Burn Prediction',
                            'Diabetes Prediction',],
                           icons=['droplet-half', 'activity', 'fire'],
                           default_index=0)

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    
    st.title("Diabetes Prediction using ML")
    
    # Getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input("Number of Pregnancies")
    with col2:
        Glucose = st.text_input("Glucose Level")
    with col3:
        BloodPressure = st.text_input("Blood Pressure Value")
    with col1:
        SkinThickness = st.text_input("Skin Thickness Value")
    with col2:
        Insulin = st.text_input("Insulin Level")
    with col3:      
        BMI = st.text_input("BMI Value")
    with col1:
        DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function Value")
    with col2:
        Age = st.text_input("Age of the Person")
    
    # Code for prediction
    diagnosis = ''
    
    # Creating a button for prediction 
    if st.button('Diabetes Test Result'):
        try:
            # Convert inputs to floats/integers as necessary
            input_data = [float(Pregnancies), float(Glucose), float(BloodPressure), 
                          float(SkinThickness), float(Insulin), float(BMI), 
                          float(DiabetesPedigreeFunction), float(Age)]
            
            # Convert to numpy array
            input_data_as_numpy_array = np.asarray(input_data, dtype=float)

            # Reshaping the array as we are predicting for one instance
            input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

            # Make prediction
            prediction = diabetes_model.predict(input_data_reshaped)

            if prediction[0] == 0:
                diagnosis = 'The person is not diabetic'
            else:
                diagnosis = 'The person is diabetic'
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

    st.success(diagnosis)

    
    
# Heart Disease Prediction Page
elif selected == 'Heart Disease Prediction':
    
    st.title('Heart Disease Prediction using ML')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
    with col2:
        sex = st.text_input('Sex')
    with col3:
        cp = st.text_input('Chest Pain types')
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
    with col3:
        exang = st.text_input('Exercise Induced Angina')
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')
    with col1:
        thal = st.text_input('thal: 0 = normal -- 1 = fixed defect --  2 = reversable defect')
    
    # Code for prediction
    heart_diagnosis = ''
    
    # Creating a button for prediction 
    if st.button('Heart Disease Test Result'):
        # Convert inputs to floats/integers as necessary
        input_data = [float(age), float(sex), float(cp), float(trestbps), float(chol), 
                      float(fbs), float(restecg), float(thalach), float(exang), 
                      float(oldpeak), float(slope), float(ca), float(thal)]
        
        # Reshape the input data as we are predicting for one instance
        input_data_reshaped = np.array(input_data).reshape(1, -1)
        
        # Make prediction
        prediction = heart_disease_model.predict(input_data_reshaped)
        
        if prediction[0] == 0:
            heart_diagnosis = 'The person does not have heart disease'
        else:
            heart_diagnosis = 'The person has heart disease'
        
        st.success(heart_diagnosis)

# Calories Burn Prediction Page
# Calories Burn Prediction Page
elif selected == 'Calories Burn Prediction':
    
    st.title('Calories Burn Prediction Using ML')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Gender = st.text_input("Gender (0 for Female, 1 for Male)")
    with col2:
        Age = st.text_input("Age")
    with col3:
        Height = st.text_input("Height (in cm)")
    with col1:
        Weight = st.text_input("Weight (in kg)")
    with col2:
        Duration = st.text_input("Duration of Exercise (in minutes)")
    with col3:
        Heart_Rate = st.text_input("Heart Rate during Exercise")
    with col1:
        Body_Temp = st.text_input("Body Temperature (in °C)")
    
    # Code for prediction
    calories_burn_diagnosis = ''
    
    # Creating a button for prediction 
    if st.button('Calories Burn Prediction'):
        try:
            # Convert inputs to floats/integers as necessary
            input_data = [float(Gender), float(Age), float(Height), float(Weight), 
                          float(Duration), float(Heart_Rate), float(Body_Temp)]
            
            # Reshape the input data as we are predicting for one instance
            input_data_reshaped = np.array(input_data).reshape(1, -1)

            # Make prediction
            prediction = calories_burn_model.predict(input_data_reshaped)
            
            calories_burn_diagnosis = f'Estimated Calories Burned: {prediction[0]:.2f} kcal'
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

    st.success(calories_burn_diagnosis)

