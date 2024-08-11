# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 06:52:10 2024

@author: Admin
"""

import numpy as np
import pickle
import streamlit as st

# Loading the trained model
loaded_model = pickle.load(open(r'C:\Users\Admin\Downloads\ML\Trained Model\trained_diabetes_model.sav', 'rb'))

# Function for diabetes prediction
def diabetes_prediction(input_data):

    # Converting input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)

    # Reshaping the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    # Make prediction
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'
  
def main():
    
    # Giving a title
    st.title("Diabetes Prediction Web App")
    
    # Getting the input data from the user
    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("Blood Pressure Value")
    SkinThickness = st.text_input("Skin Thickness Value")
    Insulin = st.text_input("Insulin Level")
    BMI = st.text_input("BMI Value")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function Value")
    Age = st.text_input("Age of the Person")
    
    # Code for prediction
    diagnosis = ''
    
    # Creating a button for prediction 
    if st.button('Diabetes Test Result'):
        # Convert inputs to floats/integers as necessary
        input_data = [float(Pregnancies), float(Glucose), float(BloodPressure), 
                      float(SkinThickness), float(Insulin), float(BMI), 
                      float(DiabetesPedigreeFunction), float(Age)]
        
        # Get prediction
        diagnosis = diabetes_prediction(input_data)
    
    st.success(diagnosis)

if __name__ == '__main__':
    main()
