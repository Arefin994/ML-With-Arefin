# -*- coding: utf-8 -*-
"""
Created on Tue 13 Aug 5:53:30 2024

@author: Arefin994
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import logging

app = FastAPI()

class ModelInput(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# Load your trained model
try:
    diabetes_model = pickle.load(open(r'trained_diabetes_model.sav', 'rb'))
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise RuntimeError("Failed to load model.")

@app.post('/diabetes_prediction')
def diabetes_pred(input_parameters: ModelInput):
    try:
        input_data = input_parameters.dict()

        input_list = [
            input_data['Pregnancies'],
            input_data['Glucose'],
            input_data['BloodPressure'],
            input_data['SkinThickness'],
            input_data['Insulin'],
            input_data['BMI'],
            input_data['DiabetesPedigreeFunction'],
            input_data['Age']
        ]

        prediction = diabetes_model.predict([input_list])

        if prediction[0] == 0:
            return {'result': 'The person is not diabetic'}
        else:
            return {'result': 'The person is diabetic'}
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed due to an internal error.")
