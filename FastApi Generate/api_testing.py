# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 06:18:52 2024

@author: Arefin994
"""

import requests

url = 'http://127.0.0.1:8000/diabetes_prediction'

input_data_for_model = {
    'Pregnancies': 6,
    'Glucose': 148,
    'BloodPressure': 72,
    'SkinThickness': 35,
    'Insulin': 0,
    'BMI': 33.6,
    'DiabetesPedigreeFunction': 0.627,
    'Age': 50
}

response = requests.post(url, json=input_data_for_model)

# Print status code and raw response content for debugging
print(f"Status Code: {response.status_code}")

try:
    response_json = response.json()
    print("JSON Response:", response_json)
except requests.exceptions.JSONDecodeError:
    print("Failed to decode JSON response.")
