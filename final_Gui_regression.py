# -*- coding: utf-8 -*-
"""
Created on Sun May  5 10:01:32 2024

@author: dell
"""

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split,GridSearchCV
from joblib import load
df = pd.read_csv("D:/ammar college/Level 3/semester2/Advanced ML/housing.csv")
# Load the saved model
svr_model = load('svr_model.joblib')
X = df.drop(columns='median_house_value')
y = df['median_house_value']


label_encoder = LabelEncoder()
uniq =X["ocean_proximity"].unique()
labels= label_encoder.fit_transform(uniq)
keys_of_encoding=dict(zip(uniq,labels))


median_value = X['total_bedrooms'].median()
X['total_bedrooms'].fillna(median_value, inplace=True)



X['ocean_proximity'] = label_encoder.fit_transform(X['ocean_proximity'])

scaler=StandardScaler()
X=scaler.fit_transform(X)


# Function to predict median house value
def predict_price(features):
    global svr_model
    # Preprocess input features
    features = pd.DataFrame(features, index=[0])
    print(features)
    features = scaler.transform(features)
    print(features)
    
    # Predict using the SVR model
    prediction = svr_model.predict(features)
    print(prediction[0])
    
    return prediction[0]

# Function to handle button click event
def predict():
    # Get user inputs
    longitude_value = entry_longitude.get()
    latitude_value = entry_latitude.get()
    House_age_value = entry_age.get()
    rooms_value = entry_rooms.get()
    bedrooms_value = entry_bedrooms.get()
    population_value = entry_population.get()
    households_value = entry_households.get()
    income_value = entry_income.get()
    proximity_value = proximity_combobox.get()
    

    # Check if any entry field is empty
    if (not longitude_value or not latitude_value or not  House_age_value or
        not rooms_value or not bedrooms_value or 
        not population_value or not households_value or
        not income_value or not proximity_value):
        messagebox.showwarning("Error", "Please fill in all fields.")
        return
    
    inputs = {
        "longitude":float(longitude_value),
        "latitude":float(latitude_value),
        'housing_median_age': float( House_age_value),
        'total_rooms': float(rooms_value),
        'total_bedrooms': float(bedrooms_value),
        'population': float(population_value),
        'households': float(households_value),
        'median_income': float(income_value),
        'ocean_proximity': proximity_value
        
    }
    inputs["ocean_proximity"]=keys_of_encoding.get(inputs["ocean_proximity"])
    print("ocean choosed: ",inputs["ocean_proximity"])
    # Predict median house value
    predicted_price = predict_price(inputs)
    
    # Show prediction
    messagebox.showinfo("Prediction", f"Predicted Median House Value: ${int(np.e**predicted_price):4d}")

# Create GUI window
window = tk.Tk()
window.title("House Price Prediction")

# Calculate the screen width and height
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

# Set the window width and height
window_width = 400
window_height = 320

# Calculate the position for the window to be centered on the screen
x_position = (screen_width - window_width) // 2
y_position = (screen_height - window_height) // 2

# Set the window's size and position
window.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")


# Create input fields
tk.Label(window, text="Housing Median Age:").grid(row=0, column=0, sticky="e",padx=(60,5),pady=5)
entry_age = tk.Entry(window)
entry_age.grid(row=0, column=1)

tk.Label(window, text="Total Rooms:").grid(row=1, column=0, sticky="e",padx=5,pady=5)
entry_rooms = tk.Entry(window)
entry_rooms.grid(row=1, column=1)

tk.Label(window, text="Total Bedrooms:").grid(row=2, column=0, sticky="e",padx=5,pady=5)
entry_bedrooms = tk.Entry(window)
entry_bedrooms.grid(row=2, column=1)

tk.Label(window, text="Population:").grid(row=3, column=0, sticky="e",padx=5,pady=5)
entry_population = tk.Entry(window)
entry_population.grid(row=3, column=1)

tk.Label(window, text="Households:").grid(row=4, column=0, sticky="e",padx=5,pady=5)
entry_households = tk.Entry(window)
entry_households.grid(row=4, column=1)

tk.Label(window, text="Median Income:").grid(row=5, column=0, sticky="e",padx=5,pady=5)
entry_income = tk.Entry(window)
entry_income.grid(row=5, column=1)

tk.Label(window, text="Ocean Proximity:").grid(row=6, column=0, sticky="e",padx=5,pady=5)
proximity_combobox = ttk.Combobox(window, values=[*keys_of_encoding.keys()])
proximity_combobox.grid(row=6, column=1)
proximity_combobox.current(0) 


tk.Label(window, text="Longitude:").grid(row=7, column=0, sticky="e",padx=5,pady=5)
entry_longitude = tk.Entry(window)
entry_longitude.grid(row=7, column=1)

tk.Label(window, text="Latitude:").grid(row=8, column=0, sticky="e",padx=5,pady=5)
entry_latitude = tk.Entry(window)
entry_latitude.grid(row=8, column=1)

# Create predict button
predict_button = tk.Button(window, text="Predict", command=predict)
predict_button.grid(row=9, columnspan=2,padx=(60,5),pady=5)
# Run GUI loop
window.mainloop()
