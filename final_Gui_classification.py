# -*- coding: utf-8 -*-
"""
Created on Sun May  5 11:54:04 2024

@author: dell
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,roc_curve, auc,r2_score,mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
import tkinter as tk
from tkinter import ttk, messagebox

df=pd.read_csv("D:/ammar college/Level 3/semester2/Advanced ML/bank.csv")

categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome','deposit']
# columnsss=df.drop(columns=['deposit']).columns.tolist()
# # columnsss=columnsss.to_list
# print(columnsss)
label_encoder = LabelEncoder()
keys_of_encoding={}
for i in categorical_cols:
    uniq =df[i].unique()
    labels= label_encoder.fit_transform(df[i].unique())
    keys_of_encoding[i]=dict(zip(uniq,labels))

for i in categorical_cols:
    df[i]=label_encoder.fit_transform(df[i])
    
X = df.drop(columns=['deposit'])  # Features
y = df['deposit']  #Target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)

X_test=scaler.transform(X_test)
###########desicion tree
Best_Parameters={'criterion': 'gini', 
                  'max_depth': 10, 
                  'max_features': None,
                  'min_samples_leaf': 5, 
                  'min_samples_split': 30}
DTModel = DecisionTreeClassifier(random_state=42,**Best_Parameters)

DTModel.fit(X_train, y_train)
target_prediction = DTModel.predict(X_test)
accuracy = accuracy_score(y_test, target_prediction)

print("Decision tree model accuracy: ",accuracy)
#####neural network
# Define the input layer
input_layer = tf.keras.Input(shape=(X_train.shape[1],))
   
hidden1 = tf.keras.layers.Dense(100, activation='relu')(input_layer)
hidden2 = tf.keras.layers.Dropout(0.3)(hidden1)  # Adding dropout for regularization
hidden3 = tf.keras.layers.Dense(50, activation='relu')(hidden2)
output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(hidden3)
# Create the model
NN_model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
   
# Compile the model( adam:Adaptive Moment Estimation)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Adjust learning rate
NN_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
# NN_model.compile(optimizer="Adam", loss='binary_crossentropy', metrics=['accuracy'])
   
# Train the model
history = NN_model.fit(X_train, y_train, epochs=10, batch_size=25, validation_split=0.25)
   
# Evaluate the model
accuracy = NN_model.evaluate(X_test, y_test)[1]

print("Neural Network model accuracy: ",accuracy)
# predictions = model.predict(X_test)
from joblib import dump
dump(NN_model, 'NN_model.joblib')
dump(history, 'history_NN_model.joblib')

# from joblib import load
# NN_model = load('NN_model.joblib')
# history = load('history_NN_model.joblib')


cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

def predict_deposit():
    age_value = age_entry.get()
    job_value = job_combobox.get()
    marital_value = marital_combobox.get()
    education_value = education_combobox.get()
    default_value = default_combobox.get()
    balance_value = balance_entry.get()
    housing_value = housing_combobox.get()
    loan_value = loan_combobox.get()
    contact_value = contact_combobox.get()
    day_value = day_entry.get()
    month_value = month_combobox.get()
    duration_value = duration_entry.get()
    campaign_value = campaign_entry.get()
    pdays_value = pdays_entry.get()
    previous_value = previous_entry.get()
    poutcome_value = poutcome_combobox.get()

    # Check if any entry field is empty
    if (not age_value or not job_value or not marital_value or not education_value or not default_value or 
        not balance_value or not housing_value or not loan_value or not contact_value or not day_value or
        not month_value or not duration_value or not campaign_value or not pdays_value or not previous_value or
        not poutcome_value):
        messagebox.showwarning("Error", "Please fill in all fields.")
        return

    # Make prediction
    input_data = {
        'age': int(age_value),'job': job_value,
        'marital': marital_value,'education': education_value,
        'default': default_value,'balance': float(balance_value),
        'housing': housing_value,'loan': loan_value,
        'contact': contact_value,'day': int(day_value),
        'month': month_value,'duration': float(duration_value),
        'campaign': int(campaign_value),'pdays': int(pdays_value),
        'previous': int(previous_value),'poutcome': poutcome_value
    }

    
    for col in cols:
        input_data[col]=keys_of_encoding[col].get(input_data[col])
   
    # Scale numerical features
    input_data=pd.DataFrame([input_data],columns=X.columns)
    
    input_num_scaled = scaler.transform(input_data)
    
    
    model=model_combobox.get()
    if model=="Desicion Tree":
        # Make prediction
        prediction = DTModel.predict(input_num_scaled)
        result=""
        if prediction[0] == 1:
            result="The customer is likely to deposit."
            messagebox.showinfo("Prediction Result", result)
        else:
            result="The customer is unlikely to deposit."
            messagebox.showerror("Prediction Result", result)
    elif model=="Neural Netwrok":
        prediction = NN_model.predict(input_num_scaled)
        print(prediction)
        result=""
        if prediction[0] > 0.5:
            result="The customer is likely to deposit."
            print("The customer is likely to deposit.")
            messagebox.showinfo("Prediction Result", result)
            # result_label.config(text=result)
        else:
            result="The customer is unlikely to deposit."
            print("The customer is unlikely to deposit.")
            messagebox.showerror("Prediction Result", result)
        


# Create Tkinter window
window = tk.Tk()
window.title("Term Deposit Prediction")


# Calculate the screen width and height
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

# Set the window width and height
window_width = 350
window_height = 560

# Calculate the position for the window to be centered on the screen
x_position = (screen_width - window_width) // 2
y_position = (screen_height - window_height) // 2

# Set the window's size and position
window.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")


# Create combo boxes for categorical features
categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

f_name="job"
tk.Label(window, text=f"{f_name.capitalize()}:").grid(row=0, column=0, sticky="e",padx=(90,5),pady=5)
job_combobox = ttk.Combobox(window, values=[*keys_of_encoding[f_name].keys()])
job_combobox.grid(row=0, column=1)
job_combobox.current(0) 


f_name="marital"
tk.Label(window, text=f"{f_name.capitalize()}:").grid(row=1, column=0, sticky="e",padx=5,pady=5)
marital_combobox = ttk.Combobox(window, values=[*keys_of_encoding[f_name].keys()])
marital_combobox.grid(row=1, column=1)
marital_combobox.current(0) 



f_name="education"
tk.Label(window, text=f"{f_name.capitalize()}:").grid(row=2, column=0, sticky="e",padx=5,pady=5)
education_combobox = ttk.Combobox(window, values=[*keys_of_encoding[f_name].keys()])
education_combobox.grid(row=2, column=1)
education_combobox.current(0) 




f_name="default"
tk.Label(window, text=f"{f_name.capitalize()}:").grid(row=3, column=0, sticky="e",padx=5,pady=5)
default_combobox = ttk.Combobox(window, values=[*keys_of_encoding[f_name].keys()])
default_combobox.grid(row=3, column=1)
default_combobox.current(0) 

f_name="housing"
tk.Label(window, text=f"{f_name.capitalize()}:").grid(row=4, column=0, sticky="e",padx=5,pady=5)
housing_combobox = ttk.Combobox(window, values=[*keys_of_encoding[f_name].keys()])
housing_combobox.grid(row=4, column=1)
housing_combobox.current(0) 


f_name="loan"
tk.Label(window, text=f"{f_name.capitalize()}:").grid(row=5, column=0, sticky="e",padx=5,pady=5)
loan_combobox = ttk.Combobox(window, values=[*keys_of_encoding[f_name].keys()])
loan_combobox.grid(row=5, column=1)
loan_combobox.current(0) 
f_name="contact"
tk.Label(window, text=f"{f_name.capitalize()}:").grid(row=6, column=0, sticky="e",padx=5,pady=5)
contact_combobox = ttk.Combobox(window, values=[*keys_of_encoding[f_name].keys()])
contact_combobox.grid(row=6, column=1)
contact_combobox.current(0) 
f_name="month"
tk.Label(window, text=f"{f_name.capitalize()}:").grid(row=7, column=0, sticky="e",padx=5,pady=5)
month_combobox = ttk.Combobox(window, values=[*keys_of_encoding[f_name].keys()])
month_combobox.grid(row=7, column=1)
month_combobox.current(0) 
f_name="poutcome"
tk.Label(window, text=f"{f_name.capitalize()}:").grid(row=8, column=0, sticky="e",padx=5,pady=5)
poutcome_combobox = ttk.Combobox(window, values=[*keys_of_encoding[f_name].keys()])
poutcome_combobox.grid(row=8, column=1)
poutcome_combobox.current(0) 


#labels and entry fields for numerical features
tk.Label(window, text="Age:").grid(row=9, column=0, sticky="e",padx=5,pady=5)
age_entry = tk.Entry(window)
age_entry.grid(row=9, column=1)

tk.Label(window, text="Balance:").grid(row=10, column=0, sticky="e",padx=5,pady=5)
balance_entry = tk.Entry(window)
balance_entry.grid(row=10, column=1)

tk.Label(window, text="Day:").grid(row=11, column=0, sticky="e",padx=5,pady=5)
day_entry = tk.Entry(window)
day_entry.grid(row=11, column=1)

tk.Label(window, text="Duration:").grid(row=12, column=0, sticky="e",padx=5,pady=5)
duration_entry = tk.Entry(window)
duration_entry.grid(row=12, column=1)

tk.Label(window, text="Campaign:").grid(row=13, column=0, sticky="e",padx=5,pady=5)
campaign_entry = tk.Entry(window)
campaign_entry.grid(row=13, column=1)

tk.Label(window, text="Pdays:").grid(row=14, column=0, sticky="e",padx=5,pady=5)
pdays_entry = tk.Entry(window)
pdays_entry.grid(row=14, column=1)

tk.Label(window, text="Previous:").grid(row=15, column=0, sticky="e",padx=5,pady=5)
previous_entry = tk.Entry(window)
previous_entry.grid(row=15, column=1)

tk.Label(window, text="Model:").grid(row=16, column=0, sticky="e",padx=5,pady=5)
model_combobox = ttk.Combobox(window, values=["Desicion Tree","Neural Netwrok"])
model_combobox.grid(row=16, column=1)
model_combobox.current(0) 

# Create a separator in the middle
separator = ttk.Separator(window, orient="horizontal")
separator.grid(row=17, column=1, sticky="ew")
# Create a button to trigger prediction
predict_button = tk.Button(window, text="Predict", command=predict_deposit)
predict_button.grid(row=18, column=1)


window.mainloop()

