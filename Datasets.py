# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 02:14:14 2024

@author: dell
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,roc_curve, auc,r2_score,mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from sklearn.svm import SVR
class Datahandl:
    
    df=pd.read_csv("D:/ammar college/Level 3/semester2/Advanced ML/bank.csv")
    
    
    def __init__(self):
        self.df=self.df
        
    def set_data(self,dataframe):
        self.df=dataframe
        
    def get_data(self):
        return self.df
    
    def shape(self):
        return self.df.shape
    
    def show_fiv(self):
        return self.df.head();
    
    # Step 2: Exploratory Data Analysis (EDA)
    
    def info(self,info_buffer):
        self.df.info(buf=info_buffer);
        return info_buffer.getvalue();
    
    def describe(self):
        return self.df.describe();
    
    def preprocessing(self,model):
        if (model=="Decision Tree" )|(model =="Neural Network"):
            ### Numeric columns Scacling
            print("the length of the columns : ",len(self.df.columns))
            numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            scaler=StandardScaler()
            X_num_scaled=pd.DataFrame(scaler.fit_transform(self.df[numeric_cols]),columns=numeric_cols)
            
            self.df=self.df.drop(columns=self.df[numeric_cols])
            
            ### Categorical columns
            categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
            self.df=pd.get_dummies(self.df,columns=categorical_cols)
            
            self.df['deposit']=self.df['deposit'].apply(lambda x: 1 if x=="yes" else 0 )
            self.df=pd.concat([X_num_scaled,self.df],axis=1)
            print("the length of the columns : ",len(self.df.columns))
        elif model == "SVR":
            #fill missing values
            print("the length of the columns : ",len(self.df.columns))
            median_value = self.df['total_bedrooms'].median()
            self.df['total_bedrooms'].fillna(median_value, inplace=True)

            # convert type from float to int
            columns_to_convert = ['total_rooms', 'total_bedrooms', 'population' , 'households']
            self.df[columns_to_convert] = self.df[columns_to_convert].astype(int)
            
            # Dealing with Right skewed distributed Data
            log_columns=["total_rooms","total_bedrooms","population","median_income","households"]
            for column in log_columns:
                self.df[column]=np.log(self.df[column])+1
            # encode columns 
            label_encoder = LabelEncoder()
            self.df['ocean_proximity'] = label_encoder.fit_transform(self.df['ocean_proximity'])
            
            # scale the data
            scaler = StandardScaler()
            self.df = pd.DataFrame(scaler.fit_transform(self.df), columns=self.df.columns)
            # self.df = self.df.join(scaled_data)
            
            print("the length of the columns : ",len(self.df.columns))
        return  self.df;
            
    def split_classification_data(self):
        
        X = self.df.drop(columns=['deposit'])  # Features
        y = self.df['deposit']  #
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        print("Classification split done and the training data size (rows, columns) : ",X_train.shape,"\
and the test data size (rows, columns) : ",X_test.shape)
        return [X_train, X_test, y_train, y_test];
    
    def split_regression_data(self):
        # Define features (X) and target (y)
        X = self.df.drop(columns='median_house_value')
        y = self.df['median_house_value']
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        print("Regression split done and the training data size (rows, columns) : ",X_train.shape,"\
and the test data size (rows, columns) : ",X_test.shape)

        return [X_train, X_test, y_train, y_test];
    
    def DecisionTreeModel(self):
        features_train,features_test,target_train,target_test=self.split_classification_data()
        # Define the hyperparameters grid
        # param_grid = {
        #     'criterion': ['gini', 'entropy'],
        #     'max_depth': [ 10, 15, 20]
        #     ,'min_samples_split': [ 20, 25, 30],
        #     'max_features': [None,6,10,14]
        # }

        # # Initialize Decision Tree Classifier
        # DTModel = DecisionTreeClassifier(random_state=42)

        # # Perform Grid Search Cross Validation
        # grid_search = GridSearchCV(DTModel, param_grid, cv=5, scoring='accuracy')
        # grid_search.fit(features_train, target_train)

        # # Get the best parameters
        # best_params = grid_search.best_params_
        # print("Best Parameters:", best_params)
        # Evaluate the best model
        # best_model = grid_search.best_estimator_
        Best_Parameters={'criterion': 'gini', 
                         'max_depth': 10, 
                         'max_features': None, 
                         'min_samples_split': 30}
        DTModel = DecisionTreeClassifier(random_state=42,**Best_Parameters)
        
        DTModel.fit(features_train, target_train)
        target_prediction = DTModel.predict(features_test)
        accuracy = accuracy_score(target_test, target_prediction)
        return target_test,target_prediction,Best_Parameters,accuracy
    
    def NeuralnetworkModel(self):
        features_train,features_test,target_train,target_test=self.split_classification_data()
        
        # Define the input layer
        input_layer = tf.keras.Input(shape=(features_train.shape[1],))

       
        hidden1 = tf.keras.layers.Dense(100, activation='relu')(input_layer)
        hidden2 = tf.keras.layers.Dropout(0.3)(hidden1)  # Adding dropout for regularization
        hidden3 = tf.keras.layers.Dense(50, activation='relu')(hidden2)
        output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(hidden3)
        
        # Create the model
        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

        # Compile the model( adam:Adaptive Moment Estimation)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Adjust learning rate
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        # model.compile(optimizer="Adam", loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        history = model.fit(features_train, target_train, epochs=10, batch_size=25, validation_split=0.25)

        # Evaluate the model
        accuracy = model.evaluate(features_test, target_test)[1]
        
        predictions = model.predict(features_test)

        return target_test,predictions,history,accuracy
    
    def SVRModel(self):
        features_train,features_test,target_train,target_test=self.split_regression_data()
        
        
        # param_grid = {
        #     'kernel': ['rbf'],
        #     'C':[1.0,10.0,20.0],
        #     'gamma': ['scale']
        # }
        
        # # Initialize SVR
        # svr = SVR()
        
        # grid_search = GridSearchCV(svr, param_grid, cv=2, scoring='neg_mean_squared_error')
        # grid_search.fit(features_train, target_train)
        # # Get the best hyperparameters
        # best_params = grid_search.best_params_
        # Initialize SVR with the best hyperparameters
        
        # print("best params: ",best_params)
        
        param_grid = {
            'kernel': 'rbf',
            'C':10.0,
            'gamma': 'scale'
        }
        
        best_svr = SVR(**param_grid)
    
        best_svr.fit(features_train, target_train)
       
        # Predictions on the testing set
        y_pred = best_svr.predict(features_test)
        # Perform grid search with cross-validation
        r2=r2_score(target_test, y_pred)
        return target_test,y_pred,r2;
   
    
    
    
    
    