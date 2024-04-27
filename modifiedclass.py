
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 02:38:16 2024

@author: dell
"""

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,roc_curve,auc
from Datasets import Datahandl
from io import StringIO
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk

class MLApp:
    dsn=""
    df=pd.DataFrame()
    task=""
    dataFrame=Datahandl()
    model_accuracy=0
    preprocessed_data=dataFrame
    prep_flag=False
    nn_target=None
    nn_pred_target=None
    nn_info=None
    Dt_target=None
    Dt_pred_target=None
    svr_target=None
    svr_pred_target=None
    run_model= False
    def __init__(self, root):
        self.root = root
        self.root.title("Machine Learning App")
        # Calculate the center position
        
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        window_width = 700
        window_height = 400
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        # Define colors
        self.bg_color = "#803D3B"
        self.button_bg_color = "#E4C59E"
        self.button_fg_color = "black"
        self.label_fg_color = "#333"
        
        # Left Frame for buttons and labels
        left_frame_width = 300
        self.left_frame = tk.Frame(root, width=left_frame_width, bg=self.bg_color)
        self.left_frame.pack(side="left", fill="y")
        # Right Frame for output text
        self.right_frame = tk.Frame(root)
        self.right_frame.pack(side="right", fill="both", expand=True)

        
        self.model_names = {
            "Decision Tree": pd.read_csv("D:/ammar college/Level 3/semester2/Advanced ML/bank.csv"),
            "Neural Network": pd.read_csv("D:/ammar college/Level 3/semester2/Advanced ML/bank.csv"),
            "SVR": pd.read_csv("D:/ammar college/Level 3/semester2/Advanced ML/housing.csv")
            # Add more datasets here if needed
        }

        self.root.configure(bg=self.bg_color)

        self.model_label = tk.Label(self.left_frame, text="Choose Model", fg=self.label_fg_color, bg=self.bg_color,font=("Times New Roman", 14,"bold"))
        self.model_label.grid(row=0, column=0, sticky="w", padx=(0,0), pady=(0,5))        
        
        self.model_var = tk.StringVar(root)
        self.model_var.set("")
        self.model_dropdown = ttk.OptionMenu(self.left_frame, self.model_var,"", "Decision Tree", "Neural Network","SVR")
        self.model_dropdown.grid(row=0, column=1, sticky="w", padx=(5,0), pady=(0,5))

        self.load_data_button = ttk.Button(self.left_frame, text="Load Dataset", command=self.load_dataset, style="Rounded.TButton")
        self.load_data_button.grid(row=1, column=0, sticky="w", padx=(5,10), pady=10)
        
        self.clear_button = ttk.Button(self.left_frame, text="Clear All", command=self.clear_All, style="Rounded.TButton")
        self.clear_button.grid(row=1, column=1, sticky="w", padx=(5,35), pady=10)
        
        self.preprocess_button = ttk.Button(self.left_frame, text="Preprocess Data", command=self.preprocess_data, style="Rounded.TButton")
        self.preprocess_button.grid(row=2, column=0, sticky="ew", padx=(5,0), pady=(5))
        
        self.plot_label = tk.Label(self.left_frame, text="Choose Plot", fg=self.label_fg_color, bg=self.bg_color,font=("Times New Roman", 14,"bold"))
        self.plot_label.grid(row=3, column=0, sticky="w", padx=(0,0),  pady=(0,10))

        self.plot_var = tk.StringVar(root)
        self.plot_var.set("")
        self.plot_dropdown = ttk.OptionMenu(self.left_frame, self.plot_var, "")
        self.plot_dropdown.grid(row=3, column=1, sticky="w", padx=(5,5), pady=(0,10))
        
        self.plot_button = ttk.Button(self.left_frame, text="Plot", command=self.plot, style="Rounded.TButton")
        self.plot_button.grid(row=4, column=0, sticky="ew",padx=5, pady=5)
        
        self.get_accuracy_button = ttk.Button(self.left_frame, text="Calculate Accuracy", command=self.get_accuracy, style="Rounded.TButton")
        self.get_accuracy_button.grid(row=10, column=0, sticky="ew", padx=5, pady=0)
        
        self.accuracy_label = tk.Label(self.left_frame, text="", fg=self.label_fg_color,font=("Times New Roman", 14,"bold"))
        self.accuracy_label.grid(row=10, column=1, sticky="ew", padx=5, pady=5)
        
        
        
        ####Output text area
        self.output_text = tk.Text(self.right_frame, height=50, width=50)
        self.output_text.grid(row=0, column=0, sticky="nsew")
        self.output_text.config(bg="white")

        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(0, weight=1)
        scrollbar = tk.Scrollbar(self.right_frame, command=self.output_text.yview)
        scrollbar.grid(row=0, column=2, padx=(0, 5), pady=10, sticky="nsew")
        self.output_text.config(yscrollcommand=scrollbar.set)
        
        # Configure style for rounded buttons
        self.style = ttk.Style()
        self.style.configure("Rounded.TButton", foreground=self.button_fg_color, background=self.button_bg_color, borderwidth=10, relief="flat", font=("Arial", 10))
        self.style.map("Rounded.TButton", background=[("active", "#45a049")])
        # Configure row and column weights for the right frame
        self.right_frame.grid_rowconfigure(0, weight=1)
        self.right_frame.grid_columnconfigure(0, weight=1)

        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")

        
    def clear_All(self):
        self.output_text.delete('1.0', tk.END)
        self.accuracy_label.config(text="")
        
    def load_dataset(self):
        self.output_text.delete('1.0', tk.END)
        self.prep_flag=False
        self.run_model=False
        model_name = self.model_var.get()  # Get the selected dataset name
        self.task=self.model_var.get() 
        if model_name in self.model_names:
            self.dataFrame.set_data(self.model_names[model_name])
            five=self.dataFrame.show_fiv()
            info_buffer = StringIO()
            info_str=self.dataFrame.info(info_buffer)
            options=None
            self.plot_var.set("")
            
            if(model_name=="Decision Tree") :
                self.dsn="Banking Market DataSet"
                options=["","HeatMap","Pair Wise","Day-Scatter Plot","Histogram","Confusion matrix"]
            elif (model_name =="Neural Network"):
                self.dsn="Banking Market DataSet"
                options=["","HeatMap","Pair Wise","Day-Scatter Plot","Histogram","Confusion matrix","ROC Curve","Tra-val accuracy","Train-val loss"]
            elif (model_name=="SVR"):
                self.dsn="California Housing prices DataSet"
                options=["","Loc-Scatter Plot","Histogram","Residuals","final"]
                
            self.plot_dropdown = ttk.OptionMenu(self.left_frame, self.plot_var,*options)
            self.plot_dropdown.grid(row=3, column=1, sticky="w", padx=(5,5), pady=(0,10))
            self.output_text.insert(tk.END, f"\t\t\tDataSet NAME: {self.dsn}\n{five.to_string(index=False)}\n-------------------------------------------------------------------------\n\
>>Data Info:\n{info_str}\n>>the Descibtion of tha data:\n{self.dataFrame.describe().to_string()}\n------------------------------------------------------------------------------------\n\
>>check if there is any missing value in the Dataset:\n{self.dataFrame.get_data().isna().any()}\n-------------------------------------------------------------------------------------\n\
 " )
        else:
            messagebox.showerror("Error", "Dataset not found.")
        self.output_text.see(tk.END)
        self.accuracy_label.config(text="")

    def get_accuracy(self):
        if (not(self.dataFrame.get_data().empty)):
            
            if (self.task == "Decision Tree"):
                self.Dt_target,self.Dt_pred_target,bestparams,acc=self.dataFrame.DecisionTreeModel()
                
                self.accuracy_label.config(text=f"{acc*100:.2f}%", fg="black", font=("Arial", 10,"italic"))
                self.output_text.insert(tk.END, f"Parameters used in the model: {bestparams}\nDecision Tree Model Accuracy: {acc*100:.2f}%\n")
                
            elif self.task == "Neural Network":
                if self.prep_flag:
                    self.nn_target,self.nn_pred_target,self.nn_info,acc = self.dataFrame.NeuralnetworkModel()
                    self.accuracy_label.config(text=f"{acc*100:.2f}%", fg="black", font=("Arial", 10,"italic"))
                    self.output_text.insert(tk.END, f"Neural Network Model Accuracy: {acc*100:.2f}%\n")
                else:
                    messagebox.showerror("Error", "Please Preprocess Data first.")
            elif self.task == "SVR":
                self.svr_target,self.svr_pred_target,acc=self.dataFrame.SVRModel()
                self.accuracy_label.config(text=f"{acc*100:.2f}%", fg="black", font=("Arial", 10,"italic"))
                self.output_text.insert(tk.END, f"Support vector regression Model Accuracy: {acc*100:.2f}%\n")
            else:
                messagebox.showerror("Error", "Selected model not implemented.")
        else:
            messagebox.showerror("Error", "Please load a dataset first.")
        self.run_model=True
        self.output_text.see(tk.END)
        
    def plot(self):
        if  self.plot_var.get() != "":
            # Create a new Toplevel window
            plot_window = tk.Toplevel(self.root)
            plot_window.title("Plot Window")
            plt.figure(figsize=(8, 8))
            if self.plot_var.get() == "HeatMap":
                # Generate the plot
                text="HeatMap of " +self.dsn
                plt.title(text)
                sns.heatmap(self.dataFrame.get_data().corr(), annot=True, cmap="viridis")
                
            elif self.plot_var.get() == "Confusion matrix":
                if self.run_model==True  and self.task == "Decision Tree":
                    cm = confusion_matrix(self.Dt_target, self.Dt_pred_target)
                    # Plot confusion matrix
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
                    text='Confusion Matrix of '+self.task
                    plt.title(text)
                    plt.ylabel('Predicted Label')
                    plt.xlabel('True Label')
        
                elif self.run_model==True  and self.task == "Neural Network":
                    pred_target_classes = (self.nn_pred_target > 0.5).astype(int)
                    # Generate confusion matrix
                    cm = confusion_matrix(self.nn_target, pred_target_classes)
                    # Plot confusion matrix
                    sns.heatmap(cm, annot=True, fmt='d', cmap='BuGn')
                    text='Confusion Matrix of '+self.task
                    plt.title(text)
                    plt.ylabel('Predicted Label')
                    plt.xlabel('True Label')
                else:
                    messagebox.showerror("Error", "Please Run the model first.")
                
            elif self.plot_var.get() == "ROC Curve":
                # if self.nn_target ==None and self.model_var.get() == "Neural Network":
                if self.run_model==True  and self.model_var.get() == "Neural Network":
                
                    fpr, tpr, _ = roc_curve(self.nn_target, self.nn_pred_target)
                    roc_auc = auc(fpr, tpr)
        
                    # Plot ROC curve
                    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
                    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('Receiver Operating Characteristic (ROC) Curve')
                    plt.legend(loc='lower right')
        
                else:
                    messagebox.showerror("Error", "Please Run the model first.")
            elif self.plot_var.get() == "Pair Wise":

                if((self.task== "Decision Tree")|(self.task== "Neural Network")):
                    # Scatter Plot
                    sns.pairplot(self.dataFrame.get_data(), vars=['age', 'balance'], hue='deposit', diag_kind='kde')
                    
                elif (self.task== "SVR"):
                    sns.pairplot(self.dataFrame.get_data(), vars=['total_rooms', 'total_bedrooms',"population","households","median_income"], hue='deposit', diag_kind='kde')
                 
            elif self.plot_var.get() == "Histogram":
                plt.suptitle('Hisotgram Plot for All featuers')
                data = self.dataFrame.get_data()
                data.hist(bins=20, figsize=(10, 6), alpha=0.5, xrot=45)  # Adjust the number of bins and alpha value as needed
                plt.tight_layout()
                
            elif self.plot_var.get() == "Loc-Scatter Plot":
                sns.scatterplot(x='longitude', y='latitude', hue='median_house_value',style='ocean_proximity', data=self.dataFrame.get_data(), palette='coolwarm', alpha=0.5)
                plt.title('Spread of Houses in California')
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                plt.grid(True)
                plt.legend(title='Median House Value')
                
            elif self.plot_var.get() == "Day-Scatter Plot":
                sns.scatterplot(x='month', y='day', hue='deposit', data=self.dataFrame.get_data(), style='deposit', palette=['blue', 'green'], alpha=0.5)
                plt.title('Time of depoites')
                plt.ylabel('days')
                plt.xlabel('months')
                plt.legend(title='Deposit', loc='upper right', bbox_to_anchor=(1.1, 1))
            elif self.plot_var.get() == "Train-val loss":
                # Plot training and validation loss
                plt.plot(self.nn_info.history['loss'], label='Training Loss')
                plt.plot(self.nn_info.history['val_loss'], label='Validation Loss')
                plt.title('Training and Validation Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
            elif self.plot_var.get() == "Tra-val accuracy":
                # Plot training and validation accuracy
                plt.plot(self.nn_info.history['accuracy'], label='Training Accuracy')
                plt.plot(self.nn_info.history['val_accuracy'], label='Validation Accuracy')
                plt.title('Training and Validation Accuracy')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.legend()
            elif self.plot_var.get() == "Residuals":
                residuals = self.svr_target- self.svr_pred_target
                # Plot residuals
                plt.figure(figsize=(10, 6))
                plt.scatter(self.svr_target, residuals, color='red', alpha=0.5)
                plt.xlabel('Actual Values')
                plt.ylabel('Residuals')
                plt.title('SVR: Residual Plot')
                
            elif self.plot_var.get() == "final":
                plt.figure(figsize=(8, 6))
                plt.scatter(self.svr_target, self.svr_pred_target, alpha=0.5)
                plt.plot([min(self.svr_target), max(self.svr_target)], [min(self.svr_target), max(self.svr_target)], '--', color='red')  # Add diagonal line for reference
                plt.title('Predictions vs. Actual Values')
                plt.xlabel('Actual Values')
                plt.ylabel('Predicted Values')  
                plt.grid(True)
                # plt.show()
            # Embed the plot in the plot window
            self.draw(plot_window)
            plt.close()
            self.output_text.see(tk.END)
            
        else:
            messagebox.showerror("Error", "Please choose a plot first.")

    def preprocess_data(self):

        if(not(self.dataFrame.get_data().empty)):
            self.prep_flag=True
            if(self.task in ["Decision Tree","Neural Network","SVR"]):
                self.dataFrame.preprocessing(self.task)
                self.output_text.insert(tk.END,f"The {self.task} Data preprocessing has been made\n>>check if there is any missing value in the Dataset:\n{self.dataFrame.get_data().isna().any()}\n {self.dataFrame.describe()}\n")
                self.output_text.see(tk.END)
                return self.dataFrame.get_data()
    
    def draw(self,plot_window):
        canvas = FigureCanvasTkAgg(plt.gcf(), master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        # Add a navigation toolbar
        toolbar = NavigationToolbar2Tk(canvas, plot_window)
        toolbar.update()
root = tk.Tk()
app = MLApp(root)

root.mainloop()
