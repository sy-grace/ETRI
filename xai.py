import numpy as np
import os
import pandas as pd
import pycaret
from pycaret.classification import *
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Directories for project
ROOT_DIR = os.getcwd() # Root directory of the project
MODEL_DIR = os.path.join(ROOT_DIR, "models") # Directory to save and load model


def shap_model(model_name, X_train, y):
    '''
    Load models and preprocess the input data.
    Use TreeExplainer to calculate the shap value and visualize the feature importance.

    NOTES: The model must be stored in MODEL_DIR.
           X_train and y must be specified according to the target values, emotionPositive and emotionTension.
    '''

    # Load the model_name and perform the data preprocessing steps in load_model.
    model = load_model(os.path.join(ROOT_DIR, MODEL_DIR, model_name))
    train_pipeline = model[:-1].transform(X_train)
    
    # Create SHAP explainer.
    explainer = shap.TreeExplainer(model.named_steps["trained_model"], device = 'gpu')
    shap_values = explainer.shap_values(train_pipeline)
    
    # SHAP values for all classes.
    shap.summary_plot(shap_values, train_pipeline)

    # See how every feaure contributes to the model output for each class
    for n in range(len(set(y))):
        shap.summary_plot(shap_values[n], train_pipeline)