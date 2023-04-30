import pycaret
from pycaret.classification import *
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pycaret.datasets import get_data 


# Directories for project
ROOT_DIR = os.getcwd() # Root directory of the project
MODEL_DIR = os.path.join(ROOT_DIR, "models") # Directory to model folder


def modeling(df, target, ignore_feature, df_name):
    '''
    Model KNN, RandomForest, DecisionTree, and Extratree Classifier models using df.
    Find the model with the best performance. 
    NOTES: The model must be stored in MODEL_DIR.
           X_train and y must be specified according to the target values, emotionPositive and emotionTension.
    '''
    # Data split trainset : testset = 7 : 3.
    df_train, df_test = train_test_split(df, test_size = 0.3, shuffle =True, random_state = 2, stratify=df[target])

    df_name =  df_name

    # set_up before pycaret.
    set_up = setup(data = df_train, 
                   target = target,
                   ignore_features = [ignore_feature],
                   
                   normalize = False, 

                   transformation = False, 
            
                   fold = 5,
                   fold_shuffle=True,
          
                   session_id = 123, # Random state number
            
                   use_gpu = True
                   )
  
    models()

    top_4_models = compare_models(include=['knn', 'rf', 'dt', 'et'], n_select=4)

    # Tunes the model specified in top_4_models.
    tuned_top_4_models = [tune_model(model) for model in top_4_models]

    print(top_4_models)
    print(tuned_top_4_models)

    # Save models
    for i, model in enumerate(tuned_top_4_models):
        save_model(model, f"{MODEL_DIR}/model_{i+1}_{type(model).__name__}_{df_name}_{target}")

    for model in tuned_top_4_models:
        # Evaluate the performance of the model on the test set.
        predict_model(model, data= df_test)
        plot_model(model)
        plot_model(estimator = model, plot = 'confusion_matrix')