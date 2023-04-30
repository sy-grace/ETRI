import os
import pandas as pd
from sklearn.preprocessing import StandardScaler


def make_dirs(ROOT_DIR):
    # Directory to dataset folder
    DATA_DIR = os.path.join(ROOT_DIR, "datasets")
    os.makedirs(DATA_DIR, exist_ok=True)

    # Directory to downloaded datasets
    RAW_DIR = os.path.join(DATA_DIR, "raw_data")
    os.makedirs(RAW_DIR, exist_ok=True)

    # Directory to sensor datasets
    SENSOR_DIR = os.path.join(DATA_DIR, "sensor_data")
    os.makedirs(SENSOR_DIR, exist_ok=True)

    # Directory to label datasets
    LABEL_DIR = os.path.join(DATA_DIR, "label_data")
    os.makedirs(LABEL_DIR, exist_ok=True)

    # Directory to sleep datasets
    SLEEP_DIR = os.path.join(DATA_DIR, "sleep_data")
    os.makedirs(SLEEP_DIR, exist_ok=True)

    # Directory to survey datasets
    SURVEY_DIR = os.path.join(DATA_DIR, "survey_data")
    os.makedirs(SURVEY_DIR, exist_ok=True)

    # Directory to lu datasets
    LU_DIR = os.path.join(DATA_DIR, "lu_data")
    os.makedirs(LU_DIR, exist_ok=True)

    # Directory to user datasets
    USER_DIR = os.path.join(DATA_DIR, "user_data")
    os.makedirs(USER_DIR, exist_ok=True)

    # Directory to save and load model
    MODEL_DIR = os.path.join(ROOT_DIR, "models")
    os.makedirs(MODEL_DIR, exist_ok=True)


def prep(df):
    '''
    Preprocess data before modeling.
    Standard scaling is performed for numerical data, and one-hot coding is performed for categorical data.
    '''

    # Change feature type
    df = df.astype({'alcohol':int})
    df = df.astype({'emotionTension':int})
    df = df.astype({'amEmotion':int})
    df = df.astype({'pmEmotion':int})
    df = df.astype({'actionSubOption':int})
    df = df.astype({'CONDITION':int})
    df = df.astype({'sleepProblem':int})
    df = df.astype({'dream':int})
    df = df.astype({'sleep':int})
    df = df.astype({'amCondition':int})

    df1 = df[['action', 'actionSubOption', 'place', 'activity', 'CONDITION', 'sleepProblem',
        'dream', 'dur', 'e4Eda', 'e4Hr', 'e4Temp', 'sleep', 'amCondition', 'amEmotion', 
        'pmEmotion', 'pmStress', 'alcohol','startHour', 'endHour',
        'sleep_score', 'total_sleep_time', 'time_in_bed', 'emotionTension', 'emotionPositive'
        ]]

    # Change categorical feature type to object
    df1 = df1.astype({'activity':object})
    df1 = df1.astype({'action':object})
    df1 = df1.astype({'CONDITION':object})
    df1 = df1.astype({'sleepProblem': object})
    df1 = df1.astype({'place':object})
    df1 = df1.astype({'actionSubOption':object})
    df1 = df1.astype({'dream':object})

    # One-hot encoding categorical feature
    one_hot = pd.get_dummies(df1)

    cols_to_standardize = ['dur', 'e4Eda', 'e4Hr', 'e4Temp', 'sleep', 'amCondition', 'amEmotion', 
        'pmEmotion', 'pmStress', 'alcohol','startHour', 'endHour',
        'sleep_score', 'total_sleep_time', 'time_in_bed']

    df_selected = one_hot[cols_to_standardize]
    scaler = StandardScaler()

    # Standard-scaling numerical feature
    df_standardized = pd.DataFrame(scaler.fit_transform(df_selected), columns=cols_to_standardize)

    # Concat categorical feature and numerical feature
    df_final = pd.concat([one_hot.drop(cols_to_standardize, axis=1), df_standardized], axis=1)

    return df_final