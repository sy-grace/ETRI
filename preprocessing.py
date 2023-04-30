# import necessary libraries
import os
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
import statistics


# Directories for project
ROOT_DIR = os.getcwd() # Root directory of the project
DATA_DIR = os.path.join(ROOT_DIR, "datasets") # Directory to dataset folder
RAW_DIR = os.path.join(DATA_DIR, "raw_data") # Directory to downloaded datasets
SENSOR_DIR = os.path.join(DATA_DIR, "sensor_data") # Directory to sensor datasets
LABEL_DIR = os.path.join(DATA_DIR, "label_data") # Directory to label datasets
SLEEP_DIR = os.path.join(DATA_DIR, "sleep_data") # Directory to sleep datasets
SURVEY_DIR = os.path.join(DATA_DIR, "survey_data") # Directory to survey datasets
LU_DIR = os.path.join(DATA_DIR, "lu_data") # Directory to lu datasets
USER_DIR = os.path.join(DATA_DIR, "user_data") # Directory to user datasets

def make_user_list(DIR):
    '''
    Get the csv file in the directory.
    Collect the userId in the csv filename in string form and return it as a list.
    '''
    csv_list = os.listdir(DIR)
    csv_user_list = []
    for csv_data in csv_list:
        csv_user_list.append(re.sub('\D', '', csv_data))
    csv_user_list.sort()
    return csv_user_list


def get_file_name(file_type, user_str):
    '''
    Add the csv extension to the filenames.
    '''
    return file_type + "_" + user_str + ".csv"


def get_file_name2(file_type, user_num):
    '''
    Add csv extension to filenames according to specific criteria.
    '''
    if user_num < 10:
        new_file = get_file_name(file_type, '00'+str(user_num))
    elif user_num < 100:
        new_file = get_file_name(file_type, '0'+str(user_num))
    else:
        new_file = get_file_name(file_type, str(user_num))
      
    return new_file


def merge_sensor_label(RAW_YEAR_DIR, user_list, FOLDER_DIR = ''):
    '''
    Merge sensor and label data and split it by user.
    Each sensor csv file is averaged and merged into one file.
    Merge label csv files for each user into one csv file.

    NOTES: Since 'dataset_2020' is different from 'dataset_2018' and 'dataset_2019', FOLDER_DIR must be specified.
    '''
    sensors_list = ['e4Eda', 'e4Hr', 'e4Temp']
    # Save each user in the label csv file address list
    for user in user_list:
        
        # Create a data frame to make the label data into one file
        df_label = pd.DataFrame()
    
        # Csv file address per user
        label_list = []
        
        # Create data frame for each sensor
        df_sensor_e4Eda = pd.DataFrame(columns = ['ts', 'e4Eda'])
        df_sensor_e4Hr = pd.DataFrame(columns = ['ts', 'e4Hr'])
        df_sensor_e4Temp = pd.DataFrame(columns = ['ts', 'e4Temp'])

        # User folder path
        USER_DIR = os.path.join(RAW_YEAR_DIR, FOLDER_DIR, user)

        # List of timestamp folders in the user folder
        ts_list = os.listdir(USER_DIR)        

        # Preprocessing by timestamp
        for ts in ts_list:
            TS_DIR = os.path.join(USER_DIR, ts) # Timestamp folder path
            
            file_list = os.listdir(TS_DIR)  # Collect the label addresses
            label_csv_list = [filename for filename in file_list if filename.endswith('.csv')]  # Select only the csv files among the folders and files within the folder.
            RAW_LABEL_DIR = os.path.join(TS_DIR, label_csv_list[0])
            label_list.append(RAW_LABEL_DIR)
            
            # Average the sensor data for each timestamp and merge them.
            for sensor in sensors_list:
                if sensor == 'e4Eda':
                    df_sensor = df_sensor_e4Eda
                elif sensor == 'e4Hr':
                    df_sensor = df_sensor_e4Hr
                elif sensor == 'e4Temp':
                    df_sensor = df_sensor_e4Temp

                RAW_SENSOR_DIR = os.path.join(TS_DIR, sensor)
                sensor_csv_list = os.listdir(RAW_SENSOR_DIR)

                # Some have no sensor data. Preprocess only with sensor data
                if len(sensor_csv_list) != 0:
                    for sensor_csv in sensor_csv_list:
                        SENSOR_CSV_DIR = os.path.join(RAW_SENSOR_DIR, sensor_csv)
                        df_raw_sensor = pd.read_csv(SENSOR_CSV_DIR)
                        mean = df_raw_sensor.iloc[:, 1].mean()  # Average the sensor data
                        timestamp = sensor_csv.split('.')[0]
                        df_sensor.loc[len(df_sensor)] = [timestamp, mean]   # Merge averaged sensor data into one file

                df_sensor = df_sensor.rename(columns={df_sensor.columns[0]: 'ts', df_sensor.columns[1]:sensor})            
        
        # Merge 3 sensor data into one file
        df_sensor_semifinal = pd.merge(df_sensor_e4Eda, df_sensor_e4Hr, on = 'ts', how = 'inner')
        df_sensor_final = pd.merge(df_sensor_semifinal, df_sensor_e4Temp, on = 'ts', how = 'inner')                    

        # Create userId               
        user_temp_str = re.sub('\D', '', user)
        if FOLDER_DIR != '':
            user_temp_str = '2' + user_temp_str
        df_sensor_final['userId'] = user_temp_str
        user_str = "{:0>3}".format(user_temp_str)
        
        # Save sensor csv file
        new_sensor_file = get_file_name('sensor', user_str)
        df_sensor_final.to_csv(os.path.join(SENSOR_DIR, new_sensor_file), index = False)

        # Concatenate label dataframes
        for label in label_list:
            new_df_label = pd.read_csv(label)
            df_label = pd.concat([df_label, new_df_label])

        # Save label csv file
        df_label = df_label.reset_index(drop = True)
        new_label_file = get_file_name('label', user_str)
        df_label.to_csv(os.path.join(LABEL_DIR, new_label_file), index=False)



def split_sleep_2019_2018():
    '''
    Sleep data preprocessing
    Split sleep_data_2019_2018 by user.
    '''
    # Read a file and save it as a dataframe
    sleep_file = 'user_sleep_2019_2018.csv'
    SLEEP_FILE = os.path.join(RAW_DIR, sleep_file)
    df_sleep = pd.read_csv(SLEEP_FILE)

    # Leave the necessary features
    df_sleep = df_sleep.loc[:, ['userId', 'startDt', 'endDt', 'sleep_score', 'total_sleep_time', 'time_in_bed']]

    userId_list = set(df_sleep.iloc[:, 0])

    # Split sleep data by user
    for userId in userId_list:
        df_userId = df_sleep[df_sleep['userId'] == userId]

        df_userId.index = [i for i in range(len(df_userId))]

        # Create 'startHour' and 'endHour' feature using 'startDt' and 'endDt' feature
        df_userId['startDate'] = df_userId['startDt'].apply(lambda x: x[:10])   # Create 'startDate' and 'endDate' feature using 'startDt' and 'endDt' feature
        df_userId['endDate'] = df_userId['endDt'].apply(lambda x: x[:10])

        df_userId['startHour'] = df_userId['startDt'].apply(lambda x: int(x[10:len(x)-3]))  # Create 'startHour' and 'endHour' feature using 'startDt' and 'endDt' feature
        df_userId['endHour'] = df_userId['endDt'].apply(lambda x: int(x[10:len(x)-3]))

        # Delete unnecessary features
        df_userId = df_userId.drop('startDt', axis=1)
        df_userId = df_userId.drop('endDt', axis=1)

        df_userId = df_userId.sort_values(by=['endDate', 'total_sleep_time'], ascending=[True, False])
        df_userId = df_userId.drop_duplicates(subset=['endDate'], keep='first')

        df_userId = df_userId.sort_values(by=['userId', 'endDate'], ascending=[True, True])

        df_userId = df_userId.loc[:, ['userId', 'startDate', 'endDate', 'startHour', 'endHour', 'sleep_score', 'total_sleep_time', 'time_in_bed']]

        new_sleep_file = get_file_name2('sleep', userId)
        
        # Save sleep data by user.
        df_userId.to_csv(os.path.join(SLEEP_DIR, new_sleep_file), index=False)



def split_sleep_2020():
    '''
    Sleep data preprocessing
    Split sleep_data_2020 by user.
    '''
    # Read a file and save it as a dataframe
    sleep_file = 'user_sleep_2020.csv'
    SLEEP_FILE = os.path.join(RAW_DIR, sleep_file)
    df_sleep = pd.read_csv(SLEEP_FILE)

    # Create 'total_sleep_time' and 'time_in_bed' feature
    df_sleep['total_sleep_time'] = df_sleep['lightsleepduration'] + df_sleep['deepsleepduration'] + df_sleep['remsleepduration']
    df_sleep['time_in_bed'] = df_sleep['total_sleep_time'] + df_sleep['wakeupduration'] + df_sleep['durationtosleep'] + df_sleep['durationtowakeup']

    # Create 'startHour' and 'endHour' feature using 'startDt' and 'endDt' feature
    df_sleep['startDt'] = pd.to_datetime(df_sleep['startDt'], unit='s').dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul')
    df_sleep['endDt'] = pd.to_datetime(df_sleep['endDt'], unit='s').dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul')

    df_sleep['startDt'] = df_sleep['startDt'].dt.strftime('%Y-%m-%d %H:%M') # Change time format
    df_sleep['endDt'] = df_sleep['endDt'].dt.strftime('%Y-%m-%d %H:%M')

    df_sleep['startDate'] = df_sleep['startDt'].apply(lambda x: x[:10]) # Create 'startDate' and 'endDate' feature using 'startDt' and 'endDt' feature
    df_sleep['endDate'] = df_sleep['endDt'].apply(lambda x: x[:10])

    df_sleep['startHour'] = df_sleep['startDt'].apply(lambda x: int(x[11:13]))  # Create 'startHour' and 'endHour'
    df_sleep['endHour'] = df_sleep['endDt'].apply(lambda x: int(x[11:13]))

    # Add 200 to userId to indicate that the data is from 2020
    df_sleep['userId'] = df_sleep['userId'].apply(lambda x: int(x[4:])+200)

    # Leave the necessary features
    df_sleep = df_sleep.loc[:, ['userId', 'startDate', 'endDate', 'startHour', 'endHour', 'sleep_score', 'total_sleep_time', 'time_in_bed']]

    userId_list = set(df_sleep.iloc[:, 0])

    # Split sleep data by user
    for userId in userId_list:
    
        df_userId = df_sleep[df_sleep['userId'] == userId]

        df_userId = df_userId.sort_values(by=['endDate', 'total_sleep_time'], ascending=[True, False])
        df_userId = df_userId.drop_duplicates(subset=['endDate'], keep='first')

        df_userId = df_userId.sort_values(by=['userId', 'endDate'], ascending=[True, True])

        df_userId.index = [i for i in range(len(df_userId))]

        new_sleep_file = get_file_name('sleep', str(userId))
        
        # Save sleep data by user.
        df_userId.to_csv(os.path.join(SLEEP_DIR, new_sleep_file), index=False)
        


def split_survey_2019_2018():
    '''
    Survey data preprocessing
    Unify format of survey_data_2019_2018 and split it by user
    '''
    # Read a file and save it as a dataframe
    survey_file = 'user_survey_2019_2018.csv'
    SURVEY_FILE = os.path.join(RAW_DIR, survey_file)
    df_survey = pd.read_csv(SURVEY_FILE)

    userId_list = set(df_survey.iloc[:, 0])

    # Data pre-processing for each user
    for userId in userId_list:
        df_userId = df_survey[df_survey['userId'] == userId]

        df_userId_am = df_userId[df_userId['amPm'] == 'am'].loc[:, ['userId', 'inputDt', 'sleep', 'sleepProblem', 'dream', 'amCondition', 'amEmotion']]
        df_userId_pm = df_userId[df_userId['amPm'] == 'pm'].loc[:, ['userId', 'inputDt', 'pmEmotion', 'pmStress', 'alcohol', 'aAmount']]
        df_userId_am.index = [i for i in range(len(df_userId_am))]
        df_userId_am['inputDt'] = df_userId_am['inputDt'].apply(lambda x: x[:10])
        df_userId_pm.index = [i for i in range(len(df_userId_pm))]
        df_userId_pm['inputDt'] = df_userId_pm['inputDt'].apply(lambda x: x[:10])

        df_userId = pd.merge(df_userId_am, df_userId_pm, left_on=['userId', 'inputDt'], right_on=['userId', 'inputDt'])
        
        new_survey_file = get_file_name2('survey', userId)
        
        # Save survey data by user
        df_userId.to_csv(os.path.join(SURVEY_DIR, new_survey_file), index=False)


def split_survey_2020():
    '''
    Survey data preprocessing
    Unify format of survey_data_2020 and split it by user
    '''
    # Read a file and save it as a dataframe
    survey_file = 'user_survey_2020.csv'
    SURVEY_FILE = os.path.join(RAW_DIR, survey_file)
    df_survey = pd.read_csv(SURVEY_FILE)

    # Align the format with the 2018, 2019 data
    df_survey['inputDt'] = df_survey['date']
    df_survey['aAmount'] = df_survey['aAmount(ml)']
    df_survey['userId'] = df_survey['userId'].apply(lambda x: int(x[4:])+200)
    df_survey['aAmount'] = df_survey['aAmount'].apply(lambda x: 0 if pd.isna(x) else x)

    userId_list = set(df_survey.iloc[:, 0])

    # Data pre-processing for each user
    for userId in userId_list:
        df_userId = df_survey[df_survey['userId'] == userId]

        df_userId_am = df_userId[df_userId['amPm'] == 'am'].loc[:, ['userId', 'inputDt', 'sleep', 'sleepProblem', 'dream', 'amCondition', 'amEmotion']]
        df_userId_pm = df_userId[df_userId['amPm'] == 'pm'].loc[:, ['userId', 'inputDt', 'pmEmotion', 'pmStress', 'alcohol', 'aAmount']]
        df_userId_am.index = [i for i in range(len(df_userId_am))]
        df_userId_am['inputDt'] = df_userId_am['inputDt'].apply(lambda x: x[:10])
        df_userId_pm.index = [i for i in range(len(df_userId_pm))]
        df_userId_pm['inputDt'] = df_userId_pm['inputDt'].apply(lambda x: x[:10])

        df_userId = pd.merge(df_userId_am, df_userId_pm, left_on=['userId', 'inputDt'], right_on=['userId', 'inputDt'])

        new_survey_file = get_file_name('survey', str(userId))

        # Save survey data by user
        df_userId.to_csv(os.path.join(SURVEY_DIR, new_survey_file), index=False)


def assemble_sensor_label():
    '''
    Combine sensor data and label data.
    '''
    # File list of label_data, sensor_data
    sensor_list = []
    label_list = []

    # Extract file names except for filename extensions
    for file in os.listdir(SENSOR_DIR):
        sensor_list.append(file.split('.')[0])
    for file in os.listdir(LABEL_DIR):
        label_list.append(file.split('.')[0])

    for sensor in sensor_list:
        for label in label_list:
            sensor_int = re.sub('\D', '', sensor)
            label_int = re.sub('\D', '', label)
            
            # Create an empty data frame
            df_lu = pd.DataFrame()

            if (sensor_int == label_int):

                # Sensor and label path
                SENSOR_FILE = os.path.join(SENSOR_DIR, sensor)
                LABEL_FILE = os.path.join(LABEL_DIR, label)
                
                # Read csv file
                df_sensor = pd.read_csv(SENSOR_FILE + '.csv')
                df_label = pd.read_csv(LABEL_FILE + '.csv')

                # Mearge sensor data and label data
                df_lu = pd.merge(df_sensor, df_label, on='ts', how='inner')

                # Save label + sensor data by lu
                df_lu.to_csv(os.path.join(LU_DIR, get_file_name('lu', label_int)), index=False)


def assembly_sleep_survey_lu():
    '''
    Merging survey, sleep and  lu csv files into one
    '''
    user_list = make_user_list(SLEEP_DIR)

    for user_str in user_list:
        user_int = int(user_str)

        # Get file names except for filename extensions
        lu_file = get_file_name('lu', user_str)
        sleep_file = get_file_name('sleep', user_str)
        survey_file = get_file_name('survey', user_str)
        
        df_lu = pd.read_csv(os.path.join(LU_DIR, lu_file))
        df_sleep = pd.read_csv(os.path.join(SLEEP_DIR, sleep_file))
        df_survey = pd.read_csv(os.path.join(SURVEY_DIR, survey_file))

        # 'date' feature of lu_data
        df_lu['date'] = pd.to_datetime(df_lu['ts'], unit='s').dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul')
        df_lu['date'] = df_lu['date'].apply(lambda x: x.strftime('%Y-%m-%d'))

        df_survey = df_survey.drop('userId', axis=1)
        df_sleep = df_sleep.drop('userId', axis=1)

        sleep_date = df_sleep['endDate'].unique()
        survey_date = df_survey['inputDt'].unique()

        # Combine df_survey and df_sleep
        temp_cols = []
        for col in df_survey.columns:
            temp_cols.append(col)
        for col in df_sleep.columns:
            temp_cols.append(col)

        df_temp = pd.DataFrame(columns=temp_cols)

        survey_len = len(df_survey)
        sleep_len = len(df_sleep)

        i = 0
        while i < survey_len:
            j = 0
            if df_survey.loc[i, 'inputDt'] in sleep_date:
                while j < sleep_len:
                    if df_survey.loc[i, 'inputDt'] == df_sleep.loc[j, 'endDate']:
                        data_list = []
                        for data in df_survey.loc[i]:
                            data_list.append(data)
                        for data in df_sleep.loc[j]:
                            data_list.append(data)
                        df_temp.loc[i] = data_list
                        break
                    else:
                        j += 1
            i += 1
        df_temp.index = [i for i in range(len(df_temp))]
        
        # Combine df_lu and df_temp
        cols = []
        for col in df_lu.columns:
            cols.append(col)
        for col in df_temp.columns:
            cols.append(col)

        df_user = pd.DataFrame(columns=cols)

        i = 0
        while i < len(df_lu):
            j = 0
            if df_lu.loc[i, 'date'] in df_temp['inputDt'].unique():
                while j < len(df_temp):
                    if df_lu.loc[i, 'date'] == df_temp.loc[j, 'inputDt']:
                        data_list = []
                        for data in df_lu.loc[i]:
                            data_list.append(data)
                        for data in df_temp.loc[j]:
                            data_list.append(data)
                        df_user.loc[i] = data_list
                        break
                    else:
                        j += 1
            i += 1
        df_user.index = [i for i in range(len(df_user))]

        USER_FILE = get_file_name('user', user_str)

        # Save user data by user
        df_user.to_csv(os.path.join(USER_DIR, USER_FILE), index=False)


def merge_into_user():
    '''
    User data integration
    A new feature 'CONDITION' is created by integrating feature 'condition' and 'condition1Option'.
    A new feature 'action' is created by integrating feature 'actionSub', 'actionSubOption', 'actionOption'.
    '''
    user_list = make_user_list(USER_DIR)
    
    df_merged = pd.DataFrame()

    for user_str in user_list:
        user_file = get_file_name('user', user_str)
        df = pd.read_csv(os.path.join(USER_DIR, user_file))

        df = df.drop(['conditionSub2Option', 'date'], axis=1)
        df['conditionSub1Option'] = df['conditionSub1Option'].fillna(0)

        # For 2018, 2019
        if int(user_str) < 200:
            # Preprocessing 'condition', and 'conditionSub1Option
            df.loc[(df['condition'] == 0) & (df['conditionSub1Option'] == 1), 'CONDITION'] = 1
            df.loc[(df['condition'] == 0) & (df['conditionSub1Option'] == 2), 'CONDITION'] = 2
            df.loc[(df['condition'] == 0) & (df['conditionSub1Option'] == 3), 'CONDITION'] = 3
            df.loc[(df['condition'] == 0) & (df['conditionSub1Option'] == 4), 'CONDITION'] = 4
            df.loc[(df['condition'] == 1) & (df['conditionSub1Option'] == 0), 'CONDITION'] = 1
            df.loc[(df['condition'] == 2) & (df['conditionSub1Option'] == 0), 'CONDITION'] = 2
            df.loc[(df['condition'] == 3) & (df['conditionSub1Option'] == 0), 'CONDITION'] = 3
            df.loc[(df['condition'] == 4) & (df['conditionSub1Option'] == 0), 'CONDITION'] = 4
            df.loc[(df['condition'] == 0) & (df['conditionSub1Option'] == 0), 'CONDITION'] = 0
            df.loc[(df['condition'] == 1) & (df['conditionSub1Option'] == 1), 'CONDITION'] = 1
            df.loc[(df['condition'] == 2) & (df['conditionSub1Option'] == 2), 'CONDITION'] = 2
            df.loc[(df['condition'] == 3) & (df['conditionSub1Option'] == 3), 'CONDITION'] = 3
            df.loc[(df['condition'] == 4) & (df['conditionSub1Option'] == 4), 'CONDITION'] = 4

            # Preprocessing 'action', 'actionSubOption', 'actionOption', and 'actionSub'
            for i in range(len(df)):
                if df.loc[i, 'actionSubOption'] == 0:
                    df.loc[i, 'actionSubOption'] = 6
                
                # Correcting typo: 'communitiy_interaction'to 'community_interaction'
                if df.loc[i, 'action'] == 'communitiy_interaction':
                    df.loc[i, 'action'] = 'community_interaction'

            df = df.fillna(0)

        # For 2020
        else:
            # Preprocessing 'condition', and 'conditionSub1Option'
            df.loc[(df['condition'] == 'ALONE') & (df['conditionSub1Option'] == 0), 'CONDITION'] = 0
            df.loc[(df['condition'] == 'WITH_MANY') & (df['conditionSub1Option'] == 0), 'CONDITION'] = 5
            df.loc[(df['condition'] == 'WITH_ONE') & (df['conditionSub1Option'] == 0), 'CONDITION'] = 5

            df.loc[(df['condition'] == 'ALONE') & (df['conditionSub1Option'] == 1), 'CONDITION'] = 1
            df.loc[(df['condition'] == 'WITH_MANY') & (df['conditionSub1Option'] == 1), 'CONDITION'] = 1
            df.loc[(df['condition'] == 'WITH_ONE') & (df['conditionSub1Option'] == 1), 'CONDITION'] = 1

            df.loc[(df['condition'] == 'ALONE') & (df['conditionSub1Option'] == 2), 'CONDITION'] = 2
            df.loc[(df['condition'] == 'WITH_MANY') & (df['conditionSub1Option'] == 2), 'CONDITION'] = 2
            df.loc[(df['condition'] == 'WITH_ONE') & (df['conditionSub1Option'] == 2), 'CONDITION'] = 2

            df.loc[(df['condition'] == 'ALONE') & (df['conditionSub1Option'] == 3), 'CONDITION'] = 3
            df.loc[(df['condition'] == 'WITH_MANY') & (df['conditionSub1Option'] == 3), 'CONDITION'] = 3
            df.loc[(df['condition'] == 'WITH_ONE') & (df['conditionSub1Option'] == 3), 'CONDITION'] = 3

            df.loc[(df['condition'] == 'ALONE') & (df['conditionSub1Option'] == 4), 'CONDITION'] = 4
            df.loc[(df['condition'] == 'WITH_MANY') & (df['conditionSub1Option'] == 4), 'CONDITION'] = 4
            df.loc[(df['condition'] == 'WITH_ONE') & (df['conditionSub1Option'] == 4), 'CONDITION'] = 4

            df.loc[(df['condition'] == 'ALONE') & (df['conditionSub1Option'] == 5), 'CONDITION'] = 5
            df.loc[(df['condition'] == 'WITH_MANY') & (df['conditionSub1Option'] == 5), 'CONDITION'] = 5
            df.loc[(df['condition'] == 'WITH_ONE') & (df['conditionSub1Option'] == 5), 'CONDITION'] = 5

            # Preprocessing 'action', 'actionSubOption', 'actionOption', and 'actionSub'
            for i in range(len(df)):
                if df.loc[i, 'actionSub'] == 'meal_amount':
                    df.loc[i, 'actionSubOption'] = None
                elif df.loc[i, 'actionSub'] == 'move_method':
                    if df.loc[i, 'actionSubOption'] in [3, 5, 6]:
                        df.loc[i, 'actionSubOption'] = 3
                    elif df.loc[i, 'actionSubOption'] == 7:
                        df.loc[i, 'actionSubOption'] = 5
                if df.loc[i, 'actionOption'] == 754:
                    df.loc[i, 'actionSubOption'] = 6
                if df.loc[i, 'actionSubOption'] > 5:
                    df.loc[i, 'actionSubOption'] = 0

            df['actionSubOption'] = df['actionSubOption'].fillna(0)

        df = df.drop(['actionOption', 'actionSub'], axis=1)
        df = df.drop(['conditionSub1Option', 'condition'], axis=1)
        
        # Preprocessing alcohol
        df['alcohol'] = df['alcohol'].fillna(0)

        # Remove None from 'CONDITION'
        df.dropna(subset=['CONDITION'])

        # Preprocessing alcohol
        df.loc[df['aAmount'] != 0, 'aAmount'] = 1
        df.drop(['alcohol'], axis=1, inplace=True)
        df = df.rename(columns={'aAmount': 'alcohol'})
        
        df_merged = pd.concat([df_merged, df], ignore_index=True)
        
    df_merged = df_merged.reset_index(drop = True)
    
    return df_merged


def labeling(col_name, df):
    '''
    Integer Encoding action and place feature.
    '''
    label = df[col_name].unique()

    # Create new dictionary
    dic = {}

    # Call LabelEncoder method
    encoder = LabelEncoder()
    encoder.fit(label)

    # Save in new dictionary
    for i, j in enumerate(encoder.classes_):
        dic[j] = i
    
    return dic


def preprocess_action_place(df):
    '''
    Function for preoprocessing action and place.
    Mapping value in labeling directory to action and place.
    '''
    # Labeling action & place
    action_dict = labeling('action', df)
    place_dict = labeling('place', df)

    df['action'] = df['action'].map(action_dict)
    df['place'] = df['place'].map(place_dict)

    df = df.drop(['inputDt', 'endDate', 'startDate'], axis=1)
    df = df[['userId', 'ts', 'e4Eda', 'e4Hr', 'e4Temp', 'action', 'actionSubOption',
        'place', 'activity', 'CONDITION', 
        'sleep', 'sleepProblem', 'dream', 'amCondition', 'amEmotion',
        'pmEmotion', 'pmStress', 'alcohol', 'startHour', 'endHour', 
        'sleep_score', 'total_sleep_time', 'time_in_bed', 'emotionPositive', 'emotionTension']]
    df = df.reset_index(drop=True)
        
    return df


 
def create_dur(df):
    '''
    Create new feature 'dur'.
    The amount of time that a user continues to behave in a particular way.
    '''
    columns = df.columns
    new_df = pd.DataFrame(columns=columns)
    new_df.rename(columns={new_df.columns[1]: 'dur'}, inplace=True)

    comp_df = df.copy() # Dataframe for comparing
    comp_df = comp_df.drop(['ts', 'e4Eda', 'e4Hr', 'e4Temp'], axis=1)
    
    i = 0
    user_list = []
    n = len(df)

    while i < n - 1:
        j = i + 1

        while comp_df.loc[i].equals(comp_df.loc[j]):
            j += 1
            if j >= n:
                break

        j -= 1
        dur = abs(df.loc[j, 'ts'] - df.loc[i, 'ts'])
        e4Eda = statistics.mean(df.loc[i:j, 'e4Eda'])
        e4Hr = statistics.mean(df.loc[i:j, 'e4Hr'])
        e4Temp = statistics.mean(df.loc[i:j, 'e4Temp'])
        new_row = []
        for k in range(len(df.loc[i])):
            new_row.append(df.iloc[i, k])
        new_row[1] = dur
        new_row[2] = e4Eda
        new_row[3] = e4Hr
        new_row[4] = e4Temp
        new_df.loc[len(new_df)] = new_row

        if df.loc[j, 'userId'] not in user_list:
            user_list.append(df.loc[j, 'userId'])

        i = j + 1

    return new_df


def preprocess_y(df):
    '''
    Preprocess 'emotionTension' and 'emotionPositive.
    EmotionPositive, emotionTension only stores data with non-zero values.
    Replace the values 1 and 2 of emotionPositive with 3 because it can cause data imbalance problems.
    '''
    df = df[df['emotionTension'] != 0]
    df = df[df['emotionTension'] != None]
    df = df[df['emotionPositive'] != 0]
    df = df[df['emotionPositive'] != None]

    df['emotionPositive'].replace(1, 3, inplace=True)
    df['emotionPositive'].replace(2, 3, inplace=True)

    return df


def preprocess():
    '''
    Create "ETRI.csv" from raw data.
    Export to one csv file after completing all preprocessing.
    '''
    
    # Specify sensor data to use in the project
    year_list = ['dataset_2018', 'dataset_2019', 'dataset_2020']

    for year in year_list:
        RAW_YEAR_DIR = os.path.join(RAW_DIR, year)
      
        if year == 'dataset_2020':
            users_list = os.listdir(RAW_YEAR_DIR)

            for users in users_list:
                FOLDER_DIR = os.path.join(RAW_YEAR_DIR, users)
                user_list = os.listdir(FOLDER_DIR)
                merge_sensor_label(RAW_YEAR_DIR, user_list, FOLDER_DIR)
        
        else:
            user_list = os.listdir(RAW_YEAR_DIR)
            merge_sensor_label(RAW_YEAR_DIR, user_list)

    # Split sleep data by user
    split_sleep_2019_2018()
    split_sleep_2020()

    # Split survey data by user
    split_survey_2019_2018()
    split_survey_2020()

    # Create lu data by assembling sensor and label data of all users
    assemble_sensor_label()
    assembly_sleep_survey_lu()

    # User data intgration and preprocess 'dur', 'action' and ylabel
    df = merge_into_user()
    df = preprocess_action_place(df)
    df = create_dur(df)
    df = preprocess_y(df)        

    # Save dataset as 'ETRI.csv'
    df.to_csv(os.path.join(DATA_DIR, 'ETRI.csv'), index=False)