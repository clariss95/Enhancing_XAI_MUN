#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE


# load the ED visits dataset that has been cleaned and preprocessed
df_event_log_dl = pd.read_csv('df_event_log_visits_for_ML_randomorder_areafix.csv')


df_event_log_dl.info()



df_event_log_dl['SEX'] = df_event_log_dl['SEX'].astype('category')
df_event_log_dl['PRESENTING_COMPLAINT_ENCODED'] = df_event_log_dl['PRESENTING_COMPLAINT_ENCODED'].astype('category')
df_event_log_dl['Is_Deceased'] = df_event_log_dl['Is_Deceased'].astype('bool')
df_event_log_dl['Died_Before_Visit'] = df_event_log_dl['Died_Before_Visit'].astype('bool')
df_event_log_dl['Died_During_Visit'] = df_event_log_dl['Died_During_Visit'].astype('bool')
df_event_log_dl['Died_After_Visit'] = df_event_log_dl['Died_After_Visit'].astype('bool')
df_event_log_dl['Is_LWBS'] = df_event_log_dl['Is_LWBS'].astype('bool')
df_event_log_dl['ED_Business_Hourly'] = df_event_log_dl['ED_Business_Hourly'].astype('category')
df_event_log_dl['TIME_OF_THE_DAY_Ordinal'] = df_event_log_dl['TIME_OF_THE_DAY_Ordinal'].astype('category')
df_event_log_dl['Visit_Season_Ordinal'] = df_event_log_dl['Visit_Season_Ordinal'].astype('category')

df_event_log_dl.info()



# Dictionary to rename the columns
rename_dict = {
    'SID': 'case:SID',
    'SEX': 'case:SEX',
    'VISIT_ID': 'case:VISIT_ID',
    'VISIT_AGE': 'case:VISIT_AGE',
    'Case_Duration_Hours': 'case:Case_Duration_Hours',
    'PRESENTING_COMPLAINT_ENCODED':'event:PRESENTING_COMPLAINT_ENCODED',
    'Is_Deceased': 'case:Is_Deceased',
    'Distance_to_Hospital': 'case:Distance_to_Hospital',
    'Is_NL_Holiday': 'case:Is_NL_Holiday',
    'Day_of_Week': 'case:Day_of_Week',
    'Is_Weekend': 'case:Is_Weekend',
    'Visit_Frequency': 'case:Visit_Frequency',
    'Prior_LWBS': 'case:Prior_LWBS',
    'FACILITY_ID_BUR': 'case:FACILITY_ID_BUR',
    'FACILITY_ID_CGH': 'case:FACILITY_ID_CGH',
    'FACILITY_ID_GBC': 'case:FACILITY_ID_GBC',
    'FACILITY_ID_HSC': 'case:FACILITY_ID_HSC',
    'FACILITY_ID_SCM': 'case:FACILITY_ID_SCM',
    'TIME_OF_THE_DAY_Ordinal': 'case:TIME_OF_THE_DAY_Ordinal',
    'Visit_Season_Ordinal': 'case:Visit_Season_Ordinal',
    'CTAS': 'event:CTAS',
    'ED_Business_Hourly': 'event:ED_Business_Hourly',
    'Inverse_CTAS': 'event:Inverse_CTAS',
    'Std_Inverse_CTAS': 'event:Std_Inverse_CTAS',
    'Mean_Inverse_CTAS': 'event:Mean_Inverse_CTAS',
    'Mean_Age': 'event:Mean_Age',
    'Area_Type': 'case:Area_Type',
    'Unique_Presenting_Complaints': 'event:Unique_Presenting_Complaints',
    'Is_LWBS': 'event:Is_LWBS',
    'Activity_Duration': 'event:Activity_Duration',
    'Disposition_Frequency': 'event:Disposition_Frequency',
    'Died_Before_Visit': 'event:Died_Before_Visit',
    'Died_During_Visit': 'event:Died_During_Visit',
    'Died_After_Visit': 'event:Died_After_Visit',
    'Num_Imaging_Tests': 'event:Num_Imaging_Tests',
    'Num_Lab_Tests': 'event:Num_Lab_Tests',
    'Daily_Imaging_Tests': 'case:Daily_Imaging_Tests',
    'Daily_Lab_Tests': 'case:Daily_Lab_Tests',
    'Activity_Admitting patient': 'event:Activity_Admitting patient',
    'Activity_Assessment': 'event:Activity_Assessment',
    'Activity_Making admit decision': 'event:Activity_Making admit decision',
    'Activity_Patient departed': 'event:Activity_Patient departed',
    'Activity_Patient discharge': 'event:Activity_Patient discharge',
    'Activity_Providing service': 'event:Activity_Providing service',
    'Activity_Triage': 'event:Activity_Triage'
}
    
# Renaming the columns
df_event_log_dl.rename(columns=rename_dict, inplace=True)

# Display the renamed columns
print(df_event_log_dl.columns)

# Dropping the 'SID' column from the DataFrame
df_event_log_dl = df_event_log_dl.drop('case:SID', axis=1)

df_event_log_dl.head()

df_event_log_dl.info()


sequence_length =  3

# Updated feature columns excluding potential leakage features


event_feature_columns = [
    'event:CTAS', 'event:PRESENTING_COMPLAINT_ENCODED',
     'event:ED_Business_Hourly', 'event:Inverse_CTAS', 'event:Std_Inverse_CTAS', 'event:Mean_Inverse_CTAS', 'event:Unique_Presenting_Complaints', 'event:Num_Imaging_Tests',
    'event:Num_Lab_Tests', 'event:Mean_Age', 'event:Activity_Admitting patient', 'event:Activity_Assessment',
    'event:Activity_Making admit decision', 'event:Activity_Patient departed', 
    'event:Activity_Patient discharge', 'event:Activity_Providing service', 
    'event:Activity_Triage', 'event:Activity_Duration', 'event:Disposition_Frequency'
]


case_feature_columns = [
    'case:SEX', 'case:VISIT_AGE', 'case:Is_Deceased', 'case:Distance_to_Hospital',
     'case:Is_NL_Holiday', 'case:Area_Type','case:Day_of_Week', 'case:Is_Weekend', 'case:Visit_Frequency', 
     'case:Prior_LWBS', 'case:Case_Duration_Hours', 'case:FACILITY_ID_BUR', 'case:FACILITY_ID_CGH', 
     'case:FACILITY_ID_GBC', 'case:FACILITY_ID_HSC', 'case:FACILITY_ID_SCM', 'case:TIME_OF_THE_DAY_Ordinal', 
     'case:Visit_Season_Ordinal', 'case:Daily_Imaging_Tests', 'case:Daily_Lab_Tests'
]

# Create meaningful feature names for the last step
last_step_event_feature_names = [f'{feature}_t{sequence_length}' for feature in event_feature_columns]

# Combine with case feature names
combined_feature_names_last_step = last_step_event_feature_names + case_feature_columns



# Define the number of event features and case features
num_event_features = len(event_feature_columns)  # number of event features
num_case_features = len(case_feature_columns)    # number of case features

# Sort by VISIT_ID before grouping
df_event_log_dl_sorted = df_event_log_dl.sort_values(by='case:VISIT_ID')
grouped = df_event_log_dl_sorted.groupby('case:VISIT_ID')

# Display the results
print(f'Number of event features: {num_event_features}')
print(f'Number of case features: {num_case_features}')

# Initialize lists to hold sequences and labels
sequences_event = []
sequences_case = []
sequence_labels = []

# Iterate over each group
for _, group in grouped:
    # Extract case attributes (assuming they are the same for each group)
    case_features = group[case_feature_columns].iloc[0].values.astype(np.float32)
    
    # Extract event features, explicitly excluding 'event:Is_LWBS'
    group_features = group[event_feature_columns].astype(float)
    group_labels = group['event:Is_LWBS'].astype(int)
    
    # Padding if necessary
    num_missing = sequence_length - len(group)
    if num_missing > 0:
        # Padding features with zeros at the beginning
        padding = np.zeros((num_missing, group_features.shape[1]))
        group_features = np.vstack((padding, group_features))
        # Padding labels with zeros at the beginning
        padded_labels = np.pad(group_labels, (num_missing, 0), 'constant')
    elif num_missing < 0:
        group_features = group_features.iloc[:sequence_length]
        group_labels = group_labels.iloc[:sequence_length]
        padded_labels = group_labels.values
    else:
        padded_labels = group_labels.values
    
    # Add to sequences
    sequences_event.append(group_features)
    sequences_case.append(case_features)
    sequence_labels.append(padded_labels[-1])

# Convert to arrays for further processing
event_features = np.array(sequences_event, dtype=np.float32)
case_features = np.array(sequences_case, dtype=np.float32)
labels = np.array(sequence_labels, dtype=np.float32)

print(event_features.shape, case_features.shape, labels.shape)

# Save the processed data to .npy files
np.save('event_features.npy', event_features)
np.save('case_features.npy', case_features)
np.save('labels.npy', labels)

print("Data saved to .npy files")

