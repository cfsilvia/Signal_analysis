import numpy as np
import pandas as pd

def interpolation_signal(filename):
    df = pd.read_excel(filename)
    signal = df["Signal"].interpolate().fillna(method="bfill").fillna(method="ffill").values
    labels = df["Labels"].values.astype(int)
    
    return signal, labels


def interpolation_original_signal(data_file,sheet_name,column_name):
    df = pd.read_excel(data_file, sheet_name = sheet_name)
    
    signal = df[column_name].interpolate().fillna(method="bfill").fillna(method="ffill").values
    
    return signal

def get_labels(data_file, sheet_name, column_name, output_file,old_labels_file, sheet_name_old, label):
    
    df = pd.read_excel(data_file, sheet_name = sheet_name)
    signal = df[column_name].interpolate().fillna(method="bfill").fillna(method="ffill").values
    
    df_labels = pd.read_excel(old_labels_file, sheet_name = sheet_name_old)
    #create array with zero  like the signal
    labels = np.zeros_like(signal)
    #create list of the intervals where there is label
    intervals = list(zip(df_labels['starting frame'],df_labels['stop frame']))
    for start, end in intervals:
        end = end + 1
        labels[start:end] = label
    #number of frames
    frame_number = df.index
    
    data_dict = {'Frame': frame_number, 'Signal': signal, 'Labels': labels}
    data = pd.DataFrame(data_dict)
    with pd.ExcelWriter(output_file + "labels_counting_yael.xlsx", engine='openpyxl',mode='w') as writer:
       data.to_excel(writer, index = False)
    
    