import numpy as np
import pandas as pd
import os



def interpolation_original_signal(data_file,sheet_name,column_name):
    df = pd.read_excel(data_file, sheet_name = sheet_name)
    
    signal = df[column_name].interpolate().fillna(method="bfill").fillna(method="ffill").values
    
    return signal

def save_as_csv(data_file,sheet_name,column_name,output_dir):
    filename = os.path.splitext(os.path.basename(data_file))[0] + '.csv'
    signal = interpolation_original_signal(data_file,sheet_name,column_name)
    full_path = os.path.join(output_dir, filename)
    signal_pd = pd.DataFrame(signal)
    signal_pd.to_csv(full_path)
    return full_path