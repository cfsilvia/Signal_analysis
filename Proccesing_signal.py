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