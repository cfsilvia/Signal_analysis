import numpy as np
import pandas as pd

def interpolation_signal(filename):
    df = pd.read_excel(filename)
    signal = df["Signal"].interpolate().fillna(method="bfill").fillna(method="ffill").values
    labels = df["Labels"].values.astype(int)
    
    return signal, labels
