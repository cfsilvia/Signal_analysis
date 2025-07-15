import vame
import pandas as pd
import Processing_signal
import os
from Vame_project import vame_project








if __name__ == "__main__":
    #Create vame project 
    working_dir = "F:/BlindMole_tracking_Juna/2025/Vame_project"
    project_name = 'Snout_left_side'
    #working_dir.mkdir(exist_ok = True)
    
    
    #data
    initial_file = r"F:\BlindMole_tracking_Juna\2025\BMR10\BMR10\output\BMR10_with_landmarks_left_ToPlot.xlsx"
    sheet_name='BM_snout_y'
    output_directory = os.path.join(working_dir, project_name, 'raw_data')
    os.makedirs(output_directory, exist_ok=True)
    column_name = "Left_side"
    csv_path= Processing_signal.save_as_csv(initial_file,sheet_name,column_name,output_directory)
    
    obj = vame_project(working_dir, project_name,csv_path)
    obj.configuration()
    a=1
    
    
    
    