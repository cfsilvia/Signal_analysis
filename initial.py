import yaml
from Annotation_Events import Annotation_Events
import tkinter as tk
from tkinter import filedialog
import Proccesing_signal
from Train_with_tweety import Train_with_tweety 
from Plot_signal import Plot_signal

def main_menu(choice, data):
    match choice:
         case '1': #annotation
             file = data['1']['data_file']
             sheet_name = data['1']['sheet_name']
             column_name = data['1']['column_name']
             output_file = data['1']['output_file']
             annotation = Annotation_Events(file, sheet_name, column_name, output_file)
             annotation()
         case '2': #training
             file = data['2']['data_file']
             output_file = data['2']['output_file']
             signal, labels = Proccesing_signal.interpolation_signal(file)
             Train_with_tweety(signal, labels,output_file)
         case '3': #plot the labels 
             file = data['3']['data_file']
             sheet_name = data['3']['sheet_name']
             column_name = data['3']['column_name']
             output_file = data['3']['output_file']
             file_labels = data['3']['file_labels']
             plotting = Plot_signal(file, sheet_name, column_name, file_labels,output_file)
             plotting._mark_labels()
         case _:
             return "Invalid option"
             
def load_settings_via_dialog():
    # hide the root window
    root = tk.Tk()
    root.withdraw()

    # open the file dialog
    file_path = filedialog.askopenfilename(
        title="Select settings YAML file",
        filetypes=[("YAML files", "*.yml *.yaml"), ("All files", "*.*")]
    )
    if not file_path:
        raise FileNotFoundError("No settings file selected.")

    # load the YAML
    with open(file_path, "r") as file:
        return yaml.safe_load(file)   
    
    
    
if __name__ == "__main__":
    try:
        data = load_settings_via_dialog()
    except FileNotFoundError as e:
        print(e)
        exit(1)
    choice = data['choice']    
    main_menu(choice,data)
    