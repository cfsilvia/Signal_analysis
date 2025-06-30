import yaml
from Annotation_Events import Annotation_Events

def main_menu(choice, data):
    match choice:
         case '1':
             file = data['1']['data_file']
             sheet_name = data['1']['sheet_name']
             column_name = data['1']['column_name']
             output_file = data['1']['output_file']
             annotation = Annotation_Events(file, sheet_name, column_name, output_file)
             annotation()
         case _:
             return "Invalid option"
             
    
    
    
    
if __name__ == "__main__":
    with open("F:/SilviaData/ScriptOnGithub/Signal_analysis/settings.yml", "r") as file: #CHANGE WHEN NECCESSARY DIRECTORY OF SETTINGS
        data = yaml.safe_load(file)
    choice = data['choice']    
    main_menu(choice,data)
    