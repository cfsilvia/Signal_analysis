'''
CLASS FOR ANNOTATE EVENTS IN ONE DIMENSIONAL SIGNAL
'''
import pandas as pd
import numpy as np

class Annotation_Events:
   def __init__(self,file,sheet_name,column_name):
       self.file = file
       self.sheet_name = sheet_name
       self.column_name = column_name
   
   def __call__(self):
        #READ EXCEL
        #Plot
        #interactive to find events in the plot