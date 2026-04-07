import pandas as pd
import numpy as np
class data_validation:
    def __init__(self, data_path):
        self.df=pd.read_csv(data_path)
    
    def check_missing_values(self):
        missing_values = self.df.isnull().sum()
        print("Missing values in each column:")
        print(missing_values)