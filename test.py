import csv
import numpy as np
import pandas as pd
import os 

pd_csv_file = pd.read_csv("negative_result.csv")



for index, row in pd_csv_file.iterrows():
   	print(os.path.isfile( "frames/" +row[0]))
    	
