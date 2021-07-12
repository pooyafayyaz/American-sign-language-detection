import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from math import sqrt
import torch
from skimage import io

class HandPatchdataset(Dataset):

    def row_to_array(data_str):
    	 return np.array(data_str[1:-1].strip().split()).reshape((-1,2))    	 
    	
    def __init__(self,data_path,impath):
        self.data = pd.read_csv(data_path)
        self.impath = impath
        self.data.iloc[:,2] = self.data.iloc[:,2].apply(row_to_array)
        self.dataset = [] 
        for index, row in data.iterrows():
       	for item in row[2]:
       		self.dataset.add([row[0],row[1],item])
       		
    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self,idx):
        try:
    
            img_name_1 = self.dataset[idx][0]
            img_name_2 = self.dataset[idx][1]
            
            img_frame_1 = self.dataset[idx][2][0]
            img_frame_2 = self.dataset[idx][2][1]
            
            image1_l = io.imread(self.impath+img_name_1+"/frame"+img_frame_1+"_l.jpg")
            image1_r = io.imread(self.impath+img_name_1+"/frame"+img_frame_1+"_r.jpg")
            
            image2_l = io.imread(self.impath+img_name_2+"/frame"+img_frame_2+"_l.jpg")
            image2_r = io.imread(self.impath+img_name_2+"/frame"+img_frame_2+"_r.jpg")

            return image1_l.tensor(),image1_r.tensor(),image2_l.tensor(),image2_r.tensor()
        
        except:
            
            return None
