import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from math import sqrt
import cv2
import torch
from skimage import io

class HandPatchdataset(Dataset):

    def row_to_array(idx,data_str):
    	 return np.array(data_str[1:-1].strip().split()).reshape((-1,2))    	 
    	
    def __init__(self,data_path,impath):
        self.data = pd.read_csv(data_path)
        self.impath = impath

        self.data.iloc[:,2] = self.data.iloc[:,2].apply(self.row_to_array)
        self.dataset = [] 
        for index, row in self.data.iterrows():
            for item in row[2]:
	            self.dataset.append([row[0],row[1],item])
        self.dataset = np.array(self.dataset,dtype=object)
        print(self.dataset.shape)
        
    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self,idx):
        try:
    
            img_name_1 = self.dataset[idx][0]
            img_name_2 = self.dataset[idx][1]
            
            img_frame_1 = self.dataset[idx][2][0]
            img_frame_2 = self.dataset[idx][2][1]
            
            image1_l = io.imread(self.impath+img_name_1+"/frame"+img_frame_1+"_l.jpg").T
            image1_r = io.imread(self.impath+img_name_1+"/frame"+img_frame_1+"_r.jpg").T
            
            image2_l = io.imread(self.impath+img_name_2+"/frame"+img_frame_2+"_l.jpg").T
            image2_r = io.imread(self.impath+img_name_2+"/frame"+img_frame_2+"_r.jpg").T
            
            return torch.from_numpy(image1_l),torch.from_numpy(image1_r),torch.from_numpy(image2_l),torch.from_numpy(image2_r)
        
        except Exception as e:
            print(e)
            return None
