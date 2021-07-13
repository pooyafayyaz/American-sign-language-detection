import csv
import pandas as pd 
import os
from natsort import natsorted
from math import sqrt
from glob import glob
import random
import csv
from tqdm import tqdm
import numpy as np

def row_to_array(idx,data_str):
	return np.array(data_str[1:-1].strip().split()).reshape((-1,2))    	 
    	
gloss = ['lightbulb', 'turndown', 'calendar', 'picture', 'bedroom', 'house', 'message', 'food', 'snow', 'time', 'turnon', 'place',
         'dim', 'wakeup', 'traffic', 'ac', 'order', 'schedule', 'direction', 'email', 'weather', 'night', 'temperature', 'raise',
         'turnoff', 'play', 'quote', 'rain', 'restaurant', 'camera', 'snooze', 'phone', 'day', 'list', 'kitchen', 'heat', 'cancel',
         'weekend', 'alarm', 'door', 'shopping', 'room', 'goodmorning', 'doorbell', 'movie', 'game', 'work', 'event', 'lock', 'sunny', 'stop']



dataset_dir = "/home/pooya/Downloads/result_GMU/result/"

all_file_names = glob(dataset_dir+"*")

csv_file = open("negetive_result.csv", "w")
pd_csv_file = pd.read_csv("result.csv")
writer = csv.writer(csv_file)

for word in tqdm(gloss):
        matching = [s for s in all_file_names if "body-"+word+"_" in s]
        for vid in tqdm(matching):
                for j in range(3):
                        gloss_pick = random.choice(gloss)
                        while (gloss_pick == word):
                                gloss_pick = random.choice(gloss)

                        matching_rand = [s for s in all_file_names if "body-"+gloss_pick+"_" in s]
                        file2 = matching_rand[random.randint(0,len(matching_rand)-1)].split("/")[-1]
                        file1 = vid.split("/")[-1]


                        data_file1 = pd_csv_file[(pd_csv_file.iloc[:,0] == file1)]
                        data_file2 = pd_csv_file[(pd_csv_file.iloc[:,1] == file2)]

                        
                        if(data_file1.shape[0]!=0):
                                data_file1 = row_to_array(0,pd_csv_file[(pd_csv_file.iloc[:,0] == file1)].iloc[0][2])[:,0]
                        else:
                                data_file1 = row_to_array(0,pd_csv_file[(pd_csv_file.iloc[:,0] == file1)].iloc[:,2].name)[:,0]
                        if(data_file2.shape[0]!=0):
                                data_file2 = row_to_array(0,pd_csv_file[(pd_csv_file.iloc[:,1] == file2)].iloc[0][2])[:,1]
                        else:
                                data_file2 = row_to_array(0,pd_csv_file[(pd_csv_file.iloc[:,1] == file2)].iloc[:,2].name)[:,1]

                        min_size = min(len(data_file1),len(data_file2))

                        combined = np.vstack((data_file1[:min_size],data_file2[:min_size])).T

                        writer.writerow([file1,file2, combined.flatten()])

