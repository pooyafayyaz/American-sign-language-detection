from motion_detection import get_motion_seq
import numpy as np
from dtaidistance import dtw_ndim
from dtaidistance import dtw
import os
from natsort import natsorted
from math import sqrt
from glob import glob
import random
import csv
from tqdm import tqdm


dataset_dir = "/Users/pooya/Downloads/result/"
raw_file_dir = "/Users/pooya/Downloads/cropped_bodypatches/"

all_file_names = glob(dataset_dir+"*")

gloss = ['lightbulb', 'turndown', 'calendar', 'picture', 'bedroom', 'house', 'message', 'food', 'snow', 'time', 'turnon', 'place',
         'dim', 'wakeup', 'traffic', 'ac', 'order', 'schedule', 'direction', 'email', 'weather', 'night', 'temperature', 'raise',
         'turnoff', 'play', 'quote', 'rain', 'restaurant', 'camera', 'snooze', 'phone', 'day', 'list', 'kitchen', 'heat', 'cancel',
         'weekend', 'alarm', 'door', 'shopping', 'room', 'goodmorning', 'doorbell', 'movie', 'game', 'work', 'event', 'lock', 'sunny', 'stop']

csv_file = open("result.csv", "w")
writer = csv.writer(csv_file)

def read_json(file):
    pose_seq = []
    
    for r, d, f in os.walk(file):
        #sort based on the frame number
        for file in natsorted(f):
            if file.endswith(".json"):

                with open(r+file, "r") as read_file:
                    data = read_file.read()
                    data = data.replace('array', 'np.array').replace("float32","np.float32")
                    try:
                        d = eval(data)
                    except:
                        return []
                    # all the key points
                    body_Pose = d[0]["keypoints"]

                    #Normilize the points
                    #Nose position
                    neck_position = (body_Pose[5] + body_Pose[6])/2
                    # Shoulder positions
                    right_shoulder = body_Pose[6]
                    left_shoulder = body_Pose[5]

                    #Calculate the distance between two shoulders
                    dist = sqrt( (left_shoulder[0] - right_shoulder[0])**2 + (left_shoulder[1] - right_shoulder[1])**2 )

                    # 1. First we have to set the origin to neck position
                    # 2. Scale all the coordinates so that the distance of right
                    # shoulder to left shoulder is one

                    right_center = np.sum(body_Pose[91:112,:], axis=0)/21
                    left_center =  np.sum(body_Pose[112:133,:], axis=0)/21

                    body_Pose[5:11,0] = (body_Pose[5:11,0] - neck_position[0])/dist
                    body_Pose[5:11,1] = (body_Pose[5:11,1] - neck_position[1])/dist

                    body_Pose[91:112,0] = (body_Pose[91:112,0] - body_Pose[91][0])
                    body_Pose[91:112,1] = (body_Pose[91:112,1] - body_Pose[91][1])

                    body_Pose[112:133,0] = (body_Pose[112:133,0] - body_Pose[112][0])
                    body_Pose[112:133,1] = (body_Pose[112:133,1] - body_Pose[112][1])

                    # Using only body and hand pose data, all other key points will be discarded
                    body_Pose = body_Pose[:,0:2]

                    body_Pose = body_Pose[np.r_[5:11, 91:133], :]

                    # Otherwise just use the body points
                    #body_Pose = body_Pose[np.r_[5:11], :]

                    pose_seq.append(body_Pose)

    return np.array(pose_seq)

def dtw_pipline(file1,file2):
    pose_seq1 = read_json(dataset_dir +file1+"/")
    pose_seq2 = read_json(dataset_dir + file2+"/")
    
    start_vid1 ,end_vid1 = get_motion_seq(raw_file_dir + file1+ ".avi")
    start_vid2 ,end_vid2 = get_motion_seq(raw_file_dir +  file1 + ".avi")
    
    if(start_vid1 == None or start_vid2 == None or end_vid1 == None or end_vid2 == None or pose_seq1 == [] or pose_seq2 == []):
        return True
    
    start_vid ,end_vid = max(start_vid1,start_vid2),max(end_vid1,end_vid2)

    _ , dtw_matrix =  dtw_ndim.warping_paths(pose_seq1.astype(np.double), pose_seq2.astype(np.double))
    dtw_matrix  = dtw_matrix[1:,1:]
    best_path = dtw.best_path(dtw_matrix)
    best_path = np.array(best_path)
    best_path = best_path[ ((start_vid<best_path) & (best_path< end_vid)).all(axis=1) ]
    
    #if(best_path.size == 0):
    #    print("Empty  ",start_vid1,start_vid2,end_vid1,end_vid2)

    
    writer.writerow([file1,file2, best_path.flatten()])


def read_dataset():
    for word in tqdm(gloss):
        matching = [s for s in all_file_names if "body-"+word+"_" in s]
        for vid in tqdm(matching):
            for j in range(3):
                file1 = vid.split("/")[-1]
                file2 = matching[random.randint(0,len(matching)-1)].split("/")[-1]
                dtw_pipline(file1,file2)

read_dataset()






