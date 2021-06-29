from motion_detection import get_motion_seq
import numpy as np
from dtaidistance import dtw_ndim
from dtaidistance import dtw
import os
from natsort import natsorted
from math import sqrt

file1 = "/Users/pooya/Downloads/result/body-lightbulb_qian_1_rgb/"
file2 = "/Users/pooya/Downloads/result/body-lightbulb_sofia_14_rgb/"

raw_file1 = "/Users/pooya/Downloads/cropped_bodypatches/body-lightbulb_qian_1_rgb.avi"
raw_file2 = "/Users/pooya/Downloads/cropped_bodypatches/body-lightbulb_sofia_14_rgb.avi"



def read_json(file):
    pose_seq = []
    for r, d, f in os.walk(file):
        #sort based on the frame number
        for file in natsorted(f):
            if file.endswith(".json"):

                with open(r+file, "r") as read_file:
                    data = read_file.read()
                    data = data.replace('array', 'np.array').replace("float32","np.float32")
                    d = eval(data)
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

pose_seq1 = read_json(file1)
pose_seq2 = read_json(file2)

print(pose_seq1.shape)
print(pose_seq2.shape)

start_vid1 ,end_vid1 = get_motion_seq(raw_file1 )
start_vid2 ,end_vid2 = get_motion_seq(raw_file2 )
start_vid ,end_vid = max(start_vid1,start_vid2),max(end_vid1,end_vid2)

print(start_vid1 ,end_vid1)
print(start_vid2 ,end_vid2)
print(start_vid ,end_vid)

_ , dtw_matrix =  dtw_ndim.warping_paths(pose_seq1.astype(np.double), pose_seq2.astype(np.double))
dtw_matrix  = dtw_matrix[1:,1:]
best_path = dtw.best_path(dtw_matrix)
best_path = np.array(best_path)
print(best_path)
print(best_path[ ((start_vid<best_path) & (best_path< end_vid)).all(axis=1) ])
