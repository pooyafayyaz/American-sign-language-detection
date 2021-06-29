from motion_detection import get_motion_seq
import numpy as np
from dtaidistance import dtw_ndim

file1 =
file2 = 

def read_json(file):
    pose_seq = []

    with open(file, "r") as read_file:
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

        return pose_seq

pose_seq1 = read_json(file1)
pose_seq2 = read_json(file2)

start_vid1 ,end_vid1 = get_motion_seq(file1)
start_vid2 ,end_vid2 = get_motion_seq(file2)

_ , dtw_matrix =  dtw_ndim.distance(pose_seq1.astype(np.double), pose_seq2.astype(np.double))
