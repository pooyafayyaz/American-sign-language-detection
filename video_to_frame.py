from natsort import natsorted
import cv2
import numpy as np
import os
from math import sqrt
from glob import glob
from tqdm import tqdm
from PIL import Image

dataset_dir = "/home/negar/Downloads/cropped_bodypatches/"
json_dir = "/home/negar/Downloads/result_GMU/result/"
all_file_names = glob(dataset_dir+"*")


def read_json(file):
    pose_seq = []
    for r, d, f in os.walk(file):
        #sort based on the frame number
        for file in natsorted(f):
            if file.endswith(".json"):

                with open(r+"/"+file, "r") as read_file:
                    data = read_file.read()
                    data = data.replace('array', 'np.array').replace("float32","np.float32")
                    try:
                        d = eval(data)
                    except:
                        return []
                    # all the key points
                    body_pose = d[0]["keypoints"]
                    right_hand = body_pose[91:112]
                    left_hand =  body_pose[112:133]
			
                    x_r_hand = int(right_hand[:,0].mean())
                    y_r_hand = int(right_hand[:,1].mean())
			
                    x_l_hand = int(left_hand[:,0].mean())
                    y_l_hand = int(left_hand[:,1].mean())
	

                    pose_seq.append([x_r_hand,y_r_hand,x_l_hand,y_l_hand])

    return np.array(pose_seq)


file_cnt = 0
for file in tqdm(all_file_names):      
        
	hand_poses = read_json(json_dir+file.split("/")[-1].split(".avi")[-2] )
	
	if(hand_poses == []):
		continue
	os.mkdir("./frames/"+file.split("/")[-1].split(".avi")[-2])

	vidcap = cv2.VideoCapture(file)
	success,image = vidcap.read()
	count = 0
	while success:
		try:
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			x_r_hand = hand_poses[count][0]
			y_r_hand = hand_poses[count][1]

			x_l_hand = hand_poses[count][2]
			y_l_hand = hand_poses[count][3]


			r_hand_patch = image[max(0, (y_r_hand-112)):max(0, (y_r_hand+112)),max(0, (x_r_hand-112)):max(0, (x_r_hand+112))]
			r_hand_patch = cv2.resize(r_hand_patch, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

			l_hand_patch = image[max(0, (y_l_hand-112)):max(0, (y_l_hand+112)),max(0, (x_l_hand-112)):max(0, (x_l_hand+112))]
			l_hand_patch = cv2.resize(l_hand_patch, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

			im = Image.fromarray(r_hand_patch)
			im.save("./frames/"+file.split("/")[-1].split(".avi")[-2] +"/frame%d_r.jpg" % count)    

			im = Image.fromarray(l_hand_patch) 
			im.save("./frames/"+file.split("/")[-1].split(".avi")[-2] +"/frame%d_l.jpg" % count)     

			success,image = vidcap.read()
			count += 1
		except:
			print(file)
			
			success,image = vidcap.read()
			count += 1
			continue

  
