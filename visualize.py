# Visualize the joins and frames 


import imutils
import cv2
import numpy as np
import os 
import sys
import argparse
from natsort import natsorted


parser = argparse.ArgumentParser()
parser.add_argument('--videos', '-videos', help="path to raw videos", type= str)
parser.add_argument('--json', '-json', help="path to result dir", type= str)
parser.add_argument('--filename', '-filename', help="file name", type= str)
parser.add_argument('--isgmu', '-isgmu', help="file name", type= bool, default= False)
args = parser.parse_args()

thick = 1

if args.isgmu:
	thick = 3 

def plotJoints(pose_data,img):

	# body lines
	cv2.line(img, (pose_data[6][0], pose_data[6][1]), (pose_data[8][0],pose_data[8][1]), (0, 102, 153), thickness=thick, lineType=8)
	cv2.line(img, (pose_data[8][0], pose_data[8][1]), (pose_data[10][0],pose_data[10][1]), (0, 153, 153), thickness=thick, lineType=8)
	cv2.line(img, (pose_data[5][0], pose_data[5][1]), (pose_data[7][0],pose_data[7][1]), (0, 153, 51), thickness=thick, lineType=8)
	cv2.line(img, (pose_data[7][0], pose_data[7][1]), (pose_data[9][0],pose_data[9][1]), (0, 153, 0), thickness=thick, lineType=8)
	cv2.line(img, (pose_data[6][0], pose_data[6][1]), (pose_data[5][0],pose_data[5][1]), (0, 51, 153), thickness=thick, lineType=8)
	
	# hand lines 
	# right hand
	cv2.line(img, (pose_data[91][0], pose_data[91][1]), (pose_data[92][0],pose_data[92][1]), (204, 0, 163), thickness=thick, lineType=8)
	cv2.line(img, (pose_data[92][0], pose_data[92][1]), (pose_data[93][0],pose_data[93][1]), (204, 0, 163), thickness=thick, lineType=8)
	cv2.line(img, (pose_data[93][0], pose_data[93][1]), (pose_data[94][0],pose_data[94][1]), (204, 0, 163), thickness=thick, lineType=8)
	cv2.line(img, (pose_data[94][0], pose_data[94][1]), (pose_data[95][0],pose_data[95][1]), (204, 0, 163), thickness=thick, lineType=8)
	
	cv2.line(img, (pose_data[91][0], pose_data[91][1]), (pose_data[96][0],pose_data[96][1]), (204, 82, 0), thickness=thick, lineType=8)
	cv2.line(img, (pose_data[96][0], pose_data[96][1]), (pose_data[97][0],pose_data[97][1]), (204, 82, 0), thickness=thick, lineType=8)
	cv2.line(img, (pose_data[97][0], pose_data[97][1]), (pose_data[98][0],pose_data[98][1]), (204, 82, 0), thickness=thick, lineType=8)
	cv2.line(img, (pose_data[98][0], pose_data[98][1]), (pose_data[99][0],pose_data[99][1]), (204, 82, 0), thickness=thick, lineType=8)
	
	cv2.line(img, (pose_data[91][0], pose_data[91][1]), (pose_data[100][0],pose_data[100][1]), (82, 204, 0), thickness=thick, lineType=8)
	cv2.line(img, (pose_data[100][0], pose_data[100][1]), (pose_data[101][0],pose_data[101][1]), (82, 204, 0), thickness=thick, lineType=8)
	cv2.line(img, (pose_data[101][0], pose_data[101][1]), (pose_data[102][0],pose_data[102][1]), (82, 204, 0), thickness=thick, lineType=8)
	cv2.line(img, (pose_data[102][0], pose_data[102][1]), (pose_data[103][0],pose_data[103][1]), (82, 204, 0), thickness=thick, lineType=8)
	
	cv2.line(img, (pose_data[91][0], pose_data[91][1]), (pose_data[104][0],pose_data[104][1]), (5, 203, 164), thickness=thick, lineType=8)
	cv2.line(img, (pose_data[104][0], pose_data[104][1]), (pose_data[105][0],pose_data[105][1]), (5, 203, 164), thickness=thick, lineType=8)
	cv2.line(img, (pose_data[105][0], pose_data[105][1]), (pose_data[106][0],pose_data[106][1]), (5, 203, 164), thickness=thick, lineType=8)
	cv2.line(img, (pose_data[106][0], pose_data[106][1]), (pose_data[107][0],pose_data[107][1]), (5, 203, 164), thickness=thick, lineType=8)
	
	cv2.line(img, (pose_data[91][0], pose_data[91][1]), (pose_data[108][0],pose_data[108][1]), (0, 0, 230), thickness=thick, lineType=8)
	cv2.line(img, (pose_data[108][0], pose_data[108][1]), (pose_data[109][0],pose_data[109][1]), (0, 0, 230), thickness=thick, lineType=8)
	cv2.line(img, (pose_data[109][0], pose_data[109][1]), (pose_data[110][0],pose_data[110][1]), (0, 0, 230), thickness=thick, lineType=8)
	cv2.line(img, (pose_data[110][0], pose_data[110][1]), (pose_data[111][0],pose_data[111][1]), (0, 0, 230), thickness=thick, lineType=8)
	
	#left hand
	
	cv2.line(img, (pose_data[112][0], pose_data[112][1]), (pose_data[113][0],pose_data[113][1]), (204, 0, 163), thickness=thick, lineType=8)
	cv2.line(img, (pose_data[113][0], pose_data[113][1]), (pose_data[114][0],pose_data[114][1]), (204, 0, 163), thickness=thick, lineType=8)
	cv2.line(img, (pose_data[114][0], pose_data[114][1]), (pose_data[115][0],pose_data[115][1]), (204, 0, 163), thickness=thick, lineType=8)
	cv2.line(img, (pose_data[115][0], pose_data[115][1]), (pose_data[116][0],pose_data[116][1]), (204, 0, 163), thickness=thick, lineType=8)
	
	cv2.line(img, (pose_data[112][0], pose_data[112][1]), (pose_data[117][0],pose_data[117][1]), (204, 82, 0), thickness=thick, lineType=8)
	cv2.line(img, (pose_data[117][0], pose_data[117][1]), (pose_data[118][0],pose_data[118][1]), (204, 82, 0), thickness=thick, lineType=8)
	cv2.line(img, (pose_data[118][0], pose_data[118][1]), (pose_data[119][0],pose_data[119][1]), (204, 82, 0), thickness=thick, lineType=8)
	cv2.line(img, (pose_data[119][0], pose_data[119][1]), (pose_data[120][0],pose_data[120][1]), (204, 82, 0), thickness=thick, lineType=8)
	
	cv2.line(img, (pose_data[112][0], pose_data[112][1]), (pose_data[121][0],pose_data[121][1]), (82, 204, 0), thickness=thick, lineType=8)
	cv2.line(img, (pose_data[121][0], pose_data[121][1]), (pose_data[122][0],pose_data[122][1]), (82, 204, 0), thickness=thick, lineType=8)
	cv2.line(img, (pose_data[122][0], pose_data[122][1]), (pose_data[123][0],pose_data[123][1]), (82, 204, 0), thickness=thick, lineType=8)
	cv2.line(img, (pose_data[123][0], pose_data[123][1]), (pose_data[124][0],pose_data[124][1]), (82, 204, 0), thickness=thick, lineType=8)
	
	cv2.line(img, (pose_data[112][0], pose_data[112][1]), (pose_data[125][0],pose_data[125][1]), (5, 203, 164), thickness=thick, lineType=8)
	cv2.line(img, (pose_data[125][0], pose_data[125][1]), (pose_data[126][0],pose_data[126][1]), (5, 203, 164), thickness=thick, lineType=8)
	cv2.line(img, (pose_data[126][0], pose_data[126][1]), (pose_data[127][0],pose_data[127][1]), (5, 203, 164), thickness=thick, lineType=8)
	cv2.line(img, (pose_data[127][0], pose_data[127][1]), (pose_data[128][0],pose_data[128][1]), (5, 203, 164), thickness=thick, lineType=8)
	
	cv2.line(img, (pose_data[112][0], pose_data[112][1]), (pose_data[129][0],pose_data[129][1]), (0, 0, 230), thickness=thick, lineType=8)
	cv2.line(img, (pose_data[129][0], pose_data[129][1]), (pose_data[130][0],pose_data[130][1]), (0, 0, 230), thickness=thick, lineType=8)
	cv2.line(img, (pose_data[130][0], pose_data[130][1]), (pose_data[131][0],pose_data[131][1]), (0, 0, 230), thickness=thick, lineType=8)
	cv2.line(img, (pose_data[131][0], pose_data[131][1]), (pose_data[132][0],pose_data[132][1]), (0, 0, 230), thickness=thick, lineType=8)
	
file_name = args.filename
print(file_name)

video_dir = args.videos
json_dir = args.json

# Read the video file 
if(args.isgmu):
	cap = cv2.VideoCapture(video_dir+file_name+".avi")
else:
	cap = cv2.VideoCapture(video_dir+file_name+".mp4")
success, img = cap.read()

# Read the pose data
pose_seq = []
for r, d, f in os.walk(json_dir+file_name):
    #sort based on the frame number
    for file in natsorted(f):
        if file.endswith(".json"):
            # open the json file and read the data
            with open(json_dir+file_name+"/"+file, "r") as read_file:
                print(file)
                data = read_file.read()
                data = data.replace('array', 'np.array').replace("float32","np.float32")
                d = eval(data)
                # all the key points 
                body_Pose = d[0]["keypoints"]
                
                pose_seq.append(body_Pose)
                
pose_seq = np.array(pose_seq)

i = 0
while success:
	# read next frame
	
	pose_data = pose_seq[i]
	cv2.putText(img, "Confidence: {}".format(np.mean(pose_data[91:133,2])), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0,0, 255), 1)
	plotJoints(pose_data,img)
	cv2.imshow("img",img)

	
	success, img = cap.read()
	
	cv2.waitKey(0)
	i+=1
	
# Close the video
cv2.destroyAllWindows()
cap.release()
