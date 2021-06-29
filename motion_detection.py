# Python Motion detection using OpenCV
import imutils
import cv2
import numpy as np
import os
from natsort import natsorted

def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()


def get_motion_seq(file):
        # Read the video file 
        cap = cv2.VideoCapture(file) 


        # Read the first 2 frames
        _,first_frame = cap.read()
        ret,second_frame = cap.read()

        # Set the threshold to remove small movments 
        MOVEMENT_THRESHOLD= 3000
        i=0
        start_motion_frame = None 
        last_motion_frame = None
        while (ret) :
                # Get the difference between two consecutive frames	
                frame_delta = cv2.absdiff(first_frame, second_frame)
                # Convert it to grayscale video 
                # Contours are easier to find in grayscale mode
                gray = cv2.cvtColor(frame_delta, cv2.COLOR_BGR2GRAY)
                # Blur the frame 
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                
                # Set the threshold 
                thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)[1]
                
                # Fill in holes
                dilate = cv2.dilate(thresh, None, iterations = 3)
                
                # Find the contours, ignore the hierarchy
                cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                
                # loop over the contours
                for c in cnts:

                        # Coordinates
                        (x, y, w, h) = cv2.boundingRect(c)
                        
                        # If the contour is too small, ignore it, otherwise, there's transient
                        # movement
                        if cv2.contourArea(c) > MOVEMENT_THRESHOLD:
                            # Set the frames for motion
                            if start_motion_frame ==None:
                                start_motion_frame = i 
                            if last_motion_frame == None or last_motion_frame < i:
                                last_motion_frame = i
                            
                            # Draw a rectangle
                            cv2.rectangle(first_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                frame_delta = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                
                #cv2.imshow("img", np.hstack((frame_delta, first_frame)))
                #cv2.imwrite("frame"+ str(i) + ".png", np.hstack((frame_delta, first_frame)))
                first_frame = second_frame
                ret,second_frame = cap.read()
                i+=1
                #cv2.waitKey(0)
                
        print(start_motion_frame,last_motion_frame)
        # Close the video
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cap.release()

        return start_motion_frame,last_motion_frame


