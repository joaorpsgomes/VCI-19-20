import numpy as np
import cv2 as cv
import argparse
from canny import tracking_realtime_img_grad,tracking_realtime_canny,match_temp_ball,match_temp_robot
from hough import hough_circle,hough_line

def nothing(x):
    print(x)

def resize(image,scl):
    w_image = int(image.shape[1] * scl / 100)
    h_image = int(image.shape[0] * scl / 100)
    dim_image = (w_image,h_image)
    image_resized = cv.resize(image, dim_image, interpolation = cv.INTER_AREA)
    return image_resized


def erosion(frame,iter):
    kernel = np.ones((3,3),np.uint8)
    erosion = cv.erode(frame,kernel,iterations = iter)
    return erosion

def dilation(frame,iter):
    kernel = np.ones((3,3),np.uint8)
    dilation = cv.dilate(frame,kernel,iterations = iter)
    return dilation

def opening(frame,iter):
    frame=erosion(frame,iter)
    frame=dilation(frame,iter)
    return frame

def closing(frame,iter):    
    frame=dilation(frame,iter)
    frame=erosion(frame,iter)
    return frame

def limit_area_to_field(frame):
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        sensitivity = 15
        lower_hsv_field=np.array([60 - sensitivity, 100, 50])
        higher_hsv_field=np.array([60 + sensitivity, 255, 255]) 
        mask_field = cv.inRange(hsv, lower_hsv_field, higher_hsv_field)
        #field range
        field_limits = cv.bitwise_and(frame, frame, mask=mask_field)
        field_limits = cv.cvtColor(field_limits, cv.COLOR_BGR2GRAY)
        ret, field_limits = cv.threshold(field_limits, 80, 255, 0)
        field_limits = opening(field_limits,1)
        field_limits = cv.GaussianBlur(field_limits,(7,7),cv.BORDER_DEFAULT)
        contours,_ = cv.findContours(field_limits, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnt = contours[-1]
        x,y,w,h = cv.boundingRect(cnt)
        #
        #mask for area of interest
        black = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)
        black = cv.rectangle(black,(0,y), (frame.shape[1], y+h-10),(255, 255, 255), -1)   #---the dimension of the ROI
        gray = cv.cvtColor(black,cv.COLOR_BGR2GRAY)               #---converting to gray
        ret,b_mask = cv.threshold(gray,127,255, 0)
        fin = cv.bitwise_and(frame,frame,mask = b_mask)
        return fin



def tracking_realtime_filtered():
    cap = cv.VideoCapture('cambada_video.mp4')
    cv.namedWindow('Original vs Filtered')

    switch = '(1)tracking_all  (2)tracking_ball (3)tracking_blue_team (4)tracking_orange_team (5)tracking_lines'
    cv.createTrackbar(switch, 'Original vs Filtered', 1, 5, nothing)

    while(True):
        ret, frame = cap.read()
        frame=resize(frame,45)

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        s = cv.getTrackbarPos(switch, 'Original vs Filtered')

        #tracking_ball
        lower_hsv_ball = np.array([22, 77, 88])
        higher_hsv_ball = np.array([41, 254, 255])
        mask_ball = cv.inRange(hsv, lower_hsv_ball, higher_hsv_ball)
        #tracking_blue_team()
        lower_hsv_blue = np.array([88, 90, 46])
        higher_hsv_blue = np.array([106, 255, 255])
        mask_blue = cv.inRange(hsv, lower_hsv_blue, higher_hsv_blue)
        #tracking_orange_team()
        lower_hsv_orange = np.array([0, 89, 0])
        higher_hsv_orange = np.array([20, 255, 196])
        mask_orange = cv.inRange(hsv, lower_hsv_orange, higher_hsv_orange)
        #tracking_lines()
        lower_hsv_lines = np.array([0, 0, 162])
        higher_hsv_lines = np.array([179, 49, 255])
        mask_lines = cv.inRange(hsv, lower_hsv_lines, higher_hsv_lines)
        


        fin = limit_area_to_field(frame)

        if s==1:
            mask  = mask_ball+mask_blue+mask_lines+mask_orange
            frame_filtered = cv.bitwise_or(fin, fin, mask=mask)
            frame_filtered = cv.cvtColor(frame_filtered, cv.COLOR_BGR2GRAY)
            ret, frame_filtered = cv.threshold(frame_filtered, 90, 255, 0)
            frame_filtered = cv.GaussianBlur(frame_filtered,(5,5),cv.BORDER_DEFAULT)

        elif s==2:   
            frame_filtered = cv.bitwise_and(fin, fin, mask=mask_ball)
            frame_filtered = cv.cvtColor(frame_filtered, cv.COLOR_BGR2GRAY)
            ret, frame_filtered = cv.threshold(frame_filtered, 127, 255, 0)        
            frame_filtered = opening(frame_filtered,1)
            frame_filtered = cv.GaussianBlur(frame_filtered,(7,7),cv.BORDER_DEFAULT)

        elif s==3:   
            frame_filtered = cv.bitwise_and(fin, fin, mask=mask_blue)
            frame_filtered = cv.cvtColor(frame_filtered, cv.COLOR_BGR2GRAY)
            ret, frame_filtered = cv.threshold(frame_filtered, 127, 255, 0)
            frame_filtered = cv.GaussianBlur(frame_filtered,(5,5),cv.BORDER_DEFAULT)

        elif s==4:
            frame_filtered = cv.bitwise_and(fin, fin, mask=mask_orange)
            frame_filtered = cv.cvtColor(frame_filtered, cv.COLOR_BGR2GRAY)
            ret, frame_filtered = cv.threshold(frame_filtered, 90, 255, 0)
            frame_filtered = cv.GaussianBlur(frame_filtered,(7,7),cv.BORDER_DEFAULT)

        elif s==5:
            frame_filtered = cv.bitwise_and(fin, fin, mask=mask_lines)
            frame_filtered = cv.cvtColor(frame_filtered, cv.COLOR_BGR2GRAY)
            ret, frame_filtered = cv.threshold(frame_filtered, 127, 255, 0)
            frame_filtered = closing(frame_filtered,1)
            frame_filtered = opening(frame_filtered,1)
            frame_filtered = cv.GaussianBlur(frame_filtered,(7,7),cv.BORDER_DEFAULT)

        
        contours, hierarchy = cv.findContours(frame_filtered, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if s!=5 and s!=1:
            for i in range(len(contours)):
                x,y,w,h = cv.boundingRect(contours[i])
                frame = cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        else:
            cv.drawContours(frame, contours, -1, (0,255,0), 3)
        cv.imshow('Original vs Filtered', frame)

        
        if(cv.waitKey(1) & 0xFF == ord('q')):
            break

def contours_apply(frame):
	_,contours,_ = cv.findContours(frame, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
	cv.drawContours(frame, contours, -1, (0,255,0), 3)
	cv.imshow('something',frame)


parser = argparse.ArgumentParser()

parser.add_argument("-r","--trackingrealtimefiltered", help="Complete tracking in real time using contours",
                    action="store_true")
parser.add_argument("-g","--trackingrealtimeimggrad", help="Tracking in real time using gradient",
                    action="store_true")
parser.add_argument("-m","--matchtempball", help="Template match ball",
                    action="store_true")
parser.add_argument("-t","--matchtemprobot", help="Template match robot",
                    action="store_true")
parser.add_argument("-y","--canny", help="Apply canny detection",
                    action="store_true")
parser.add_argument("-c","--houghcircle", help="Apply hough methot with circle",
                    action="store_true")
parser.add_argument("-l","--houghline", help="Apply hough methot with lines",action="store_true")

                  

args = parser.parse_args()


if args.trackingrealtimefiltered:
    tracking_realtime_filtered()
elif args.trackingrealtimeimggrad:
    tracking_realtime_img_grad()
elif args.matchtempball:
    match_temp_ball()
elif args.matchtemprobot:
    match_temp_robot()
elif args.canny:
    tracking_realtime_canny()
elif args.houghcircle:
    hough_circle()
elif args.houghline:
    hough_line()

