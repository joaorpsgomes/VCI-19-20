import cv2 
import numpy as np
import argparse

def nothing(x):
    print(x)

def resize(image,scl):
    w_image = int(image.shape[1] * scl / 100)
    h_image = int(image.shape[0] * scl / 100)
    dim_image = (w_image,h_image)
    image_resized = cv2.resize(image, dim_image, interpolation = cv2.INTER_AREA)
    return image_resized

def erosion(frame,iter):
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(frame,kernel,iterations = iter)
    return erosion

def dilation(frame,iter):
    kernel = np.ones((5,5),np.uint8)
    dilation = cv2.dilate(frame,kernel,iterations = iter)
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
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sensitivity = 15
        lower_hsv_field=np.array([60 - sensitivity, 100, 50])
        higher_hsv_field=np.array([60 + sensitivity, 255, 255])
        mask_field = cv2.inRange(hsv, lower_hsv_field, higher_hsv_field)
        #field range
        field_limits = cv2.bitwise_and(frame, frame, mask=mask_field)
        field_limits = cv2.cvtColor(field_limits, cv2.COLOR_BGR2GRAY)
        ret, field_limits = cv2.threshold(field_limits, 80, 255, 0)
        field_limits = opening(field_limits,1)
        field_limits = cv2.GaussianBlur(field_limits,(7,7),cv2.BORDER_DEFAULT)
        contours,_ = cv2.findContours(field_limits, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[-1]
        x,y,w,h = cv2.boundingRect(cnt)
        #
        #mask for area of interest
        black = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)
        black = cv2.rectangle(black,(0,y), (frame.shape[1], frame.shape[0]),(255, 255, 255), -1)   #---the dimension of the ROI
        gray = cv2.cvtColor(black,cv2.COLOR_BGR2GRAY)               #---converting to gray
        ret,b_mask = cv2.threshold(gray,127,255, 0)
        fin = cv2.bitwise_and(frame,frame,mask = b_mask)
        #cv.imshow('Original vs Filtered', fin)
        return fin

def tracking_realtime_img_grad():
    cap = cv2.VideoCapture('cambada_video.mp4')

    while(True):
        ret, frame = cap.read()
        frame=resize(frame,50)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        #tracking_ball
        lower_hsv_ball = np.array([22, 114, 88])
        higher_hsv_ball = np.array([41, 254, 255])
        mask_ball = cv2.inRange(hsv, lower_hsv_ball, higher_hsv_ball)
        #tracking_blue_team()
        lower_hsv_blue = np.array([91, 78, 46])
        higher_hsv_blue = np.array([101, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_hsv_blue, higher_hsv_blue)
        #tracking_orange_team()
        lower_hsv_orange = np.array([0, 89, 0])
        higher_hsv_orange = np.array([20, 255, 196])
        mask_orange = cv2.inRange(hsv, lower_hsv_orange, higher_hsv_orange)
        #tracking_lines()
        lower_hsv_lines = np.array([0, 0, 162])
        higher_hsv_lines = np.array([179, 49, 255])
        mask_lines = cv2.inRange(hsv, lower_hsv_lines, higher_hsv_lines)

        frame = limit_area_to_field(frame)
        
        mask  = mask_ball+mask_blue+mask_lines+mask_orange
        frame_filtered = cv2.bitwise_or(frame, frame, mask=mask)
        frame_filtered = cv2.cvtColor(frame_filtered, cv2.COLOR_BGR2GRAY) 
        sobelx = cv2.Sobel(frame_filtered,cv2.CV_64F,1,0,ksize=5)
        sobely = cv2.Sobel(frame_filtered,cv2.CV_64F,0,1,ksize=5)
        scharrx = cv2.Scharr(frame_filtered, cv2.CV_64F,1,0)
        scharry = cv2.Scharr(frame_filtered, cv2.CV_64F,0,1)

        cv2.imshow('Original', frame_filtered)
        cv2.imshow('sobelx', sobelx)
        cv2.imshow('sobely', sobely)
        cv2.imshow('scharrx', scharrx)
        cv2.imshow('scharry', scharry)
        

        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break


def tracking_realtime_laplace():
    cap = cv2.VideoCapture('cambada_video.mp4')
    cv2.namedWindow('Original vs Filtered')

    switch = '(1)tracking_all  (2)tracking_ball (3)tracking_blue_team (4)tracking_orange_team (5)tracking_lines'
    cv2.createTrackbar(switch, 'Original vs Filtered', 1, 5, nothing)

    while(True):
        ret, frame = cap.read()
        frame=resize(frame,75)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        s = cv2.getTrackbarPos(switch, 'Original vs Filtered')

        #tracking_ball
        lower_hsv_ball = np.array([22, 114, 88])
        higher_hsv_ball = np.array([41, 254, 255])
        mask_ball = cv2.inRange(hsv, lower_hsv_ball, higher_hsv_ball)
        #tracking_blue_team()
        lower_hsv_blue = np.array([91, 78, 46])
        higher_hsv_blue = np.array([101, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_hsv_blue, higher_hsv_blue)
        #tracking_orange_team()
        lower_hsv_orange = np.array([0, 89, 0])
        higher_hsv_orange = np.array([20, 255, 196])
        mask_orange = cv2.inRange(hsv, lower_hsv_orange, higher_hsv_orange)
        #tracking_lines()
        lower_hsv_lines = np.array([0, 0, 162])
        higher_hsv_lines = np.array([179, 49, 255])
        mask_lines = cv2.inRange(hsv, lower_hsv_lines, higher_hsv_lines)

        frame = limit_area_to_field(frame)
        
        if s==1:
            mask  = mask_ball+mask_blue+mask_lines+mask_orange
            frame_filtered = cv2.bitwise_or(frame, frame, mask=mask)
            frame_filtered = cv2.cvtColor(frame_filtered, cv2.COLOR_BGR2GRAY)
            frame_filtered = cv2.Laplacian(frame_filtered, cv2.CV_64F)
            frame_filtered = np.uint8(np.absolute(frame_filtered))  

        elif s==2:   
            frame_filtered = cv2.bitwise_and(frame, frame, mask=mask_ball)
            frame_filtered = cv2.cvtColor(frame_filtered, cv2.COLOR_BGR2GRAY)
            frame_filtered = cv2.Laplacian(frame_filtered, cv2.CV_64F)
            frame_filtered = np.uint8(np.absolute(frame_filtered))       

        elif s==3:   
            frame_filtered = cv2.bitwise_and(frame, frame, mask=mask_blue)
            frame_filtered = cv2.cvtColor(frame_filtered, cv2.COLOR_BGR2GRAY)
            frame_filtered = cv2.Laplacian(frame_filtered, cv2.CV_64F)
            frame_filtered = np.uint8(np.absolute(frame_filtered))   

        elif s==4:
            frame_filtered = cv2.bitwise_and(frame, frame, mask=mask_orange)
            frame_filtered = cv2.cvtColor(frame_filtered, cv2.COLOR_BGR2GRAY)
            frame_filtered = cv2.Laplacian(frame_filtered, cv2.CV_64F)
            frame_filtered = np.uint8(np.absolute(frame_filtered))

        elif s==5:
            frame_filtered = cv2.bitwise_and(frame, frame, mask=mask_lines)
            frame_filtered = cv2.cvtColor(frame_filtered, cv2.COLOR_BGR2GRAY)
            frame_filtered = cv2.Laplacian(frame_filtered, cv2.CV_64F)
            frame_filtered = np.uint8(np.absolute(frame_filtered))   

        # show thresholded image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        numpy_horizontal = np.hstack((frame, frame_filtered))
        cv2.imshow('Original vs Filtered', numpy_horizontal)

        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break

def tracking_realtime_canny():
    cap = cv2.VideoCapture('cambada_video.mp4')
    cv2.namedWindow('Original vs Filtered')

    switch = '(1)tracking_all  (2)tracking_ball (3)tracking_blue_team (4)tracking_orange_team (5)tracking_lines'
    cv2.createTrackbar(switch, 'Original vs Filtered', 1, 5, nothing)

    while(True):
        ret, frame = cap.read()
        frame=resize(frame,75)

        cv2.imshow('Orig', frame)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        s = cv2.getTrackbarPos(switch, 'Original vs Filtered')

     #tracking_ball
        lower_hsv_ball = np.array([22, 114, 88])
        higher_hsv_ball = np.array([41, 254, 255])
        mask_ball = cv2.inRange(hsv, lower_hsv_ball, higher_hsv_ball)
        #tracking_blue_team()
        lower_hsv_blue = np.array([91, 78, 46])
        higher_hsv_blue = np.array([101, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_hsv_blue, higher_hsv_blue)
        #tracking_orange_team()
        lower_hsv_orange = np.array([0, 89, 0])
        higher_hsv_orange = np.array([20, 255, 196])
        mask_orange = cv2.inRange(hsv, lower_hsv_orange, higher_hsv_orange)
        #tracking_lines()
        lower_hsv_lines = np.array([0, 0, 162])
        higher_hsv_lines = np.array([179, 49, 255])
        mask_lines = cv2.inRange(hsv, lower_hsv_lines, higher_hsv_lines)

        frame = limit_area_to_field(frame)
        
        if s==1:
            mask  = mask_ball+mask_blue+mask_lines+mask_orange
            frame_filtered = cv2.bitwise_or(frame, frame, mask=mask)
            frame_filtered = cv2.cvtColor(frame_filtered, cv2.COLOR_BGR2GRAY)
            canny = cv2.Canny(frame_filtered,100,200,L2gradient=True) 

        elif s==2:   
            frame_filtered = cv2.bitwise_and(frame, frame, mask=mask_ball)
            frame_filtered = cv2.cvtColor(frame_filtered, cv2.COLOR_BGR2GRAY)
            canny = cv2.Canny(frame_filtered,150,200,L2gradient=True)

        elif s==3:   
            frame_filtered = cv2.bitwise_and(frame, frame, mask=mask_blue)
            frame_filtered = cv2.cvtColor(frame_filtered, cv2.COLOR_BGR2GRAY)
            canny = cv2.Canny(frame_filtered,100,200,L2gradient=True)

        elif s==4:
            frame_filtered = cv2.bitwise_and(frame, frame, mask=mask_orange)
            frame_filtered = cv2.cvtColor(frame_filtered, cv2.COLOR_BGR2GRAY)
            canny = cv2.Canny(frame_filtered,100,200,L2gradient=True)

        elif s==5:
            frame_filtered = cv2.bitwise_and(frame, frame, mask=mask_lines)
            frame_filtered = cv2.cvtColor(frame_filtered, cv2.COLOR_BGR2GRAY)
            canny = cv2.Canny(frame_filtered,100,200,L2gradient=True)  

        cv2.imshow('With Contours', canny)

        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break

def match_temp_ball():
    template = cv2.imread('1.png')
    cv2.imshow('template', template)
    t1 = cv2.imread('1.png', 0)

    cap = cv2.VideoCapture('cambada_2.mp4')

    while(True):
        ret, frame = cap.read()
        frame = resize(frame,40)

        framegray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        w, h = t1.shape[::-1]
        res1 = cv2.matchTemplate(framegray, t1, cv2.TM_CCOEFF_NORMED)

        threshold = 0.8
        loc = np.where(res1 >= threshold)

        for pt in zip(*loc[::-1]):
            cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0,255,0), 2)

        cv2.imshow('detected', frame)

        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break

def match_temp_robot():
    template = cv2.imread('3.png')
    cv2.imshow('template', template)
    t2 = cv2.imread('3.png', 0)

    cap = cv2.VideoCapture('cambada_2.mp4')

    while(True):
        ret, frame = cap.read()
        frame = resize(frame,40)

        framegray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        w, h = t2.shape[::-1]
        res1 = cv2.matchTemplate(framegray, t2, cv2.TM_CCOEFF_NORMED)

        threshold = 0.75
        loc = np.where(res1 >= threshold)

        for pt in zip(*loc[::-1]):
            cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0,255,0), 2)

        cv2.imshow('detected', frame)

        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break