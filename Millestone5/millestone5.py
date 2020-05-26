import numpy as np
import cv2 as cv
import math
import argparse


def nothing(x):
    print()

def resize(image,scl):
    w_image = int(image.shape[1] * scl / 100)
    h_image = int(image.shape[0] * scl / 100)
    dim_image = (w_image,h_image)
    image_resized = cv.resize(image, dim_image, interpolation = cv.INTER_AREA)
    return image_resized


def __draw_label(img, text, pos, bg_color):
    font_face = cv.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    color = (0, 0, 0)
    thickness = cv.FILLED
    margin = 2
    txt_size = cv.getTextSize(text, font_face, scale, thickness)
    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin
    cv.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
    cv.putText(img, text, pos, font_face, scale, color, 1, cv.LINE_AA)


def limit_area_to_field(frame):
    
        #mask for area of interest
        black = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)
        black = cv.rectangle(black,(0,52), (426, frame.shape[1]),(255, 255, 255), -1)   #---the dimension of the ROI
        gray = cv.cvtColor(black,cv.COLOR_BGR2GRAY)               #---converting to gray
        ret,b_mask = cv.threshold(gray,127,255, 0)
        fin = cv.bitwise_and(frame,frame,mask = b_mask)
        return fin

def pixel2meter(pixels):
    meter_per_pixel=0.025
    meter=pixels*meter_per_pixel
    return meter


def tracking_with_ID():
    cap = cv.VideoCapture('cambada_2.mp4')
    cv.namedWindow('Result')
    ret, frame = cap.read()
    frame=resize(frame,25)
    height, width = frame.shape[:2]
	            
    while(True):
        ret, frame = cap.read()
        if ret==True:
            frame=resize(frame,25)
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            #tracking_ball
            lower_hsv_ball = np.array([22, 77, 88])
            higher_hsv_ball = np.array([41, 254, 255])
            mask_ball = cv.inRange(hsv, lower_hsv_ball, higher_hsv_ball)
            #tracking_blue_team()
            lower_hsv_blue = np.array([80, 90, 46])
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

            mask  = mask_ball+mask_orange
            frame_filtered = cv.bitwise_or(fin, fin, mask=mask)
            frame_filtered = cv.cvtColor(frame_filtered, cv.COLOR_BGR2GRAY)
            ret, frame_filtered = cv.threshold(frame_filtered, 90, 255, 0)
            frame_filtered = cv.GaussianBlur(frame_filtered,(7,7),cv.BORDER_DEFAULT)
            #cv.imshow('filtred',frame_filtered)
            
            frame_filtered_robot = cv.bitwise_and(fin, fin, mask=mask_blue)
            frame_filtered_robot = cv.cvtColor(frame_filtered_robot, cv.COLOR_BGR2GRAY)
            ret, frame_filtered_robot = cv.threshold(frame_filtered_robot, 127, 255, 0)
            frame_filtered_robot = cv.GaussianBlur(frame_filtered_robot,(7,7),cv.BORDER_DEFAULT)
        
            
            
            contours, hierarchy = cv.findContours(frame_filtered, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            contours_robot, hierarchy_robot = cv.findContours(frame_filtered_robot, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            l=len(contours)
            #print(l)
            for i in range(len(contours)):
                x,y,w,h = cv.boundingRect(contours[i])

                if l==1:
                    __draw_label(frame, 'Ball 1', (x,y-3), (255,0,0))
                if l==2:
                    if i==0:
                        __draw_label(frame, 'Ball 2', (x,y-3), (255,0,0))
                    if i==1:
                        __draw_label(frame, 'Ball 1', (x,y-3), (255,0,0))
                if l==3:
                    if i==0:
                        __draw_label(frame, 'Ball 3', (x,y-3), (255,0,0))
                    if i==1:
                        __draw_label(frame, 'Ball 2', (x,y-3), (255,0,0))
                    if i==2:
                        __draw_label(frame, 'Ball 1', (x,y-3), (255,0,0))
                if l==4:
                    if i==0:
                        __draw_label(frame, 'Ball 4', (x,y-3), (255,0,0))
                    if i==1:
                        __draw_label(frame, 'Ball 3', (x,y-3), (255,0,0))
                    if i==2:
                        __draw_label(frame, 'Ball 2', (x,y-3), (255,0,0))
                    if i==3:
                        __draw_label(frame, 'Ball 1', (x,y-3), (255,0,0))
                if l==5:
                    if i==0:
                        __draw_label(frame, 'Ball 5', (x,y-3), (255,0,0))
                    if i==1:
                        __draw_label(frame, 'Ball 4', (x,y-3), (255,0,0))
                    if i==2:
                        __draw_label(frame, 'Ball 3', (x,y-3), (255,0,0))
                    if i==3:
                        __draw_label(frame, 'Ball 2', (x,y-3), (255,0,0))
                    if i==4:
                        __draw_label(frame, 'Ball 1', (x,y-3), (255,0,0))
                if l==6:
                    if i==0:
                        __draw_label(frame, 'Ball 6', (x,y-3), (255,0,0))
                    if i==1:
                        __draw_label(frame, 'Ball 5', (x,y-3), (255,0,0))
                    if i==2:
                        __draw_label(frame, 'Ball 4', (x,y-3), (255,0,0))
                    if i==3:
                        __draw_label(frame, 'Ball 3', (x,y-3), (255,0,0))
                    if i==4:
                        __draw_label(frame, 'Ball 2', (x,y-3), (255,0,0))
                    if i==5:
                        __draw_label(frame, 'Ball 1', (x,y-3), (255,0,0))

            
                frame = cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            lc=len(contours_robot)
            for i in range(len(contours_robot)):
                x,y,w,h = cv.boundingRect(contours_robot[i])

                if lc==1:
                    __draw_label(frame, 'Robot 1', (x,y-3), (255,0,0))
                if lc>=2:
                    if i==0:
                        __draw_label(frame, 'Robot 2', (x,y-3), (255,0,0))
                    if i==1:
                        __draw_label(frame, 'Robot 1', (x,y-3), (255,0,0))
                frame = cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            #cv.drawContours(frame, contours, -1, (0,255,0), 3)
            cv.imshow('Result', frame)
            cv.waitKey(10)

        
            if(cv.waitKey(1) & 0xFF == ord('q')):
                break
        else:
            cap.release()
            out.release()
            cv.destroyAllWindows()
            break
        



def optical_flow():
    cap = cv.VideoCapture('cambada_2.mp4')
    ret, frame = cap.read()
    frame=resize(frame,25)
    height, width = frame.shape[:2]
        
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                        qualityLevel = 0.01,
                        minDistance = 1,
                        blockSize = 50)
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (30,30),
                    maxLevel = 1,
                    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0,255,(100,3))

    #Inicialization of variables
    ret, old_frame = cap.read()
    old_frame=resize(old_frame,25)
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    c=[[[1,1]]]
    c=np.array(c,np.float32)
    mask_of = np.zeros_like(old_frame)
    offset=14
    margin=5+offset
    x1=0
    x2=0
    x3=0
    x4=0
    x5=0
    x6=0
    dist_ball1=0
    dist_ball2=0
    dist_ball3=0
    dist_ball4=0
    dist_ball5=0
    dist_ball6=0
    dist_robot1=0
    dist_robot2=0
    

    while(True):
        c=[[[1,1]]]
        c=np.array(c)
        ret, frame = cap.read()
        if ret==True:
            frame=resize(frame,25)
            
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

            #tracking_ball
            lower_hsv_ball = np.array([22, 77, 88])
            higher_hsv_ball = np.array([41, 254, 255])
            mask_ball = cv.inRange(hsv, lower_hsv_ball, higher_hsv_ball)
            #tracking_blue_team()
            lower_hsv_blue = np.array([80, 90, 46])
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
            
            # mask to diferrent balls
            mask  = mask_ball+mask_orange
            frame_filtered = cv.bitwise_or(fin, fin, mask=mask)
            frame_filtered = cv.cvtColor(frame_filtered, cv.COLOR_BGR2GRAY)
            ret, frame_filtered = cv.threshold(frame_filtered, 90, 255, 0)
            frame_filtered = cv.GaussianBlur(frame_filtered,(7,7),cv.BORDER_DEFAULT)
            
            # mask to robots
            frame_filtered_robot = cv.bitwise_and(fin, fin, mask=mask_blue)
            frame_filtered_robot = cv.cvtColor(frame_filtered_robot, cv.COLOR_BGR2GRAY)
            ret, frame_filtered_robot = cv.threshold(frame_filtered_robot, 127, 255, 0)
            frame_filtered_robot = cv.GaussianBlur(frame_filtered_robot,(7,7),cv.BORDER_DEFAULT)
        
        
            contours, hierarchy = cv.findContours(frame_filtered, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            contours_robot, hierarchy_robot = cv.findContours(frame_filtered_robot, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            l=len(contours)
            for i in range(len(contours)):
                x,y,w,h = cv.boundingRect(contours[i])
                b=[[[np.float32(x+offset),np.float32(y+offset)]]]       # auxiliar array                
                c=np.append(c,b,axis=0)                                 # auxiliar array  
                c=np.array(c,np.float32)
                
                if l==1:
                    __draw_label(frame, 'Ball 1', (x,y-3), (255,0,0))
                    x1=x
                    y1=y
                if l==2:
                    if i==0:
                        __draw_label(frame, 'Ball 2', (x,y-3), (255,0,0))
                        x2=x
                        y2=y
                    if i==1:
                        __draw_label(frame, 'Ball 1', (x,y-3), (255,0,0))
                        x1=x
                        y1=y
                if l==3:
                    if i==0:
                        __draw_label(frame, 'Ball 3', (x,y-3), (255,0,0))
                        x3=x
                        y3=y
                    if i==1:
                        __draw_label(frame, 'Ball 2', (x,y-3), (255,0,0))
                        x2=x
                        y2=y
                    if i==2:
                        __draw_label(frame, 'Ball 1', (x,y-3), (255,0,0))
                        x1=x
                        y1=y
                if l==4:
                    if i==0:
                        __draw_label(frame, 'Ball 4', (x,y-3), (255,0,0))
                        x4=x
                        y4=y
                    if i==1:
                        __draw_label(frame, 'Ball 3', (x,y-3), (255,0,0))
                        x3=x
                        y3=y
                    if i==2:
                        __draw_label(frame, 'Ball 2', (x,y-3), (255,0,0))
                        x2=x
                        y2=y
                    if i==3:
                        __draw_label(frame, 'Ball 1', (x,y-3), (255,0,0))
                        x1=x
                        y1=y
                if l==5:
                    if i==0:
                        __draw_label(frame, 'Ball 5', (x,y-3), (255,0,0))
                        x5=x
                        y5=y
                    if i==1:
                        __draw_label(frame, 'Ball 4', (x,y-3), (255,0,0))
                        x4=x
                        y4=y
                    if i==2:
                        __draw_label(frame, 'Ball 3', (x,y-3), (255,0,0))
                        x3=x
                        y3=y
                    if i==3:
                        __draw_label(frame, 'Ball 2', (x,y-3), (255,0,0))
                        x2=x
                        y2=y
                    if i==4:
                        __draw_label(frame, 'Ball 1', (x,y-3), (255,0,0))
                        x1=x
                        y1=y
                if l==6:
                    if i==0:
                        __draw_label(frame, 'Ball 6', (x,y-3), (255,0,0))
                        x6=x
                        y6=y
                    if i==1:
                        __draw_label(frame, 'Ball 5', (x,y-3), (255,0,0))
                        x5=x
                        y5=y
                    if i==2:
                        __draw_label(frame, 'Ball 4', (x,y-3), (255,0,0))
                        x4=x
                        y4=y
                    if i==3:
                        __draw_label(frame, 'Ball 3', (x,y-3), (255,0,0))
                        x3=x
                        y3=y
                    if i==4:
                        __draw_label(frame, 'Ball 2', (x,y-3), (255,0,0))
                        x2=x
                        y2=y
                    if i==5:
                        __draw_label(frame, 'Ball 1', (x,y-3), (255,0,0))
                        x1=x
                        y1=y

            
                frame = cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            lc=len(contours_robot)
            for i in range(len(contours_robot)):
                x,y,w,h = cv.boundingRect(contours_robot[i])
                b=[[[np.float32(x+offset),np.float32(y+offset)]]]
                b=np.array(b,np.float32)
                c=np.append(c,b,axis=0)
                c=np.array(c)
                c=c.astype(float)

                if lc==1:
                    __draw_label(frame, 'Robot 1', (x,y-3), (255,0,0))
                    xr1=x
                    yr1=y
                if lc>=2:
                    if i==0:
                        __draw_label(frame, 'Robot 2', (x,y-3), (255,0,0))
                        xr2=x
                        yr2=y
                    if i==1:
                        __draw_label(frame, 'Robot 1', (x,y-3), (255,0,0))
                        xr1=x
                        yr1=y
                frame = cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

            #cv.drawContours(frame, contours, -1, (0,255,0), 3)
            cv.imshow('Original ', frame)
            #out.write(frame)


            p0=np.array(c,np.float32)
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # calculate optical flow
            p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            # Select good points
            good_new = p1[st==1]
            good_old = p0[st==1]
            # draw the tracks
            for i,(new,old) in enumerate(zip(good_new, good_old)):
                a,b = new.ravel()
                t,d = old.ravel()

                if t<=xr1+margin and t>=xr1-margin and d<=yr1+margin and d>=yr1-margin:
                    m=math.sqrt((a-t)*(a-t)+(b-d)*(b-d))
                    dist_robot1=dist_robot1+m
                    dist_robot1_meter=pixel2meter(dist_robot1)
                    print("Distance Robot1 : %f meters" % dist_robot1_meter)
                
                if t<=xr2+margin and t>=xr2-margin and d<=yr2+margin and d>=yr2-margin:
                    m=math.sqrt((a-t)*(a-t)+(b-d)*(b-d))
                    dist_robot2=dist_robot2+m
                    dist_robot2_meter=pixel2meter(dist_robot2)
                    print("Distance Robot2 : %f meters " % dist_robot2_meter)
                
                if t<=x1+margin and t>=x1-margin and d<=y1+margin and d>=y1-margin:
                    m=math.sqrt((a-t)*(a-t)+(b-d)*(b-d))
                    dist_ball1=dist_ball1+m
                    dist_ball1_meter=pixel2meter(dist_ball1)
                    print("Distance Ball1 : %f meters" % dist_ball1_meter)
                
                if t<=x2+margin and t>=x2-margin and d<=y2+margin and d>=y2-margin:
                    m=math.sqrt((a-t)*(a-t)+(b-d)*(b-d))
                    dist_ball2=dist_ball2+m
                    dist_ball2_meter=pixel2meter(dist_ball2)
                    print("Distance Ball2 : %f meters" % dist_ball2_meter)
                
                if t<=x3+margin and t>=x3-margin and d<=y3+margin and d>=y3-margin:
                    m=math.sqrt((a-t)*(a-t)+(b-d)*(b-d))
                    dist_ball3=dist_ball3+m
                    dist_ball3_meter=pixel2meter(dist_ball3)
                    print("Distance Ball3 : %f meters" % dist_ball3_meter)
                
                if t<=x4+margin and t>=x4-margin and d<=y4+margin and d>=y4-margin:
                    m=math.sqrt((a-t)*(a-t)+(b-d)*(b-d))
                    dist_ball4=dist_ball4+m
                    dist_ball4_meter=pixel2meter(dist_ball4)
                    print("Distance Ball4 : %f meters" % dist_ball4_meter)
                
                if t<=x5+margin and t>=x5-margin and d<=y5+margin and d>=y5-margin:
                    m=math.sqrt((a-t)*(a-t)+(b-d)*(b-d))
                    dist_ball5=dist_ball5+m
                    dist_ball5_meter=pixel2meter(dist_ball5)
                    print("Distance Ball5 : %f meters" % dist_ball5_meter)
                
                if t<=x6+margin and t>=x6-margin and d<=y6+margin and d>=y6-margin:
                    m=math.sqrt((a-t)*(a-t)+(b-d)*(b-d))
                    dist_ball6=dist_ball6+m
                    dist_ball6_meter=pixel2meter(dist_ball6)
                    print("Distance Ball6 : %f meters" % dist_ball6_meter)
               
                
                mask_of = cv.line(mask_of, (a,b),(t,d), color[i].tolist(), 2)
                frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)

            img = cv.add(frame,mask_of)
            cv.imshow('result',img)
            k = cv.waitKey(30) & 0xff
            cv.waitKey(30)
            old_gray = frame_gray.copy()

            if(cv.waitKey(1) & 0xFF == ord('q')):
                break
        else:
            cap.release()
            cv.destroyAllWindows()
            break
        






parser = argparse.ArgumentParser()

parser.add_argument("-t","--trackingID", help="Tracking balls and robots with ID",
                    action="store_true")
parser.add_argument("-p","--opticalflow", help="Tracking distance traveled by the ball and robots ",action="store_true")

                  
args = parser.parse_args()


if args.trackingID:
    tracking_with_ID()
elif args.opticalflow:
    optical_flow()
