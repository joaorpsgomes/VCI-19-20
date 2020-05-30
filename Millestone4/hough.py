import cv2 
import numpy as np

def nothing(x):
    print(x)


def erosion(frame,iter):
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(frame,kernel,iterations = iter)
    return erosion

def dilation(frame,iter):
    kernel = np.ones((3,3),np.uint8)
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

def resize(image,scl):
    w_image = int(image.shape[1] * scl / 100)
    h_image = int(image.shape[0] * scl / 100)
    dim_image = (w_image,h_image)
    image_resized = cv2.resize(image, dim_image, interpolation = cv2.INTER_AREA)
    return image_resized

'''
    Perform ball detection using the Hough transform
'''
def hough_circle():
    cap = cv2.VideoCapture('cambada_video.mp4')
    cv2.namedWindow('image')

    while(1):
        ret, frame = cap.read()
        frame=resize(frame,100)
        frame_ori=limit_area_to_field(frame)                    # limit interest area
        hsv = cv2.cvtColor(frame_ori, cv2.COLOR_BGR2HSV)        # convert BRG to HSV
        lower_hsv_ball = np.array([22, 77, 88])
        higher_hsv_ball = np.array([41, 254, 255])
        mask_ball = cv2.inRange(hsv, lower_hsv_ball, higher_hsv_ball)       # range HSV level

        img = cv2.bitwise_and(frame_ori, frame_ori, mask=mask_ball)         # merge between frame and ball mask
        kernel = np.ones((5,5),np.uint8)
        #img = cv2.dilate(img,kernel,iterations = 1)
        #img = cv2.erode(img,kernel,iterations = 1)
        #cv2.imshow('erosion',img)


        cimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)                         # convert frame BRG to GRAY
        #cimg = cv2.medianBlur(cimg,5)
        thresh = cv2.threshold(cimg, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Aply hough circles methods
        circles = cv2.HoughCircles(thresh,cv2.HOUGH_GRADIENT,3,100,param1=50,param2=30,minRadius=1,maxRadius=25)

        '''
        HOUGH_GRADIENT: Define the detection method. Currently this is the only one available in OpenCV.
        dp =: The inverse ratio of resolution.
        min_dist = gray.rows/16: Minimum distance between detected centers.
        param_1 =: Upper threshold for the internal Canny edge detector.
        param_2 = : Threshold for center detection.
        min_radius =: Minimum radius to be detected. If unknown, put zero as default.
        max_radius =: Maximum radius to be detected. If unknown, put zero as default.
        '''

        # draw circles in correspond position
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x,y,r) in circles:
                cv2.circle(frame, (x,y), r, (36,255,12), 3)

        cv2.imshow('image', frame)
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break

    cv2.destroyAllWindows()

'''
    Perform line detection using the Hough transform    
'''
def hough_line():
    cap = cv2.VideoCapture('cambada_video.mp4')
    #cv2.namedWindow('Result')
    while(1):
        ret, frame_real = cap.read()
        frame_real=resize(frame_real,80)
        frame_ori=limit_area_to_field(frame_real)

        hsv = cv2.cvtColor(frame_ori, cv2.COLOR_BGR2HSV)
        lower_hsv_lines = np.array([0, 0, 162])
        higher_hsv_lines = np.array([179, 49, 255])
        mask_lines = cv2.inRange(hsv, lower_hsv_lines, higher_hsv_lines)                # range HSV level

        frame = cv2.bitwise_and(frame_ori, frame_ori, mask=mask_lines)                  # merge between frame and ball mask
        #kernel = np.ones((5,5),np.uint8)
        #frame = cv2.dilate(frame,kernel,iterations = 1)
        #frame = cv2.erode(frame,kernel,iterations = 1)


        #cv2.imshow('frame_with_mask', frame)
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)                                   # convert BRG to GRAY
        edges = cv2.Canny(gray,50,150,apertureSize = 3)                                 # aplly canny edges 
        lines = cv2.HoughLinesP(edges,10,np.pi/180,10,minLineLength=0,maxLineGap=0)
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(frame_real,(x1,y1),(x2,y2),(0,255,0),2)

        cv2.imshow('result', frame_real)

        if(cv2.waitKey(1) & 0xFF == ord('q')):
                break