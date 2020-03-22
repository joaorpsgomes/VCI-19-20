import cv2 
import numpy as np




def resize(image,scl):
    w_image = int(image.shape[1] * scl / 100)
    h_image = int(image.shape[0] * scl / 100)
    dim_image = (w_image,h_image)
    image_resized = cv2.resize(image, dim_image, interpolation = cv2.INTER_AREA)
    return image_resized

def nothing(x):
    print(x)

def init_trackbar():
    cv2.namedWindow('image')
    ilowH = 0
    ihighH = 180

    ilowS = 0
    ihighS = 255
    ilowV = 0
    ihighV = 255

    # create trackbars for color change
    cv2.createTrackbar('lowH','image',ilowH,179,nothing)
    cv2.createTrackbar('highH','image',ihighH,179,nothing)

    cv2.createTrackbar('lowS','image',ilowS,255,nothing)
    cv2.createTrackbar('highS','image',ihighS,255,nothing)

    cv2.createTrackbar('lowV','image',ilowV,255,nothing)
    cv2.createTrackbar('highV','image',ihighV,255,nothing)

def trackbar_realtime():
    while(True):
        img=cv2.imread('cambada_image.jpg',1) 
        frame=resize(img,60)
        
        ilowH = cv2.getTrackbarPos('lowH', 'image')
        ihighH = cv2.getTrackbarPos('highH', 'image')
        ilowS = cv2.getTrackbarPos('lowS', 'image')
        ihighS = cv2.getTrackbarPos('highS', 'image')
        ilowV = cv2.getTrackbarPos('lowV', 'image')
        ihighV = cv2.getTrackbarPos('highV', 'image')

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array([ilowH, ilowS, ilowV])
        higher_hsv = np.array([ihighH, ihighS, ihighV])
        mask = cv2.inRange(hsv, lower_hsv, higher_hsv)

        frame = cv2.bitwise_and(frame, frame, mask=mask)

        # show thresholded image
        cv2.imshow('image', frame)


        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break

def tracking_ball():
    
    while(True):
        frame=cv2.imread('cambada_image.jpg',1) 
        frame=resize(frame,60)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array([22, 77, 88])
        higher_hsv = np.array([41, 254, 255])
        mask = cv2.inRange(hsv, lower_hsv, higher_hsv)

        frame = cv2.bitwise_and(frame, frame, mask=mask)

        # show thresholded image
        cv2.imshow('image', frame)


        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break

def tracking_blue_team():
    while(True):
        frame=cv2.imread('cambada_image.jpg',1) 
        frame=resize(frame,60)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array([88, 90, 46])
        higher_hsv = np.array([106, 255, 255])
        mask = cv2.inRange(hsv, lower_hsv, higher_hsv)

        frame = cv2.bitwise_and(frame, frame, mask=mask)

        # show thresholded image
        cv2.imshow('image', frame)


        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break

def tracking_orange_team():
    while(True):
        frame=cv2.imread('cambada_image.jpg',1) 
        frame=resize(frame,60)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array([0, 89, 0])
        higher_hsv = np.array([20, 255, 196])
        mask = cv2.inRange(hsv, lower_hsv, higher_hsv)

        frame = cv2.bitwise_and(frame, frame, mask=mask)

        # show thresholded image
        cv2.imshow('image', frame)


        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break


def tracking_lines():
    while(True):
        frame=cv2.imread('cambada_image.jpg',1) 
        frame=resize(frame,60)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #lower_hsv = np.array([0, 0, 0])
        #higher_hsv = np.array([179, 33, 255])
        lower_hsv = np.array([0, 0, 162])
        higher_hsv = np.array([179, 49, 255])
        mask = cv2.inRange(hsv, lower_hsv, higher_hsv)

        frame = cv2.bitwise_and(frame, frame, mask=mask)

        # show thresholded image
        cv2.imshow('image', frame)


        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break

def tracking_realtime():
    #init_trackbar()
    #trackbar_realtime()
    cap = cv2.VideoCapture('cambada_video.mp4')
    cv2.namedWindow('image')

    switch = '(1)tracking_ball  (2)tracking_blue_team (3)tracking_orange_team (4)tracking_lines'
    cv2.createTrackbar(switch, 'image', 1, 5, nothing)

    while(True):
        ret, frame = cap.read()
        frame=resize(frame,60)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        s = cv2.getTrackbarPos(switch, 'image')

        #tracking_ball
        lower_hsv_ball = np.array([22, 77, 88])
        higher_hsv_ball = np.array([41, 254, 255])
        mask_ball = cv2.inRange(hsv, lower_hsv_ball, higher_hsv_ball)
        #tracking_blue_team()
        lower_hsv_blue = np.array([88, 90, 46])
        higher_hsv_blue = np.array([106, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_hsv_blue, higher_hsv_blue)
        #tracking_orange_team()
        lower_hsv_orange = np.array([0, 89, 0])
        higher_hsv_orange = np.array([20, 255, 196])
        mask_orange = cv2.inRange(hsv, lower_hsv_orange, higher_hsv_orange)
        #tracking_lines()
        lower_hsv_lines = np.array([0, 0, 162])
        higher_hsv_lines = np.array([179, 49, 255])
        mask_lines = cv2.inRange(hsv, lower_hsv_lines, higher_hsv_lines)
        
        if s==1:
            mask = mask_ball+mask_blue+mask_lines+mask_orange
            frame = cv2.bitwise_or(frame, frame, mask=mask)
        elif s==2:   
            frame = cv2.bitwise_and(frame, frame, mask=mask_ball)
        elif s==3:   
            frame = cv2.bitwise_and(frame, frame, mask=mask_blue)
        elif s==4:
            frame = cv2.bitwise_and(frame, frame, mask=mask_orange)
        elif s==5:
            frame = cv2.bitwise_and(frame, frame, mask=mask_lines)
        

        #frame = cv2.bitwise_and(frame, frame, mask=mask_ball)
        #frame = cv2.bitwise_and(frame, frame, mask=mask_blue)
        #frame = cv2.bitwise_and(frame, frame, mask=mask_orange)
        #frame = cv2.bitwise_and(frame, frame, mask=mask_lines)

        # show thresholded image
        cv2.imshow('image', frame)


        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break
        
        #return frame


def tracking_realtime_gray():

    kernel = np.ones((1,1),np.uint8)
    

    while(True):
        frame=tracking_realtime()
        #img=cv2.imread('img_test.png',1)
        #img=resize(img,60)

        gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

        cv2.imwrite('filtred.jpg',closing)
        cv2.imshow('filtred',closing)

        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break


"""
init_trackbar()
trackbar_realtime()

tracking_ball()
tracking_blue_team()
tracking_orange_team()
tracking_lines()
"""
tracking_realtime()

#tracking_realtime_gray()




    
cv2.destroyAllWindows()