import cv2 
import numpy as np
from matplotlib import pyplot as plt
import argparse


def resize(image,scl):
    w_image = int(image.shape[1] * scl / 100)
    h_image = int(image.shape[0] * scl / 100)
    dim_image = (w_image,h_image)
    image_resized = cv2.resize(image, dim_image, interpolation = cv2.INTER_AREA)
    return image_resized

def nothing(x):
    print(x)

def setMouseCallback():
    window_name = 'Detect_mouse'

    img = np.zeros((512,512,3), np.uint8)

    cv2.namedWindow(window_name)


    def events(event, x,y,flags, params):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            print('Left double click \n')
        elif event==cv2.EVENT_MOUSEMOVE and flags==cv2.EVENT_FLAG_CTRLKEY:
            print('Mouse move \n') 
        elif event==cv2.EVENT_LBUTTONDOWN:
            print('Left click \n') 
        elif event==cv2.EVENT_RBUTTONDOWN:
            print('Rigth click \n')
        elif event==cv2.EVENT_MBUTTONDOWN:
            print('Middle click\n')
        elif x==100 and y==100:
            print('X=100 and Y==100\n')


    cv2.setMouseCallback(window_name, events)


    while(True):
        cv2.imshow(window_name, img)
        if(cv2.waitKey(1) & 0xFF == ord('q')):
                break


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

    init_trackbar()

    while(True):
        img=cv2.imread('cambada_image.png',1) 
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

    cap = cv2.VideoCapture('cambada_video.mp4')
    cv2.namedWindow('image')

    switch = '(0) original_video (1)tracking_all  (2)tracking_ball (3)tracking_blue_team (4)tracking_orange_team (5)tracking_lines'
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




def tracking_realtime_filtered():
    cap = cv2.VideoCapture('cambada_video.mp4')
    cv2.namedWindow('Original vs Filtered')

    switch = '(1)tracking_all  (2)tracking_ball (3)tracking_blue_team (4)tracking_orange_team (5)tracking_lines'
    cv2.createTrackbar(switch, 'Original vs Filtered', 1, 5, nothing)

    while(True):
        ret, frame = cap.read()
        frame=resize(frame,60)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        s = cv2.getTrackbarPos(switch, 'Original vs Filtered')

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
            mask  = mask_ball+mask_blue+mask_lines+mask_orange
            frame_filtered = cv2.bitwise_or(frame, frame, mask=mask)
            frame_filtered = cv2.cvtColor(frame_filtered, cv2.COLOR_BGR2GRAY)
            frame_filtered = opening(frame_filtered,1)
            frame_filtered = erosion(frame_filtered,1)

        elif s==2:   
            frame_filtered = cv2.bitwise_and(frame, frame, mask=mask_ball)
            frame_filtered = cv2.cvtColor(frame_filtered, cv2.COLOR_BGR2GRAY)        
            frame_filtered = opening(frame_filtered,1)

        elif s==3:   
            frame_filtered = cv2.bitwise_and(frame, frame, mask=mask_blue)
            frame_filtered = cv2.cvtColor(frame_filtered, cv2.COLOR_BGR2GRAY)
            frame_filtered = opening(frame_filtered,1)

        elif s==4:
            frame_filtered = cv2.bitwise_and(frame, frame, mask=mask_orange)
            frame_filtered = cv2.cvtColor(frame_filtered, cv2.COLOR_BGR2GRAY)
            frame_filtered = opening(frame_filtered,1)

        elif s==5:
            frame_filtered = cv2.bitwise_and(frame, frame, mask=mask_lines)
            frame_filtered = cv2.cvtColor(frame_filtered, cv2.COLOR_BGR2GRAY)
            frame_filtered = closing(frame_filtered,1)
            frame_filtered = opening(frame_filtered,1)
            frame_filtered = erosion(frame_filtered,1)

        # show thresholded image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        numpy_horizontal = np.hstack((frame, frame_filtered))
        cv2.imshow('Original vs Filtered', numpy_horizontal)
        #cv2.imshow('image', frame)


        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break

#Image Segmentation with Distance Transform and Watershed Algorithm 

def simplest_thresholds():

    img = cv2.imread('coins.jpg')

    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
    ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
    ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
    ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)

    titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
    images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

    for i in range(6):
        plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])

    plt.show()


def seg_algorithms_img():

    while(1):

        img = cv2.imread('coins.jpg')

        cv2.imshow('Original Image',img)

        #We start with finding an approximate estimate of the coins. For that, we can use the Otsu's binarization. 

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        cv2.imshow('Image', thresh)

        #To remove any small holes in the object, we can use morphological closing.

        # noise removal
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

        cv2.imshow('Image with noise removal', opening)
        # sure background area
        sure_bg = cv2.dilate(opening,kernel,iterations=3)
        cv2.imshow('Sure Background area', sure_bg)
        # Finding sure foreground area
        #dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
        #ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
        sure_fg = cv2.erode(opening,kernel,iterations=3)
        cv2.imshow('Sure Foreground area', sure_fg)
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)
        cv2.imshow('Unknown region', unknown)

        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers+1
        # Now, mark the region of unknown with zero
        markers[unknown==255] = 0

        markers = cv2.watershed(img,markers)
        img[markers == -1] = [255,0,0]

        cv2.imshow('Result', img)

        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break

def region_growing():
    def get8n(x, y, shape):
        out = []
        maxx = shape[1]-1
        maxy = shape[0]-1

        #top left
        outx = min(max(x-1,0),maxx)
        outy = min(max(y-1,0),maxy)
        out.append((outx,outy))

        #top center
        outx = x
        outy = min(max(y-1,0),maxy)
        out.append((outx,outy))

        #top right
        outx = min(max(x+1,0),maxx)
        outy = min(max(y-1,0),maxy)
        out.append((outx,outy))

        #left
        outx = min(max(x-1,0),maxx)
        outy = y
        out.append((outx,outy))

        #right
        outx = min(max(x+1,0),maxx)
        outy = y
        out.append((outx,outy))

        #bottom left
        outx = min(max(x-1,0),maxx)
        outy = min(max(y+1,0),maxy)
        out.append((outx,outy))

        #bottom center
        outx = x
        outy = min(max(y+1,0),maxy)
        out.append((outx,outy))

        #bottom right
        outx = min(max(x+1,0),maxx)
        outy = min(max(y+1,0),maxy)
        out.append((outx,outy))

        return out

    def region_growing(img, seed):
        list = []
        outimg = np.zeros_like(img)
        list.append((seed[0], seed[1]))
        processed = []
        while(len(list) > 0):
            pix = list[0]
            outimg[pix[0], pix[1]] = 255
            for coord in get8n(pix[0], pix[1], img.shape):
                if img[coord[0], coord[1]] != 0:
                    outimg[coord[0], coord[1]] = 255
                    if not coord in processed:
                        list.append(coord)
                    processed.append(coord)
            list.pop(0)
            #cv2.imseg_algorithms_img()tKey(1)
        return outimg

    def on_mouse(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            print( 'Seed: ' + str(x) + ', ' + str(y), img[y,x])
            clicks.append((y,x))

    clicks = []
    image = cv2.imread('coins.jpg', 0)
    ret, img = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    cv2.namedWindow('Input')
    cv2.setMouseCallback('Input', on_mouse, 0, )
    cv2.imshow('Input', img)
    cv2.waitKey()
    seed = clicks[-1]
    out = region_growing(img, seed)
    cv2.imshow('Region Growing', out)
    cv2.waitKey()

cv2.destroyAllWindows()

parser = argparse.ArgumentParser()

parser.add_argument("-m","--MouseCallback", help="Test MouseCallBack",
                    action="store_true")
parser.add_argument("-t","--TrackbarRealTime", help="Trackbar in an image",
                    action="store_true")
parser.add_argument("-r","--TrackingRealTime", help="Tracks all game components in real time",
                    action="store_true")
parser.add_argument("-g","--TrackingRealTimeFiltered", help="Tracks all game components in real time in a grayscale",
                    action="store_true")
parser.add_argument("-s","--SimplestThresholds", help="Exploring the basic thresholds in an image",
                    action="store_true")
parser.add_argument("-a","--SegmentationAlgorithms", help="Exploring image Segmentation with Distance Transform(commented)/Erosion and Watershed Algorithm",
                    action="store_true")
parser.add_argument("-w","--RegionGrowing", help="Exploring Region Growing",
                    action="store_true")                   

args = parser.parse_args()


if args.MouseCallback:
	setMouseCallback()
elif args.TrackbarRealTime:
    trackbar_realtime()
elif args.TrackingRealTime:
    tracking_realtime()
elif args.TrackingRealTimeFiltered:
	tracking_realtime_filtered()
elif args.SimplestThresholds:
	simplest_thresholds()
elif args.SegmentationAlgorithms:
	seg_algorithms_img()
elif args.RegionGrowing:
	region_growing()