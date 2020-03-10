from __future__ import print_function
import numpy as np
import cv2
import time
import argparse




def changeColorSpace(colorspace):
	
	cap = cv2.VideoCapture(0)

	while(True):

		ret, frame= cap.read()

		cv2.imshow('frame',frame)
		if colorspace=='hsv':
			frame_2=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
			cv2.imshow('hsv',frame_2)
		if colorspace=='yuv':
			frame_3=cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
			cv2.imshow('yuv',frame_3)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()


def resize(image,scl):
    w_image = int(image.shape[1] * scl / 100)
    h_image = int(image.shape[0] * scl / 100)
    dim_image = (w_image,h_image)
    image_resized = cv2.resize(image, dim_image, interpolation = cv2.INTER_AREA)
    return image_resized

def estimateFrameRate(num_frame=30):
	cap = cv2.VideoCapture(0)
	
	start =time.time()

	for i in range(num_frame):
		_,frame =cap.read()

	stop = time.time()

	print("Capture {0:4d} frames in {1:3.2f} seconds   => frame rate = {2:3.2f}".format(num_frame,stop-start,num_frame/(stop-start)))


def histogram_equalization():

	while(True):
		img=cv2.imread('dog.jpg',1) 
		img=resize(img,40)
		img_cvt= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		equalization= cv2.equalizeHist(img_cvt)
		
		cv2.imshow('original',img)
		cv2.imshow('Equalization',equalization)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		cv2.imwrite('dog_normalized.jpg',equalization)
	
	cv2.destroyAllWindows()

def histogram_calculation(colorspace):
		
	cap = cv2.VideoCapture(0)

	fourcc = cv2.VideoWriter_fourcc(*'MJPG')        

	while(cap.isOpened()):
		ret, frame = cap.read()

		if ret==True:
			frame = cv2.flip(frame,1)       # flip frame

			if colorspace=='hsv':
				frame=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
				#cv2.imshow('hsv',frame)
			if colorspace=='yuv':
				frame=cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
				#cv2.imshow('yuv',frame)
			if colorspace=='rgb':
				frame=frame
				#cv2.imshow('yuv',frame)

			brg_planes=cv2.split(frame) # Separate the source image in its three R,G and B planes.
			histsize=256
			histrange=(0,256)       # the upper boundary is exclusive
			accumulate=False

			b_hist=cv2.calcHist(brg_planes,[0],None,[histsize],histrange,accumulate=accumulate)
			r_hist=cv2.calcHist(brg_planes,[1],None,[histsize],histrange,accumulate=accumulate)
			g_hist=cv2.calcHist(brg_planes,[2],None,[histsize],histrange,accumulate=accumulate)

			hist_w = 512
			hist_h = 400
			bin_w = int(round( hist_w/histsize ))
			histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
			histImage2 = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
			histImage3 = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)

			cv2.normalize(b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
			cv2.normalize(g_hist, g_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
			cv2.normalize(r_hist, r_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)

			for i in range(1, histsize):
				cv2.line(histImage, ( bin_w*(i-1), hist_h - int(np.round(b_hist[i-1])) ),
					( bin_w*(i), hist_h - int(np.round(b_hist[i])) ),
					( 255, 0, 0), thickness=2)

				cv2.line(histImage2, ( bin_w*(i-1), hist_h - int(np.round(g_hist[i-1])) ),
					( bin_w*(i), hist_h - int(np.round(g_hist[i])) ),
					( 0, 255, 0), thickness=2)

				cv2.line(histImage3, ( bin_w*(i-1), hist_h - int(np.round(r_hist[i-1])) ),
					( bin_w*(i), hist_h - int(np.round(r_hist[i])) ),
					( 0, 0, 255), thickness=2)


			cv2.imshow('real-time video',frame)
			if colorspace=='rgb':
				cv2.imshow('Blue',histImage)
				cv2.imshow('Green',histImage2)
				cv2.imshow('Red', histImage3)
			elif colorspace=='yuv':
				cv2.imshow('Luma',histImage3)
				cv2.imshow('Blue-Luma',histImage2)
				cv2.imshow('Red-luma',histImage)
			elif colorspace=='hsv':
				cv2.imshow('Brightness - (0..100)',histImage)
				cv2.imshow('Saturation - (0..100)',histImage2)
				cv2.imshow('Hue - (0..360)',histImage3)	
			if cv2.waitKey(20) & 0xFF == ord('q'):
				break
			

	cap.release()
	cv2.destroyAllWindows()


def BlurImages():

	while(1):
		img1 = cv2.imread('gaussnoise.jpg', cv2.IMREAD_UNCHANGED)
		img2 = cv2.imread('median.jpg', cv2.IMREAD_UNCHANGED)
		img3 = cv2.imread('bilat.jpg', cv2.IMREAD_UNCHANGED)
		img4 = cv2.imread('landscape.jpg', cv2.IMREAD_UNCHANGED)

		kernel = np.ones((10,10),np.float32)/25       

		gaussian = cv2.GaussianBlur(img1,(25,25),0)
		median = cv2.medianBlur(img2,5)
		bilat = cv2.bilateralFilter(img3, 15, 75, 75)
		filter2d = cv2.filter2D(img4,-1,kernel)
		average = cv2.blur(img4,(10,10))                       

		cv2.imshow('Original before Gaussian Blur',img1)
		cv2.imshow('Gaussian Blur',gaussian)
		cv2.imshow('Original before Median Blur',img2)
		cv2.imshow('Median Blur', median)
		cv2.imshow('Original before Bilateral Filtering',img3)
		cv2.imshow('Bilateral Filtered',bilat)
		cv2.imshow('Original before 2D Filtering or Averaging',img4)
		cv2.imshow('2D Filtered', filter2d)
		cv2.imshow('Averaged image', average)

		if cv2.waitKey(20) & 0xFF == ord('q'):
			break
	
		
cv2.destroyAllWindows()


import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-f","--FrameRate", help="print frame rate",
                    action="store_true")
parser.add_argument("-e","--HistoEqual", help="show the histogram equalization",
                    action="store_true")
parser.add_argument("-b","--BlurImages", help="Apply gaussian and blur filters to the acquired images",
                    action="store_true")
parser.add_argument("-c","--HistoCalc", help="show the histogram in real time (rbg,hsv,yuv)")
                   
parser.add_argument("-a","--ChangeColor", help="Change the color spaces")

                    

args = parser.parse_args()


if args.FrameRate:
	estimateFrameRate()
elif args.HistoEqual:
    histogram_equalization()
elif args.BlurImages:
    BlurImages()
elif args.HistoCalc:
	histogram_calculation(str(args.HistoCalc))
elif args.ChangeColor:
	changeColorSpace(str(args.ChangeColor))




