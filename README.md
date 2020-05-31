# VCI-19-20
Projecto para a cadeira de Visão por computador 2019/20

## Milestone1

* Usage Examples:
	
	* python3 Millestone.py -e
		* Estimete Frame rate
	* python3 Millestone.py -s
		* Saves an image with the name 'image.png'
	* python3 Millestone.py -p
		* Show the video capture by camera
	* python3 Millestone.py -w
		* Play video in realt time with watermark
	* python3 Millestone.py -b 1.1
		* Play video in real time with brigthness aplified 1.1 times
	* python3 Millestone.py -c 4
		* Play video in real time with constrast aplified 4 times
	* python3 Millestone.py -t
		* Play Video real time with White balance
	* python3 Millestone.py -i
		* Play Video real time with Intensity Normalized
	* python3 Millestone.py -v
		* Save Video recorded in real time
	* python3 Millestone.py -m
		* Show Video with an image as a watermark

* Import:
	* .numpy
	* .cv2
	* .time
	* .argparse

## Milleston2

* Usage Examples:
	
	* python3 Millestone.py -e
		* show the histogram equalization
	* python3 Millestone.py -b
		* Apply gaussian and blur filters to the acquired images
	* python3 Millestone.py -c rgb
		* Show histogram of video recorded in realtime, options:
			* rgb
			* hsv
			* yuv
	* python3 Millestone.py -a hsv
		* Change the color spaces, options:
			* rgb
			* hsv
			* yuv		

* Import:
	* .numpy
	* .cv2
	* .time
	* .argparse

## Milestone3

* Usage Examples:
	
	* python3 Millestone.py -m
		* Test MouseCallBack
	* python3 Millestone.py -t
		* Trackbar in an image
	* python3 Millestone.py -r
		* Tracks all game components in real time
	* python3 Millestone.py -g
		* Tracks all game components in real time in a grayscale
	* python3 Millestone.py -s
		* Exploring the basic thresholds in an image
	* python3 Millestone.py -a
		* Exploring image Segmentation with Distance
        * Transform(commented)/Erosion and Watershed Algorithm
	* python3 Millestone.py -w
		* Exploring Region Growing

* Import: 
	* .numpy
	* .cv2
	* .argparse

## Milestone4

* Usage Examples:
	
	* python3 Millestone.py -r
		* Complete tracking in real time using contours
	* python3 Millestone.py -g
		* Tracking in real time using gradient
	* python3 Millestone.py -m
		* Template match ball
	* python3 Millestone.py -t
		* Template match robot
	* python3 Millestone.py -y
		* Apply canny detection
	* python3 Millestone.py -c
		* Apply hough methot with circle
	* python3 Millestone.py -l
		* Apply hough methot with lines

* Import: 
	* .numpy
	* .cv2
	* .argparse
	* .tracking_realtime_img_grad from canny.py
	* .tracking_realtime_canny from canny.py
	* .match_temp_ball from canny.py
	* .match_temp_robot from canny.py
	* .hough_circle from hough.py
	* .hough_line from hough.py

## Milestone5

* Usage Examples:
	
	* python3 Millestone.py -t
		* Tracking balls and robots with ID
	* python3 Millestone.py -p
		* Tracking distance traveled by the ball and robots
	* python3 Millestone.py -v
		* Multi tracking algorithms videos
	* python3 multi_object_tracking.py -v cambada_2.mp4 -t csrt
		* Play cambada_2.mp4 using csrt Tracker. To use it press 's' and select with mouse area to track thenspace to confirm. The tracker options are:
			* csrt
			* kcf
			* boosting
			* mil
			* tld
			* medianflow
			* mosse

* Import: 
	* .numpy
	* .cv2
	* .argparse
	* .math
	* .os


## Packages:

* sudo apt-get install python3-pip
* pip3 install opencv-python
* pip3 install opencv-contrib-python
* pip3 install numpy
* pip3 install argparse
* pip3 install python-time
* pip3 install opencv-python tensorflow
* pip3 install cvlib
* pip3 install imutils






	
