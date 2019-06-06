# -*- coding: utf-8 -*-
"""
Created on Thu May  9 11:49:34 2019

@author: Bhavesh
"""

''' IMPORTING USEFUL LIBRARIES '''
# for measuring time taken for detection
import time

# for image processing 
import cv2

# for using mathematical constants and methods
import numpy as np


class LaneDetectionClass:
    
    ''' COLOR SPACE CONVERSION METHODS '''
    ### convert image from hsl color space to rgb color space
    def hls2rgb(self,img):
        return cv2.cvtColor(img,cv2.COLOR_HLS2RGB)
    
    ### convert image from rgb color space to bgr color space
    def rgb2bgr(self,img):
        return cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    
    ### convert image from bgr color space to rgb color space
    def bgr2rgb(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    ### convert image from rgb color space to hsl color space
    def rgb2hls(self,img):
        return cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
    
    ### convert image from rgb color space to grayscale
    def grayscale(self,img):
        return cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    
    ''' THIS METHOD EXTRACTS ONLY WHITE AND YELLOW COMPONENTS FROM ORIGINAL 
        HLS IMAGE AND RETURNS A BINARY IMAGE '''
    def compute_white_yellow(self,hls_img):
        # Compute a binary thresholded image where yellow is isolated from HLS components
        img_hls_yellow_bin = np.zeros_like(hls_img[:,:,0])
        img_hls_yellow_bin[((hls_img[:,:,0] >= 15) & (hls_img[:,:,0] <= 35))
                     & ((hls_img[:,:,1] >= 30) & (hls_img[:,:,1] <= 204))
                     & ((hls_img[:,:,2] >= 115) & (hls_img[:,:,2] <= 255))                
                    ] = 1
        # Compute a binary thresholded image where white is isolated from HLS components
        img_hls_white_bin = np.zeros_like(hls_img[:,:,0])
        img_hls_white_bin[((hls_img[:,:,0] >= 0) & (hls_img[:,:,0] <= 255))
                     & ((hls_img[:,:,1] >= 200) & (hls_img[:,:,1] <= 255))
                     & ((hls_img[:,:,2] >= 0) & (hls_img[:,:,2] <= 255))                
                    ] = 1
        # Now combine both
        img_hls_white_yellow_bin = np.zeros_like(hls_img[:,:,0])
        img_hls_white_yellow_bin[(img_hls_yellow_bin == 1) | (img_hls_white_bin == 1)] = 1
        return img_hls_white_yellow_bin
    
    ### takes input binary image and gives binary soft output
    def blur(self,bin_img):
        
        # Applying the Gaussian Blur on the input image with kernel size = 11
        # This size of kernel has been chosen after trial and error on many images
        kernel_size = 11
        
        return cv2.GaussianBlur(bin_img, (kernel_size, kernel_size), 0)
    
    ### takes input rgb image and gives binary output
    def onlyLanes(self,input_img):
        
        # Determinig dimensions of input image
        rows, cols = input_img.shape[:2]
        
        # Determining the region of Interest (ROI)
        bottom_left  = [cols*0.1, rows*0.95]
        top_left     = [cols*0.4, rows*0.55]
        bottom_right = [cols*0.9, rows*0.95]
        top_right    = [cols*0.6, rows*0.55] 
        vertices     = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
        
        # Creation of mask
        mask         = np.zeros_like(input_img)
        cv2.fillPoly(mask, vertices, 255)
        
        # Applying the mask on the input image
        input_img    = cv2.bitwise_and(input_img, mask)
        
        return input_img
    
    ### takes input binary image and returns list of lines; (not image with lines)
    def hough_lines(self,img):
        return cv2.HoughLinesP(img, rho=1, theta=np.pi/180, threshold=20, minLineLength=50, maxLineGap=100)
    
    # this method draws circles at end points and centroid of detected lines.
    def draw_circles(self,image,lines):
        
        # If there is at least one line detected from the image, do:
        if not lines is None:
            
            # For each line detected from the image, do:
            for line in lines:
                
                # For both end points of the line, do:
                for x1,y1,x2,y2 in line:
                    
                    # Find the centroid of the line
                    x3,y3 = int(0.5*(x1+x2)), int(0.5*(y1+y2))
                    
                    # Find the slope of the line
                    slope = (y2-y1)/(x2-x1)
                    
                    # If the slope is Positive, then it is right lane line
                    if slope >= 0:
                        cv2.circle(image, center = (x1,y1),radius = 5, color=(255,0,0), thickness = -1)
                        cv2.circle(image, center = (x2,y2),radius = 5, color=(255,0,0), thickness = -1)
                        cv2.circle(image, center = (x3,y3),radius = 5, color=(255,0,0), thickness = -1)
                    
                    # If slope is negative, then it is left lane line
                    else:
                        cv2.circle(image, center = (x1,y1),radius = 5, color=(0,0,255), thickness = -1)
                        cv2.circle(image, center = (x2,y2),radius = 5, color=(0,0,255), thickness = -1)
                        cv2.circle(image, center = (x3,y3),radius = 5, color=(0,0,255), thickness = -1)
        
        # Return the image with circles drawn at appropriate places
        return image
    
    # this method finds and draws a rectangle around the cars.
    def detect_cars(self,image):
        
        '''
        ---> This Haar cascade is trained with almost 8000 true and false images
            of cars.
        ---> https://github.com/shaanhk/New-GithubTest
        ---> This is the link for more information.
        ---> The trained Haar cascade is stored in an .xml file.
        ---> This xml file must be in same folder as this file.
        '''
        
        # Loading the trained cascade
        car_cascade = cv2.CascadeClassifier('cars.xml')
        
        # Grayscaling the input image
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Applying cascade and detecting vehicles of various size
        cars = car_cascade.detectMultiScale(gray, 1.3,2,minSize=(70,70),maxSize=(200,200))
        
        # For each detected vehicle, do:
        for (x,y,w,h) in cars: 
            
            # draw a rectangle of blue color around it
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
        
        # Return the image with drawn rectangles in it around detected vehicles.
        return image
    
    # this method processes one frame and returns the final output image to be added to the output video
    def process(self, image):
        
        # Converting image from RGB to HLS color space
        img_hls = self.rgb2hls(np.uint8(image))
        
        # Extracting only white and yellow components
        white_yellow = self.compute_white_yellow(img_hls)
        
        # Applying region of interest mask
        regions      = self.onlyLanes(white_yellow)
        
        # Remove noise from the image and make it smoother
        blurred      = self.blur(regions)
        
        # Find lines by applying probabilistic Hough Trandform
        lines        = self.hough_lines(blurred)
        
        # Draw circles on detected lanes
        temp         = self.draw_circles(image,lines)
        
        # Detect cars from the image
        output       = self.detect_cars(temp)
        
        # Return the output
        return output
''' The class Ends Here '''


# this method takes input as a video that is to be processed
def process_video(video_input):
    
    '''
    ---> This image takes video name as input.
    ---> The video extension must be mp4
    ---> The output video will be .avi
    '''
    inpt = video_input + '.mp4'
    otpt = video_input + 'out.avi'
    
    # Create an object of lane Detector class
    detector = LaneDetectionClass()
    
    # Read the video
    cap = cv2.VideoCapture(inpt)
    
    # Determining the dimensions of frame
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    # Preparing an ouput stream to write into output file
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(otpt,fourcc, 20, (frame_width,frame_height))
    
    # A couter for counting number of frames
    cnt = 0
    
    # while the input video has more frames, do:
    while cap.isOpened():
        
        # Read the current frame from input video
        ret,image = cap.read()
        
        # If read successfully, do:
        if ret==True:
            
            # start measuring time from now
            start = time.time()
            
            # If it is the first frame of video, do:
            if cnt==0:
                # Start the ultimate timer for measuring processing time for entire video
                etrm_strt = start
            
            # Convert the image from BGR color space to RGB color space
            img = detector.bgr2rgb(image)
            
            # Process the image
            output_image = detector.process(img)
            
            # Take another reading of time after processing the image
            end = time.time()
            
            # Write the processed image to output video file
            out.write(detector.rgb2bgr(output_image))
            
            # Print the time taken to process current frame
            print(end - start)
            
            # One more frame processed
            cnt += 1
        
        # If not successfullly read, the video has ended
        else:
            
            # Stop the ultimate timer
            etrm_end = time.time()
            
            # Pring average time taken per frame
            print('Average Time : '+str((etrm_end-etrm_strt)/cnt)+'s')
            
            break
    
    # Close the input video file
    cap.release()
    
    # Close and save the output Video file
    out.release()
    
    # Return the number of frames processed
    return cnt

if __name__ == '__main__':
    
    # Start a timer to measure total time taken for 
    # 1. Opening the input and output flies
    # 2. Reading input frame by frame, processing and adding it to output file
    # 3. Saving and closing the output file
    start = time.time()
    
    # Processing the video file.
    # NOTE : The extension must be excluded from input file name
    # Only change the video name for processing another video
    cnt = process_video('solidWhiteRight')
    
    # stop the timer
    end = time.time()
    
    # Print the time taken
    print("Total : "+str((end-start))+"s for "+str(cnt)+"frames")

# ============================================================================ #
#	References:

#	1. https://towardsdatascience.com/teaching-cars-to-see-advanced-lane-detection-using-computer-vision-87a01de0424f
#	2. https://towardsdatascience.com/finding-lane-lines-on-the-road-30cf016a1165
#	3. https://www.opencv.org
#	4. https://www.stackoverflow.com
#	5. https://www.quora.com
#	6. https://www.geeksforgeeks.org/opencv-python-program-vehicle-detection-video-frame/
#	7. https://github.com/shaanhk/New-GithubTest