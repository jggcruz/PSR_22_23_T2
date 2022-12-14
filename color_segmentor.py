import cv2
import numpy as np 
import json
import timeit

global_minimum_blue = 0
global_maximum_blue = 255
global_minimum_green = 0
global_maximum_green = 255
global_minimum_red = 0
global_maximum_red = 255


def trackbar_min_blue(blue_min):
    global global_minimum_blue
    global_minimum_blue = blue_min
    print("Minimum thereshold is " + str(blue_min) + " for color blue.")


def trackbar_max_blue(blue_max):
    global global_maximum_blue
    global_maximum_blue = blue_max
    print("Maximum thereshold is " + str(blue_max) + " for color blue.")


def trackbar_min_green(green_min):
    global global_minimum_green
    global_minimum_green = green_min
    print("Minimum thereshold is " + str(green_min) + " for color green.")


def trackbar_max_green(green_max):
    global global_maximum_green
    global_maximum_green = green_max
    print("Maximum thereshold is " + str(green_max) + "for color blue.")


def trackbar_min_red(red_min):
    global global_minimum_red
    global_minimum_red = red_min
    print("Minimum thereshold is " + str(red_min) + " for color red.")


def trackbar_max_red(red_max):
    global global_maximum_red
    global_maximum_red = red_max
    print("Maximum thereshold is " + str(red_max) + " for color red.")


def main(): 

    window = 'Color Segmentation'
    global blue_min, green_min, red_min, blue_max, green_max, red_max
    
    blue_min = 0
    green_min = 0
    red_min = 0
    blue_max = 255
    green_max = 255
    red_max = 255

    #load video from webcam
    video = cv2.VideoCapture(0)

    #Trackbars
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('Minimun Blue', window, 0, 255, trackbar_min_blue)
    cv2.createTrackbar('Maximun Blue', window, 255, 255, trackbar_max_blue)
    cv2.createTrackbar('Minimun Green', window, 0, 255, trackbar_min_green)
    cv2.createTrackbar('Maximun Green', window, 255, 255, trackbar_max_green)
    cv2.createTrackbar('Minimun Red', window, 0, 255, trackbar_min_red)
    cv2.createTrackbar('Maximun Red', window, 255, 255, trackbar_max_red)

        #Check if camera opened correctly
    if not video.isOpened():  
        print("failed to open cam")
    else:
        while True:   
            #Capture all frames from the video
            ret, frame = video.read()
            
            #Save the original. Usable for threshold comparision
            video_copy = frame.copy()

            # if user wants HSV mode
            # if args['hue_saturation_value']:
            #     image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            #     cv2.imshow('HSV', image)

            dic_blue = {'blue_min': global_minimum_blue, 'blue_max': global_maximum_blue}
            dic_green = {'green_min': global_minimum_green, 'green_max': global_maximum_green}
            dic_red = {'red_min': global_minimum_red, 'red_max': global_maximum_red}

            threshold={'threshold':{'blue': dic_blue, 'green': dic_green, 'red': dic_red}}

            #inRange function for segmentation requires a np.array
            lower_thresholds=np.array([threshold['threshold']['blue']['blue_min'],
            threshold['threshold']['green']['green_min'],
            threshold['threshold']['red']['red_min']])

            upper_thresholds=np.array([threshold['threshold']['blue']['blue_max'],
            threshold['threshold']['green']['green_max'],
            threshold['threshold']['red']['red_max']])

            #masking the image 
            mask=cv2.inRange(video_copy,lower_thresholds, upper_thresholds)

            cv2.imshow(window, mask)
                
            cv2.imshow('Original',frame)    
            k=cv2.waitKey(20)
            

            #Press 'q' to quit the program
            if k == ord('q'):
                image_gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                moments= cv2.moments(image_gray)
                X = int(moments ["m10"] / moments["m00"])
                Y = int(moments ["m01"] / moments["m00"])
                cv2.putText(frame, 'Goodbye web cam!', (X,Y), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255), 2)
                break
            
            #save to directory
            elif k == ord('w'):
                file_name = 'threshold.json'
                with open(file_name, 'w') as file_handle:
                        print('Writting thresholds to file' + file_name + '!')
                        file_handle.write(json.dumps(threshold))
            
if __name__ == '__main__':
    main()