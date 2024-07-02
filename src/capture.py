import cv2 
  
NAME = "Duy"


video = cv2.VideoCapture("/home/hungdv/tcgroup/Jetson/insightface/face_detection_insightface/src/IMG_5272.MOV") 
if (video.isOpened() == False):  
    print("Error reading video file") 
  
frame_width = int(video.get(3)) 
frame_height = int(video.get(4)) 
   
size = (frame_width, frame_height) 
   
result = cv2.VideoWriter(f'{NAME}.avi',  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         20, size) 

capture = False
while(True): 
    ret, frame = video.read() 
    if ret == True:  
        if capture:
            result.write(frame) 
        cv2.imshow('Frame', frame) 
        k = cv2.waitKey(1)
        if  k == ord('q'): 
            break
        elif k == ord(' '):
            print("Start record" if capture else "Stop record ")
            capture = not capture

        
    else: 
        break

video.release() 
result.release() 

cv2.destroyAllWindows() 
   
print("The video was successfully saved") 