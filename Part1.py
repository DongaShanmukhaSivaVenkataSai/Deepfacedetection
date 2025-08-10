import os
import uuid
import cv2
import time
#creating path for images
Images_path=os.path.join('Data','Images')
#creating folders
os.makedirs(Images_path,exist_ok=True)
#intiating vedio capture
cap=cv2.VideoCapture(0)
number_of_images=25
for i in range(number_of_images):
    #noting the return value and image in numpyarray
    ret,frame=cap.read()
    #creating unique id and adding jpg for image file
    id=str(uuid.uuid1())+".jpg"
    img_name=os.path.join(Images_path,id)
    if not ret:
        print("Image capturation failed")
        continue
    
    cv2.imwrite(img_name,frame)
    cv2.imshow('image_captured',frame)
    #waiting for 1 sec
    time.sleep(1)
    #break if q is pressed
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()