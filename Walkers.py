import cv2

print(cv2.__version__)

# Create our body classifier
body_clasifier=cv2.CascadeClassifier("haarcascade_fullbody.xml")

# Initiate video capture for video file
cap = cv2.VideoCapture('walking.avi')

# Loop once video is successfully loaded
while True:
    
    # Read first frame
    ret, frame = cap.read()

    #Convert Each Frame into Grayscale
    sorce=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #cv2.imshow("frames",sorce)
    # Pass frame to our body classifier
    bodies=body_clasifier.detectMultiScale(sorce,1.2,3)
    #cv2.imshow("frame",bodies)
    
    # Extract bounding boxes for any bodies identified
    for (x,y,w,h)in bodies:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        cv2.imshow('output',frame)

    if cv2.waitKey(1) == 32: #32 is the Space Key
        break

cap.release()
cv2.destroyAllWindows()
