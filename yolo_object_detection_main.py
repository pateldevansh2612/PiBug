import cv2
import numpy as np
import glob
import random


# Load Yolo
net = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_testing.cfg")

# Name custom object
classes = ["Bird"]

# Images path
images_path = glob.glob(r"C:\Users\pc2\Desktop\birddataset\final\testing\*.jpg")



layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
# Insert here the path of your images
random.shuffle(images_path)
# loop through all the images
for img_path in images_path:
    # Loading image
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.9:
                # Object detected
                print("class_id:"+str(class_id))
                l1.append(class_id)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                cv2.rectangle(img, (x,y),(x+w,y+h) , (255, 0, 0), 5)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    #l1=[l1.append(i) for i in indexes]
    
    #print("printing indices")
    #print(indexes)
    #print("printing boxes")
    #print(boxes)
    #print("printing confidence")
    #print(confidences)
    
    cv2.imshow("Image", img)
    key = cv2.waitKey(0)

cv2.destroyAllWindows()
