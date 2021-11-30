!pip install inflection

import inflection    # Camel Case
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
from imutils.video import FPS
import json
import time
import glob
import warnings
warnings.filterwarnings("ignore")



"""Camel case"""

def camel(string):
    return inflection.camelize(string, False)


def no_helmet_detection(img, CONFIG_FILE, WEIGHT_FILE,NAMES):
  #img: output image of people counting taken as input
  #CONFIG_FILE: Confihuration file
  #WEIGHT_FILE: Pretrained weights file
  #NAMES: Class names for model
  #c_nohelmet: Counter for no helmet class as output
  #out: Output image after bbox prediction as output

  #Loading Weight and cfg files
  net = cv2.dnn.readNet(CONFIG_FILE, WEIGHT_FILE)

  # Reading class names 
  classes = []

  with open(NAMES) as cls:
    classes = cls.read().splitlines()

  print(classes)

  k = 0

  # No helmet counter initialization
  c_nohelmet = 0
  height, width,_ = img.shape
  blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),swapRB=True, crop=False)

  # give images input to darknet module
  net.setInput(blob)

  # Get the output layers of model
  out_names= net.getUnconnectedOutLayersNames()
  layerOutputs = net.forward(out_names)

  # Generating detection in the form of bounding box, confidence and class id
  boxes=[]
  confidences=[]
  class_ids=[]

  for out in layerOutputs:
    for detection in out:
      scores = detection[5:]
      class_id = np.argmax(scores)
      confidence = scores[class_id]
      if confidence > 0.4:
        centerX= int(detection[0]* width)
        centerY= int(detection[1]* height)
        w= int(detection[2]* width)
        h= int(detection[3]* height)
        x = int(centerX - (w/ 2))
        y = int(centerY - (h/ 2))

        boxes.append([x, y, w, h])
        confidences.append(float(confidence))
        class_ids.append(class_id)

  # To avoid overlapping bounding boxes NMS is performed
  indexes= cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.4)
  font= cv2.FONT_HERSHEY_PLAIN
  colors= np.random.uniform(0, 255, size=(len(boxes),3))
  indexes= np.array(indexes)
  #print(indexes)
  if len(indexes) > 0:
    for i in indexes.flatten():
      x, y, w, h= boxes[i]
      #print(x,y,w,h)
      label= str(classes[class_ids[i]])
      if label == 'noHelmet':
        c_nohelmet = c_nohelmet+1
        confidence= str(round(confidences[i],2))
        color= colors[i]
        cv2.rectangle(img, (x, y), (x+w,y+h), 2)
        out = cv2.putText(img, label, (x, y+20), font, 1, (255, 255, 0), 2)
      else:
        continue
       

      k = k+1   
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  return (c_nohelmet, out)