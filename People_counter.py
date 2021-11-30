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

################################### Color Classifier ####################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,AdaBoostClassifier,GradientBoostingClassifier,VotingClassifier
import pickle



"""Camel case"""

def camel(string):
    return inflection.camelize(string, False)

"""People_count"""

def People_Count(img,CONFIG_FILE, WEIGHT_FILE, NAMES):
  #img: image is taken as input
  #CONFIG_FILE: Confihuration file
  #WEIGHT_FILE: Pretrained weights file
  #NAMES: Class names for model
  #c_people: Count of number of people as output
  #im_people: Image with printed count on it as output



  # To read darknet module using config file and yolov3 weights
  net = cv2.dnn.readNet(WEIGHT_FILE, 
                        CONFIG_FILE)
  
  #Reading class names
  classes = []
  with open(NAMES, "r") as f:
      classes = [line.strip() for line in f.readlines()]
  layer_names = net.getLayerNames()
  output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
  colors = np.random.uniform(0, 255, size=(len(classes), 3))
  
  # People counter initialized
  c_people = 0  
  
  # Getting output layer of darknet
  height, width,_ = img.shape
  blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),swapRB=True, crop=False)
  net.setInput(blob)
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
      if confidence > 0.3:
        centerX= int(detection[0]* width)
        centerY= int(detection[1]* height)
        w= int(detection[2]* width)
        h= int(detection[3]* height)
        x = int(centerX - (w/ 2))
        y = int(centerY - (h/ 2))

        boxes.append([x, y, w, h])
        confidences.append(float(confidence))
        class_ids.append(class_id)

  indexes= cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.4)
  indexes= np.array(indexes)
  font= cv2.FONT_HERSHEY_PLAIN
  colors= np.random.uniform(0, 255, size=(len(boxes),3))


  for i in indexes.flatten():
    x, y, w, h= boxes[i]
    label= str(classes[class_ids[i]])

    # Person counter
    if label == 'person':
      c_people = c_people+1
    else:
      continue
    confidence= str(round(confidences[i],2))
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  im_people = cv2.putText(img, camel("People Counter: ") + str(c_people) , (10, 20), font, 1.5, (255,0,0), 2)
  return (c_people, im_people)