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



def image(PATH, CONFIG_FILE, WEIGHT_FILE):

  net = cv2.dnn.readNet(CONFIG_FILE, WEIGHT_FILE)

  # Define classes
  classes=['Helmet']

  colors = ['White_hat', 'Orange_hat', 'Blue_hat', 'Red_hat', 'Yellow_hat']
  colors[0] = 0
  colors[1] = 0
  colors[2] = 0
  colors[3] = 0
  colors[4] = 0
  
  img = cv2.imread(PATH)
  height, width,_ = img.shape

  blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),swapRB=True, crop=False)

  # give images input to darknet module
  net.setInput(blob)

  #  Get the output layers of model
  out_names= net.getUnconnectedOutLayersNames()
  layerOutputs = net.forward(out_names)

  # To store detections like class id, probability, bounding box parameters new lists are formed.
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
  clr= np.random.uniform(0, 255, size=(len(boxes),3))
  indexes= np.array(indexes)
  #print(indexes)

  

  if len(indexes) > 0:
    for i in indexes.flatten():
      x, y, w, h= boxes[i]
      #print(x,y,w,h)
      label= str(classes[class_ids[i]])
      confidence= str(round(confidences[i],2))
      color= clr[i]
      cv2.rectangle(img, (x, y), (x+w,y+h), 2)
      cr=[x,y,w,h]
      
      for i in range(4):
        if cr[i]<0:
          cr[i]=0
        else:
          continue

      # People counter
      people, img_1 = People_Count(img,"/content/drive/MyDrive/person_yolov4.cfg",
                              "/content/drive/MyDrive/person_yolov4.weights",
                              "/content/drive/MyDrive/person_coco.names")

      # No helmet detection
      nhelmet, img_2 = no_helmet_detection(img_1,'/content/drive/MyDrive/head_yolov4_roboflow_binary.cfg',
                                      '/content/drive/MyDrive/head_custom-yolov4-detector_best_binary.weights',
                                      '/content/drive/MyDrive/head_classes.names')
      if (nhelmet == 0):
        img_out = img_1      
      else:
        img_out = img_2

      
      crop = img_out[cr[1]:cr[1]+cr[3], cr[0]:cr[0]+cr[2]]
      crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) #convert it to RGB channel
      avg = np.average(crop_rgb, axis = (0,1))
      avg = avg.reshape(1, -1)
      #print(avg)
      #avg_t = avg.T
      
      # Load RGB classifier

      model = pickle.load(open('/content/model.pkl','rb'))

      color = model.predict(avg)
      cv2.putText(img_out, str(color), (x, y+20), font, 1, (255, 255, 0), 2)
      #print(color)
    
      if color == 'white':
        colors[0] = colors[0]+1
      elif color == 'orange':
        colors[1] = colors[1]+1
      elif color == 'blue':
        colors[2] = colors[2]+1
      elif color == 'red':
        colors[3] = colors[3]+1
      elif color == 'yellow':
        colors[4] = colors[4]+1
        
      helmet = colors[0]+colors[1]+colors[2]+colors[3]+colors[4]

  cv2_imshow(img_out)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  #print(colors)
  print(json.dumps({camel('White_hat'): colors[0],camel('Orange_hat'): colors[1],camel('Blue_hat'): colors[2], camel('Red_hat'): colors[3], camel('Yellow_hat'): colors[4], camel('no_helmet'): nhelmet, camel("helmet"):helmet ,camel("People_count"):people}, indent=6))



image('/content/img_1.jpg', '/content/custom-yolov4-detector (3).cfg',
     '/content/drive/MyDrive/Only_helmet_final_1K.weights')
