#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import keras 
import sys

import cv2
from keras.models import load_model
import numpy as np

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.inference import load_image
from utils.preprocessor import preprocess_input


# In[ ]:





# In[8]:


# parameters for loading data and images
detection_model_path = 'C:\\Users\\l1f15bscs0049\\Desktop\\haarcascade_frontalface_default.xml'
image_path = 'C:\\Users\\l1f15bscs0049\\Desktop\\t5.png'
font = cv2.FONT_HERSHEY_SIMPLEX

# hyper-parameters for bounding boxes shape
gender_offsets = (30, 60)
gender_offsets = (10, 10)


# loading models
face_detection = load_detection_model(detection_model_path)



# loading images
rgb_image = load_image(image_path, grayscale=False)
gray_image = load_image(image_path, grayscale=True)
gray_image = np.squeeze(gray_image)
gray_image = gray_image.astype('uint8')

faces = detect_faces(face_detection, gray_image)
for face_coordinates in faces:
    x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
    rgb_face = rgb_image[y1:y2, x1:x2]



    rgb_face = preprocess_input(rgb_face, False)
    rgb_face = np.expand_dims(rgb_face, 0)
    

   

    

    color = (0, 0, 255)
    draw_bounding_box(face_coordinates, rgb_image, color)
   

bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
cv2.imwrite('C:\\Users\\l1f15bscs0049\\Desktop\\testing5.png', bgr_image)


# In[4]:


draw_bounding_box()


# In[8]:


img_p = 'C:\\Users\\l1f15bscs0049\\Desktop\\testing.png'
img = cv2.imread(img_p)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face_cascade =  cv2.CascadeClassifier('C:\\Users\\l1f15bscs0049\\Desktop\\haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x,y,w,h) in faces:
    print("x ",x,"y ",y,"w ",w,"h ",h)
    
    


# In[ ]:




