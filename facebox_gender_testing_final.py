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





# In[2]:


np.array=[]

# parameters for loading data and images
detection_model_path = 'C:\\Users\\l1f15bscs0049\\Desktop\\haarcascade_frontalface_default.xml'
gender_model_path = 'C:\\Users\\l1f15bscs0049\\Desktop\\simple_CNN.81-0.96.hdf5'
image_path = 'C:\\Users\\l1f15bscs0049\\Desktop\\imgt.jpg'
gender_labels = get_labels('imdb')
font = cv2.FONT_HERSHEY_SIMPLEX

# hyper-parameters for bounding boxes shape
gender_offsets = (30, 60)
gender_offsets = (10, 10)


# loading models
face_detection = load_detection_model(detection_model_path)
gender_classifier = load_model(gender_model_path, compile=False)

# getting input model shapes for inference
gender_target_size = gender_classifier.input_shape[1:3]

# loading images
rgb_image = load_image(image_path, grayscale=False)
gray_image = load_image(image_path, grayscale=True)
gray_image = np.squeeze(gray_image)
gray_image = gray_image.astype('uint8')

faces = detect_faces(face_detection, gray_image)
for face_coordinates in faces:
    x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
    rgb_face = rgb_image[y1:y2, x1:x2]


    try:
        rgb_face = cv2.resize(rgb_face, (gender_target_size))
        
    except:
        continue

    rgb_face = preprocess_input(rgb_face, False)
    rgb_face = np.expand_dims(rgb_face, 0)
    gender_prediction = gender_classifier.predict(rgb_face)
    #print(gender_prediction)
    gender_label_arg = np.argmax(gender_prediction)
    #np.array.append(gender_label_arg)
    #print genders
    #print(gender_label_arg)
    
   
    #np.array
    
    #print(np.array)
    
    gender_text = gender_labels[gender_label_arg]
    if gender_text == 'woman':
        np.array.append(0)
    else:
        np.array.append(1)
    #np.array.append(gender_text)
    print(gender_text)
    if gender_text == gender_labels[0]:
        color = (0, 0, 255)
    else:
        color = (255, 0, 0)

    draw_bounding_box(face_coordinates, rgb_image, color)
    draw_text(face_coordinates, rgb_image, gender_text, color, 0, -20, 1, 2)
    
    print(gender_label_arg)
bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
cv2.imwrite('C:\\Users\\l1f15bscs0049\\Desktop\\predicted_test_image.png', bgr_image)


# In[ ]:





# In[ ]:





# In[ ]:




