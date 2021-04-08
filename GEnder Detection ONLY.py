#!/usr/bin/env python
# coding: utf-8

# In[4]:


import keras 
import sys
import os 
import os.path
import cv2
from keras.models import load_model
from collections import Counter
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





# In[6]:



# parameters for loading data and images
image_path = 'C:\\Users\\l1f15bscs0049\\Desktop\\t5.png'
detection_model_path = 'C:\\Users\\l1f15bscs0049\\Desktop\\haarcascade_frontalface_default.xml'
gender_model_path = 'C:\\Users\\l1f15bscs0049\\Desktop\\simple_CNN.81-0.96.hdf5'
gender_labels = get_labels('imdb')
font = cv2.FONT_HERSHEY_SIMPLEX

# hyper-parameters for bounding boxes shape
gender_offsets = (30, 60)
gender_offsets = (10, 10)
emotion_offsets = (20, 40)
emotion_offsets = (0, 0)

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

#creating a file
save_path = 'C:\\Users\\l1f15bscs0049\\Desktop'
completeName = os.path.join(save_path, "hellojee.txt")         
file = open(completeName, "a")
    
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
    gender_label_arg = np.argmax(gender_prediction)
    gender_text = gender_labels[gender_label_arg]
    #print(gender_label_arg)
    file.write(str(gender_label_arg))
    file.write("\n")
    
    if gender_text == gender_labels[0]:
        color = (0, 0, 255)
    else:
        color = (255, 0, 0)

    draw_bounding_box(face_coordinates, rgb_image, color)
    draw_text(face_coordinates, rgb_image, gender_text, color, 0, -20, 1, 2)

bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
cv2.imwrite('C:\\Users\\l1f15bscs0049\\Desktop\\t1a.png', bgr_image)
file.close()

from collections import Counter
with open(completeName, "r") as f:
    cd = Counter(int(line.split(None, 1)[0]) for line in f)
    #print(cd)

    
women_count = cd[0]
men_count = cd[1]

#call a function 
    
 
    
file.close()


# In[5]:


print(cd[0])
print("\n",cd[1])


# In[7]:


import os.path

save_path = 'C:\\Users\\l1f15bscs0049\\Desktop'

completeName = os.path.join(save_path, "hellojee.txt")         

file = open(completeName, "a")
file1.write("toFile")


from collections import Counter
with open('abc.txt') as f:
    c = Counter(int(line.split(None, 1)[0]) for line in f)
    print c

file1.close()


# In[29]:


path = 'C:\\Users\\l1f15bscs0049\\Desktop\\t4.jpg'
gender_detection(path)


# In[ ]:


f.write()


# In[ ]:




