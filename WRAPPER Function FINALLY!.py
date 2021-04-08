#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import warnings
import os 
import cv2 
import numpy as np 


# In[24]:


def play_ad (path):

    # Create a VideoCapture object and read from input file 
    cap = cv2.VideoCapture(path) 
    
    # Check if camera opened successfully 
    if (cap.isOpened()== False):  
      print("Error opening video  file") 

    # Read until video is completed 
    while(cap.isOpened()): 

      # Capture frame-by-frame 
      ret, frame = cap.read() 
      if ret == True: 

        # Display the resulting frame 
        cv2.imshow('Frame', frame) 

        # Press Q on keyboard to  exit 
        if cv2.waitKey(25) & 0xFF == ord('q'): 
          break

      # Break the loop 
      else:  
        break

    # When everything done, release  
    # the video capture object 
    cap.release()

    
    


# In[29]:


path = "D:\\ads.csv"
df = pd.read_csv(path)

warnings.filterwarnings('ignore')
ads= df[["File_path"]] [df.Gender == 0] [df.AD_Status == "active"]
#ads

arr = ads.to_numpy()
x,y=ads.shape

for i in range (0,x):
    play_ad(arr[i][0])
    #print("playing ad\n")
cv2.destroyAllWindows() 

   


# In[6]:


print(arr)


# In[ ]:


def play(self):
        from os import startfile
        startfile(self.path)

path = 'C:\\Users\\l1f15bscs0049\\Desktop\\test2.mp4'      
play(path)        


# In[ ]:


path1()


# In[ ]:




