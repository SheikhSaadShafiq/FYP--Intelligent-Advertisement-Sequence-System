#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import os 


# In[10]:


def frame_extraction(path):
    cap= cv2.VideoCapture(path)

    #get fps
    fps =int(cap.get(cv2.CAP_PROP_FPS)) 
    print('fps of the video is',fps)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('frame count of the video is ' , frame_count)
    
    duration = int(frame_count/fps)
    print('duration of the video is ',duration)

    #extract 1 frame per second 
    path = 'C:\\Users\\l1f15bscs0049\\Desktop\\frames'
    count= 1
    for i in range (0,frame_count,fps):
        frame_no = (i /(duration*fps))

        print(i,frame_no)

        cap.set(cv2.CAP_PROP_POS_FRAMES,i);
        ret, frame = cap.read()

        if ret != False:    
            #cv2.imwrite('frame'+str(i)+'.jpg',frame)
            #cv2.imwrite('C:\\Users\\l1f15bscs0049\\Desktop\\frames\\frame'+str(i)+'jpg',frame)
            cv2.imwrite(os.path.join(path, 'frame'+ str(count) +'.jpg'),frame)
            count=count + 1
    cap.release()
    cv2.destroyAllWindows()


# In[9]:


frame_extraction('C:\\Users\\l1f15bscs0049\\Desktop\\test2.mp4')


# In[ ]:




