#!/usr/bin/env python
# coding: utf-8

# In[6]:


import keras 
import sys
import os 
import cv2
from keras.models import load_model
import numpy as np
import pandas as pd
import warnings

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.inference import load_image
from utils.preprocessor import preprocess_input


# In[ ]:





# In[ ]:






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

    

    
    
def Wrapper_func(count):

        #wrapper Function code below
        path = "D:\\ad.csv"
        df = pd.read_csv(path)

        warnings.filterwarnings('ignore')
        ads= df[["File_path"]] [df.Gender == count] [df.AD_Status == "active"]
        #ads

        arr = ads.to_numpy()
        x,y=ads.shape
       # print(arr[1][0])
        for i in range (0,x):
            print("playing ad\n")
            play_ad(arr[i][0])
            cv2.destroyAllWindows()  
    


def random_ads():
    #wrapper Function code below
        path = "D:\\ad.csv"
        df = pd.read_csv(path)

        warnings.filterwarnings('ignore')
        ads= df[["File_path"]] [df.AD_Status == "active"]
        #ads

        arr = ads.to_numpy()
        x,y=ads.shape

        for i in range (0,x):
            play_ad(arr[i][0])
            print("playing ad\n")
        cv2.destroyAllWindows()  
    

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

        #print(i,frame_no)

        cap.set(cv2.CAP_PROP_POS_FRAMES,i);
        ret, frame = cap.read()

        if ret != False:    
            #cv2.imwrite('frame'+str(i)+'.jpg',frame)
            #cv2.imwrite('C:\\Users\\l1f15bscs0049\\Desktop\\frames\\frame'+str(i)+'jpg',frame)
            cv2.imwrite(os.path.join(path, 'frame'+ str(count) +'.jpg'),frame)
            count=count + 1
    print('Frame Extraction Done')
    cap.release()
    cv2.destroyAllWindows()
    




def detection (image_path):


    # parameters for loading data and images
    detection_model_path = 'C:\\Users\\l1f15bscs0049\\Desktop\\haarcascade_frontalface_default.xml'
    
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
    count1 = 0
    faces = detect_faces(face_detection, gray_image)
    for face_coordinates in faces:
        x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
        rgb_face = rgb_image[y1:y2, x1:x2]



        rgb_face = preprocess_input(rgb_face, False)
        rgb_face = np.expand_dims(rgb_face, 0)






        color = (0, 0, 255)
        draw_bounding_box(face_coordinates, rgb_image, color)
        count1 = count1 + 1


    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    #cv2.imwrite('C:\\Users\\l1f15bscs0049\\Desktop\\test_cases\\testt2.png', bgr_image)
    return count1




def gender_detection(image_path):
        # parameters for loading data and images
        detection_model_path = 'C:\\Users\\l1f15bscs0049\\Desktop\\haarcascade_frontalface_default.xml'
        gender_model_path = 'C:\\Users\\l1f15bscs0049\\Desktop\\simple_CNN.81-0.96.hdf5'
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
            #print(gender_label_arg)

        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        #cv2.imwrite('C:\\Users\\l1f15bscs0049\\Desktop\\a.png', bgr_image)
        print('\n\tGender Detection Done')
        
        file.close()
        
        #check men women count
        from collections import Counter
        with open(completeName, "r") as f:
            cd = Counter(int(line.split(None, 1)[0]) for line in f)
        #print(cd)


        women_count = cd[0]
        men_count = cd[1]
       # print(women_count)
        #print(men_count)
        #print(cd[0])
        #print(cd[1])
        os.remove(completeName)
        print("file removed")
        #call a wrapper function 
        if(women_count > men_count):
            print("Women detected")
            Wrapper_func(0)
            
            
        elif(men_count > women_count):
            print("men detected")
            Wrapper_func(1)
        
        else: 
            print("no Detection\n Random Ad's playing\n")
            random_ads()
            


        file.close()

        



# In[ ]:





# In[ ]:





# In[7]:


import keras 
import sys
import os 
import cv2
from keras.models import load_model
import numpy as np
import pandas as pd
import warnings


from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.inference import load_image
from utils.preprocessor import preprocess_input



frame_extraction('C:\\Users\\l1f15bscs0049\\Desktop\\test6.mp4')
#gender_detection('C:\\Users\\l1f15bscs0049\\Desktop\\frames\\frame1.jpg')



image_path = 'C:\\Users\\l1f15bscs0049\\Desktop\\frames\\frame1.jpg'
image_path2 = 'C:\\Users\\l1f15bscs0049\\Desktop\\frames\\frame2.jpg'
image_path3 = 'C:\\Users\\l1f15bscs0049\\Desktop\\frames\\frame3.jpg'
image_path4 = 'C:\\Users\\l1f15bscs0049\\Desktop\\frames\\frame4.jpg'
image_path5 = 'C:\\Users\\l1f15bscs0049\\Desktop\\frames\\frame5.jpg'
image_path6 = 'C:\\Users\\l1f15bscs0049\\Desktop\\frames\\frame6.jpg'


a=0
b=0
c=0
d=0
e=0 
f=0

a = detection(image_path)
b = detection(image_path2)
c = detection(image_path3)
d = detection(image_path4)
e = detection(image_path5)
f = detection(image_path6)
print('Face Detection done')

print(a,b,c,d,e,f)
warnings.filterwarnings('ignore')
if a>b and a>c and a>d and a>e and a>f:
    gender_detection(image_path)
elif b>a and b>c and b>d and b>e and b>f:
    gender_detection(image_path2)
elif c>a and c>b and c>d and c>e and c>f:
    gender_detection(image_path3)
elif d>a and d>b and d>c and c>e and d>f:
    gender_detection(image_path4)
elif e>a and e>b and e>c and e>d and e>f:
    gender_detection(image_path5)
elif f>a and f>b and f>c and f>d and f>e:
    gender_detection(image_path6)
else : 
    print("\n\tSorry, No gender found! \n\tTRY AGAIN\nRandom Ad's Playing")
    random_ads()
    
    
    
    

    


# In[7]:


ppath= 'C:\\Users\\l1f15bscs0049\\Desktop\\t5.png'
gender_detection(ppath)


# In[ ]:




