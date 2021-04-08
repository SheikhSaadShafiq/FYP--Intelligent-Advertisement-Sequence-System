#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


path = "D:\\ad.csv"


# In[4]:


df = pd.read_csv(path)


# In[9]:


df.head()


# In[5]:


df.columns


# In[7]:


import warnings
warnings.filterwarnings('ignore')
ads= df[["AdName" ,"File_path"]] [df.Gender == 1] [df.AD_Status == "active"]
ads


# In[8]:


ad1= df[["AdName"]] [df.Gender == 0]
ad1


# In[6]:


import warnings
warnings.filterwarnings('ignore')
ads= df[["AdName" ,"File_path"]] [df.AD_Status == "active"]
ads


# In[ ]:


ads

