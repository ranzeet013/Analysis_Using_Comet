#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[2]:


dataframe = pd.read_csv('API_ITA_DS2.csv')


# In[3]:


dataframe.head()


# In[4]:


dataframe.tail()


# In[5]:


dataframe.columns


# In[6]:


dataframe.drop(['Country Name', 'Country Code', 'Unnamed: 65', 'Indicator Code'], axis = 1, inplace = True)


# In[7]:


dataframe.head()


# In[8]:


dataframe = dataframe[dataframe['Indicator Name'].str.contains('(% of GDP)')]


# In[9]:


dataframe.head()


# In[10]:


dataframe = dataframe.transpose()


# In[11]:


dataframe.columns = dataframe.iloc[0] 


# In[12]:


dataframe = dataframe[1:]


# In[13]:


dataframe


# In[14]:


dataframe.isna().sum()


# In[15]:


dataframe.dropna(thresh = 30, axis = 1, inplace = True)
dataframe = dataframe.iloc[10:]


# In[16]:


dataframe


# In[17]:


import matplotlib.pyplot as plt
import numpy as np

def plot_indicator(ts, indicator):
    xmin = np.min(ts.index)
    xmax = np.max(ts.index)
    plt.figure(figsize=(15, 6))
    plt.plot(ts)
    plt.title(indicator)
    plt.xticks(np.arange(xmin, xmax + 1, 1), rotation=45)
    plt.xlim(xmin, xmax)
    plt.grid()
    plt.show() 


# In[19]:


from comet_ml import Experiment


# In[24]:


for indicator in dataframe.columns:
    ts = dataframe[indicator]
    ts.dropna(inplace=True)
    ts.index = ts.index.astype(int)
    fig = plot_indicator(ts,indicator)


# In[ ]:




