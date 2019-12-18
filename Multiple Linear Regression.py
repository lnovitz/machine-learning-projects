#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from pylab import rcParams

import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] =5, 4


# In[1]:


import seaborn as sb
sb.set_style('whitegrid')
from collections import Counter


# In[5]:


address = 'C:/Users/pc14103/Documents/Analysis/Lynda/enrollment_forecast.csv'

enroll = pd.read_csv(address)
enroll.columns = ['year', 'roll', 'unem', 'hgrad', 'inc']
enroll.head()


# In[6]:


sb.pairplot(enroll)


# In[7]:


print(enroll.corr())


# In[8]:


enroll_data = enroll[['unem', 'hgrad']].values

enroll_target = enroll[['roll']].values

enroll_data_names = ['unem', 'hgrad']

X, y = scale(enroll_data), enroll_target


# In[9]:


missing_values = X==np.NAN
X[missing_values == True]


# In[10]:


LinReg = LinearRegression(normalize=True)

LinReg.fit(X, y)

print(LinReg.score(X, y))

