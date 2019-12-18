#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

from pylab import rcParams

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 10,8


# In[3]:


rooms = 2*np.random.randn(100, 1)+3
rooms[1:10]


# In[4]:


price = 265 + 6*rooms + abs(np.random.randn(100, 1))
price[1:10]


# In[5]:


plt.plot(rooms, price, 'r^')
plt.xlabel("# of Rooms, 2019 Average")
plt.ylabel("2019 Average Home Price, 1000s USD")
plt.show()


# In[7]:


X = rooms
y = price

LinReg = LinearRegression()
LinReg.fit(X,y)
print(LinReg.intercept_, LinReg.coef_)


# In[8]:


print(LinReg.score(X,y))

