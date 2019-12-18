#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import sklearn
from sklearn.preprocessing import scale
import sklearn.metrics as sm
from sklearn.metrics import confusion_matrix, classification_report


# In[3]:


from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(7,4))


# In[31]:


print(iris)


# In[5]:


iris = datasets.load_iris()

X = scale(iris.data)
y = pd.DataFrame(iris.target)
variable_names = iris.feature_names
X[0:10]


# In[ ]:





# In[6]:


clustering = KMeans(n_clusters = 3, random_state=5)

clustering.fit(X)


# In[ ]:





# In[7]:


iris_df = pd.DataFrame(iris.data)
iris_df.columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']
y.columns = ['Targets']


# In[8]:


color_theme = np.array(['darkgray', 'lightsalmon', 'powderblue'])

plt.subplot(1,2,1)

plt.scatter(x=iris_df.Petal_Length, y=iris_df.Petal_Width, c=color_theme[iris.target], s=50)
plt.title('Ground Truth Classification')

plt.subplot(1,2,2)

plt.scatter(x=iris_df.Petal_Length, y=iris_df.Petal_Width, c=color_theme[clustering.labels_], s=50)
plt.title('K-Means Classification')


# In[ ]:





# In[9]:


relabel = np.choose(clustering.labels_, [2, 0, 1]).astype(np.int64)

plt.subplot(1,2,1)

plt.scatter(x=iris_df.Petal_Length, y=iris_df.Petal_Width, c=color_theme[iris.target], s=50)
plt.title('Ground Truth Classification')

plt.subplot(1,2,2)

plt.scatter(x=iris_df.Petal_Length, y=iris_df.Petal_Width, c=color_theme[relabel], s=50)
plt.title('K-Means Classification')


# In[10]:


print(classification_report(y, relabel))


# In[22]:


data = pd.read_csv("C:/Users/pc14103/Documents/Analysis/Lynda/Clustering_Sample.csv")


# In[34]:


data = data.fillna(0)


# In[35]:


data.head(5)


# In[36]:


df = data.iloc[:,[1,2,3,4,5,6,7,8]]


# In[37]:


df.head(5)


# In[39]:


scaled_df = scale(df)


# In[40]:


print(scaled_df)


# In[52]:


clustering = KMeans(n_clusters = 3, random_state=5)

clustering.fit(scaled_df)


# In[53]:


color_theme = np.array(['darkgray', 'lightsalmon', 'powderblue'])

plt.scatter(x=data.WEB_LOGIN_DAYS_2017, y=data.AGE, c=color_theme[clustering.labels_], s=50)
plt.title('K-Means Classification')

plt.show()


# In[50]:


clustering.labels_

