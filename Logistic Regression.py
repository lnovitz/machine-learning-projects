#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn

from pandas import Series, DataFrame
from pylab import rcParams
from sklearn import preprocessing


# In[2]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 5, 4
sb.set_style('whitegrid')


# In[5]:


address = 'C:/Users/pc14103/Documents/Analysis/Lynda/titanic-training-data.csv'
titanic_training = pd.read_csv(address)
titanic_training.columns = ['PassengerID', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
print(titanic_training.head())


# In[6]:


print(titanic_training.info())


# Survived (0 = No, 1 = Yes)
# Pclass - Passenger Class
# SipSp - number of siblings/spouses aboard
# parch - number of parents/children aboard
# ticket - ticket number
# fare - british pounds
# embarked - port (C = France, Q = UK, S = Ireland)

# In[8]:


#is our target var binary? Yes
sb.countplot(x='Survived', data=titanic_training, palette='hls')


# In[9]:


#do we have missing values? yes
titanic_training.isnull().sum()


# In[10]:


titanic_training.describe()


# In[11]:


#what variables are relevant for prediciting survival?
#survived, Pclass, Sex, Age, SibSp, Parch, Fare
titanic_data = titanic_training.drop(['Name', 'Ticket', 'Cabin'], axis=1)
titanic_data.head()


# In[13]:


sb.boxplot(x='Parch', y='Age', data=titanic_data, palette='hls')


# In[ ]:


#want to predict age based on Parch, # of relatives


# In[14]:


Parch_groups = titanic_data.groupby(titanic_data['Parch'])
Parch_groups.mean()


# In[15]:


def age_approx(cols):
    Age = cols[0]
    Parch = cols[1]
    
    if pd.isnull(Age):
        if Parch == 0:
            return 32
        elif Parch == 1:
            return 24
        elif Parch == 2:
            return 17
        elif Parch == 3:
            return 33
        elif Parch == 4:
            return 45
        #elif Parch == 5:
        #    return 39
        #elif Parch == 6:
        #    return 43
        else:
            return 30
    else: return Age


# In[16]:


titanic_data['Age'] = titanic_data[['Age', 'Parch']].apply(age_approx, axis=1)
titanic_data.isnull().sum()
                                    


# In[17]:


titanic_data.dropna(inplace=True)
titanic_data.reset_index(inplace=True, drop=True)

print(titanic_data.info())


# In[18]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()


# In[21]:


gender_cat = titanic_data['Sex']
gender_encoded = label_encoder.fit_transform(gender_cat)
gender_encoded[0:5] # 1 = male, 0 = female


# In[22]:


titanic_data.head()


# In[23]:


gender_DF = pd.DataFrame(gender_encoded, columns=['male_gender'])
gender_DF.head()


# In[24]:


embarked_cat = titanic_data['Embarked']
embarked_encoded = label_encoder.fit_transform(embarked_cat)
embarked_encoded[0:100] #oh no, multinomial, can only deal with binary. need to change with one hot encoder


# In[25]:


from sklearn.preprocessing import OneHotEncoder
binary_encoder = OneHotEncoder(categories='auto')
embarked_1hot = binary_encoder.fit_transform(embarked_encoded.reshape(-1,1)) #reshape as single column matrix
embarked_1hot_mat = embarked_1hot.toarray()
embarked_DF = pd.DataFrame(embarked_1hot_mat, columns = ['C', 'Q', 'S'])
embarked_DF.head()


# In[26]:


titanic_data.drop(['Sex', 'Embarked'], axis=1, inplace=True)
titanic_data.head()


# In[27]:


titanic_dmy = pd.concat([titanic_data, gender_DF, embarked_DF], axis=1, verify_integrity=True).astype(float)
titanic_dmy[0:5]


# In[29]:


#check for independence between features/ correlation between varialbes
sb.heatmap(titanic_dmy.corr())


# In[30]:


#Fare and Pclass have too strong a correlation, drop!
titanic_dmy.drop(['Fare', 'Pclass'], axis=1, inplace=True)
titanic_dmy.head()


# In[31]:


#rule of thumb, 50 records per predictive feature. 6 predictive features (passenger ID, age, sibsp, parch, male, C/Q/S). 
#6 times 50 = 300 total records minimum. we have 800 at least. good to go! 
titanic_dmy.info()


# In[32]:


X_train, X_test, y_train, y_test = train_test_split(titanic_dmy.drop('Survived', axis=1),
                                                   titanic_dmy['Survived'], test_size=0.2,
                                                   random_state=200) #set seed


# In[33]:


print(X_train.shape)
print(y_train.shape)


# In[34]:


X_train[0:5]


# In[ ]:


# we treated missing values yay and encoded stuff yay


# In[35]:


LogReg = LogisticRegression(solver='liblinear')
LogReg.fit(X_train, y_train)


# In[36]:


y_pred = LogReg.predict(X_test)


# In[37]:


#how well did our model perform?
print(classification_report(y_test, y_pred))


# In[38]:


# k fold cross validation
y_train_pred = cross_val_predict(LogReg, X_train, y_train, cv=5)
confusion_matrix(y_train, y_train_pred) #377 correct, 180 correct, 91 incoorect, 63 incorrrect


# In[39]:


precision_score(y_train, y_train_pred)


# In[40]:


#lets make a test prediction
titanic_dmy[863:864]


# In[41]:


test_passenger = np.array([866, 40, 0, 0, 0, 0, 0, 1]).reshape(1,-1)
print(LogReg.predict(test_passenger))
print(LogReg.predict_proba(test_passenger))

