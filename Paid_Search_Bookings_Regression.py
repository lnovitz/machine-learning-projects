import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')
social = pd.read_csv('BookingSearchAnalysis_raw.csv')
bookings = pd.read_csv('Bookings.csv')
social.head(5)
bookings.head(5)

dataset = pd.merge(social, bookings, left_on='Day', right_on='BOOKTRANS_DT', how='outer')
cols = ['Cost', 'Impressions', 'Clicks', 'BOOKINGS']
dataset[cols] = dataset[cols].fillna(0.0).applymap(np.int64)
dataset.head(5)

dataset.plot(x='Clicks', y='BOOKINGS', style='o') 
plt.title('Clicks vs Bookings')  
plt.xlabel('Clicks')  
plt.ylabel('Bookings')  
plt.show()

plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(dataset['BOOKINGS'])

X = dataset['Clicks'].values.reshape(-1,1)
y = dataset['BOOKINGS'].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()  
regressor.fit(X_train, y_train) #training the algorithm

#To retrieve the intercept:
print(regressor.intercept_)
#For retrieving the slope:
print(regressor.coef_)

