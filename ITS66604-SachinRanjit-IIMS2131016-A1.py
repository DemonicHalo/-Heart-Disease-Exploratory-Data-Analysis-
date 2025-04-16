#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[5]:


data = pd.read_csv("diabetes.csv")
data
#load the dataset


# In[6]:


data['Glucose'].replace(0,np.nan, inplace = True)
data['BloodPressure'].replace(0,np.nan, inplace = True)
data['SkinThickness'].replace(0,np.nan, inplace = True)
data['Insulin'].replace(0,np.nan, inplace = True)
data['BMI'].replace(0,np.nan, inplace = True)
data['Age'].replace(0,np.nan, inplace = True)
data


# In[7]:


data.isnull().sum()


# In[10]:


mean_gl = data['Glucose'].astype('float').mean(axis=0)
data['Glucose'].replace(np.nan, mean_gl, inplace = True)

mean_st = data['SkinThickness'].astype('float').mean(axis=0)
data['SkinThickness'].replace(np.nan, mean_st, inplace = True)

mean_bp = data['BloodPressure'].astype('float').mean(axis=0)
data['BloodPressure'].replace(np.nan, mean_bp, inplace = True)

mean_insulin = data['Insulin'].astype('float').mean(axis=0)
data['Insulin'].replace(np.nan, mean_insulin, inplace = True)

mean_bmi = data['BMI'].astype('float').mean(axis=0)
data['BMI'].replace(np.nan, mean_bmi, inplace = True)

mean_age = data['Age'].astype('float').mean(axis=0)
data['Age'].replace(np.nan, mean_age, inplace = True)


# In[11]:


data.isnull().sum()


# In[13]:


#question no 2 starts


# In[14]:


#question no 2 starts
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# here taking a dataset x containing features and y containing labels with Glucose 
X = data[['Glucose']]
y = data['Outcome']


#Now splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[15]:


from sklearn.ensemble import RandomForestClassifier

#Now Initialize the Random Forest Classifier
model = RandomForestClassifier()
#Now Training the model
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

# Predict on the testing data
y_pred = model.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)




# In[16]:


new_data_point = [[172]]  
prediction = model.predict(new_data_point)
print("Prediction:", prediction)
#Prediction for a new data point


# In[22]:


new_data_point = [[140]]  
prediction = model.predict(new_data_point)
print("Prediction:", prediction)
#Prediction for a new data point


# In[23]:


new_data_point = [[90]]  
prediction = model.predict(new_data_point)
print("Prediction:", prediction)
#Prediction for a new data point


# In[17]:


#question no 3
data = data[["Glucose", "Insulin"]]


# In[18]:


msk=np.random.rand(len(df))<0.8
msk
msk=np.random.rand(len(df))<0.8
train=data[msk]
test=data[~msk]



# In[19]:


from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['Glucose']])
train_y = np.asanyarray(train[['Insulin']])
regr.fit (train_x, train_y)
print ('Coefficients: ', regr.coef_) #this means slope (m)
print ('Intercept: ',regr.intercept_)


# In[20]:


import matplotlib.pyplot as plt
plt.scatter(train["Glucose"], train["Insulin"],  color='maroon')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-g')
plt.xlabel("Glucose")
plt.ylabel("Insulin")
plt.show()


# In[21]:


from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['Glucose']])
#Here creating a list out of an array (only selects x variable)

test_y = np.asanyarray(test[['Insulin']]) 
#Here creating a list out of an array (only selects y variable)
test_y_ = regr.predict(test_x) #prediction model 

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y_) )


# In[ ]:




