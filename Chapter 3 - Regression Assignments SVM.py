#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Regression Assignments SVM


# In[1]:


import pandas as pd


# In[2]:


dataset=pd.read_csv("insurance_pre.csv")


# In[3]:


dataset


# In[4]:


dataset=pd.get_dummies(dataset,drop_first=True)


# In[5]:


dataset.astype(int)


# In[6]:


dataset.columns


# In[7]:


independent=dataset[['age', 'bmi', 'children','sex_male', 'smoker_yes']]


# In[8]:


independent


# In[9]:


dependent=dataset[['charges']]


# In[10]:


dependent


# In[11]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(independent,dependent,test_size=0.30,random_state=0)


# In[19]:


#Introducing Standard Scaler
from sklearn.preprocessing import StandardScaler

sc_X=StandardScaler()
sc_Y=StandardScaler()

x_train=sc_X.fit_transform(x_train)
x_test=sc_X.transform(x_test)

y_train=sc_Y.fit_transform(y_train)
y_test=sc_Y.transform(y_test)

y_train=sc_Y.fit_transform(y_train.reshape(-1,1)).ravel()


# In[20]:


x_train


# In[21]:


x_test


# In[22]:


y_train


# In[23]:


y_test


# In[24]:


from sklearn.svm import SVR
regressor=SVR(kernel='linear')
regressor.fit(x_train,y_train)


# In[14]:


y_pred=regressor.predict(x_test)


# In[25]:


from sklearn.metrics import r2_score
r_score=r2_score(y_test,y_pred)
r_score


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




