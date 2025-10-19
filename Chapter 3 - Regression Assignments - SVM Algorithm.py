#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Assignment - Regression Assignments - SVM algorithm


# In[2]:


import pandas as pd


# In[3]:


dataset=pd.read_csv("insurance_pre.csv")


# In[4]:


dataset


# In[5]:


dataset=pd.get_dummies(dataset,drop_first=True)


# In[6]:


dataset.astype(int)


# In[7]:


dataset.columns


# In[8]:


independent=dataset[['age', 'bmi', 'children','sex_male', 'smoker_yes']]


# In[9]:


independent


# In[10]:


dependent=dataset[['charges']]


# In[11]:


dependent


# In[12]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(independent,dependent,test_size=0.30,random_state=0)


# In[13]:


from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(x_train,y_train)


# In[14]:


y_pred=regressor.predict(x_test)


# In[16]:


from sklearn.metrics import r2_score
r_score=r2_score(y_pred,y_test)
r_score


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




