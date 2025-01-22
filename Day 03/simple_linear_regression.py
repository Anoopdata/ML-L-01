#!/usr/bin/env python
# coding: utf-8

# # Predict the marks obtained by a student based on hours of study

# ### *Importing Libraries*

# In[2]:


import pandas as pd
import numpy as np


# ### *Load Dataset*

# In[3]:


dataset = pd.read_csv('01Students.csv')
df = dataset.copy()


# ### *Summarize Dataset*

# In[4]:


print(df.shape)
print(df.head())


# ### *Segregate Dataset into X(Input/IndependentVariable) & Y(Output/DependentVariable)*

# In[5]:


X = df.iloc[:, :-1]
Y = df.iloc[:, -1]


# ### *Splitting Dataset into Train & Test*

# In[6]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=12)


# ### *Feature Scaling*
# ### we scale our data to make all the features contribute equally to the result

# In[ ]:





# ### *Training*

# In[7]:


from sklearn.linear_model import LinearRegression
std_reg = LinearRegression()
std_reg.fit(x_train, y_train)


# ### *Prediction for all Test Data*

# In[8]:


y_predict = std_reg.predict(x_test)


# In[9]:


y_predict


# In[10]:


y_test


# ### *Evaluating Model*

# In[11]:


from sklearn.metrics import mean_squared_error, r2_score

rscore = r2_score(y_test, y_predict)
print(rscore)


# In[15]:


rmsr = (mean_squared_error(y_test, y_predict))**0.5
print(rmsr)


# ### *Plotting*

# In[16]:


import matplotlib.pyplot as plt

plt.scatter(x_test, y_test)
plt.plot(x_test, y_predict, 'ro:')
plt.show()


# In[ ]:





# In[ ]:




