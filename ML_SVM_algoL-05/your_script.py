#!/usr/bin/env python
# coding: utf-8

# ### *Importing Libraries*

# In[1]:


import pandas as pd
import numpy as np


# ### *Load Dataset*

# In[2]:


dataset = pd.read_csv('loan_data.csv')
LoanPrep = dataset.copy()


# ### *Summarize Dataset*

# In[3]:


print(LoanPrep.dtypes)


# In[4]:


LoanPrep.isnull().sum()


# In[5]:


LoanPrep = LoanPrep.dropna()


# ### *Data Preprocessing*

# In[6]:


LoanPrep = LoanPrep.drop(['gender'], axis=1)


# In[7]:


LoanPrep.head(10)


# In[8]:


LoanPrep = pd.get_dummies(LoanPrep, drop_first=True)
LoanPrep.head(10)


# ### *Feature Scaling*

# In[10]:


from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()

LoanPrep['income'] = scalar.fit_transform(LoanPrep[['income']])
LoanPrep['loanamt'] = scalar.fit_transform(LoanPrep[['loanamt']])


# In[11]:


LoanPrep.head(10)


# ### *Segregate Dataset into X(Input/IndependentVariable) & Y(Output/DependentVariable)*

# In[12]:


x = LoanPrep.iloc[:, :-1]
y = LoanPrep.iloc[:, -1]


# ### *Splitting Dataset into Train & Test*

# In[13]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state = 1234)


# In[ ]:





# ### *Training*

# In[14]:


from sklearn.svm import SVC
svc = SVC(kernel='sigmoid')
svc.fit(x_train, y_train)


# ### *Prediction for all Test Data*

# In[15]:


y_predict = svc.predict(x_test)


# ### *Evaluating Model*

# In[16]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict)
print(cm)

score = svc.score(x_test,y_test)
print(score)


# In[17]:


(104+26)/(104+26+28+1)


# In[ ]:




