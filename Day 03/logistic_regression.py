#!/usr/bin/env python
# coding: utf-8

# # predict whether the customer will buy the product or not

# ### *Importing Libraries*

# In[1]:


import pandas as pd
import numpy as np


# ### *Load Dataset*

# In[2]:


dataset = pd.read_csv('DigitalAd_dataset.csv')
df = dataset.copy()


# ### *Summarize Dataset*

# In[3]:


print(df.shape)
print(df.head())


# ### *Segregate Dataset into X(Input/IndependentVariable) & Y(Output/DependentVariable)*

# In[4]:


X = df.iloc[:, :-1]
Y = df.iloc[:, -1]


# ### *Splitting Dataset into Train & Test*

# In[18]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)


# ### *Feature Scaling*
# ### we scale our data to make all the features contribute equally to the result

# In[19]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# ### *Training*

# In[20]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)


# In[21]:


age = int(input("enter age : "))
sal = int(input("enter your salary : "))
newcust = [[age,sal]]
result = model.predict(sc.transform(newcust))
print(result)

if result == 1:
    print("Customer will Buy")
else:
    print("Customer won't buy")


# ### *Prediction for all Test Data*

# In[22]:


y_pred = model.predict(x_test)
print(np.concatenate((y_p)))


# ### *Evaluating Model*

# In[23]:


from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix : \n", cm)


# In[16]:


(47+19)/(47+4+10+19)


# In[ ]:





# In[25]:


acu_score = accuracy_score(y_test, y_pred)*100

print(f"Accuracy score : {acu_score:.2f}%")


# In[ ]:




