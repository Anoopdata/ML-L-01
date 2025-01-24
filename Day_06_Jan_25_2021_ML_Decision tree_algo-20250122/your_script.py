#!/usr/bin/env python
# coding: utf-8

# ### *Importing Libraries*

# In[1]:


import pandas as pd
import numpy as np
from sklearn.datasets import load_iris


# ### *Load Dataset*

# In[2]:


dataset = load_iris()


# ### *Summarize Dataset*

# In[3]:


print(dataset.data.shape)
print(dataset.target)


# In[4]:


# input - sepal_length, sepal_width, petal_length, petal_width
# output - 0 - Setosa, 1 - Versicolour, 2 - Virginica


# ### *Segregate Dataset into X(Input/IndependentVariable) & Y(Output/DependentVariable)*

# In[5]:


x = dataset.data
y = dataset.target


# ### *Splitting Dataset into Train & Test*

# In[6]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state=0)


# ### *Find the best max_depth value*

# In[7]:


accurary = []
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

for i in range(1,10):
    model = DecisionTreeClassifier(max_depth= i, random_state=0)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    score = accuracy_score(y_test, pred)
    accurary.append(score)
    
plt.figure(figsize=(12,6))
plt.plot(range(1,10), accurary, color='blue', marker='o')
plt.title("Finding best depth value")
plt.xlabel("predicted value")
plt.ylabel("score")


# ### *Training*

# In[8]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
model.fit(x_train, y_train)


# In[9]:


sl, sw, pl, pw = 6.0, 2.9,4.5,1.5
result = model.predict([[sl, sw, pl, pw]])

if result == 0:
    print("setosa")
elif result == 1:
    print("versicolour")
else:
    print("virginica")


# ### *Prediction for all Test Data*

# In[10]:


y_pred = model.predict(x_test)


# ### *Evaluating Model*

# In[11]:


from sklearn.metrics import accuracy_score, confusion_matrix
print("Accuracy : ", accuracy_score(y_test, y_pred)*100)
print("Confusion matrix : \n", confusion_matrix(y_test, y_pred))


# In[ ]:


from nbconvert import ScriptExporter

notebook_path = "/home/an/Downloads/Day_06_Jan_25_2021_ML_Decision tree_algo-20250122/Decision Tree.ipynb"
python_script_path = "your_script.py"

# Convert notebook to a Python script
exporter = ScriptExporter()
script, _ = exporter.from_filename(notebook_path)

# Save the script
with open(python_script_path, "w") as f:
    f.write(script)

