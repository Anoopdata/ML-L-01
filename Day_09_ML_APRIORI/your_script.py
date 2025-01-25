#!/usr/bin/env python
# coding: utf-8

# ### Importing the basic libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ### Importing the dataset

# In[2]:


dataset = pd.read_csv('dataset.csv')
print(dataset.shape)
print(dataset.head(5))


# ### Data Pre-Processing

# In[5]:


transactions = []
for i in range(0, 7500):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20) if str(dataset.values[i,j]) != 'nan'])


# In[6]:


transactions


# ### Training APRIORI

# In[ ]:


# pip install apyori


# In[7]:


from apyori import apriori
rules = apriori(transactions= transactions, min_support = 0.003, min_confidence=0.2, min_lift =3)


# ### Result

# In[9]:


results = list(rules)
results


# In[ ]:


RelationRecord(items=frozenset({'light cream', 'chicken'}), 
               support=0.004533333333333334, 
               ordered_statistics=[OrderedStatistic(items_base=frozenset({'light cream'}), 
                                                    items_add=frozenset({'chicken'}), 
                                                    confidence=0.2905982905982906, 
                                                    lift=4.843304843304844)])


# In[11]:


lhs = [tuple(result[2][0][0]) for result in results]
rhs = [tuple(result[2][0][1]) for result in results]
support_ = [result[1] for result in results]
confidence_ = [result[2][0][2] for result in results]
lift_ = [result[2][0][3] for result in results]


# In[12]:


newdataset = pd.DataFrame(zip(lhs, rhs, support_, confidence_, lift_), 
                          columns=['Item purchased', 'possibility','Support', 'Confidence', 'Lift'])


# In[13]:


newdataset


# In[ ]:


from nbconvert import ScriptExporter

# Specify the path to your notebook
notebook_path = "/home/an/Downloads/Day_09_Feb_01_2022_ML_APRIORI-20250122/Apiriori.ipynb"
script_path = "your_script.py"

# Convert notebook to script
exporter = ScriptExporter()
body, _ = exporter.from_filename(notebook_path)

# Save as .py file
with open(script_path, 'w') as f:
    f.write(body)

print(f"Converted {notebook_path} to {script_path}")

