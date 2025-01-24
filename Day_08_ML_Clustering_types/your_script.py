#!/usr/bin/env python
# coding: utf-8

# ### *Importing Libraries*

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# ### *Load Dataset*

# In[2]:


dataset = pd.read_csv('dataset.csv')


# ### *Summarize Dataset*

# In[4]:


print(dataset.shape)
print(dataset.head(10))


# ### *Segregate*

# In[ ]:





# ### *Mapping*

# In[5]:


dataset['Gender'] = dataset['Gender'].map({'Male':0, 'Female':1}).astype(int)
print(dataset.head())


# ### *Finding the Optimized K Value*

# In[7]:


from sklearn.cluster import KMeans
lst = []
for i in range(1,11):
    km = KMeans(n_clusters=i, random_state=0)
    km.fit(dataset)
    lst.append(km.inertia_)

plt.plot(range(1,11),lst,color='blue', marker='o')
plt.title('Optimal K Value')
plt.xlabel('Number of Clusters')
plt.ylabel('inertia')
plt.show()


# # Training

# In[14]:


model = KMeans(n_clusters=5, random_state=0)
y_means = model.fit_predict(dataset)
y_means


# ### *Visualize*

# In[15]:


labels = model.labels_
labels = pd.DataFrame(labels)

df = pd.concat([dataset, labels], axis=1)
df = df.rename(columns={0:'label'})

df.plot.scatter(x='Annual Income (k$)', y= 'Spending Score', c='label', colormap='Set1')


# # using AgglomerativeClustering

# In[18]:


from sklearn.cluster import AgglomerativeClustering
model1 = AgglomerativeClustering(n_clusters = 5, linkage='average')

y_mean = model1.fit_predict(dataset)
y_mean


# In[20]:


labels = model1.labels_
labels = pd.DataFrame(labels)

df = pd.concat([dataset, labels], axis=1)
df = df.rename(columns={0:'label'})

df.plot.scatter(x='Annual Income (k$)', y= 'Spending Score', c='label', colormap='Set1')
plt.title('Hierarchical Clustering')
plt.show()


# In[ ]:




