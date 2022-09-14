#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#ayesha Zaheer
#20i0974 
#AI K 


# In[26]:


import numpy as np
import pandas as pd
import statistics as st
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# In[1]:


#task8
outcomes=['A','B','C','D','E']
str1 = ""
prob = []
for i in outcomes:
    str1 = ""
    str1 = str1+ i
    str3 = str1
    for j in outcomes:
        str1 = str3
        str1 = str1+ j
        str2 = str1
        for k in outcomes:
            str1 = str2
            str1 += k
            prob.append(str1)
prob


# In[2]:


count_1=0
for i in range(0,125):
    var=prob[i]
    if(var[0]==var[1] or var[1]==var[2] or var[0]==var[2]):
        count_1=count_1+1
print(count_1)

#probability of event = number of favourable outcomes/ total number of outcomes
success_prob= count_1/125
print(success_prob)


# In[3]:


count_2=0
for i in range(0,125):
    var=prob[i]
    if(var[0]!=var[1] and var[1]!=var[2] and var[0]!=var[2]):
        count_2=count_2+1
print(count_2)

failure_prob= count_2/125
print(failure_prob)


# In[4]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.set_ylabel('Probability')
ax.set_title('Probability Distribution function')
title = ['go to the same city', 'dont go to the same city']
probs = [52,48]
ax.bar(title,probs)
plt.show()


# In[ ]:


#Is it a discrete distribution or a continuous one?
#it is discrete because fixed values are outputs and the graph is also not a curve 
#it is univariate 


# In[6]:


df = pd.read_csv("C:\\Users\\ayesh\\Downloads\\Housing.csv - Housing.csv.csv")
df


# In[7]:


#Apply univariate analysis on a feature in the given dataset
meann=df['stories'].mean()
print("The mean for stories is:",meann)


# In[8]:


standard_dev=df['stories'].std()
print("standard devitation of stories is:" ,standard_dev)


# In[9]:


variance=standard_dev**2
print("variance of stories is " ,variance)


# In[10]:


z_score=abs(stats.zscore(df['stories']) )
z_score


# In[11]:


column = ['mainroad', 'guestroom', 'basement', 'hotwaterheating','airconditioning', 'prefarea', 'furnishingstatus']
df[column] = df[column].apply(LabelEncoder().fit_transform)
x = df.drop(['furnishingstatus'], axis = 1)
y = df['furnishingstatus']

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=51)
classifier_1 = tree.DecisionTreeClassifier()
classifier_1 = classifier_1.fit(X_train, y_train)
prediction=classifier_1.predict(X_test)
classifier_1.score(X_test,y_test)
print ("Accuracy : " , accuracy_score(y_test,prediction)*100)  
print("Report : \n", classification_report(y_test, prediction))
print("F1 Score : ",f1_score(y_test, prediction, average='macro')*100)


# In[12]:


np_array=np.array(df.drop(['furnishingstatus'], axis=1))
m1 = np.mean(np_array,axis = 0)
ones1 = np.ones(np_array.shape)
print(ones1.shape)
mean1 = m1 * ones1
Z1 = np_array - mean1
Standard=np.std(np_array,axis=0)
z=Z1/Standard
print(z)


# In[14]:


X=z
y = df['furnishingstatus']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=51)
classifier_1 = tree.DecisionTreeClassifier()
classifier_1 = classifier_1.fit(X_train, y_train)
prediction=classifier_1.predict(X_test)
classifier_1.score(X_test,y_test)
print ("Accuracy : " , accuracy_score(y_test,prediction)*100)  
print("Report : \n", classification_report(y_test, prediction))
print("F1 Score : ",f1_score(y_test, prediction, average='macro')*100)


# In[ ]:


#Classifiers performed better on the normalized data as all of the features were given equal weights


# In[16]:


plt.scatter(df['area'], df['length'])
plt.xlabel("length") 
plt.ylabel("area") 
plt.show()


# In[18]:


plt.scatter(df['area'], df['width'])
plt.xlabel("width") 
plt.ylabel("area") 
plt.show()


# In[19]:


fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot( projection='3d')

ax.scatter(df['area'], df['width'], df['length'], c='blue')
plt.title("3D") 
plt.xlabel("length") 
plt.ylabel("area") 
plt.show()


# In[24]:


covarience=np.cov(df['bedrooms'],df['area'],bias=True)
print(covarience)

print(" ")
correlation=np.corrcoef(df['bedrooms'],df['area'])
print(correlation)


# In[25]:


np.corrcoef(df['area'],df['length'])

plt.scatter(df['area'],df['length'])
plt.plot(np.unique(df['area']), np.poly1d(np.polyfit(df['area'], df['length'], 1))(np.unique(df['area'])), color='blue')


# In[ ]:




