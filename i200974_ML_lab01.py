#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import KFold


# In[4]:


dataframe=pd.read_csv("C:\\Users\\ayesh\\Downloads\\ECG200_TRAIN.csv",delimiter='  ')
dataframe


# In[5]:


#picking out the label column of 1s and -1s  
#array=df.to_numpy()
#data_labels =df["-1.0000000e+00"]
#data_labels


# In[6]:


# data_features=df.iloc[0:99,1:97]
# slicee=int(0.7*len(array))
# seventy=df[:slicee]
# thirty=df[slicee:]
# len(thirty) 


# In[12]:


#making 2 new dataframes one w postive and one with negative values 
dataframe_positve=dataframe[dataframe["-1.0000000e+00"]==1.0]
print(dataframe_positve)
print(dataframe_positve['-1.0000000e+00'].value_counts()) 
#postive 1s have 69 rows 

dataframe_negative=dataframe[dataframe["-1.0000000e+00"]==-1.0]
print(dataframe_negative)
print(dataframe_negative['-1.0000000e+00'].value_counts())
#negative 1s have 30 rows 


# In[15]:


#now further split the dataframes into 4 parts maintaining 17:7 ratio 
var1=int(69/4)
var2=int(30/4)
print(var1,":",var2)


# In[21]:


#concatinate the dfs and check their ratios with value counts 
dataframe_1 = pd.concat([dataframe_positve[:var1], dataframe_negative[:var2]])
print(dataframe_1)
d1=dataframe_1['-1.0000000e+00'].value_counts()
print(d1)
#shuffle the dataframe 
dataframe_1 = shuffle(dataframe_1)
print(dataframe_1)

dataframe_2 = pd.concat([dataframe_positve[var1:2*var1], dataframe_negative[var2:2*var2]])
print(dataframe_2)
d2=dataframe_2['-1.0000000e+00'].value_counts()
print(d2)
#shuffle the dataframe 
dataframe_2 = shuffle(dataframe_2)
print(dataframe_2)

dataframe_3 = pd.concat([dataframe_positve[2*var1:3*var1], dataframe_negative[2*var2:3*var2]])
print(dataframe_3)
d3=dataframe_3['-1.0000000e+00'].value_counts()
print(d3)
#shuffle the dataframe 
dataframe_3 = shuffle(dataframe_3)
print(dataframe_3)

dataframe_4 = pd.concat([dataframe_positve[3*var1:70], dataframe_negative[3*var2:31]])
print(dataframe_4)
d4=dataframe_4['-1.0000000e+00'].value_counts()
print(d4)
#shuffle the dataframe 
dataframe_4 = shuffle(dataframe_4)
print(dataframe_4)


# In[58]:


x_dataframe = pd.concat([dataframe_1, dataframe_2, dataframe_3],axis=0)
y_dataframe = dataframe_4
#print(x)
#print(y)
x_train = x_dataframe.drop('-1.0000000e+00',axis=1)
y_train = x_dataframe["-1.0000000e+00"]
x_test = y_dataframe.drop('-1.0000000e+00',axis=1)
y_test = y_dataframe["-1.0000000e+00"]

classifier = KNeighborsClassifier()
classifier = classifier.fit(x_train, y_train)
pred = classifier.predict(x_test)
classifier.score(x_test,y_test)
accuracies_1=[]
print ("Accuracy : " , accuracy_score(y_test,pred)*100)  
accuracies_1.append(accuracy_score(y_test,pred)*100)
print("Report : \n", classification_report(y_test, pred))
print("F1 Score : ",f1_score(y_test, pred, average='macro')*100)


# In[59]:


x_dataframe = pd.concat([dataframe_1, dataframe_2, dataframe_4],axis=0)
y_dataframe = dataframe_3
#print(x)
#print(y)
x_train = x_dataframe.drop('-1.0000000e+00',axis=1)
y_train = x_dataframe["-1.0000000e+00"]
x_test = y_dataframe.drop('-1.0000000e+00',axis=1)
y_test = y_dataframe["-1.0000000e+00"]

classifier = KNeighborsClassifier()
classifier = classifier.fit(x_train, y_train)
pred = classifier.predict(x_test)
classifier.score(x_test,y_test)
print ("Accuracy : " , accuracy_score(y_test,pred)*100)  
accuracies_1.append(accuracy_score(y_test,pred)*100)
print("Report : \n", classification_report(y_test, pred))
print("F1 Score : ",f1_score(y_test, pred, average='macro')*100)


# In[60]:


x_dataframe = pd.concat([dataframe_2, dataframe_3, dataframe_4],axis=0)
y_dataframe = dataframe_1
#print(x)
#print(y)
x_train = x_dataframe.drop('-1.0000000e+00',axis=1)
y_train = x_dataframe["-1.0000000e+00"]
x_test = y_dataframe.drop('-1.0000000e+00',axis=1)
y_test = y_dataframe["-1.0000000e+00"]

classifier = KNeighborsClassifier()
classifier = classifier.fit(x_train, y_train)
pred = classifier.predict(x_test)
classifier.score(x_test,y_test)
print ("Accuracy : " , accuracy_score(y_test,pred)*100)  
accuracies_1.append(accuracy_score(y_test,pred)*100)
print("Report : \n", classification_report(y_test, pred))
print("F1 Score : ",f1_score(y_test, pred, average='macro')*100)


# In[61]:


x_dataframe = pd.concat([dataframe_1, dataframe_3, dataframe_4],axis=0)
y_dataframe = dataframe_2
#print(x)
#print(y)
x_train = x_dataframe.drop('-1.0000000e+00',axis=1)
y_train = x_dataframe["-1.0000000e+00"]
x_test = y_dataframe.drop('-1.0000000e+00',axis=1)
y_test = y_dataframe["-1.0000000e+00"]

classifier = KNeighborsClassifier()
classifier = classifier.fit(x_train, y_train)
pred = classifier.predict(x_test)
classifier.score(x_test,y_test)
print ("Accuracy : " , accuracy_score(y_test,pred)*100)  
accuracies_1.append(accuracy_score(y_test,pred)*100)
print("Report : \n", classification_report(y_test, pred))
print("F1 Score : ",f1_score(y_test, pred, average='macro')*100)


# In[62]:


#now we will use sk learn functions and libaraies inorder to compare the accuracies 


# In[63]:


x = dataframe.drop('-1.0000000e+00',axis=1)
y = dataframe['-1.0000000e+00']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 39)
classifier = KNeighborsClassifier()
classifier = classifier.fit(x_train, y_train)
p = classifier.predict(x_test)
classifier.score(x_test,y_test)
print ("Accuracy : " , accuracy_score(y_test,p)*100)  
print("Report : \n", classification_report(y_test, p))
print("F1 Score : ",f1_score(y_test, p, average='macro')*100)


# In[64]:


#now apply the k fold cross validation 
print(accuracies_1)


# In[74]:


k_fold = KFold(n_splits = 4,shuffle = True)
accuracies_2=[]
num=1
for a, b in k_fold.split(x): 
    x_train = x.iloc[a]
    x_test  = x.iloc[b]
    y_train = y.iloc[a]
    y_test  = y.iloc[b]    
    
    classifier = KNeighborsClassifier()
    classifier = classifier.fit(x_train, y_train)
    prediction = classifier.predict(x_test)
    classifier.score(x_test,y_test)
    print ("Accuracy " ,num, " is : ", accuracy_score(y_test,prediction)*100)
    num=num+1
    accuracies_2.append(accuracy_score(y_test,prediction)*100)


# In[75]:


sum1=sum(accuracies_1)
avg1 = (sum1/4)
print(avg1)

sum2=sum(accuracies_2)
avg2 = sum2/4
print(avg2)


# In[ ]:


#Average Accuracy Results using Sklearn are better than that of numpy

