#!/usr/bin/env python
# coding: utf-8

# # Task 1

# In[7]:


class University:
    __campus=6 #__ is private _ is protected
    def __init__(self,uname,location):
        self.uname = uname
        self.location = location
    def __display2(self):
        print("This is a protected function")
    def display(self):
        #print("\nName: ", self.name, "\nLocation: ", self.location)
        print(f'\nName: {self.uname} \nLocation: {self.location}')
   
class Department(University): #Inheritance
        def __init__(self,name,program, u_name, u_location):
            super().__init__(u_name,u_location)
            #University.__init__()
            self.name = name
            self.program=program
        def display(self):
            super().display()
            print(f"\nName: {self.name} \nPrograms: {self.program}")


# In[8]:


obj2 =University("FAST", "Islamabad")
obj2.__display2()


# In[29]:


obj1=Department("AI",5,"FAST",'Islamabad')
obj1.display()


# # Task 2

# In[30]:


names = ['ayesha', 'zaheer', 'hello', 'sister']

print("Array: ")
for i, value in enumerate(names,1):
    print(i,value)


# # Task 3

# In[11]:


string_1 = ['aahed', 'I', 'ayesha', 'sucks', 'abaca', 'frog']
c = 0
length=len(string_1)
for i in range(0,length):
    if(len(string_1[i])>=2):
        f = string_1[i][0]
        l= string_1[i][-1]
        if(f==l):
            c=c+1
print("Strings with length greater than 2 and the same first and last letters are : ")
print(c)


# In[6]:


string_2 = ['aahed', 'xenia', 'ayesha', 'sucks', 'xerox', 'frog']
for name in string_2:
    if name[0:1]=='x':
        string_2.remove(name)
for name in string_2:
    string_2.sort()
print(string_2)


# In[1]:


tuplee =[(7, 6), (4, 9), (0, 1)]
for i in range(0, len(tuplee)):   
    for j in range(0, (len(tuplee)-i-1)):   
        if (tuplee[j][-1] > tuplee[j + 1][-1]):
            var = tuplee[j] 
            tuplee[j]= tuplee[j + 1] 
            tuplee[j + 1]= var 
print(tuplee)


# In[3]:


list_3=[1, 2, 2]
temp = []
length=len(list_3)
for i in range(0,(length-1)):
    
    if(list_3[i]!=list_3[i+1]):
        temp.append(list_3[i])
print(temp)


# In[7]:


array_1 = [5,1,3,7,8]
array_2 = [4,2,9,6]
#sort array1
for i in range(0, len(array_1)):   
    for j in range(0, (len(array_1)-i-1)):   
        if (array_1[j] > array_1[j + 1]):
            var = array_1[j] 
            array_1[j]= array_1[j + 1] 
            array_1[j + 1]= var 
print(array_1)
#sort array2
for i in range(0, len(array_2)):   
    for j in range(0, (len(array_2)-i-1)):   
        if (array_2[j] > array_2[j + 1]):
            var = array_2[j] 
            array_2[j]= array_2[j + 1] 
            array_2[j + 1]= var 
print(array_2)

#merge
merge_array=array_1+array_2
print(merge_array)
#sort merge array 
for i in range(0, len(merge_array)):   
    for j in range(0, (len(merge_array)-i-1)):   
        if (merge_array[j] > merge_array[j + 1]):
            var = merge_array[j] 
            merge_array[j]= merge_array[j + 1] 
            merge_array[j + 1]= var 
print(merge_array)


# In[22]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataframe = pd.read_csv("drinks.csv")
dataframe


# In[14]:


#Find the rows in the dataset
num_rows=len(dataframe.index)
num_rows


# In[17]:


#Drop na values
nan_values=dataframe.dropna(axis = 0, inplace = False)
nan_values
#checking if the nan values have been dropped 
num_rows=len(nan_values.index)
print(num_rows)


# In[20]:


#A function that takes name of the coloumn and outputs the data in the column
def namecol(name):
    outputs=dataframe.loc[:,name]
    print(outputs)
    
name=input('enter name of column ')
namecol(name)


# In[21]:


#plot histograms of data
dataframe.hist()


# In[24]:


#2D Scatterplot
imagee=plt.scatter(dataframe.country, dataframe.spirit_servings)
print(imagee)

#3D Scatterplot
x_axis = dataframe['beer_servings']
y_axis = dataframe['spirit_servings']
z_axis = dataframe['wine_servings']
matplot = plt.figure().gca(projection='3d')
matplot.scatter(x_axis,y_axis,z_axis)
matplot.set_xlabel('Spirit')
matplot.set_ylabel('Wine')
matplot.set_zlabel('Beer')
plt.show()


# In[29]:


#pie
cou = dataframe['country'].head(5)
var = dataframe['wine_servings'].head(5)
figure = plt.figure(figsize =(11, 8))
print(plt.pie(var, labels = cou))

#bar chart
figure = plt.figure(figsize = (11, 6))
plt.bar(var, cou, color ='blue', width = 6.2)
plt.ylabel("Country")
plt.xlabel("wine Servings")
plt.title("wine Servings across countries")
print(plt.show())


# In[ ]:




