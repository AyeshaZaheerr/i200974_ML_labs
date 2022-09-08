#!/usr/bin/env python
# coding: utf-8

# In[ ]:


y=6
def display():
    print(y)
    #y=7
display()
#First searches in the local scope than global scope 


# In[1]:


y=6
def min():
    res=min([1,2,3]) #self call hojaye gi 
    print(res)
min()
    


# In[70]:


#task1
import numpy as np
vector=np.random.randint(9,size=(5,1))
vector

#task2
vectorrr=[]
last_index=vector[len(vector)-1]
#print("the original vector :")
#print(vector)
for i in range(0,len(vector)-1):
    #print(vector[i])
    vectorrr.append(vector[i])
    
summ=0   
#print("vector without the last index")
for i in vectorrr:
    #print(i)
    summ=summ+i
#print("the sum of the indxes")
#print(summ)
#last_index
l=(summ/last_index)*(-1)
#print(l)

new_vector=np.ones(vector.shape)
new_vector 

new_vector[-1]=l
new_vector


dot_product=np.dot(vector.T,new_vector)
dot_product


# In[72]:


np.arccos(0)


# In[88]:



#task3
sq=[]
for i in new_vector:
    #print(i)
    sq.append(pow(i,2))
summ=np.sum(sq)
print(summ)
magnitude_b=pow(summ,1/2)
magnitude_b

sq=[]
for i in vector:
    #print(i)
    sq.append(pow(i,2))
summ=np.sum(sq)
print(summ)
magnitude_a=pow(summ,1/2)
magnitude_a


# In[97]:


projection=(np.dot(vector.T,new_vector))/magnitude_b
projection=projection*new_vector
projection


# In[ ]:


#task4
#we cannnot recreate the original vector from since the projection is zero 


# In[5]:


#Take two vectors and project one on the other, and find the initial vectors from the projections.
#task5
import numpy as np 
orthon_1 = np.array([2,1])
orthon_1 = orthon_1.reshape(2,-1)
#print(orthon_1)

#print("2nd matrix ")
orthon_2 = np.array([1,4])
orthon_2 = orthon_2.reshape(2,-1)
#print(orthon_2)

sq=[]
for i in orthon_2:
    #print(i)
    sq.append(pow(i,2))
summ=np.sum(sq)
#print(summ)
mag_orthon_2=pow(summ,1/2)
mag_orthon_2

dot_product=np.dot(orthon_1.T,orthon_2)

projection=(dot_product/mag_orthon_2)
projection=projection*orthon_2
print("projection of orthon_1 on orthon_2")
print(projection)

#angle=np.arccos(dot_product)
#print(angle)



# In[6]:


import numpy as np 
orthon_1 = np.array([2,1])
orthon_1 = orthon_1.reshape(2,-1)
#print(orthon_1)

#print("2nd matrix ")
orthon_2 = np.array([1,4])
orthon_2 = orthon_2.reshape(2,-1)
#print(orthon_2)

sq=[]
for i in orthon_1:
    #print(i)
    sq.append(pow(i,2))
summ=np.sum(sq)
#print(summ)
mag_orthon_1=pow(summ,1/2)
mag_orthon_1

dot_product=np.dot(orthon_2.T,orthon_1)

projection=(dot_product/mag_orthon_1)
projection=projection*orthon_1
print("projection of orthon_2 on orthon_1")
print(projection)


# In[8]:


#task6


# In[ ]:




