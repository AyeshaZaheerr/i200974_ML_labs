#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collections import defaultdict


# In[2]:


b=defaultdict(int) #int is factory function 


# In[3]:


b['apples']=6
b['bananas']=7


# In[4]:


b['onions']


# In[5]:


b['apples']


# In[6]:


b2=defaultdict(float)
b2['onions']


# In[8]:


def myvalue():
    return 1/7

b=defaultdict(myvalue)


# In[9]:


b['areeba']


# In[30]:


import pandas as pd
import numpy as np


# In[31]:


pos = "I love mangoes Mangoes are a lot tastier than other fruits I like mangoes because they are good Apples are good too but I prefer mangoes People say mango is the king of fruit so that is why mangoes are good Mangoes are good because they are juicy"
neg = "Eating too many mangoes can have a neg_string impact on your health Mangoes are good but not too good I like apples more than mangoes I don’t like mangoes because they are expensive Mangoes are not good.I don’t like mangoes"

pos_string = pos.split()
pos_string1 = []
for i in pos_string:
    temp = i.lower()
    pos_string1.append(temp)

neg_string = neg.split()    
neg_string1 = []
for i in neg_string:
    temp = i.lower()
    neg_string1.append(temp)
    

# document and their unique values and counts
total_sentence = pos_string1 + neg_string1
unique_words = np.unique(total_sentence)
unique_count = len(unique_words)
        
print(unique_count)


# In[32]:


p_positive = {}
p_negative = {}
def prob_words_in_unique(x,pos_string1,neg_string1,unique_count):
    
    # unique words
    #finding the probs of positive words inn unique 
    for i in x:
        count_positive = 0
        for j in pos_string1:
            if i == j:
                count_positive=count_positive+1
        p_positive[i] = ((count_positive+1)/(len(pos_string1) + unique_count))
        
    print(p_positive)
    
    #finding the probs of negative words in unique 
    for i in x:   
        count_neg = 0
        for k in neg_string1:
            if i == k:
                count_neg=count_neg+1
        p_negative[i] = ((count_neg+1)/(len(neg_string1) + unique_count))
    print(p_negative)
    
def prob_words_notin_unique(x,pos_string1,neg_string1,unique_count): #coverting to dict and updating key de rahay hain
    
    len_pos=len(pos_string1)
    p_positive.update({x:(1)/(len_pos) + unique_count})
     
    len_neg=len(neg_string1)  
    p_negative.update({x:(1)/(len_neg + unique_count)})
    
prob_words_in_unique(unique_words,pos_string1,neg_string1,unique_count) 


test_sentence = "apples are shit"
x = test_sentence.split()
# Multiplying the probabilites
probsneg = []
probspos = []
for i in x:
    if i in unique_words:
        probspos.append(p_positive[i])
        probsneg.append(p_negative[i])
    else:
        prob_words_notin_unique(i,pos_string1,neg_string1,unique_count)
        print(i)
        print(p_positive[i])
        probspos.append(p_positive[i])
        probsneg.append(p_negative[i])

len_neg=len(neg_string1) 
len_pos=len(pos_string1)

# probability of neg & pos sentences
probn = len_neg/len_pos+len_neg
probp = len_pos/len_pos+len_neg

prob_1 = probp * np.prod(probspos)
prob_2 = probn * np.prod(probsneg)


print('Probabilites\n\n')
print('Positives',prob_1)
print('Negatives',prob_2)

if prob_1 > prob_2:
    print("Sentence is positive")
else:
    print("Sentence is negative")


# In[ ]:




