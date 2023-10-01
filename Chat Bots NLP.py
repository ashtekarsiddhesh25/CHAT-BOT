#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import nltk
import string
import random


# In[2]:


f = open("C:/Users/Siddhesh Ashtekar/OneDrive/Desktop/data.txt",'r',errors='ignore')
raw_doc = f.read()


# In[3]:


raw_doc


# # Preprocessing raw doc

# In[4]:


raw_doc = raw_doc.lower() #converting entire text to lowercase
nltk.download('punkt') #using the punkt tokeniser
nltk.download('wordnet') #using the wordnet directory


# In[5]:


raw_doc


# In[6]:


sentence_tokens = nltk.sent_tokenize(raw_doc)
word_tokens = nltk.word_tokenize(raw_doc)


# In[7]:


sentence_tokens[:5] # end up with 5 sentences from the raw doc


# In[8]:


word_tokens[:5]


# # Text Processing Steps

# In[9]:


from nltk.stem import WordNetLemmatizer
  
lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punc_dict = dict((ord(punct), None)for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punc_dict)))


# In[10]:


# define Greet functions'
greet_inputs = ('hello' , 'hi' , 'whassup' , 'how are you?')
greet_responses = ('hi' , 'Hey' ,'Hey There!','Hello sir!!')
def greet(sentence):
    for word in sentence.split():
        if word.lower() in greet_inputs:
            return random.choice(greet_responses)


# In[11]:


# Response generation of bots
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[12]:


# Generating response to the user 
def response(user_response):
    robo1_response = ''
    TfidfVec = TfidfVectorizer(tokenizer = LemNormalize , stop_words = 'english')
    tfidf = TfidfVec.fit_transform(sentence_tokens)
    vals = cosine_similarity(tfidf[-1],tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if (req_tfidf == 0):
        robo1_response = robo1_response + "I am sorry sir .Unable to understand you!"
        return robo1_response
    else:
        robo1_response = robo1_response + sentence_tokens[idx]
        return robo1_response


# In[ ]:


# Defining the chart flow 
flag = True
print("Hello sir ! . How can i help you ")
while(flag == True):
    user_response = input()
    user_response = user_response.lower()
    if(user_response != 'bye'):
        if(user_response == 'thank you' or user_response == 'thanks'):
            flag = False
            print('Bot: Your Welcome..')
        else:
            if(greet(user_response) != None):
                print('Bot '+greet(user_response))
            else:
                sentence_tokens.append(user_response)
                word_tokens = word_tokens + nltk.word_tokenize(user_response)
                final_words = list(set(word_tokens))
                print('Bot: ',end = '')
                print(response(user_response))
                sentence_tokens.remove(user_response)
    else:
        flag = False
        print('Bot: Goodbye Sir!')
        


# In[ ]:




