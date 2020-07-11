#!/usr/bin/env python
# coding: utf-8

# In[155]:


from itertools import chain, combinations
from scipy.sparse import csr_matrix
import string
import nltk
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import pandas as pd


def tokenize(doc, keep_internal_punct=False):
    splitList = []
    lowercasedoc = doc.lower()
    
    if(keep_internal_punct == False):
        finalList = re.sub('(\W+)', " ", lowercasedoc).split()
    else:
        regexString='[\w_][^\s]*[\w_]|[\w_]'
        finalList = re.findall(regexString, lowercasedoc)
    return np.array([d.lower() for d in finalList if len(d)>=3])


def create_csv(vocab,xlist,qr):
    global wordlist
    row=[]
    col=[]
    datalist=[]
    increment=0
    
    for xl in xlist:
       
        for a,data in nltk.FreqDist(xl).items():
            voc_keys = vocab.keys()
            if a in voc_keys:
                datalist.append(data)
                row.append(increment)
                col.append(vocab[a])
        increment=increment+1
    return csr_matrix((np.array(datalist),(np.array(row),np.array(col))),shape=(increment,len(vocab.keys())))



# In[156]:


def create_lables(tw,pw,labels):
    for i in range(tw):
        labels.append(1)
    for i in range(pw):
        labels.append(0)
    return labels


# In[157]:


def token_features(tokens, feats):
    ###TODO
    output_List = Counter(tokens)
    
    for k,v in output_List.items():
        feats['token='+k]= v


# In[161]:


def classification():
    dump=['USA-classify.csv','UK-classify.csv','India-classify.csv']
    x=[]
    y=[]
    qr = 0
    labels=[]
    
    
    tweets_list_pos=pd.read_csv("pos.csv",sep='\t')['tweets'].tolist()
    tw = 0
    tweets_list_neg=pd.read_csv("neg.csv",sep='\t')['tweets'].tolist()
    pw = 0
    for tw1 in tweets_list_pos:
       
        x.extend(list(tokenize(tw1)))
    for tw2 in tweets_list_pos:
        
        y.append(list(tokenize(tw2)))
        tw = tw+1
      
    for twp1 in tweets_list_neg:
        
        x.extend(list(tokenize(twp1)))
    for twp2 in tweets_list_neg:
        
        y.append(list(tokenize(twp2)))
        pw = pw + 1
        
    labels = create_lables(tw,pw,labels)
    
    
    vocab={}
   
    i=0
    
    for key,values in nltk.FreqDist(x).items():
        if values>=2:
            vocab[key]=i
            i+=1
   
    classifier=LogisticRegression()
    classifier.fit(create_csv(vocab,y,qr),labels)
    positive=0
    negative=0
    for location_city in dump:
        y=[]
        
        tweetsn=pd.read_csv(location_city,sep='\t')['text'].tolist()
        for t in tweetsn:
            l=list(tokenize(t))
            y.append(l)
        
        
        lab=list(classifier.predict(create_csv(vocab,y,qr).toarray()))
        total_tweets = str(len(classifier.predict(create_csv(vocab,y,qr).toarray())))
        print(location_city+" total tweets : "+total_tweets)
        pos_tweets = str(lab.count(1))
        neg_tweets = str(lab.count(0)) 
        print("positive tweets:"+pos_tweets)
        print("negative tweets:"+neg_tweets)
        Total_positive=+int(pos_tweets)
        Total_negative=+int(neg_tweets)
        positive=positive+lab.count(1)
        negative=negative+lab.count(0)
        
        write_to_file(positive,negative)


# In[162]:


def write_to_file(positive,negative):
    classifyfile = open('classify.txt', 'w+')
    if classifyfile.mode == 'w+':
        classifyfile.write("Number of instances per class found for positive tweets:"+str(positive))
        classifyfile.write("\nNumber of instances per class found for negative tweets::"+str(negative))
        classifyfile.close()


# In[163]:


def main():
    """ Main method. You should not modify this. """
    print("In main method")
   
    classification()
    
if __name__ == '__main__':
    main()


# In[ ]:





# In[ ]:




