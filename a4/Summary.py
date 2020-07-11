#!/usr/bin/env python
# coding: utf-8

# In[1]:


import networkx as nx
import matplotlib.pyplot as plt
import sys
import pandas as pd
from TwitterAPI import TwitterAPI
import numpy as np
import time
from collections import Counter
from nltk.tokenize import word_tokenize


# In[3]:


def main():
    outputfile = list()
    outputfile.append('collect.txt')
    outputfile.append('cluster.txt')
    outputfile.append('classify.txt')    
    summary_file = open("sumarize.txt","w+")
    for i in outputfile:
        f=open(i,'r')
        if f.mode=='r':
            readFiles=f.read()
        lines=readFiles.splitlines()
        for write_line in lines:
            if summary_file.mode =='w+':
                summary_file.write("\n " +write_line)
                
if __name__ == '__main__':
    main()


# In[ ]:




