#!/usr/bin/env python
# coding: utf-8

# In[23]:


from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
from TwitterAPI import TwitterAPI
from nltk.tokenize import word_tokenize
import pandas as pd
import networkx as networkx
import matplotlib.pyplot as plt
import numpy as np


# In[24]:


def create_graph(quor = 0):
    data_list = []
    edges_l = []
    csv_list = ['USA-cluster.csv','UK-cluster.csv','India-cluster.csv']
    for csv in csv_list:
        data_list.append(csv);
    graph=nx.Graph()
    for file in data_list:
        frame=pd.read_csv(file,sep='\t')
        for index,row in frame.iterrows():
            row_friend = row['friends'].split(',')
            l_friends=list(row['friends'].split(','))
            for ls in l_friends:
                edges_point = (row['Id'],ls)
                edges_l.append(edges_point)
    graph = graph_made(edges_l)
    return graph


# In[25]:


def graph_made(edges_l):
    graph=nx.Graph()
    graph.add_edges_from(edges_l)
    nx.draw(graph)
    #nx.draw_networkx(graph)
    plt.figure(figsize=(100,100))
    plt.savefig("cluster.png", format = "PNG")
    n=str(nx.nodes(graph))
    return graph


# In[26]:


def girvan_newman_cluster(depth=0, min_nodes = 10, max_nodes =800, max_n = 0, index = 0):
    
    graph=create_graph()
    if graph.order() == 1:
        return [graph.nodes()]
    com_result=[]
    file_nam = "cluster.txt"
    file1=open(file_nam,"w+")

    component = get_components(graph)
    
    length_comp = len(component)
    if(length_comp>1):
        for ids,eg in enumerate(component):
            length_eg_node = len(eg.nodes())
            if max_n<length_eg_node:
                max_n=length_eg_node
                index=ids
    else:
        index=0
    i=1;
    indenting = '   ' * depth
    com_index = component[index].nodes()
    length_com_index = len(com_index)
    len_nodes = str(length_com_index)
    print("Selected cluster with "+len_nodes)
    components=[component[index]]
    for ins,comp in enumerate(component):
        if ins!=index:
            com_result.append(comp)
    while len(components) == 1:
        gr=nx.Graph(components[0])
        edge_b = get_betweenenss(gr)
        edge_bg= edge_b.items()
        edge = sorted(edge_bg, key=lambda x: x[1], reverse=True)[0][0]
        print(indenting + 'removing ' + str(edge))
        gr.remove_edge(*edge)
        #comp = [co for co in nx.connected_component_subgraphs(g)]
        comp = get_components(gr)
        com = len(comp)
        if( com <=1 ):
            components=[gr]
        else:
            components=comp
            
            
    com_result.extend(components)
    res_comp = len(com_result)
    string_res_com = str(res_comp)
    
    if file1.mode=="w+":
        res_com = len(com_result)
        file1.write("\n Number of communities discovered="+str(res_com))
        g_nodes_result = (len(graph.nodes())/len(com_result))
        file1.write("\n Average number of user per community ="+str(g_nodes_result))
        file1.write("\n")
        file1.close()
    
    return components


# In[27]:


def get_components(graph):
    """
    A helper function you may use below.
    Returns the list of all connected components in the given graph.
    """
    return [c for c in nx.connected_component_subgraphs(graph)]


# In[28]:


def get_betweenenss(graph):
    """
    A helper function you may use below.
    Returns the list of all connected components in the given graph.
    """
    return nx.edge_betweenness_centrality(graph)


# In[29]:


def main():
    print("In method")
    girvan_newman_cluster()
if __name__ == '__main__':
    main()


# In[ ]:





# In[ ]:




