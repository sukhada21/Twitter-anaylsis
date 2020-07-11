#!/usr/bin/env python
# coding: utf-8

# In[24]:


import networkx as nx
import urllib.request
import pickle

def jaccard_wt(graph, node):
    """
    The weighted jaccard score, defined in bonus.md.
    Args:
      graph....a networkx graph
      node.....a node to score potential new edges for.
    Returns:
      A list of ((node, ni), score) tuples, representing the 
                score assigned to edge (node, ni)
                (note the edge order)
    """
    temp_list = []
    neighbors_list = set(graph.neighbors(node))  
    result = []
    graph_node =graph.nodes()
    val = {}
    for n in graph_node:
        if ((n not in neighbors_list) & (n != node)):
            list_neigh = set(graph.neighbors(n))
            temp_reult= 0
            interse=0
            temp_1=0
            temp_2=0
            grph_node_list = graph.neighbors(node)
            n_node_list = graph.neighbors(n)
            for ind in grph_node_list:
                temp_1+=(graph.degree(ind))
            for i in (neighbors_list & list_neigh):
                degree_grp = graph.degree(i)
                interse+=(1/degree_grp)
                temp_list.append(interse)
            for inn in n_node_list:
                temp_2+=(graph.degree(inn))
            divide = (1/temp_1)+(1/temp_2)
            result.append(((node,n),interse/(divide)))

    finalresult = sorted(result, key=lambda x: (-x[1],x[0][1]))
    return finalresult





