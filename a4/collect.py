#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
from TwitterAPI import TwitterAPI
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np


# In[59]:


consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''


# In[60]:


def get_twitter():
    """ Construct an instance of TwitterAPI using the tokens you entered above.
    Returns:
      An instance of TwitterAPI.
    """
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)


# In[61]:


def robust_request(twitter, resource, params, max_tries=5):
    """ If a Twitter request fails, sleep for 15 minutes.
    Do this at most max_tries times before quitting.
    Args:
      twitter .... A TwitterAPI object.
      resource ... A resource string to request; e.g., "friends/ids"
      params ..... A parameter dict for the request, e.g., to specify
                   parameters like screen_name or count.
      max_tries .. The maximum number of tries to attempt.
    Returns:
      A TwitterResponse object, or None if failed.
    """
    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)


# In[62]:


def get_friends(twitter, screen_name):
    """
    Retrieve the Twitter user objects for each screen_name.
    Params:
        twitter........The TwitterAPI object.
        screen_names...A list of strings, one per screen_name
    Returns:
        A list of dicts, one per user, containing all the user information
        (e.g., screen_name, id, location, etc)
    Here we are using twitter.request to get the friend of a particular username,
    getting the number of users to return per page as 200
    
    """
    returnList=[]
    #string= screen_name+'&count=5000'
    #response=robust_request(twitter,"friends/list",{'screen_name':screen_name,'count':'200'},5)
    count = 200
    twitter_response = twitter.request('friends/list',{'screen_name' :screen_name,'count':count})
    #for tr in twitter_response:
    returnList.append(tr['screen_name'] for tr in twitter_response)
    return returnList


# In[68]:


def get_tweets(twitter,value):
    """
    Retrives twitter object, searching for tweets that contains "" , the request returns maximum 200 requests per page.
    geocode specifys the location of which the data should be collected from.
    """
    twitter_response = twitter.request('search/tweets',{'q':'#CaptainMarvel','count':'200','geocode':value})
    return twitter_response


# In[73]:


def prepare_data(twitter):
    """
    This method prepares data required for properssing and classifying for clusters.
    """
    users = list()
    data = {}

    coordinates = {"USA":('41.8781,-87.6298,50mi'),"UK":('51.5074,-0.1278,50mi'),"India":('36.778261,-119.41793239999998,50mi')}
    
    for city, loca in coordinates.items():
        user_list = []
        cnt = 0
        result_1 = []
        twitter_response = get_tweets(twitter,loca)
        for r in twitter_response:
            result_1.append(r)
            val = r['user']['screen_name']
            if not val in user_list:
                if cnt < 5:
                    user_list.append(val)
                users.append(val)
            else:
                continue
            cnt = cnt +1
        data[city]=user_list  
        frame = pd.DataFrame(result_1)
        csv_name = city+"-classify.csv"
        frame.to_csv(csv_name,sep='\t')
    print(data)
    return data,users


# In[74]:


def data_dump(twitter):
    """
    We save the dumped data into files with city name respectivity and just will help us in classifying and clustering the data.
    """
    data,user = prepare_data(twitter)
    write_results(user)

    for location,user in data.items():
        id = 'Id'
        friend_id = 'friends'
        col = [id,friend_id]
        DF_new = pd.DataFrame(columns=col)
        for per in user:
            get_data_frame = [per,get_friends(twitter,per)]
            DF_new.loc[len(DF_new)]= get_data_frame
        file_name = location+"-cluster.csv"
        DF_new.to_csv(file_name,sep='\t')


# In[75]:


def write_results(users):
    """
    Method to write information to files
    """
    user_array = np.array(users)
    unique_user=np.unique(user_array)
    file = open("collect.txt", 'w+')
    if file.mode=='w+':
        no_of_users = str(len(unique_user))
        msg_collected = str(len(users))
        file.write("Number of users Collected: "+no_of_users)
        file.write("\nNumber of messages Collected: "+msg_collected)
        file.close()


# In[77]:


def main():
    """ Main method. You should not modify this. """
    twitter = get_twitter()
    data_dump(twitter)

if __name__ == '__main__':
    main()


# In[ ]:





# In[128]:





# In[129]:





# In[ ]:





# In[ ]:




