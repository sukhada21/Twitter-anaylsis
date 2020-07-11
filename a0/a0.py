# coding: utf-8

"""
CS579: Assignment 0
Collecting a political social network

In this assignment, I've given you a list of Twitter accounts of 4
U.S. presedential candidates from the previous election.

The goal is to use the Twitter API to construct a social network of these
accounts. We will then use the [networkx](http://networkx.github.io/) library
to plot these links, as well as print some statistics of the resulting graph.

1. Create an account on [twitter.com](http://twitter.com).
2. Generate authentication tokens by following the instructions [here](https://developer.twitter.com/en/docs/basics/authentication/guides/access-tokens.html).
3. Add your tokens to the key/token variables below. (API Key == Consumer Key)
4. Be sure you've installed the Python modules
[networkx](http://networkx.github.io/) and
[TwitterAPI](https://github.com/geduldig/TwitterAPI). Assuming you've already
installed [pip](http://pip.readthedocs.org/en/latest/installing.html), you can
do this with `pip install networkx TwitterAPI`.

OK, now you're ready to start collecting some data!

I've provided a partial implementation below. Your job is to complete the
code where indicated.  You need to modify the 10 methods indicated by
#TODO.

Your output should match the sample provided in Log.txt.
"""

# Imports you'll need.
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
from TwitterAPI import TwitterAPI
from itertools import combinations

consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''


# This method is done for you.
def get_twitter():
    """ Construct an instance of TwitterAPI using the tokens you entered above.
    Returns:
      An instance of TwitterAPI.
    """
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)


def read_screen_names(filename):
	with open(filename , 'r') as f:
                #Content_list is the list that contains the read lines.     
                #return content_list = f.readlines()
                #print(content_list)
				data=f.read().splitlines()
	return data



# I've provided the method below to handle Twitter's rate limiting.
# You should call this method whenever you need to access the Twitter API.
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


def get_users(twitter, screen_names):
	r = twitter.request('users/lookup',{'screen_name' :screen_names})
	return r
    


def get_friends(twitter, screen_name):
	tr = twitter.request('friends/ids',{'screen_name' :screen_name})
	return tr
	>>> twitter = get_twitter()
	
def add_all_friends(twitter, users):
	for us in users:
		listoffriends=get_friends(twitter,us['screen_name'])
		datalist = listoffriends.json()
		us.update({'friends':datalist['ids']})
	
	
def print_num_friends(users):
	for u in users:
		#count = (u[len('friends')])
		listoffriends= u['friends']
		count = len(listoffriends)
		print(u['screen_name'],count)
    


def count_friends(users):
	c= Counter()
	for u in users:
		listofids=u['friends']
		c.update(listofids)
	#print(c)
	return c

def friend_overlap(users):
	
	overlap=0
	tuplelist=[]
	
	for i in range(len(users)):
	
		a=set()
		b=set()
		firstuser=users[i]
		getids=firstuser['friends']
		getfirstname=firstuser['screen_name']
		a.update(getids)
		#print (len(getids))
		new=i+1
		if new<len(users):
			seconduser=users[new]
			getsecondids=seconduser['friends']
			getsecondname=seconduser['screen_name']
			b.update(getsecondids)
			#print (len(getsecondids))
		overlap=(len(a&b))
		returnTuple=(getfirstname,getsecondname,overlap)
		tuplelist.append(returnTuple)
	print (tuplelist)
	return tuplelist
	
def copy_overlap(users):
	tuplelist=[]
	comb = combinations(users,2)
	for a,b in comb:
		aset=set()
		bset=set()
		overlap=0
		getids=a['friends']
		getfirstname=a['screen_name']
		aset.update(getids)
		getsecondids=b['friends']
		getsecondname=b['screen_name']
		bset.update(getsecondids)
		overlap=(len(aset&bset))
		returnTuple=(getfirstname,getsecondname,overlap)
		tuplelist.append(returnTuple)
	returnList = sorted(tuplelist, key = lambda x: float(x[2]), reverse = True)	
	return returnList

def followed_by_hillary_and_donald(users, twitter):
	h=set()
	d=set()
	followedList=[]
	followedbyhillary=get_friends(twitter,'HillaryClinton')
	hillarfollowerlist = followedbyhillary.json()
	h.update(hillarfollowerlist['ids'])
	followedbydonald=get_friends(twitter,'realDonaldTrump')
	donaldfollowerlist = followedbydonald.json()
	d.update(donaldfollowerlist['ids'])
	commonid=(h&d)
	for c in commonid:
		commonusername = twitter.request('users/lookup',{'user_id' :c})
		followed_by_friend = commonusername.json()
		screenNames=followed_by_friend[0]
		followedList.append(screenNames['screen_name'])
	return followedList	


def create_graph(users, friend_counts):
	GF=nx.Graph()
	for u in users:
		GF.add_node(u['screen_name'])
	for k,v in friend_counts.items():
		if (v>1):
			GF.add_node(k)
			for us in users:
				friendsList= us['friends']
				if k in friendsList:
					GF.add_edge(us['screen_name'],k)
	return GF
				


def draw_network(graph, users, filename):
	lableList = {}
	for u in users:
		lableList[u['screen_name']]=(u['screen_name'])
	pos=nx.spring_layout(graph)
	nx.draw_networkx(graph,pos,with_labels=False,node_size=50)
	nx.draw_networkx_labels(graph,pos,lableList)
	nx.draw_networkx_edges(graph,pos,width=0.5,edge_color='ivory')
	plt.axis('off')
	fig=plt.figure(1)
	fig.set_size_inches(15, 7)
	fig.savefig(filename)
	


def main():
    """ Main method. You should not modify this. """
    twitter = get_twitter()
    screen_names = read_screen_names('candidates.txt')
    print('Established Twitter connection.')
    print('Read screen names: %s' % screen_names)
    users = sorted(get_users(twitter, screen_names), key=lambda x: x['screen_name'])
    print('found %d users with screen_names %s' %
          (len(users), str([u['screen_name'] for u in users])))
    add_all_friends(twitter, users)
    print('Friends per candidate:')
    print_num_friends(users)
    friend_counts = count_friends(users)
    print('Most common friends:\n%s' % str(friend_counts.most_common(5)))
    print('Friend Overlap:\n%s' % str(copy_overlap(users)))
    print('User followed by Hillary and Donald: %s' % str(followed_by_hillary_and_donald(users, twitter)))

    graph = create_graph(users, friend_counts)
    print('graph has %s nodes and %s edges' % (len(graph.nodes()), len(graph.edges())))
    draw_network(graph, users, 'network.png')
    print('network drawn to network.png')


if __name__ == '__main__':
    main()

# That's it for now! This should give you an introduction to some of the data we'll study in this course.
