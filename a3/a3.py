from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/p9wmkvbqt1xr6lc/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()
	
def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())
	
def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.

    Note: you may modify the movies parameter directly; no need to make
    a new copy.
    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.

    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    """
    ###TODO
    genereList = []
    genre = movies["genres"]
    for i in genre :
        genereList.append(tokenize_string(i))
    movies['tokens'] = pd.Series(genereList,index = movies.index)
    return movies

def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i

    Params:
      movies...The movies DataFrame
    Returns:
      A tuple containing:
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
    """
    ###TODO
    tokenList = movies['tokens'].tolist()
    featureList=[]
    csrMatrixList=[]
    vocab = {}
    termuniquedoc = {}
    for i in tokenList:
        for j in i:
            featureList.append(j)
    for i in sorted(set(featureList)):
        vocab[i] = len(vocab)
    for i in tokenList:
        for j in set(i):
            if j not in termuniquedoc:
                termuniquedoc[j] = 1
            else:
                termuniquedoc[j] += 1
    lengthVocab = len(vocab)
    for i in tokenList:
        indices = []
        data = []
        row = []
        termdict =dict(Counter(i))
        for j in set(i):
            row.append(0)
            indices.append(vocab[j])  
            max_k = max(termdict.values())
            log_value = math.log10(len(movies) / termuniquedoc[j])
            tfidf = termdict[j] / max_k * log_value
            data.append(tfidf)          
        matrix = csr_matrix((data,(row,indices)),shape=(1,lengthVocab))
        csrMatrixList.append(matrix)
    movies['features'] = csrMatrixList   
    return (movies, vocab)
	
def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]
	
def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      A float. The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    """
    ###TODO
    den = np.linalg.norm(a.toarray()) * np.linalg.norm(b.toarray())
    numbr = np.dot(a.toarray(),b.toarray().T)
    return numbr[0][0]/den
	
def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.

    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.

    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.

    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """
    ###TODO

    rating_movieid = ratings_test['movieId']
    rating_userid =ratings_test['userId']
    zipFile = zip(rating_movieid,rating_userid)
    valList = []
    tempList = []
    for id,rating in zipFile:      
        total =0
        counting = 0
        positiveRating = False
        ratingList=[]
        movie_id = movies['movieId']
        testData=movies[movie_id==id]['features'].iloc[0]
        ratingValue = ratings_train.loc[ratings_train['userId']==rating].iterrows()
        for i,r in ratingValue: 
            if id != r['movieId']:
                vector_a = movies[movie_id==int(r['movieId'])]['features'].iloc[0]
                valueC=cosine_sim(vector_a,testData)
                ratingR = r['rating']
                ratingList.append(ratingR)
                if (valueC>0):
                    total+=valueC
                    counting += valueC * ratingR
                    positiveRating=True
        if positiveRating == False :   
             valList.append(np.mean(ratingList))
        else:
            valList.append(counting/total)
    return  np.asarray(valList)
	
def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()
	
def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


if __name__ == '__main__':
    main()