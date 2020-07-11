#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collections import Counter, defaultdict
from itertools import chain, combinations
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import string
from scipy.sparse import csr_matrix
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import tarfile
import urllib.request
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer




def download_data():
    """ Download and unzip data.
    DONE ALREADY.
    """
    url = 'https://www.dropbox.com/s/8oehplrobcgi9cq/imdb.tgz?dl=1'
    urllib.request.urlretrieve(url, 'imdb.tgz')
    tar = tarfile.open("imdb.tgz")
    tar.extractall()
    tar.close()





def read_data(path):
    """
    Walks all subdirectories of this path and reads all
    the text files and labels.
    DONE ALREADY.

    Params:
      path....path to files
    Returns:
      docs.....list of strings, one per document
      labels...list of ints, 1=positive, 0=negative label.
               Inferred from file path (i.e., if it contains
               'pos', it is 1, else 0)
    """
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'pos', '*.txt'))])
    data = [(1, open(f).readlines()[0]) for f in sorted(fnames)]
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'neg', '*.txt'))])
    data += [(0, open(f).readlines()[0]) for f in sorted(fnames)]
    data = sorted(data, key=lambda x: x[1])
    return np.array([d[1] for d in data]), np.array([d[0] for d in data])


# In[291]:


def tokenize(doc, keep_internal_punct=False):
    """
    Tokenize a string.
    The string should be converted to lowercase.
    If keep_internal_punct is False, then return only the alphanumerics (letters, numbers and underscore).
    If keep_internal_punct is True, then also retain punctuation that
    is inside of a word. E.g., in the example below, the token "isn't"
    is maintained when keep_internal_punct=True; otherwise, it is
    split into "isn" and "t" tokens.

    Params:
      doc....a string.
      keep_internal_punct...see above
    Returns:
      a numpy array containing the resulting tokens.

    >>> tokenize(" Hi there! Isn't this fun?", keep_internal_punct=False)
    array(['hi', 'there', 'isn', 't', 'this', 'fun'], dtype='<U5')
    >>> tokenize("Hi there! Isn't this fun? ", keep_internal_punct=True)
    array(['hi', 'there', "isn't", 'this', 'fun'], dtype='<U5')
    """
    ###TODO
    splitList = []
    lowercasedoc = doc.lower()
    stop_words = set(stopwords.words('english'))
    porter=PorterStemmer()
    token_words=word_tokenize(lowercasedoc)
    
    if(keep_internal_punct == False):
        finalList = re.sub('(\W+)', " ", lowercasedoc).split()          
        for l in finalList:
            if l not in stop_words:
                splitList.append(l);
        
    else:
        regex='[\w_][^\s]*[\w_]|[\w_]'
        finalList = re.findall(regex, lowercasedoc)
        for l in finalList:
            if l not in stop_words:
                splitList.append(l);
    return np.array(splitList)

def token_features(tokens, feats):
    """
    Add features for each token. The feature name
    is pre-pended with the string "token=".
    Note that the feats dict is modified in place,
    so there is no return value.

    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.

    >>> feats = defaultdict(lambda: 0)
    >>> token_features(['hi', 'there', 'hi'], feats)
    >>> sorted(feats.items())
    [('token=hi', 2), ('token=there', 1)]
    """
    ###TODO
    output_List = Counter(tokens)
    
    for k,v in output_List.items():
        feats['token='+k]= v


# In[2]:


tokenize(" Hi there! Isn't this fun?", keep_internal_punct=False)


# In[4]:


def token_pair_features(tokens, feats, k=3):
    """
    Compute features indicating that two words occur near
    each other within a window of size k.

    For example [a, b, c, d] with k=3 will consider the
    windows: [a,b,c], [b,c,d]. In the first window,
    a_b, a_c, and b_c appear; in the second window,
    b_c, c_d, and b_d appear. This example is in the
    doctest below.
    Note that the order of the tokens in the feature name
    matches the order in which they appear in the document.
    (e.g., a__b, not b__a)

    Params:
      tokens....array of token strings from a document.
      feats.....a dict from feature to value
      k.........the window size (3 by default)
    Returns:
      nothing; feats is modified in place.

    >>> feats = defaultdict(lambda: 0)
    >>> token_pair_features(np.array(['a', 'b', 'c', 'd']), feats)
    >>> sorted(feats.items())
    [('token_pair=a__b', 1), ('token_pair=a__c', 1), ('token_pair=b__c', 2), ('token_pair=b__d', 1), ('token_pair=c__d', 1)]
    """
    ###TODO
    pass
    tempList = []
    length_token = len(tokens)-k+1
    for i in range(0,length_token):
        comb = combinations(tokens[i:i+k],2) 
        for item in list(comb):
            tempList.append('__'.join(item))
    output_List = Counter(tempList)
    
    for k,v in output_List.items():
        feats['token_pair='+k]=v


# In[297]:


neg_words = set(['bad', 'hate', 'horrible', 'worst', 'boring'])
pos_words = set(['awesome', 'amazing', 'best', 'good', 'great', 'love', 'wonderful'])


# In[298]:


def lexicon_features(tokens, feats):
    """
    Add features indicating how many time a token appears that matches either
    the neg_words or pos_words (defined above). The matching should ignore
    case.

    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.

    In this example, 'LOVE' and 'great' match the pos_words,
    and 'boring' matches the neg_words list.
    >>> feats = defaultdict(lambda: 0)
    >>> lexicon_features(np.array(['i', 'LOVE', 'this', 'great', 'boring', 'movie']), feats)
    >>> sorted(feats.items())
    [('neg_words', 1), ('pos_words', 2)]
    """
    ###TODO
    feats['neg_words']=0
    feats['pos_words']=0
    for i in tokens:
        t=i.lower()
        if (t in neg_words):
            feats['neg_words'] +=1
        elif(t in pos_words):
            feats['pos_words'] +=1


# In[300]:


def featurize(tokens, feature_fns):
    """
    Compute all features for a list of tokens from
    a single document.

    Params:
      tokens........array of token strings from a document.
      feature_fns...a list of functions, one per feature
    Returns:
      list of (feature, value) tuples, SORTED alphabetically
      by the feature name.

    >>> feats = featurize(np.array(['i', 'LOVE', 'this', 'great', 'movie']), [token_features, lexicon_features])
    >>> feats
    [('neg_words', 0), ('pos_words', 2), ('token=LOVE', 1), ('token=great', 1), ('token=i', 1), ('token=movie', 1), ('token=this', 1)]
    """
    ###TODO
    feats = defaultdict(lambda: 0)
    for f in feature_fns:
        f(tokens, feats)
    return sorted(feats.items(),key=lambda x:x[0])


# In[361]:


def vectorize(tokens_list, feature_fns, min_freq, vocab=None):
    """
    Given the tokens for a set of documents, create a sparse
    feature matrix, where each row represents a document, and
    each column represents a feature.

    Params:
      tokens_list...a list of lists; each sublist is an
                    array of token strings from a document.
      feature_fns...a list of functions, one per feature
      min_freq......Remove features that do not appear in
                    at least min_freq different documents.
    Returns:
      - a csr_matrix: See https://goo.gl/f5TiF1 for documentation.
      This is a sparse matrix (zero values are not stored).
      - vocab: a dict from feature name to column index. NOTE
      that the columns are sorted alphabetically (so, the feature
      "token=great" is column 0 and "token=horrible" is column 1
      because "great" < "horrible" alphabetically),

    >>> docs = ["Isn't this movie great?", "Horrible, horrible movie"]
    >>> tokens_list = [tokenize(d) for d in docs]
    >>> feature_fns = [token_features]
    >>> X, vocab = vectorize(tokens_list, feature_fns, min_freq=1)
    >>> type(X)
    <class 'scipy.sparse.csr.csr_matrix'>
    >>> X.toarray()
    array([[1, 0, 1, 1, 1, 1],
           [0, 2, 0, 1, 0, 0]], dtype=int64)
    >>> sorted(vocab.items(), key=lambda x: x[1])
    [('token=great', 0), ('token=horrible', 1), ('token=isn', 2), ('token=movie', 3), ('token=t', 4), ('token=this', 5)]
    """
    ###TODO
    field = []
    attributes = []
    data = []
    index = 0
    labellist=[]
    resultedVocab=[]
    new_vocab = defaultdict(list)
    docmap = defaultdict(dict)
    len_of_token = len(tokens_list)
    if vocab == None:
        modified_vocabulary = {}
        for doc in range(len_of_token):
            feats = featurize(tokens_list[doc], feature_fns)
            featuresdict = dict(feats)
            docmap[doc] = featuresdict
            for feature in featuresdict:         
                new_vocab[feature].append(doc)       

        for i in sorted(new_vocab):
            if len(new_vocab[i]) >= min_freq:
                modified_vocabulary[i] = index
                index += 1
        sorted_vocb=sorted(modified_vocabulary.keys())
        for i in sorted_vocb:
            for doc in sorted(new_vocab[i]):
                if i in docmap[doc]:
                    field.append(doc)
                    data.append(docmap[doc][i])
                    attributes.append(modified_vocabulary[i])   
        csr = csr_matrix((data, (field, attributes)), shape=(len(tokens_list), len(modified_vocabulary)),dtype=np.int64)
        return csr, modified_vocabulary
    elif vocab != None:
        for doc in range(len_of_token):
            featuresdict = dict(featurize(tokens_list[doc],feature_fns))        
            for feature in featuresdict:
                if feature in vocab:                                      
                    data.append(featuresdict[feature])
                    field.append(doc)
                    attributes.append(vocab[feature])
  
        csr = csr_matrix((data,(field,attributes)), shape=(len(tokens_list),len(vocab)),dtype=np.int64)
        return csr, vocab


# In[308]:


def accuracy_score(truth, predicted):
    """ Compute accuracy of predictions.
    DONE ALREADY
    Params:
      truth.......array of true labels (0 or 1)
      predicted...array of predicted labels (0 or 1)
    """
    return len(np.where(truth==predicted)[0]) / len(truth)


# In[309]:


def cross_validation_accuracy(clf, X, labels, k):
    """
    Compute the average testing accuracy over k folds of cross-validation. You
    can use sklearn's KFold class here (no random seed, and no shuffling
    needed).

    Params:
      clf......A LogisticRegression classifier.
      X........A csr_matrix of features.
      labels...The true labels for each instance in X
      k........The number of cross-validation folds.

    Returns:
      The average testing accuracy of the classifier
      over each fold of cross-validation.
    """
    ###TODO
    pass
    kflod = KFold(k,random_state=None, shuffle=False)
    accuracy_score_all=[]
    for train , test in kflod.split(X):
        clf.fit(X[train],labels[train])
        predict = clf.predict(X[test])
        accuracy_score_all.append(accuracy_score(labels[test],predict))
    average = np.mean(accuracy_score_all) 
    return average


# In[310]:


def eval_all_combinations(docs, labels, punct_vals,
                          feature_fns, min_freqs):
    """
    Enumerate all possible classifier settings and compute the
    cross validation accuracy for each setting. We will use this
    to determine which setting has the best accuracy.

    For each setting, construct a LogisticRegression classifier
    and compute its cross-validation accuracy for that setting.

    In addition to looping over possible assignments to
    keep_internal_punct and min_freqs, we will enumerate all
    possible combinations of feature functions. So, if
    feature_fns = [token_features, token_pair_features, lexicon_features],
    then we will consider all 7 combinations of features (see Log.txt
    for more examples).

    Params:
      docs..........The list of original training documents.
      labels........The true labels for each training document (0 or 1)
      punct_vals....List of possible assignments to
                    keep_internal_punct (e.g., [True, False])
      feature_fns...List of possible feature functions to use
      min_freqs.....List of possible min_freq values to use
                    (e.g., [2,5,10])

    Returns:
      A list of dicts, one per combination. Each dict has
      four keys:
      'punct': True or False, the setting of keep_internal_punct
      'features': The list of functions used to compute features.
      'min_freq': The setting of the min_freq parameter.
      'accuracy': The average cross_validation accuracy for this setting, using 5 folds.

      This list should be SORTED in descending order of accuracy.

      This function will take a bit longer to run (~20s for me).
    """
    ###TODO
    final_List= []
    token_list= []
    clf = LogisticRegression()
    allcombinations=[]
    for l in range(1,len(feature_fns)+1):
            feature_list = list(combinations(feature_fns,l))
            allcombinations.extend(feature_list)
    for p in punct_vals:
        for feature in allcombinations:
            for m in min_freqs:
                if feature:
                    X,vocab= vectorize([tokenize(d,p) for d in docs], feature, min_freq=m,vocab=None)
                    tempdict={}
                    tempdict['punct']=p
                    tempdict['features']=feature
                    tempdict['min_freq']=m
                    avg = cross_validation_accuracy(clf, X, labels, 5)
                    tempdict['accuracy']=avg
                    final_List.append(tempdict)
    return sorted(final_List, key=lambda x: (-x['accuracy'], -x['min_freq']))


# In[374]:


def plot_sorted_accuracies(results):
    """
    Plot all accuracies from the result of eval_all_combinations
    in ascending order of accuracy.
    Save to "accuracies.png".
    """
    ###TODO
    x =[]
    y =[]
    
    listtoplot = []
    for r in results:
        listtoplot.append(r["accuracy"])

    sorted_list = sorted(listtoplot)
    plt.plot(sorted_list)
    plt.ylabel("Accuracy")
    plt.xlabel("Setting")
    plt.savefig("accuracies.png")


# In[312]:


def mean_accuracy_per_setting(results):
    """
    To determine how important each model setting is to overall accuracy,
    we'll compute the mean accuracy of all combinations with a particular
    setting. For example, compute the mean accuracy of all runs with
    min_freq=2.

    Params:
      results...The output of eval_all_combinations
    Returns:
      A list of (accuracy, setting) tuples, SORTED in
      descending order of accuracy.
    """
    mean_dict = {}
    min_dict =defaultdict(list)
    punct_dict=defaultdict(list)
    feature_dict=defaultdict(list)
    final_list = []
    
    for i in results:
        temp_list=[]
        min_freq_key= 'min_freq='+str(i['min_freq'])
        min_dict[min_freq_key].append(i['accuracy'])
        punct_key = 'punct='+str(i['punct'])
        punct_dict[punct_key].append(i['accuracy'])
        
        for f in i['features']:
            temp_list.append(str(f.__name__))
            str_name = " ".join(str(x) for x in temp_list)
        
        feature_key = 'features='+str_name
        feature_dict[feature_key].append(i['accuracy'])
    
    mean_dict.update(min_dict)
    mean_dict.update(punct_dict)
    mean_dict.update(feature_dict)
    
    for key, value in mean_dict.items():
        final_list.append((np.mean(value),key))
        
    return sorted(final_list, key=lambda k:k[0])[::-1]


# In[313]:


def fit_best_classifier(docs, labels, best_result):
    """
    Using the best setting from eval_all_combinations,
    re-vectorize all the training data and fit a
    LogisticRegression classifier to all training data.
    (i.e., no cross-validation done here)

    Params:
      docs..........List of training document strings.
      labels........The true labels for each training document (0 or 1)
      best_result...Element of eval_all_combinations
                    with highest accuracy
    Returns:
      clf.....A LogisticRegression classifier fit to all
            training data.
      vocab...The dict from feature name to column index.
    """
    ###TODO
    token_list = []
    clf = LogisticRegression()
    keep_punchtuation = best_result['punct']
    min_freq=best_result['min_freq']
    feature_fn=best_result['features']
    
    X, vocab = vectorize([tokenize(d,keep_punchtuation) for d in docs], feature_fn, min_freq)
    clf.fit(X,labels)
    return clf, vocab


# In[314]:


def top_coefs(clf, label, n, vocab):
    """
    Find the n features with the highest coefficients in
    this classifier for this label.
    See the .coef_ attribute of LogisticRegression.

    Params:
      clf.....LogisticRegression classifier
      label...1 or 0; if 1, return the top coefficients
              for the positive class; else for negative.
      n.......The number of coefficients to return.
      vocab...Dict from feature name to column index.
    Returns:
      List of (feature_name, coefficient) tuples, SORTED
      in descending order of the coefficient for the
      given class label.
    """
    ###TODO
    finalList = []
    
    for v in vocab:
        finalList.append((v,clf.coef_[0][vocab[v]]))
        
    if label == 1:
        return sorted(finalList,key=lambda x:-x[1])[:n]
    elif label == 0:
        sortedList = sorted(finalList, key=lambda x: x[1])
        
        temp_list =[]
        
        for t in sortedList[:n]:
            temp_list.append((t[0],-1*t[1]))
            
        return temp_list


# In[349]:


def parse_test_data(best_result, vocab):
    """
    Using the vocabulary fit to the training data, read
    and vectorize the testing data. Note that vocab should
    be passed to the vectorize function to ensure the feature
    mapping is consistent from training to testing.

    Note: use read_data function defined above to read the
    test data.

    Params:
      best_result...Element of eval_all_combinations
                    with highest accuracy
      vocab.........dict from feature name to column index,
                    built from the training data.
    Returns:
      test_docs.....List of strings, one per testing document,
                    containing the raw.
      test_labels...List of ints, one per testing document,
                    1 for positive, 0 for negative.
      X_test........A csr_matrix representing the features
                    in the test data. Each row is a document,
                    each column is a feature.
    """
    ###TODO
    docs, labels = read_data(os.path.join('data', 'test'))
    test_docs = docs
    test_labels = labels
    keep_punchtuation = best_result["punct"]
    min_freq = best_result["min_freq"]
    feature = best_result["features"]
    tokens=[] 
    X_test, vocab_new = vectorize([tokenize(d, keep_punchtuation)for d in docs], feature, min_freq,vocab)
    return test_docs,test_labels,X_test


# In[371]:


def print_top_misclassified(test_docs, test_labels, X_test, clf, n):
    """
    Print the n testing documents that are misclassified by the
    largest margin. By using the .predict_proba function of
    LogisticRegression <https://goo.gl/4WXbYA>, we can get the
    predicted probabilities of each class for each instance.
    We will first identify all incorrectly classified documents,
    then sort them in descending order of the predicted probability
    for the incorrect class.
    E.g., if document i is misclassified as positive, we will
    consider the probability of the positive class when sorting.

    Params:
      test_docs.....List of strings, one per test document
      test_labels...Array of true testing labels
      X_test........csr_matrix for test data
      clf...........LogisticRegression classifier fit on all training
                    data.
      n.............The number of documents to print.

    Returns:
      Nothing; see Log.txt for example printed output.
    """
    ###TODO
    final_list =[]
    predictedvalues = clf.predict(X_test)
    probability=clf.predict_proba(X_test)
    incorrect_values =np.where(predictedvalues!=test_labels)
    
     #result.append((test_labels[ind],predicted_value[ind],probabilities[ind][predicted_value[ind]],test_docs[ind])) 
    
    for i in incorrect_values[0]:
        final_list.append((probability[i][predictedvalues[i]],predictedvalues[i],test_labels[i],test_docs[i]))
        
    sorted_list = sorted(final_list,key=lambda x: x[0], reverse= True)[:n]
    
    for s in sorted_list:
        print('\n'+"truth=",s[2]," predicted=",s[1]," proba=",s[0])
        print(str(s[3]))
        #print('\n',str(s[3]))


# In[375]:


def main():
    """
    Put it all together.
    ALREADY DONE.
    """
    feature_fns = [token_features, token_pair_features, lexicon_features]
    # Download and read data.
    download_data()
    docs, labels = read_data(os.path.join('data', 'train'))
    # Evaluate accuracy of many combinations
    # of tokenization/featurization.
    results = eval_all_combinations(docs, labels,
                                    [True, False],
                                    feature_fns,
                                    [2,5,10])
    # Print information about these results.
    best_result = results[0]
    worst_result = results[-1]
    print('best cross-validation result:\n%s' % str(best_result))
    print('worst cross-validation result:\n%s' % str(worst_result))
    plot_sorted_accuracies(results)
    print('\nMean Accuracies per Setting:')
    print('\n'.join(['%s: %.5f' % (s,v) for v,s in mean_accuracy_per_setting(results)]))

    # Fit best classifier.
    clf, vocab = fit_best_classifier(docs, labels, results[0])

    # Print top coefficients per class.
    print('\nTOP COEFFICIENTS PER CLASS:')
    print('negative words:')
    top_coefs(clf, 0, 5, vocab)
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 0, 5, vocab)]))
    print('\npositive words:')
    top_coefs(clf, 1, 5, vocab)
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 1, 5, vocab)]))
    # Parse test data
    test_docs, test_labels, X_test = parse_test_data(best_result, vocab)

    # Evaluate on test set.
    predictions = clf.predict(X_test)
    print('testing accuracy=%f' %
          accuracy_score(test_labels, predictions))
    print('\nTOP MISCLASSIFIED TEST DOCUMENTS:')
    print_top_misclassified(test_docs, test_labels, X_test, clf, 5)
    
if __name__ == '__main__':
    main()


# In[ ]:




