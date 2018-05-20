# -*- coding: utf-8 -*-
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import fbeta_score
from nltk.classify import ClassifierI
from statistics import mode
#import matplotlib.pyplot as plt
import nltk

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf*100

def extract(word_list):
    all_word_types=['J','V','N']
    ret_list=[]
    temp=nltk.pos_tag(word_list)
    for w in temp:
        if (w[1][0]) in all_word_types:
            ret_list.append(w[0].lower())
        
    return ret_list

def find_feats(document):
    words=nltk.word_tokenize(document)
    words=extract(words)
    features={}
    for w in word_feats:
         features[w]=w in words
    return features

temp_var=open('word_feats.pickle','rb')
word_feats=pickle.load(temp_var)
temp_var.close()

temp_var=open('train_data.pickle','rb')
train_set=pickle.load(temp_var)
temp_var.close()


temp_var=open('test_data.pickle','rb')
test_set=pickle.load(temp_var)
temp_var.close()

save_clf=open('OrigNB.pickle','rb')
Classifier=pickle.load(save_clf)
save_clf.close()

save_clf=open('Logistic_reg.pickle','rb')
LogisticRegression_classifier=pickle.load(save_clf)
save_clf.close()

save_clf=open('LinearSVC.pickle','rb')
linsvc_bestclf=pickle.load(save_clf)
save_clf.close()

save_clf=open('NuSVC.pickle','rb')
NuSVC_bestclf=pickle.load(save_clf)
save_clf.close()

save_clf=open('randomforest.pickle','rb')
rf_bestclf=pickle.load(save_clf)
save_clf.close()



# In[6]:

def find_pred(clf,test_set):
    y_pred=[clf.classify(test_set[i][0]) for i in range(len(test_set))]
    y_true=[test_set[i][1] for i in range(len(test_set))]
    y_pred=pd.Series(y_pred).apply(lambda x:0 if x=='neg' else 1)
    y_true=pd.Series(y_true).apply(lambda x:0 if x=='neg' else 1)
    accuracy=accuracy_score(y_true,y_pred)*100
    f_beta=fbeta_score(y_true,y_pred,beta=0.5)
    return accuracy,f_beta


voted_classifier = VoteClassifier(Classifier,
                                  LogisticRegression_classifier,
                                  linsvc_bestclf,
                                  NuSVC_bestclf,
                                  rf_bestclf)


# In[10]:

def sentiment(text,ground_truth_flag=False):
    
    if ground_truth_flag==True:
        feats=text[0]
        return voted_classifier.classify(feats),voted_classifier.confidence(feats)
    else: 
        feats = find_feats(text)
        return voted_classifier.classify(feats),voted_classifier.confidence(feats)
    


