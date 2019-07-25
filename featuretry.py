
import nltk
import numpy as np
import scipy as sp
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer
import pickle
import text_features
import topic
import heapq

print ('Pickling out')
pos_data=np.load('posproc.npy')
neg_data=np.load('negproc.npy')
print ('Number of  sarcastic tweets :', len(pos_data))
print ('Number of  non-sarcastic tweets :', len(neg_data))

print ('Training topics')

topic_mod = topic.topic(nbtopic=200,alpha='symmetric')
topic_mod.fit(np.concatenate((pos_data,neg_data)))
np.save('topicsave',topic_mod)
print("Topic saved")
print ('Feature eng')
# label set
cls_set = ['Non-Sarcastic','Sarcastic']
featuresets = [] 

index=0
for tweet in pos_data:
    if (np.mod(index,10000)==0):
        print ("Positive tweet processed: ",index)
    featuresets.append((text_features.dialogue_act_features(tweet,topic_mod),cls_set[1]))
    index+=1
 
index=0
for tweet in neg_data:
    if (np.mod(index,10000)==0):
        print ("Negative tweet processed: ",index)
    featuresets.append((text_features.dialogue_act_features(tweet,topic_mod),cls_set[0]))
    index+=1
        
featuresets=np.array(featuresets)

np.save('featuresave1',featuresets)
print("Finished")
