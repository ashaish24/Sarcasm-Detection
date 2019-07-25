from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
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
import argparse
import heapq
pos_data=np.load('posproc.npy')
neg_data=np.load('negproc.npy')
#topic_mod = np.load('topicsave.npy')
topic_mod = topic.topic(nbtopic=200,alpha='symmetric')
topic_mod.fit(np.concatenate((pos_data,neg_data)))
print ('Feature eng')
# label set
cls_set = ['Non-Sarcastic','Sarcastic']
featuresets = [] 

featuresets=np.load('featuresave.npy')
targets=(featuresets[0::,1]=='Sarcastic').astype(int)

print ('Dictionnary vectorizer')
vec = DictVectorizer()
featurevec = vec.fit_transform(featuresets[0::,0])


file_Name = "vecdict.p"
fileObject = open(file_Name,'wb') 
pickle.dump(vec, fileObject)
fileObject.close()

print ('Feature splitting')

order=shuffle(range(len(featuresets)))
targets=targets[order]
featurevec=featurevec[order,0::]


size = int(len(featuresets) * .3) 

trainvec = featurevec[size:,0::]
train_targets = targets[size:]
testvec = featurevec[:size,0::]
test_targets = targets[:size]

print ('Training')


pos_p=(train_targets==1)
neg_p=(train_targets==0)
ratio = np.sum(neg_p.astype(float))/np.sum(pos_p.astype(float))
new_trainvec=trainvec
new_train_targets=train_targets
for j in range(int(ratio-1.0)):
    new_trainvec=sp.sparse.vstack([new_trainvec,trainvec[pos_p,0::]])
    new_train_targets=np.concatenate((new_train_targets,train_targets[pos_p]))    

classifier = LinearSVC(C=0.1,penalty='l2',dual=True)
classifier.fit(new_trainvec,new_train_targets)

'''
file_Name = "classif.p"
fileObject = open(file_Name,'wb') 
pickle.dump(classifier, fileObject)
fileObject.close()
'''

print ('Most important features')

print ('grams:')

coeff = vec.inverse_transform(classifier.coef_[0])[0]
largest = heapq.nlargest(int(100/2.0), coeff, key=coeff.get)
smallest = heapq.nsmallest(int(100/2.0), coeff, key=coeff.get)
for j in range(int(100/2.0)):
    print (largest[j], coeff[largest[j]])
for j in range(int(100/2.0)):
    print (smallest[j], coeff[smallest[j]])


print ('sentiment:')

print ('Positive sentiment', coeff['Positive sentiment'])
print ('Positive sentiment 1/2', coeff['Positive sentiment 1/2'])
print ('Positive sentiment 2/2', coeff['Positive sentiment 2/2'])
print ('Positive sentiment 1/3', coeff['Positive sentiment 1/3'])
print ('Positive sentiment 2/3', coeff['Positive sentiment 2/3'])
print ('Positive sentiment 3/3', coeff['Positive sentiment 3/3'])
print ('Negative sentiment', coeff['Negative sentiment'])
print ('Negative sentiment 1/2', coeff['Negative sentiment 1/2'])
print ('Negative sentiment 2/2', coeff['Negative sentiment 2/2'])
print ('Negative sentiment 1/3', coeff['Negative sentiment 1/3'])
print ('Negative sentiment 2/3', coeff['Negative sentiment 2/3'])
print ('Negative sentiment 3/3', coeff['Negative sentiment 3/3'])

print ('Blob sentiment', coeff['Blob sentiment'])
print ('Blob subjectivity', coeff['Blob subjectivity'])
print ('Blob sentiment 1/2', coeff['Blob sentiment 1/2'])
print ('Blob sentiment 2/2', coeff['Blob sentiment 2/2'])
print ('Blob subjectivity 1/2', coeff['Blob subjectivity 1/2'])
print ('Blob subjectivity 2/2', coeff['Blob subjectivity 2/2'])
print ('Blob sentiment 1/3', coeff['Blob sentiment 1/3'])
print ('Blob sentiment 2/3', coeff['Blob sentiment 2/3'])
print ('Blob sentiment 3/3', coeff['Blob sentiment 3/3'])
print ('Blob subjectivity 1/3', coeff['Blob subjectivity 1/3'])
print ('Blob subjectivity 2/3', coeff['Blob subjectivity 2/3'])
print ('Blob subjectivity 3/3', coeff['Blob subjectivity 3/3'])

print ('topics:')


topics_tag=[]
topics_coeff=[]
topics_num=[]
for j in range(200):
    topics_tag.append('Topic :' +str(j))
    topics_coeff.append(coeff[topics_tag[j]])
    topics_num.append(j)
topics_tag=np.array(topics_tag)
topics_num=np.array(topics_num)
topics_coeff=np.array(topics_coeff)

topics_num=topics_num[topics_coeff.argsort()]
topics_tag=topics_tag[topics_coeff.argsort()]
topics_coeff=topics_coeff[topics_coeff.argsort()]
for j in range(10):
    print (topics_coeff[j], topic_mod.get_topic(topics_num[j]))
for j in range(190,200):
    print (topics_coeff[j], topic_mod.get_topic(topics_num[j]))


print ('Validating')

output = classifier.predict(testvec)
print (classification_report(test_targets, output, target_names=cls_set))


basic_test=["I like eating food","I love when the baby screams",\
            'I just love when you make me feel like shit',"She didn't like the recently released movie", \
            "Isn'it great when your girlfriend dumps you ?", "Sun rises in the east.",'I love spamming !']

print("Basic test")
feature_basictest=[]
for tweet in basic_test: 
    feature_basictest.append(text_features.dialogue_act_features(tweet,topic_mod))
feature_basictest=np.array(feature_basictest) 
feature_basictestvec = vec.transform(feature_basictest)

print (basic_test)
'''
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--text", required=True,
	help="Enter text")
args = vars(ap.parse_args())
feature_basictest=[]
tweet=args["t"]
feature_basictest.append(feature_extract.dialogue_act_features(tweet,topic_mod))
feature_basictest=np.array(feature_basictest) 
feature_basictestvec = vec.transform(feature_basictest)
'''
print (classifier.predict(feature_basictestvec))
#y_score = classifier.decision_function(feature_basictestvec)
print (classifier.decision_function(feature_basictestvec))
'''precision, recall, _ = precision_recall_curve(y_test, y_score)

plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.show()
'''
