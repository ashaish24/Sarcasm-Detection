# Sarcasm-Detection
Detects whether a given sentence is sarcastic or non-sarcastic. It uses SVM classifier to classify the sentence as sarcastic or
not, by training the classifier using the features extracted from the given text.

### Dataset Collection 
Twitter API can be used to stream tweets with the label sarcasm (sarcastic texts) and other tweets that dont have the label sarcasm (non-sarcastic texts). 

### Preprocessing 
Removal of all the hashtags, http links, non ASCII characters and tweets that start with the @ symbol. Removal of @tagging and any mention
of the word sarcasm or sarcastic. If after this stage the tweet is not empty and contains at least 3 words, it is added to the dataset list.

### Feature Extraction 
Five features are extracted from the tweets:
Feature 1 - n-grams - To extract those, each tweet is tokenized, stemmed, uncapitalized. Porter Stemmer algorithm is used for stemming the words to their base word. nltk package is used for extracting the unigram and bigram from each sentence.

Feature 2 - Sentiments - First the tweet slang and emojis must be converted to a recognisable word. (example: :-) –good, u –you). A tweet is broken up into two and three parts. Sentiment scores are calculated using two libraries (SentiWordNet and TextBlob). Positive and negative sentiment scores are collected for the overall tweet as well as each individual part. Furthermore, the contrast between the parts are inserted into the features. SentiWordNet and TextBlob libraries are used to extract sentiment from sentence.

Feature 3 - Topic
The Python library gensim which implements topic modeling using Latent Dirichlet Allocation (LDA) is used to learn the topics. The
collection of topics for each tweet is then inserted into the features.

Feature 4- Parts of Speech -The parts of speech in each tweet are counted and inserted into the features. SentiWordNet library is used, which contains parts of speech for each word.

Feature 5 - Capitalizations - A binary flag indicating whether the tweet contains at least 4 tokens that start with a capitalization is inserted into the features. isupper() function is used to get the count of capital letters in each word.

## SVM classifier Algorithm 4.2 is used for classifying sarcastic tweets.
