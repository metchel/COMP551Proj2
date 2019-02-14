import os
import sys
import numpy as np
import re
import csv
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.model_selection import train_test_split
from naive_bayes import BernoulliNaiveBayes
from preprocessing import Preprocessor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

pos_examples = [open('../data/train/pos/' + f).read() for f in os.listdir('../data/train/pos')]
neg_examples = [open('../data/train/neg/' + f).read() for f in os.listdir('../data/train/neg')]


test = {}
for n in range(25000):
    filename = '../data/test/{}.txt'.format(n)
    test[n] = open(filename, 'r').read()

X = pos_examples + neg_examples
y = [1 if i < len(pos_examples) else 0 for i in range(len(pos_examples) + len(neg_examples))]

train_X, validate_X, train_y, validate_y = train_test_split(X, y, test_size=0.01)

cv = CountVectorizer(binary=True, analyzer='word', lowercase=True, stop_words='english', max_features=100000, ngram_range=(1, 2))
tfidf = TfidfVectorizer(binary=False, stop_words='english', max_features=150000, ngram_range = (1, 2))

pp = Preprocessor()
train_X, validate_X = pp.process_data(train_X), pp.process_data(validate_X)
#train_features = pp.process_data(train_X, cv, True)
#validate_features = pp.process_data(validate_X, cv, False)
#train_features_2 = pp.process_data(train_X, tfidf, True)
#validate_features_2 = pp.process_data(validate_X, tfidf, False)

#train_features = cv.fit_transform(train_X, train_y)
#validate_features = cv.transform(validate_X)
train_features_2 = tfidf.fit_transform(train_X, train_y)
validate_features_2 = tfidf.transform(validate_X)

test_X = pp.process_data(test.values())
test_features = tfidf.transform(test_X)

"""
nb = BernoulliNB()
nb.fit(train_features, train_y)
accuracy_train = accuracy_score(train_y, nb.predict(train_features))
accuracy_validate = accuracy_score(validate_y, nb.predict(validate_features))
"""

for c in [1, 10, 100]:
    print('C = {}'.format(c))
    lr = LogisticRegression(C=c)
    lr.fit(train_features_2, train_y)
    accuracy_train_2 = accuracy_score(train_y, lr.predict(train_features_2))
    accuracy_validate_2 = accuracy_score(validate_y, lr.predict(validate_features_2))
    print('TRAINING NB_TFIDF: {}'.format(accuracy_train_2))
    print('VALIDATING NB_TFIDF: {}'.format(accuracy_validate_2))

"""
mnb = MultinomialNB()
mnb.fit(train_features, train_y)
accuracy_train_3 = accuracy_score(train_y, mnb.predict(train_features))
accuracy_validate_3 = accuracy_score(validate_y, mnb.predict(validate_features))
"""

"""
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(train_features_2, train_y)
accuracy_train_3 = accuracy_score(train_y, rfc.predict(train_features_2))
accuracy_validate_3 = accuracy_score(validate_y, rfc.predict(validate_features_2))


#print('TRAINING NB: {}'.format(accuracy_train))
#print('VALIDATING NB: {}'.format(accuracy_validate))
#print('TRAINING NB_TFIDF: {}'.format(accuracy_train_2))
#print('VALIDATING NB_TFIDF: {}'.format(accuracy_validate_2))
"""

lr_2 = LogisticRegression(C=10)
lr_2.fit(train_features_2, train_y)
predictions = lr_2.predict(test_features)


with open('results.csv', 'w') as results:
    writer = csv.writer(results, delimiter=',')
    writer.writerow(['Id', 'Category'])
    for i in range(25000):
        writer.writerow([i, predictions[i]])
