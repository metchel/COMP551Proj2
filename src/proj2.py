#!/usr/local/bin/python3

import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
from os import path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def extract_training_samples(file):
    training_samples = []

    for line in open(path.relpath(file), encoding='ISO-8859-1'):
        training_samples.append(line.strip())
    
    return training_samples

def clean_training_samples(samples):
    REPLACE_WITH_EMPTY = re.compile("(\.)|(\,)|(\:)|(\;)|(\()|(\))|(\[)|(\])|(\?)|(\!)")
    REPLACE_WITH_SPACE = re.compile("(\-)")

    samples = [REPLACE_WITH_EMPTY.sub("", line.lower().strip()) for line in samples]
    samples = [REPLACE_WITH_SPACE.sub(" ", line) for line in samples]

    return samples

def lemmatize(words):
    lemmatizer = WordNetLemmatizer()
    new_words = []
    for word in words:
        new_words.append(lemmatizer.lemmatize(word))

    return new_words

def stem(words):
    stemmer = PorterStemmer()
    new_words = []
    for word in words:
        new_words.append(stemmer.stem(word))

    return new_words

def normalize(samples):
    new_samples = []

    for line in samples:
        words = line.split()
        line = ''
        words = stem(words)
        words = lemmatize(words)
        line = ' '.join(words)
        new_samples.append(line)

    return new_samples

def train_logistic_regression(train_X, train_y, test_X, test_y, coef):
    lr = LogisticRegression(C=coef)
    lr.fit(train_X, train_y)

    return generate_result(lr, test_X, test_y)

def select_param_logistic_regression(train_X, train_y, test_X, test_y, params):
    results = []
    for param in params:
        results.append((param, (train_logistic_regression(train_X, train_y, test_X, test_y, param))))

    return results

def select_best_logreg(results):
    best_logreg = -1, (-1, [[]])
    for result in results:
        c, (accuracy_score, conf_matrix) = result
        c_best, (best_acc_score, best_conf_matrix) = best_logreg
        if accuracy_score > best_acc_score:
            best_logreg = result

    return best_logreg

def train_naive_bayes(train_X, train_y, test_X, test_y):
    nb = MultinomialNB()
    nb.fit(train_X, train_y)

    return generate_result(nb, test_X, test_y)

def train_svm(train_X, train_y, test_X, test_y):
    lsvm = LinearSVC()
    lsvm.fit(train_X, train_y)

    return generate_result(lsvm, test_X, test_y)

def generate_result(model, test_X, test_y):
    return (
        accuracy_score(test_y, model.predict(test_X)),
        confusion_matrix(test_y, model.predict(test_X))
    )

def main():

    # Data labels: 1 = positive review, 0 = negative review
    POS_LABEL = 1
    NEG_LABEL = 0

    pos_samples = extract_training_samples('../data/rt-polaritydata/rt-polaritydata/rt-polarity.pos')
    neg_samples = extract_training_samples('../data/rt-polaritydata/rt-polaritydata/rt-polarity.neg')

    X = pos_samples + neg_samples
    # Preprocess data
    X = normalize(clean_training_samples(X))

    # Label vector y
    y = [POS_LABEL if i < len(pos_samples) else NEG_LABEL for i in range(len(pos_samples) + len(neg_samples))]

    # Split [X][y] into 80% training data 20% test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.80, test_size = 0.20)
    
    # Use only the training data to create the vector [[word1, word2, ..., wordn], [01...00], ..., [00...00]]
    # Stop words and ngrams are handled here
    cv = CountVectorizer(binary=True, analyzer='word', lowercase=True, stop_words='english', ngram_range=(1, 2))
    X_train_dtm = cv.fit_transform(X_train)
    X_test_dtm = cv.transform(X_test)
 
    LOGREG = 'LogisticRegression'
    print('------------------' + '\n' + LOGREG + '\n' + '------------------' + '\n')
    # Select logistic resgression model with best accuracy, varying C param 
    C_PARAMETER_RANGE = [0.001, 0.01, 0.1, 1, 10, 100]
    lr_results = select_param_logistic_regression(X_train_dtm, y_train, X_test_dtm, y_test, C_PARAMETER_RANGE)
    c, (lr_acc, lr_conf_mat) = select_best_logreg(lr_results)
    print('C Value: ' +  str(c) + '\n' + 'Accuracy Score: ' + str(lr_acc) + '\n' + 'Confusion Matrix: ' + '\n' + str(lr_conf_mat) + '\n\n')

    NB = 'MultinomialNB'
    print('------------------' + '\n' + NB + '\n' + '------------------' + '\n')
    # Multinomial Naive Bayes with default params
    nb_result = train_naive_bayes(X_train_dtm, y_train, X_test_dtm, y_test)
    print('Accuracy Score: ' + str(nb_result[0]) + '\n' + 'Confusion Matrix: ' + '\n' + str(nb_result[1]) + '\n\n')


    SVC = 'LinearSVC'
    print('------------------' + '\n' + SVC + '\n' + '------------------' + '\n')
    # SVC with default params
    svm_result = train_svm(X_train_dtm, y_train, X_test_dtm, y_test)
    print('Accuracy Score: ' + str(svm_result[0]) + '\n' + 'Confusion Matrix: ' + '\n' + str(svm_result[1]) + '\n\n')
