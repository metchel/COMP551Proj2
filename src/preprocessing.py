from sklearn.feature_extraction.text import CountVectorizer
import nltk
import numpy as np
from nltk.stem import PorterStemmer, WordNetLemmatizer
from scipy.sparse import hstack
class Preprocessor():
    def __init__(self, _word_count=True, _all_caps=True, _exclamations=True, _questions=True):
        self.vectorizer = None
        self.is_vectorized = False
        self.word_count = _word_count
        self.all_caps = _all_caps
        self.exclamations = _exclamations
        self.questions = _questions

    def extract_features(self, text):
        features = dict()
        if self.exclamations:
            features['exclamations'] = text.count('!')
        if self.questions:
            features['questions'] = text.count('?')
        tokens = nltk.word_tokenize(text)
        if self.word_count:
            features['word_count'] = len(tokens)
        if self.all_caps:
            all_caps = [w for w in tokens if w.isupper()]
            features['all_caps'] = len(all_caps)

        return list(features.values())

    def vectorize(self, data, MAX_FEATURES=5000):
        if not self.is_vectorized:
            cv = CountVectorizer(binary=True, max_features=MAX_FEATURES, stop_words='english', analyzer='word', lowercase=True, ngram_range=(1, 2))
            word_features = cv.fit_transform(data)
            self.vectorizer = cv
            self.is_vectorized = True
            print(cv.get_feature_names())
            return word_features.toarray()
        else:
            cv.transform(data)

    def process_data(self, data):
        print("Beginning processing")
        features = []

        non_word_features = []
        word_features = []
        lemmatizer = WordNetLemmatizer()
        stemmer = PorterStemmer()
        X = []
        cleaned_data = []
        for datum in data:
            tokened = nltk.word_tokenize(datum)
            stemmed = [stemmer.stem(w) for w in tokened]
            lemmad = [lemmatizer.lemmatize(w) for w in stemmed]
            cleaned_data.append(" ".join(lemmad))

        print("End processing")
        return cleaned_data

"""
TEST = [
'THIS MOVIE IS like SOOO BAD!!! WTF were they thinking making this??? LOL it sucks.',
'This movie was FRIGGIN EPIC!!!! The best part was the middle of it.',
'I really liked the main actor in this movie. The way that his hair swayed with the wind was quite awesome. Did you guys also like his hair??',
'There are at least 10 awesome things that this movie didn\'t do and hence I enjoyed it very little.'
]


pp = Preprocessor()
features = pp.process_data(TEST)
print(features)
"""
