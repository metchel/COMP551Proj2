import math
import numpy as np
from scipy import sparse

class BernoulliNaiveBayes():
    def __init__(self):
        self.class_probabilities = {0: 0, 1: 0}
        self.feature_probabilities = {0: [], 1: []}

    def fit(self, X, y):
        if isinstance(X, list):
            X = np.array(X)
        n, m = X.shape
        class_counts = {0: 0, 1: 0}
        feature_counts = {0: np.zeros(m), 1: np.zeros(m)}

        for y_i in y:
            class_counts[y_i] += 1

        sparse_matrix = sparse.csr_matrix(X).nonzero()
        (row, col) = sparse_matrix
        for i in range(len(row)):
            c = y[row[i]]
            feature_counts[c][col[i]] += 1

        self.class_probabilities = {0: math.log(class_counts[0]/float(n)), 1: math.log(class_counts[1]/float(n))}
        self.feature_probabilities = {
            0: [math.log((feature_count + 1)/float(class_counts[0] + 2)) for feature_count in feature_counts[0]],
            1: [math.log((feature_count + 1)/float(class_counts[1] + 2)) for feature_count in feature_counts[1]]
        }

        return self

    def predict(self, X):
        if isinstance(X, sparse.csr.csr_matrix):
            X = X.toarray()
        
        predictions = []
        for i, x_i in enumerate(X):
            features = [j for j in range(len(x_i)) if x_i[j] == 1]
            prob_1 = self.class_probabilities[1] + sum([self.feature_probabilities[1][i] for i in features])
            prob_0 = self.class_probabilities[0] + sum([self.feature_probabilities[0][i] for i in features])

            if prob_1 >= prob_0:
                predictions.append(1)
            if prob_1 < prob_0:
                predictions.append(0)

        return predictions

