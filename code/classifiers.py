import numpy as np

def sgn(scalar):
    return 1 if scalar >= 0 else -1

class Perceptron:

    def __init__(self, dim, epochs=20, learning_rate=0.1):
        self.eta = learning_rate
        self.epochs = epochs
        self.w = np.random.uniform(-0.1, 0.1, size=dim)
        self.b = np.random.uniform(-0.1, 0.1)
        self.t = 0
        self.mistakes = 0

    def update(self, x, y, weight=1.0):
        y_prime = sgn(np.dot(self.w, x) + self.b)

        if y_prime != y:
            self.w += weight * self.eta * y * x
            self.b += weight * self.eta * y
            self.mistakes += 1

    def label_binarize(self, labels):
        l1, l2 = np.unique(labels)
        labels[labels == l1] = -1
        labels[labels == l2] = +1
        return labels

    def fit(self, X, Y, sample_weight=None):
        if len(np.unique(Y)) > 2:
            raise ValueError("Error: More than 2 class labels found.")

        X = (X * sample_weight[:, np.newaxis]) if sample_weight is not None else X

        Y = self.label_binarize(Y)


        for epoch in range(self.epochs):
            indices = np.random.permutation(len(X))
            X = X[indices]
            Y = Y[indices]

            for idx in range(len(X)):
                # weight = sample_weight[idx] if sample_weight is not None else 1
                self.update(X[idx], Y[idx], weight=1)
                self.t += 1

        self.set_coef_intercept()

        return self

    def set_coef_intercept(self):
        self.coef_ = [self.w]
        self.intercept_ = self.b

    def decision_function(self, X):
        return np.array([np.dot(self.w, x) + self.b for x in X])


    def predict(self, X):
        return np.array([sgn(np.dot(self.w, x) + self.b) for x in X])

    def score(self, X, Y):
        return np.average(self.label_binarize(Y) == self.predict(X))

class AveragedPerceptron(Perceptron):
    def __init__(self, dim, epochs=20, learning_rate=0.1):
        super().__init__(dim, epochs=epochs, learning_rate=learning_rate)
        self.avg_w = np.zeros(dim)
        self.avg_b = 0.

    def update(self, x, y):
        y_prime = sgn(np.dot(self.w, x) + self.b)

        if y_prime != y:
            self.w += self.eta * y * x
            self.b += self.eta * y
            self.mistakes += 1

        self.avg_w += self.w
        self.avg_b += self.b

    def set_coef_intercept(self):
        self.coef_ = [self.avg_w]
        self.intercept_ = self.avg_b

    def predict(self, X):
        return np.array([sgn(np.dot(self.avg_w, x) + self.avg_b) for x in X])