import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.datasets import make_blobs
from sklearn.datasets import load_svmlight_file
from sklearn.feature_selection import SelectKBest

import matplotlib.pyplot as plt

class Data(object):
    """
    Base class for data objects
    """

    def __init__(self):
        self.X = None
        self.Y = None

    def generate(self, n, d):
        """
        Generate `n` points of `d` dimension

        :param n: number of data points to generate
        :type n: int
        :param d: dimension of each data point
        :type d: int
        """

        self.n = n
        self.d = d
        self.X = np.random.rand(n, d)
        self.Y = np.random.choice([0, 1], size=n)

    def visualize(self, method='PCA', fraction=1.0):
        if fraction < 1.0:
            X_sampled = self.X[np.random.choice(self.n, int(fraction * self.n), replace=False), :]
        else:
            X_sampled = self.X

        if self.d == 2:
            X_proj = self.X
        else:
            if method == 'TSNE':
                X_proj = TSNE().fit_transform(X_sampled)
            elif method == 'PCA':
                X_proj = PCA(n_components=2, whiten=True).fit_transform(X_sampled)

        x, y = X_proj[:, 0], X_proj[:, 1]
        # plt.scatter(x, y, c=self.Y, s=50)
        return x, y

    def sample(self, k):
        X_sampled = self.X[np.random.choice(self.n, k, replace=False), :]
        return X_sampled

    def sample_k_nearest(self, point, k):
        dist = cdist(np.expand_dims(point, axis=0), self.X)
        least_dist_k = np.argsort(dist[0])[:k]
        return self.X[least_dist_k]

    def split(self, test_fraction= 0.2):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=test_fraction)


class AlmostLS(Data):

    def generate(self, n, d, seperation=4.0, pos_fraction=0.5):
        """
        Generate `n` points of `d` dimension

        :param n: number of data points to generate
        :type n: int
        :param d: dimension of each data point
        :type d: int
        """
        self.n = n
        self.d = d

        # 95% of data lies in a 2*sigma ball around the mean.
        # For unit variance normal, the 2*sigma ball mean the centers
        # should be at distance of 4 in the d dimensional space.
        mean1, mean2 = np.array([0.] * d), np.array([seperation / np.sqrt(d)] * d)
        cov = np.eye(d)

        # mean1, mean2 = np.random.random_sample((d,)), np.random.random_sample((d,)) + np.power(seperation , 1 / (2*d))
        dist_mean = np.linalg.norm(mean1 - mean2, ord=2)
        # print(dist_mean)
        # cov = np.eye(d)

        frac = int(1 / pos_fraction)

        cluster1 = np.random.multivariate_normal(mean1, cov, size=n // frac)
        cluster2 = np.random.multivariate_normal(mean2, cov, size=(n - n//frac))

        self.X = np.vstack((cluster1, cluster2))
        self.Y = np.array([0]*(n // frac) + [1]*(n - n // frac))

        # Make blobs
        # if True:
        #     self.X, self.Y = make_blobs(n_samples=self.n, n_features=self.d, centers=2)


class BadlyNLS(AlmostLS):

    def generate(self, n, d, seperation=1.0, pos_fraction=0.5):
        super().generate(n, d, seperation=seperation, pos_fraction=pos_fraction)


class LocallyLS(Data):

    def generate(self, n, d, centres=10):
        self.n = n
        self.d = d
        self.X, Y = make_blobs(n_samples=n, n_features=d, centers=20, cluster_std=5)
        self.Y =np.array([1 if y % 2 == 0 else -1 for y in Y])


class Moon(Data):
    def generate(self, n, d):
        self.n = n
        self.d = d
        self.X, Y = make_moons(n, noise=0.2)
        self.Y = Y % 2


class CompData(Data):

    def generate(self, n, d):
        self.X, self.Y = load_svmlight_file('../data/competitive/data.train')
        self.X = self.X.toarray()[:5000]
        self.Y = self.Y[:5000]
        self.n = len(self.X)
        self.d = self.X.shape[1]
