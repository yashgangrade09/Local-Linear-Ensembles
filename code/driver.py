import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron as skPerceptron, LogisticRegression
from classifiers import Perceptron, AveragedPerceptron
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture, GMM
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm
from data import *
import seaborn as sns
sns.set_style('darkgrid')

# Set seed for reproducibility
# np.random.seed(20)

# n_comp = 9


def generate_data(n, d):
    # data = BadlyNLS()
    # data = AlmostLS()
    data = LocallyLS()
    # data = Moon()
    # data = CompData()
    data.generate(n, d)
    return data


def train_classifier(data, probs):
    # classifier = Perceptron(data.X.shape[1], epochs=30)
    # classifier = KNeighborsClassifier(n_neighbors=10)
    classifier = skPerceptron(max_iter=1000, tol=1e-4)
    # classifier = LogisticRegression()
    # print(data.X_train.flags)
    # classifier = SVC()
    # to_fit = np.ascontiguousarray(data.X_train)
    classifier.fit(data.X_train, data.Y_train, sample_weight=probs)
    # classifier.fit(data.X_train, data.Y_train)

    # classifier.set_coef_intercept()
    # classifier.fit(data.X_train, data.Y_train)

    # print('Accuracy:', classifier.score(data.X_test, data.Y_test))
    return classifier


def plot_linear_classifier(weight, intercept):
    slope = -(1/(weight[1] / weight[0]))
    x = np.arange(-100, 100, 0.1)
    y = [slope*xcord - intercept/weight[0] for xcord in x]
    plt.plot(x, y)


def fit_gmm(X, n=10):
    gmm_mdl = GaussianMixture(n_components=n)
    # gmm_mdl = BayesianGaussianMixture(n_components=n, max_iter=1000, n_init=3)
    gmm_mdl.fit(X)
    return gmm_mdl


def weighted_linear_clfs(data, weighing_model):
    probs = weighing_model.predict_proba(data.X_train)
    clfs = []
    for col in range(probs.shape[1]):
        thresholded_prob = probs.copy()
        thresholded_prob[thresholded_prob < 0.1] = 0
        clf = train_classifier(data, np.ascontiguousarray(thresholded_prob[:, col]))
        clfs.append(clf)
    return clfs


def predict_weighted(X, clfs, gmm):
    predictions = np.empty((len(X), len(clfs)))

    for idx, classifier in enumerate(clfs):
        predictions[:, idx] = classifier.predict(X)

    probs = gmm.predict_proba(X)
    weighted_preds = np.sum(np.multiply(predictions, probs), axis=1)
    # weighted_preds = np.average(predictions, axis=1)
    return np.round(weighted_preds)


from matplotlib.patches import Ellipse


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))

def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)

def locally_linear_model(data, gmm_model):
    classifiers = weighted_linear_clfs(data, gmm_model)
    preds = predict_weighted(data.X_test, classifiers, gmm_model)
    return np.average(preds == data.Y_test)

def global_model(data):
    clf = train_classifier(data, None)
    return clf.score(data.X_test, data.Y_test)

def gmm__ncomp_tests():
    n_comps = [2,3,4,5,6,7,8,9,10,20,30,40,50, 60, 70, 80, 90, 100]
    d = generate_data(num_datapoints, dim)
    d.split(test_fraction=0.5)
    llm_accs = []
    global_accs = []
    for n_comp in tqdm(n_comps):
        gmm = fit_gmm(d.X_train, n=n_comp)
        llm_accs.append(locally_linear_model(d, gmm))
        global_accs.append(global_model(d))


    plt.plot(n_comps, llm_accs)
    # plt.title("Effect of number of components 'k' of GMM")
    plt.xlabel('Number of gaussian components')
    plt.ylabel('Accuracy')
    plt.savefig('../figures/num_comp.png')

    return llm_accs, global_accs, n_comps

def num_points():
    n_comp = 20
    num_points_list = [100, 500, 1000, 5000, 10000, 20000, 30000, 40000, 50000, 100000]

    llm_accs = []
    global_accs = []
    for num_datapoints in tqdm(num_points_list):
        d = generate_data(num_datapoints, dim)
        d.split(test_fraction=0.5)
        gmm = fit_gmm(d.X_train, n=n_comp)
        llm_accs.append(locally_linear_model(d, gmm))
        global_accs.append(global_model(d))

    plt.plot(num_points_list, llm_accs, label='Locally linear model')
    plt.plot(num_points_list, global_accs, label='Global model')
    # plt.title("Effect of number of components 'k' of GMM")
    plt.xlabel('Number of gaussian components')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('../figures/num_points.png')

num_datapoints = 10000
dim = 20

gmm__ncomp_tests()
num_points()