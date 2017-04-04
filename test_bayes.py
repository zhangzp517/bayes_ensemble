import numpy as np
import pickle

from sklearn import naive_bayes
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import log_loss

from scipy import stats

fl = '/users/zgallegos/documents/school/math_538/project/data/australian.txt'

X, y = load_svmlight_file(fl)
X = X.toarray()

def create_ensemble(classifier, X, num, num_features):
    """
    """
    estimators = [None] * num
    n_feats = X.shape[1]

    for i in range(num):

        estimators[i] = clone(classifier)
        estimators[i].sub_feats = np.random.choice(
            range(n_feats), size = num_features, replace = False)

    return estimators


def fit_ensemble(ensemble, X, y, partial = False, classes = None):
    """
    """
    for e in ensemble:

        X_ = X[:, e.sub_feats]
        
        if not partial:

            e.fit(X_, y)

        else:

            e.partial_fit(X_, y, classes = classes)


def predict_vote(ensemble, X):
    """
    """
    ensemble_preds = [e.predict(X[:, e.sub_feats]) for e in ensemble]

    return stats.mode(ensemble_preds)[0]


def update_bayes(ensemble, i, X, y, a, b, theta):
    """
    """
    for e in ensemble:

        e_pred = e.predict_proba(X[:, e.sub_feats])
        e.g[i] = log_loss(y, e_pred, labels = [-1, 1])

        num = a + (i + 1)
        den = b + theta * sum(e.g[0:(i+1)])
        e.lam[i] = num / den


def predict_bayes(ensemble, i, X):
    """
    """
    lams = np.array([e.g[i] for e in ensemble])
    
    l_n = [None] * len(ensemble)
    l_p = [None] * len(ensemble)

    for i, e in enumerate(ensemble):

        e_pred = e.predict_proba(X[:, e.sub_feats])

        l_n[i] = log_loss([-1], e_pred, labels = [-1, 1])
        l_p[i] = log_loss([1], e_pred, labels = [-1, 1])

    l_n = np.array(l_n)
    l_p = np.array(l_p)

    if sum(lams * l_p) <= sum(lams * l_n):

        return 1

    else:

        return -1


def online_test(single, ensemble, X, y, classes, start_prop):
    """
    """

    X_start, X_stream, y_start, y_stream = train_test_split(
        X, y, test_size = 1 - start_prop)

    """
    get sizes, add attributes
    """

    stream_size = X_stream.shape[0]

    single_preds = [None] * stream_size
    vote_preds = [None] * stream_size
    bayes_preds = [None] * stream_size

    for e in ensemble:
        """
        loss and posterior means for bayes
        """
        e.g = [None] * stream_size
        e.lam = [None] * stream_size

    """
    initial fit
    """

    single.fit(X_start, y_start)
    fit_ensemble(ensemble, X_start, y_start)

    """
    online predictions
    """

    for j in range(stream_size):

        x_ = X_stream[j].reshape(1, -1)
        y_ = y_stream[j].reshape(1,)

        single_preds[j] = single.predict(x_)
        vote_preds[j] = predict_vote(ensemble, x_)

        if j == 0:

            bayes_preds[j] = predict_vote(ensemble, x_)

        else:

            bayes_preds[j] = predict_bayes(ensemble, j-1, x_)

        update_bayes(ensemble, j, x_, y_, 1, 1, .1)

        single.partial_fit(x_, y_, classes = classes)
        fit_ensemble(ensemble, x_, y_, partial = True, classes = classes)

    single_preds = np.array(single_preds).reshape(y_stream.shape)
    vote_preds = np.array(vote_preds).reshape(y_stream.shape)
    bayes_preds = np.array(bayes_preds).reshape(y_stream.shape)

    return single_preds, vote_preds, bayes_preds, y_stream



if __name__ == '__main__':

    n_trials = 5

    sing = []
    vote = []
    bayes = []

    for i in range(n_trials):

        nb = naive_bayes.GaussianNB()
        e = create_ensemble(nb, X, 100, 3)

        s, v, b, y1 = online_test(nb, e, X, y, (-1, 1), .1)

        sing.append(1 - np.mean(s == y1))
        vote.append(1 - np.mean(v == y1))
        bayes.append(1 - np.mean(b == y1))

    print('singular predictor overall error: %s' % np.mean(sing))
    print('ensemble voting overall error: %s' % np.mean(sing))
    print('bayesian posterior estimation of ensemble'
          'weights overall error: %s' % np.mean(sing))






