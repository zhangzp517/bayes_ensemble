import numpy as np
import pickle
import os

import matplotlib.pyplot as plt

from sklearn import naive_bayes
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import log_loss, accuracy_score

from scipy import stats


def create_ensemble(classifier, X, num, num_features):
    """
    """
    estimators = [None] * num
    n_feats = X.shape[1]

    for i in range(num):

        estimators[i] = clone(classifier)
        estimators[i].sub_feats = list(np.sort(np.random.choice(
            range(n_feats), size = num_features, replace = False)))

    return estimators


def fit_ensemble(ensemble, x, y, partial = False, classes = None):
    """
    """
    for e in ensemble:

        x_ = x[:, e.sub_feats]
        
        if not partial:

            e.fit(x_, y)

        else:

            e.partial_fit(x_, y, classes = classes)

    return ensemble


def predict_vote(ensemble, X, classes):
    """
    """
    ensemble_preds = [e.predict_proba(X[:, e.sub_feats]) for e in ensemble]
    pred = np.concatenate(ensemble_preds).sum(axis = 0)

    return classes[np.argmax(pred)]


def online_test(ensemble, single, X, y, classes, start_prop):
    """
    """
    X_start, X_stream, y_start, y_stream = train_test_split(
        X, y, test_size = 1 - start_prop)

    stream_size = X_stream.shape[0]

    single_preds = [None] * stream_size
    vote_preds = [None] * stream_size

    """
    initial fit
    """

    single.fit(X_start, y_start)
    ensemble = fit_ensemble(ensemble, X_start, y_start)

    for j in range(stream_size):

        x_ = X_stream[j].reshape(1, -1)
        y_ = y_stream[j].reshape(1,)

        single_preds[j] = single.predict(x_)
        vote_preds[j] = predict_vote(ensemble, x_, classes)

        single.partial_fit(x_, y_, classes = classes)
        ensemble = fit_ensemble(ensemble, x_, y_, partial = True, classes = classes)

    single_preds = np.array(single_preds).reshape(y_stream.shape)
    vote_preds = np.array(vote_preds).reshape(y_stream.shape)

    return single_preds, vote_preds, y_stream




if __name__ == 'main':

    use_datasets = ['mushrooms.txt', 'australian.txt', 'heart.txt',
                    'ionosphere.txt', 'sonar.txt']
    use_classes = [(1, 2), (-1, 1), (-1, 1), (-1, 1), (-1, 1)]

    os.chdir('/users/zgallegos/documents/school/math_538/project/data')


    for j, k in enumerate(use_datasets):

        fl = use_datasets[j]
        cls = use_classes[j]

        X, y = load_svmlight_file(fl)
        X = X.toarray()
        y = y.astype(int)

        perc = .75
        feats = int(np.floor(.75 * X.shape[1]))

        n_trials = 5

        sing = []
        vote = []

        for i in range(n_trials):

            single = naive_bayes.BernoulliNB()
            ensemble = create_ensemble(naive_bayes.BernoulliNB(), X, 100, feats)

            s, v, y_ = online_test(ensemble, single, X, y, cls, .1)

            sing.append(1 - accuracy_score(y_, s))
            vote.append(1 - accuracy_score(y_, v))

        print(k)
        print('single mean error %s' % np.mean(sing))
        print('vote mean error %s' % np.mean(vote))

        indx = range(1, len(y_) + 1)
        cum_sing = 1 - np.cumsum(s == y_) / indx
        cum_vote = 1 - np.cumsum(v == y_) / indx

        plt.plot(indx, cum_sing)
        plt.plot(indx, cum_vote)
        plt.legend(['singular', 'voting'], loc = 'upper right')
        plt.show()


