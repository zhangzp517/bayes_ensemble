import numpy as np
import pandas as pd
import os
import re

from sklearn import naive_bayes
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from sklearn.datasets import load_svmlight_file


def create_ensemble(classifier, X, num, num_features):
    """
    create a uniform ensemble of <num> classifiers, each of which uses
    <num_features> randomly-sampled features of the data
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
    fit the ensemble, each classifier with its own set of features
    if <partial>, do an online fit
    """
    for e in ensemble:

        x_ = x[:, e.sub_feats]
        
        if not partial:

            e.fit(x_, y)

        else:

            e.partial_fit(x_, y, classes = classes)

    return ensemble


def update_bayes(ensemble, i, x, y, classes, a, b, theta):
    """
    update loss and ensemble weights
    """
    t = i + 1

    for e in ensemble:

        x_ = x[:, e.sub_feats]
        pred = e.predict_proba(x_)
        e.loss[i] = log_loss(y, pred, labels = classes)

        num = a + t
        den = b + theta * np.sum(e.loss[0:t])
        e.lam[i] = num / den

    return ensemble


def predict_bayes(ensemble, i, X, classes):
    """
    ensemble prediction using posterior ensemble weights
    """
    m = len(ensemble)
    neg_label, pos_label = classes
    
    if i == -1:

        return predict_vote(ensemble, X, classes)

    curr_lams = np.array([e.lam[i] for e in ensemble])

    loss_neg = [None] * m
    loss_pos = [None] * m

    for i, e in enumerate(ensemble):

        pred = e.predict_proba(X[:, e.sub_feats])

        loss_neg[i] = log_loss([neg_label], pred, labels = classes)
        loss_pos[i] = log_loss([pos_label], pred, labels = classes)

    loss_neg = np.array(loss_neg)
    loss_pos = np.array(loss_pos)

    probs = [sum(curr_lams * loss_neg), sum(curr_lams * loss_pos)]

    return classes[np.argmin(probs)]


def predict_vote(ensemble, X, classes):
    """
    ensemble prediction by soft voting
    """
    ensemble_preds = [e.predict_proba(X[:, e.sub_feats]) for e in ensemble]
    pred = np.concatenate(ensemble_preds).sum(axis = 0)

    return classes[np.argmax(pred)]


def online_test(ensemble, single, X, y, classes, start_prop, **kwargs):
    """
    perform the tests
    """
    X_start, X_stream, y_start, y_stream = train_test_split(
        X, y, test_size = 1 - start_prop)

    stream_size = X_stream.shape[0]

    single_preds = [None] * stream_size
    vote_preds = [None] * stream_size
    bayes_preds = [None] * stream_size

    for e in ensemble:

        e.loss = [None] * stream_size
        e.lam = [None] * stream_size


    """
    initial fit
    """

    single.fit(X_start, y_start)
    ensemble = fit_ensemble(ensemble, X_start, y_start)

    for j in range(stream_size):

        x_ = X_stream[j].reshape(1, -1)
        y_ = y_stream[j].reshape(1,)

        """
        update bayes loss and weights
        """

        ensemble = update_bayes(ensemble, j, x_, y_, classes, **kwargs)

        """
        make predictions
        """

        single_preds[j] = single.predict(x_)
        vote_preds[j] = predict_vote(ensemble, x_, classes)
        bayes_preds[j] = predict_bayes(ensemble, j - 1, x_, classes)

        """
        partial fit
        """

        single.partial_fit(x_, y_, classes = classes)
        ensemble = fit_ensemble(ensemble, x_, y_, partial = True, classes = classes)

    single_preds = np.array(single_preds).reshape(y_stream.shape)
    vote_preds = np.array(vote_preds).reshape(y_stream.shape)
    bayes_preds = np.array(bayes_preds).reshape(y_stream.shape)

    return single_preds, vote_preds, bayes_preds, y_stream



if __name__ == '__main__':

    """
    run the tests. five data files, ten trials each
    """

    os.chdir('/users/zgallegos/documents/school/math_538/project/data')

    use_datasets = ['heart.txt', 'mushrooms.txt', 'australian.txt',
                    'ionosphere.txt', 'sonar.txt']
    use_classes = [(-1, 1), (1, 2), (-1, 1), (-1, 1), (-1, 1)]

    n_trials = 10
    feat_perc = .5 # percentage of the features ensemble classifiers get

    for j, k in enumerate(use_datasets):
        
        fl_name = re.search('^.+?(?=\.txt)', k).group(0)
        err_fl = fl_name + '_errors.csv'

        cls = use_classes[j]

        X, y = load_svmlight_file(k)
        X = X.toarray()
        y = y.astype(int)

        feats = int(np.floor(feat_perc * X.shape[1]))

        sing = []
        vote = []
        bayes = []

        trlz = range(n_trials)

        for i in trlz:

            cumerr_fl = fl_name + '_cumerrors_trial_%s.csv' % i

            single = naive_bayes.BernoulliNB()
            ensemble = create_ensemble(naive_bayes.BernoulliNB(), X, 100, feats)

            s, v, b, y_ = online_test(ensemble, single, X, y, cls, 
                            start_prop = .1, a = 1, b = 1, theta = .1)

            indx = range(1, len(y_) + 1)
            cum_sing = 1 - np.cumsum(s == y_) / indx
            cum_vote = 1 - np.cumsum(v == y_) / indx
            cum_bayes = 1- np.cumsum(b == y_) / indx

            to_write = np.stack((indx, cum_sing, cum_vote, cum_bayes), axis = 1)
            
            df = pd.DataFrame(to_write, columns = ['index', 'single', 'vote', 'bayes'])
            df.to_csv(os.path.join('results', 'cumulative', cumerr_fl))

            sing.append(1 - accuracy_score(y_, s))
            vote.append(1 - accuracy_score(y_, v))
            bayes.append(1 - accuracy_score(y_, b))

        to_write = np.stack((trlz, sing, vote, bayes), axis = 1)

        df = pd.DataFrame(to_write, columns = ['trial', 'single', 'vote', 'bayes'])
        df.to_csv(os.path.join('results', 'accuracy', err_fl))





