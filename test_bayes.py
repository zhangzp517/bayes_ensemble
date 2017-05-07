import pandas as pd
import numpy as np
import os
import re

from sklearn import naive_bayes
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import SGDClassifier


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


def fit_sgd_ensemble(ensemble, X, y):
    sgd = SGDClassifier(loss='log')
    ensemble_prediction = []
    for e in ensemble:

        X_ = X[:, e.sub_feats]

        e.fit(X_, y)
        ensemble_prediction.append(e.predict_proba(X_)[:,1])
    return sgd.fit(np.array(ensemble_prediction).transpose(),y)


def predict_sgd(ensemble, fitted_sgd, X):
    """
    """
    ensemble_preds = [e.predict_proba(X[:, e.sub_feats])[:,1][0] for e in ensemble]
    return fitted_sgd.predict(np.array(ensemble_preds).reshape(1,-1))


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


def output_loss(ensemble, dataset, trial):
    """
    """
    end_lams = np.array([e.lam[-1] for e in ensemble])

    min_ind = np.argmin(end_lams)
    max_ind = np.argmax(end_lams)
    med_ind = np.argsort(end_lams)[len(end_lams) // 2]

    lams_towrite = np.stack((ensemble[min_ind].lam,
                             ensemble[med_ind].lam,
                             ensemble[max_ind].lam), axis = 1)

    loss_towrite = np.stack((ensemble[min_ind].loss,
                             ensemble[med_ind].loss,
                             ensemble[max_ind].loss), axis = 1)

    df = pd.DataFrame(lams_towrite, columns = ['lam_min', 'lam_med', 'lam_max'])
    df.to_csv(os.path.join('results', 'loss', dataset + '_lam_samp_' + str(trial) + '.csv'))

    df = pd.DataFrame(loss_towrite, columns = ['loss_min', 'loss_med', 'loss_max'])
    df.to_csv(os.path.join('results', 'loss', dataset + '_loss_samp_' + str(trial) + '.csv'))


def online_test(ensemble, single, X, y, classes, start_prop, do_sgd, **kwargs):
    """
    perform the tests
    """
    X_start, X_stream, y_start, y_stream = train_test_split(
        X, y, test_size = 1 - start_prop)

    stream_size = X_stream.shape[0]

    single_preds = [None] * stream_size
    vote_preds = [None] * stream_size
    bayes_preds = [None] * stream_size
    sgd_preds = [None] * stream_size

    for e in ensemble:

        e.loss = [None] * stream_size
        e.lam = [None] * stream_size


    """
    initial fit
    """

    single.fit(X_start, y_start)
    ensemble = fit_ensemble(ensemble, X_start, y_start)

    if do_sgd:

        fitted_sgd = (fit_sgd_ensemble(ensemble, X_start, y_start))

    _X_ = X_start
    _y_ = y_start

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

        if do_sgd:

            sgd_preds[j] = predict_sgd(ensemble, fitted_sgd, x_)[0]

        """
        partial fit
        """

        single.partial_fit(x_, y_, classes = classes)
        ensemble = fit_ensemble(ensemble, x_, y_, partial = True, classes = classes)

        _X_=(np.vstack((_X_,x_)))
        _y_=(np.append(_y_,y_))

        if do_sgd:

            fitted_sgd = fit_sgd_ensemble(ensemble, _X_, _y_)

    single_preds = np.array(single_preds).reshape(y_stream.shape)
    vote_preds = np.array(vote_preds).reshape(y_stream.shape)
    bayes_preds = np.array(bayes_preds).reshape(y_stream.shape)
    sgd_preds = np.array(sgd_preds).reshape(y_stream.shape)

    return (ensemble, single_preds, vote_preds, 
            bayes_preds, sgd_preds, y_stream)




if __name__ == '__main__':

    """
    run the tests. five data files, ten trials each
    """

    os.chdir('/users/zgallegos/documents/school/math_538/project/data')

    use_datasets = ['heart.txt', 'australian.txt', 'ionosphere.txt', 
                    'sonar.txt', 'mushrooms.txt']
    use_classes = [(-1, 1), (-1, 1), (-1, 1), (-1, 1), (1, 2)]

    use_datasets = ['mushrooms.txt']
    use_classes = [(1, 2)]

    n_trials = 10
    feat_perc = .5 # percentage of the features ensemble classifiers get

    for j, k in enumerate(use_datasets):

        if k == 'mushrooms.txt':

            do_sgd = False

        else:

            do_sgd = True
        
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
        sgd = []

        trlz = range(n_trials)

        for i in trlz:

            cumerr_fl = fl_name + '_cumerrors_trial_%s.csv' % i

            single = naive_bayes.BernoulliNB()
            ensemble = create_ensemble(naive_bayes.BernoulliNB(), X, 100, feats)

            e, s, v, b, d, y_ = online_test(ensemble, single, X, y, cls, .1, do_sgd, 
                                                a = 1, b = 1, theta = .1)

            indx = range(1, len(y_) + 1)
            cum_sing = 1 - np.cumsum(s == y_) / indx
            cum_vote = 1 - np.cumsum(v == y_) / indx
            cum_bayes = 1- np.cumsum(b == y_) / indx
            cum_sgd = 1- np.cumsum(d == y_) / indx

            to_write = np.stack((indx, cum_sing, cum_vote, cum_bayes, cum_sgd), axis = 1)
            
            df = pd.DataFrame(to_write, columns = ['index', 'single', 'vote', 'bayes', 'sgd'])
            df.to_csv(os.path.join('results', 'cumulative', cumerr_fl))

            sing.append(1 - accuracy_score(y_, s))
            vote.append(1 - accuracy_score(y_, v))
            bayes.append(1 - accuracy_score(y_, b))
            sgd.append(1 - accuracy_score(y_, d))

            output_loss(e, fl_name, i)

        to_write = np.stack((trlz, sing, vote, bayes, sgd), axis = 1)

        df = pd.DataFrame(to_write, columns = ['trial', 'single', 'vote', 'bayes', 'sgd'])
        df.to_csv(os.path.join('results', 'accuracy', err_fl))






