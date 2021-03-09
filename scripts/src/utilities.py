from sklearn.model_selection import StratifiedKFold
import numpy as np
from random import randint, random

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import CountVectorizer

import dimension_reduce as dr
from de import de_rf, de_lr, de_nb, de_mlpn, de_knn
from de import de_smote

def get_result_statistics(test_y, predictions):
    tn, fp, fn, tp = confusion_matrix(test_y, predictions).ravel()
    PD = tp / (tp + fn)
    PF = fp / (fp + tn)
    PREC = tp / (tp + fp)
    F_MEASURE = 2 * PD * PREC / (PD + PREC)
    G_MEASURE = 2 * PD * (1 - PF) / (PD + 1 - PF)
    return tn, fp, fn, tp, PD, PF, PREC, F_MEASURE, G_MEASURE

def result_statistics(test_y, predictions):
    tn, fp, fn, tp = confusion_matrix(test_y, predictions).ravel()
    print("TN, FP, FN, TP: ", (tn, fp, fn, tp))

    PD = tp / (tp + fn)
    PF = fp / (fp + tn)
    PREC = tp / (tp + fp)
    F_MEASURE = 2 * PD * PREC / (PD + PREC)
    G_MEASURE = 2 * PD * (1 - PF) / (PD + 1 - PF)
    print("pd: ", PD)
    print("pf: ", PF)
    print("prec: ", PREC)
    print("f-measure: ", F_MEASURE)
    print("g-measure: ", G_MEASURE)
   
def random_forest(features, target, n_estimators, min_samples_leaf, min_samples_split, max_leaf_nodes, max_features,
                  max_depth):
    rf = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                min_samples_split=min_samples_split, max_leaf_nodes=max_leaf_nodes,
                                max_features=max_features, max_depth=max_depth)
    rf.fit(features, target)
    return rf


def random_forest_default(features, target):
    rf = RandomForestClassifier()
    rf.fit(features, target)
    return rf


def KNN(features, target, n_neighbors, leaf_size):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, leaf_size=leaf_size)
    knn.fit(features, target)
    return knn


def KNN_default(features, target):
    knn = KNeighborsClassifier()
    knn.fit(features, target)
    return knn


def logistic_regression(features, target, c, max_iter, verbose):
    lr = LogisticRegression(C=c, max_iter=max_iter, solver="lbfgs")
#    print("++++++++++++Paras: Features++++++++++++++", features)
#    print("++++++++++++Paras: target, c, max_iter++++++++++++++", target)
#    print("++++++++++++Paras: taret, c, max_iter++++++++++++++", c)
#    print("++++++++++++Paras: taret, c, max_iter++++++++++++++", max_iter)        
    lr.fit(features, target)
    return lr


def logistic_regression_default(features, target):
    lr = LogisticRegression()
    lr.fit(features, target)
    return lr


def naive_bayes(features, target, var_smoothing):
    nb = GaussianNB(var_smoothing=var_smoothing)
    nb.fit(features, target)
    return nb

#def naive_bayes(features, target, var_smoothing):
#    nb = MultinomialNB(var_smoothing=var_smoothing)
#    nb.fit(features, target)
#    return nb

def naive_bayes_default(features, target):
    nb = GaussianNB()
    nb.fit(features, target)
    return nb


def multilayer_perceptron(features, target, alpha, learning_rate_init, power_t, max_iter, momentum, n_iter_no_change):
    mlpn = MLPClassifier(alpha=alpha, learning_rate_init=learning_rate_init, power_t=power_t, max_iter=max_iter,
                         momentum=momentum, n_iter_no_change=n_iter_no_change, solver="sgd")
    mlpn.fit(features, target)
    return mlpn


def multilayer_perceptron_default(features, target):
    mlpn = MLPClassifier()
    mlpn.fit(features, target)
    return mlpn


def data_transfer(X, y, T, l):
    global train_x
    global train_y
    global test_x
    global test_y
    
    train_x = X
    train_y = y
    test_x = T
    test_y = l


def rf_tuning(n_estimators, min_samples_leaf, min_samples_split, max_leaf_nodes, max_features, max_depth):
    rf = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                min_samples_split=min_samples_split, max_leaf_nodes=max_leaf_nodes,
                                max_features=max_features, max_depth=max_depth)
    rf.fit(train_x, train_y)
    predictions = rf.predict(test_x)
    tn, fp, fn, tp = confusion_matrix(test_y, predictions).ravel()
    if tp + fn == 0:
        PD = 0.0
    else:
        PD = tp / (tp + fn)
    if fp + tn == 0:
        PF = 0.0
    else:
        PF = fp / (fp + tn)
    if PD + 1 - PF == 0:
        G_MEASURE = 0.0
    else:
        G_MEASURE = 2 * PD * (1 - PF) / (PD + 1 - PF)
    return G_MEASURE


def nb_tuning(var_smoothing):
    nb = GaussianNB(var_smoothing=var_smoothing)
    nb.fit(train_x, train_y)
    predictions = nb.predict(test_x)
    tn, fp, fn, tp = confusion_matrix(test_y, predictions).ravel()
    if tp + fn == 0:
        PD = 0.0
    else:
        PD = tp / (tp + fn)
    if fp + tn == 0:
        PF = 0.0
    else:
        PF = fp / (fp + tn)
    if PD + 1 - PF == 0:
        G_MEASURE = 0.0
    else:
        G_MEASURE = 2 * PD * (1 - PF) / (PD + 1 - PF)
    return G_MEASURE


def lr_tuning(c, max_iter, verbose):
    lr = LogisticRegression(C=c, max_iter=max_iter, solver="lbfgs")
    lr.fit(train_x, train_y)
    predictions = lr.predict(test_x)
    tn, fp, fn, tp = confusion_matrix(test_y, predictions).ravel()
    if tp + fn == 0:
        PD = 0.0
    else:
        PD = tp / (tp + fn)
    if fp + tn == 0:
        PF = 0.0
    else:
        PF = fp / (fp + tn)
    if PD + 1 - PF == 0:
        G_MEASURE = 0.0
    else:
        G_MEASURE = 2 * PD * (1 - PF) / (PD + 1 - PF)
    return G_MEASURE


def mlpn_tuning(alpha, learning_rate_init, power_t, max_iter, momentum, n_iter_no_change):
    mlpn = MLPClassifier(alpha=alpha, learning_rate_init=learning_rate_init, power_t=power_t, max_iter=max_iter,
                         momentum=momentum, n_iter_no_change=n_iter_no_change, solver="sgd")
    mlpn.fit(train_x, train_y)
    predictions = mlpn.predict(test_x)
    tn, fp, fn, tp = confusion_matrix(test_y, predictions).ravel()
    if tp + fn == 0:
        PD = 0.0
    else:
        PD = tp / (tp + fn)
    if fp + tn == 0:
        PF = 0.0
    else:
        PF = fp / (fp + tn)
    if PD + 1 - PF == 0:
        G_MEASURE = 0.0
    else:
        G_MEASURE = 2 * PD * (1 - PF) / (PD + 1 - PF)
    return G_MEASURE


def knn_tuning(n_neighbors, leaf_size):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, leaf_size=leaf_size)
    knn.fit(train_x, train_y)
    predictions = knn.predict(test_x)
    tn, fp, fn, tp = confusion_matrix(test_y, predictions).ravel()
    if tp + fn == 0:
        PD = 0.0
    else:
        PD = tp / (tp + fn)
    if fp + tn == 0:
        PF = 0.0
    else:
        PF = fp / (fp + tn)
    if PD + 1 - PF == 0:
        G_MEASURE = 0.0
    else:
        G_MEASURE = 2 * PD * (1 - PF) / (PD + 1 - PF)
    return G_MEASURE


def parse_results(result):
    tmp = result[1:-1].rsplit(', ', 1)
    parameters = tmp[0][7:-2].replace('\n', '').replace(' ', '').split(',')
    parameters = [float(i) for i in parameters]
    result = float(tmp[1])
    return parameters, result




def my_smote(data, num, k=5, r=1):
    corpus = []
    if len(data) < k:
        k = len(data) - 1
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', p=r).fit(data)
    distances, indices = nbrs.kneighbors(data)
    for i in range(0, num):
        mid = randint(0, len(data) - 1)
        nn = indices[mid, randint(1, k - 1)]
        datamade = []
        for j in range(0, len(data[mid])):
            gap = random()
            datamade.append((data[nn, j] - data[mid, j]) * gap + data[mid, j])
        corpus.append(datamade)
    corpus = np.array(corpus)
    corpus = np.vstack((corpus, np.array(data)))
    return corpus


def balance(data_train, train_label, m=0, r=0, neighbors=0):
    pos_train = []
    neg_train = []
    for j, i in enumerate(train_label):
        if i == 1:
            pos_train.append(data_train[j])
        else:
            neg_train.append(data_train[j])
    pos_train = np.array(pos_train)
    neg_train = np.array(neg_train)

    if len(pos_train) < len(neg_train):
        pos_train = my_smote(pos_train, m, k=neighbors, r=r)
        if len(neg_train) < m:
            m = len(neg_train)
        neg_train = neg_train[np.random.choice(len(neg_train), m, replace=False)]
    # print(pos_train,neg_train)
    data_train1 = np.vstack((pos_train, neg_train))
    label_train = [1] * len(pos_train) + [0] * len(neg_train)
    return data_train1, label_train


def rf_smotuned(m, r, neighbours):
#    lab = [y for x in train_y.values.tolist() for y in x]
    lab =[]
    for x in train_y:
        lab.append(x)
    train_balanced_x, train_balanced_y = balance(train_x, lab, m=m, r=r, neighbors=neighbours)
    rf = RandomForestClassifier()
    rf.fit(train_balanced_x, train_balanced_y)
    predictions = rf.predict(test_x)
    tn, fp, fn, tp = confusion_matrix(test_y, predictions).ravel()
    if tp + fn == 0:
        PD = 0.0
    else:
        PD = tp / (tp + fn)
    if fp + tn == 0:
        PF = 0.0
    else:
        PF = fp / (fp + tn)
    if PD + 1 - PF == 0:
        G_MEASURE = 0.0
    else:
        G_MEASURE = 2 * PD * (1 - PF) / (PD + 1 - PF)
    return G_MEASURE
def nb_smotuned(m, r, neighbours):
#    lab = [y for x in train_y.values.tolist() for y in x]
    lab =[]
    for x in train_y:
        lab.append(x)
    train_balanced_x, train_balanced_y = balance(train_x.values, lab, m=m, r=r, neighbors=neighbours)
#    print("I'm here:")
    nb = GaussianNB()
    nb.fit(train_balanced_x, train_balanced_y)
    predictions = nb.predict(test_x)
    tn, fp, fn, tp = confusion_matrix(test_y, predictions).ravel()
    if tp + fn == 0:
        PD = 0.0
    else:
        PD = tp / (tp + fn)
    if fp + tn == 0:
        PF = 0.0
    else:
        PF = fp / (fp + tn)
    if PD + 1 - PF == 0:
        G_MEASURE = 0.0
    else:
        G_MEASURE = 2 * PD * (1 - PF) / (PD + 1 - PF)
    return G_MEASURE