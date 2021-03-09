from sklearn.model_selection import StratifiedKFold
import numpy as np
from random import randint, random

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors

from de import de_rf, de_lr, de_nb, de_mlpn, de_knn
from de import de_smote

def data_transfer(X, y, T, l):
    global train_x
    global train_y
    global test_x
    global test_y
    
    train_x = X
    train_y = y
    test_x = T
    test_y = l

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


def k_fold_crossvalidation(dataset, model):
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]

    global train_x
    global train_y
    global test_x
    global test_y

    evaluation_scores = []
    skf = StratifiedKFold(n_splits=5)
    for train_index, test_index in skf.split(X, y):
        # print("Train Index: ", train_index, "\n")
        # print("Test Index: ", test_index)
        train_x, test_x = X.iloc[train_index], X.iloc[test_index]
        train_y, test_y = y.iloc[train_index], y.iloc[test_index]

        if model == "RF":
            de_result = list(de_rf(rf_tuning, bounds=[(10, 150), (1, 20), (2, 20), (2, 50), (0.01, 1), (1, 10)]))
            parameter, result = parse_results(str(de_result[-1]))
            evaluation_scores.append(result)
        elif model == "NB":
            de_result = list(de_nb(nb_tuning, bounds=[(0.0, 1.0)]))
            parameter, result = parse_results(str(de_result[-1]))
            evaluation_scores.append(result)
        elif model == "MLP":
            de_result = list(
                de_mlpn(mlpn_tuning, bounds=[(0.0001, 0.001), (0.001, 0.01), (0.1, 1), (50, 300), (0.1, 1), (10, 100)]))
            parameter, result = parse_results(str(de_result[-1]))
            evaluation_scores.append(result)
        elif model == "LR":
            de_result = list(de_lr(lr_tuning, bounds=[(1, 10), (50, 200), (0, 10)]))
            parameter, result = parse_results(str(de_result[-1]))
            evaluation_scores.append(result)
        elif model == "KNN":
            de_result = list(de_knn(knn_tuning, bounds=[(1, 10), (10, 100)]))
            parameter, result = parse_results(str(de_result[-1]))
            evaluation_scores.append(result)

    return np.mean(evaluation_scores)


def k_fold_crossvalidation_smote(dataset, model):
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1:]

    global train_x
    global train_y
    global test_x
    global test_y

    evaluation_scores = []
    skf = StratifiedKFold(n_splits=10)
    for train_index, test_index in skf.split(X, y):
        train_x, test_x = X.iloc[train_index], X.iloc[test_index]
        train_y, test_y = y.iloc[train_index], y.iloc[test_index]

        lab = [y for x in train_y.values.tolist() for y in x]
        train_balanced_x, train_balanced_y = balance(train_x.values, lab, m=50, r=2, neighbors=5)

        if model == "RF":
            rf = RandomForestClassifier()
            rf.fit(train_balanced_x, train_balanced_y)
            predictions = rf.predict(test_x)
            tn, fp, fn, tp = confusion_matrix(test_y, predictions).ravel()
            PD = tp / (tp + fn)
            PF = fp / (fp + tn)
            G_MEASURE = 2 * PD * (1 - PF) / (PD + 1 - PF)
            evaluation_scores.append(G_MEASURE)
        elif model == "NB":
            nb = GaussianNB()
            nb.fit(train_balanced_x, train_balanced_y)
            predictions = nb.predict(test_x)
            tn, fp, fn, tp = confusion_matrix(test_y, predictions).ravel()
            PD = tp / (tp + fn)
            PF = fp / (fp + tn)
            G_MEASURE = 2 * PD * (1 - PF) / (PD + 1 - PF)
            evaluation_scores.append(G_MEASURE)
        elif model == "MLP":
            mlpn = MLPClassifier()
            mlpn.fit(train_balanced_x, train_balanced_y)
            predictions = mlpn.predict(test_x)
            tn, fp, fn, tp = confusion_matrix(test_y, predictions).ravel()
            PD = tp / (tp + fn)
            PF = fp / (fp + tn)
            G_MEASURE = 2 * PD * (1 - PF) / (PD + 1 - PF)
            evaluation_scores.append(G_MEASURE)
        elif model == "LR":
            lr = LogisticRegression()
            lr.fit(train_balanced_x, train_balanced_y)
            predictions = lr.predict(test_x)
            tn, fp, fn, tp = confusion_matrix(test_y, predictions).ravel()
            PD = tp / (tp + fn)
            PF = fp / (fp + tn)
            G_MEASURE = 2 * PD * (1 - PF) / (PD + 1 - PF)
            evaluation_scores.append(G_MEASURE)
        elif model == "KNN":
            knn = KNeighborsClassifier()
            knn.fit(train_balanced_x, train_balanced_y)
            predictions = knn.predict(test_x)
            tn, fp, fn, tp = confusion_matrix(test_y, predictions).ravel()
            PD = tp / (tp + fn)
            PF = fp / (fp + tn)
            G_MEASURE = 2 * PD * (1 - PF) / (PD + 1 - PF)
            evaluation_scores.append(G_MEASURE)

    # print(evaluation_scores)
    return np.mean(evaluation_scores)


def result_statistics(test_y, predictions):
    tn, fp, fn, tp = confusion_matrix(test_y, predictions).ravel()
    print("TN, FP, FN, TP: ", (tn, fp, fn, tp))

    PD = tp / (tp + fn)
    PF = fp / (fp + tn)
    PREC = tp / (tp + fp)
    F_MEASURE = 2 * PD * PREC / (PD + PREC)
    G_MEASURE = 2 * PD * (1 - PF) / (PD + 1 - PF)
    return tn, fp, fn, tp, PD, PF, PREC, F_MEASURE, G_MEASURE
    

def get_result_statistics(test_y, predictions):
    tn, fp, fn, tp = confusion_matrix(test_y, predictions).ravel()
    PD = tp / (tp + fn)
    PF = fp / (fp + tn)
    PREC = tp / (tp + fp)
    F_MEASURE = 2 * PD * PREC / (PD + PREC)
    G_MEASURE = 2 * PD * (1 - PF) / (PD + 1 - PF)
    return tn, fp, fn, tp, PD, PF, PREC, F_MEASURE, G_MEASURE
#    print("pd: ", PD)
#    print("pf: ", PF)
#    print("prec: ", PREC)
#    print("f-measure: ", F_MEASURE)
#    print("g-measure: ", G_MEASURE)


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


def k_fold_crossvalidation_smotuned(dataset, model):
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1:]

    global train_x
    global train_y
    global test_x
    global test_y

    evaluation_scores = []
    skf = StratifiedKFold(n_splits=5)
    for train_index, test_index in skf.split(X, y):
        train_x, test_x = X.iloc[train_index], X.iloc[test_index]
        train_y, test_y = y.iloc[train_index], y.iloc[test_index]

        if model == "RF":
            de_result = list(de_smote(rf_smotuned, bounds=[(50, 400), (1, 6), (5, 21)]))
            parameter, result = parse_results(str(de_result[-1]))
            evaluation_scores.append(result)
        elif model == "NB":
            de_result = list(de_smote(nb_smotuned, bounds=[(50, 400), (1, 6), (5, 21)]))
            parameter, result = parse_results(str(de_result[-1]))
            evaluation_scores.append(result)
        elif model == "MLP":
            de_result = list(de_smote(mlpn_smotuned, bounds=[(50, 400), (1, 6), (5, 21)]))
            parameter, result = parse_results(str(de_result[-1]))
            evaluation_scores.append(result)
        elif model == "LR":
            de_result = list(de_smote(lr_smotuned, bounds=[(50, 400), (1, 6), (5, 21)]))
            parameter, result = parse_results(str(de_result[-1]))
            evaluation_scores.append(result)
        elif model == "KNN":
            de_result = list(de_smote(knn_smotuned, bounds=[(50, 400), (1, 6), (5, 21)]))
            parameter, result = parse_results(str(de_result[-1]))
            evaluation_scores.append(result)

    return np.mean(evaluation_scores)


def rf_smotuned(m, r, neighbours):
    lab = [y for x in train_y.values.tolist() for y in x]
    train_balanced_x, train_balanced_y = balance(train_x.values, lab, m=m, r=r, neighbors=neighbours)
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
    lab = [y for x in train_y.values.tolist() for y in x]
    train_balanced_x, train_balanced_y = balance(train_x.values, lab, m=m, r=r, neighbors=neighbours)
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


def lr_smotuned(m, r, neighbours):
    lab = [y for x in train_y.values.tolist() for y in x]
    train_balanced_x, train_balanced_y = balance(train_x.values, lab, m=m, r=r, neighbors=neighbours)
    lr = LogisticRegression()
    lr.fit(train_balanced_x, train_balanced_y)
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


def mlpn_smotuned(m, r, neighbours):
    lab = [y for x in train_y.values.tolist() for y in x]
    train_balanced_x, train_balanced_y = balance(train_x.values, lab, m=m, r=r, neighbors=neighbours)
    mlpn = MLPClassifier()
    mlpn.fit(train_balanced_x, train_balanced_y)
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


def knn_smotuned(m, r, neighbours):
    lab = [y for x in train_y.values.tolist() for y in x]
    train_balanced_x, train_balanced_y = balance(train_x.values, lab, m=m, r=r, neighbors=neighbours)
    knn = KNeighborsClassifier()
    knn.fit(train_balanced_x, train_balanced_y)
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
