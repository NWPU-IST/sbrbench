import pandas as pd
import numpy as np
import time
import warnings

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from utilities import k_fold_crossvalidation_smote, balance, result_statistics

warnings.filterwarnings('ignore')

TRAIN_PATH = r"../data/text-frequency/within/camel-farsectwo.csv"
TEST_PATH = r"../data/text-frequency/within/camel-test.csv"


def read_data(path):
    data = pd.read_csv(path)
    data = data.drop(['id'], axis=1)
    data = data.sample(frac=1)
    return data


def prediction(train_dataset_path, test_dataset_path):
    train_dataset = read_data(train_dataset_path)
    test_dataset = read_data(test_dataset_path)

    print(train_dataset_path)
    print("-------------------------------------")
    print("")

    # K-fold cross-validation
    learners = ["RF", "NB", "MLP", "LR", "KNN"]
    results = []
    for l in learners:
        print("K CrossValidation of ", l, "...")
        kfold_result = k_fold_crossvalidation_smote(train_dataset, l)
        print(kfold_result)
        results.append(kfold_result)
        print("completed...\n")

    best_learner = learners[results.index(max(results))]
    print("best learner: ", best_learner)
    print("best score: ", max(results))
    print("")

    global train_x
    global test_x
    global train_y
    global test_y

    train_x = train_dataset.iloc[:, :-1]
    train_y = train_dataset.iloc[:, -1:]

    test_x = test_dataset.iloc[:, :-1]
    test_y = test_dataset.iloc[:, -1:]

    # use smote on the whole train-dataset
    lab = [y for x in train_y.values.tolist() for y in x]
    train_balanced_x, train_balanced_y = balance(train_x.values, lab, m=50, r=2, neighbors=5)

    if best_learner == "RF":
        rf = RandomForestClassifier()
        rf.fit(train_balanced_x, train_balanced_y)
        predictions = rf.predict(test_x)
        result_statistics(test_y, predictions)
    elif best_learner == "NB":
        nb = GaussianNB()
        nb.fit(train_balanced_x, train_balanced_y)
        predictions = nb.predict(test_x)
        result_statistics(test_y, predictions)
    elif best_learner == "MLP":
        mlpn = MLPClassifier()
        mlpn.fit(train_balanced_x, train_balanced_y)
        predictions = mlpn.predict(test_x)
        result_statistics(test_y, predictions)
    elif best_learner == "LR":
        lr = LogisticRegression()
        lr.fit(train_balanced_x, train_balanced_y)
        predictions = lr.predict(test_x)
        result_statistics(test_y, predictions)
    elif best_learner == "KNN":
        knn = KNeighborsClassifier()
        knn.fit(train_balanced_x, train_balanced_y)
        predictions = knn.predict(test_x)
        result_statistics(test_y, predictions)


def main():
    prediction(TRAIN_PATH, TEST_PATH)


if __name__ == "__main__":
    main()
