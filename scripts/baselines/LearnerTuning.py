import pandas as pd
import numpy as np
import time
import warnings

from utilities import parse_results, k_fold_crossvalidation
from de import de_rf, de_lr, de_nb, de_mlpn, de_knn
from utilities import rf_tuning, nb_tuning, mlpn_tuning, lr_tuning, knn_tuning
from utilities import result_statistics
from utilities import random_forest, KNN, logistic_regression, naive_bayes, multilayer_perceptron

warnings.filterwarnings('ignore')

TRAIN_PATH = r"../data/text-frequency/within/ambari-train.csv"
TEST_PATH = r"../input_matrix/revised/ambari-test.csv"


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

    # k-fold cross validation
    # learners = ["RF", "NB", "MLP", "LR", "KNN"]
    learners = ["NB","RF"]
    results = []
    for l in learners:
        print("K CrossValidation of ", l, "...")
        kfold_result = k_fold_crossvalidation(train_dataset, l)
        print(kfold_result)
        results.append(kfold_result)
        print("completed...\n")

    best_learner = learners[results.index(max(results))]
    print("best learner: ", best_learner)
    print("best score: ", max(results))
    print("")

    # train with the whole train-dataset
    global train_x
    global test_x
    global train_y
    global test_y

    train_x = train_dataset.iloc[:, :-1]
    train_y = train_dataset.iloc[:, -1]

    test_x = test_dataset.iloc[:, :-1]
    test_y = test_dataset.iloc[:, -1]

    if best_learner == "RF":
        print("----------Tuning Random Forest with DE----------")
        rf_tuning_start_time = time.time()
        de_rf_result = list(de_rf(rf_tuning, bounds=[(10, 150), (1, 20), (2, 20), (2, 50), (0.01, 1), (1, 10)]))
        print("RF_Tuned: ", de_rf_result[-1])
        parameter, result = parse_results(str(de_rf_result[-1]))
        print("para: ", parameter)
        print("result: ", result)
        rf = random_forest(train_x, train_y, np.int(np.round(parameter[0])), np.int(np.round(parameter[1])),
                           np.int(np.round(parameter[2])),
                           np.int(np.round(parameter[3])), parameter[4], np.int(np.round(parameter[5])))
        rf_predictions = rf.predict(test_x)
        result_statistics(test_y, rf_predictions)
        print("--- Random Forest Tuning Time: %s seconds ---" % (time.time() - rf_tuning_start_time))
        print("")
    elif best_learner == "NB":
        print("----------Tuning Naive Bayes with DE----------")
        nb_tuning_start_time = time.time()
        de_nb_result = list(de_nb(nb_tuning, bounds=[(0.0, 1.0)]))
        print("NB_Tuned: ", de_nb_result[-1])
        parameter, result = parse_results(str(de_nb_result[-1]))
        print("para: ", parameter)
        print("result: ", result)
        nb = naive_bayes(train_x, train_y, parameter[0])
        nb_predictions = nb.predict(test_x)
        result_statistics(test_y, nb_predictions)
        print("--- Naive Bayes Tuning Time: %s seconds ---" % (time.time() - nb_tuning_start_time))
        print("")
    elif best_learner == "MLP":
        print("----------Tuning Multilayer Perceptron with DE----------")
        mlpn_tuning_start_time = time.time()
        de_mlpn_result = list(
            de_mlpn(mlpn_tuning, bounds=[(0.0001, 0.001), (0.001, 0.01), (0.1, 1), (50, 300), (0.1, 1), (10, 100)]))
        print("MP_Tuned: ", de_mlpn_result[-1])
        parameter, result = parse_results(str(de_mlpn_result[-1]))
        print("para: ", parameter)
        print("result: ", result)
        mlpn = multilayer_perceptron(train_x, train_y, parameter[0], parameter[1], parameter[2],
                                     np.int(np.round(parameter[3])),
                                     parameter[4], np.int(np.round(parameter[5])))
        mlpn_predictions = mlpn.predict(test_x)
        result_statistics(test_y, mlpn_predictions)
        print("--- Multilayer Percepptron Tuning Time: %s seconds ---" % (time.time() - mlpn_tuning_start_time))
        print("")
    elif best_learner == "LR":
        print("----------Tuning Logistic Regression with DE----------")
        lr_tuning_start_time = time.time()
        de_lr_result = list(de_lr(lr_tuning, bounds=[(1, 10), (50, 200), (0, 10)]))
        print("LR_Tuned: ", de_lr_result[-1])
        parameter, result = parse_results(str(de_lr_result[-1]))
        print("para: ", parameter)
        print("result: ", result)
        lr = logistic_regression(train_x, train_y, parameter[0], np.int(np.round(parameter[1])), parameter[2])
        lr_predictions = lr.predict(test_x)
        result_statistics(test_y, lr_predictions)
        print("--- Logistic Regression Tuning Time: %s seconds ---" % (time.time() - lr_tuning_start_time))
        print("")
    elif best_learner == "KNN":
        print("----------Tuning KNN with DE----------")
        knn_tuning_start_time = time.time()
        de_knn_result = list(de_knn(knn_tuning, bounds=[(1, 10), (10, 100)]))
        print("KNN_Tuned: ", de_knn_result[-1])
        parameter, result = parse_results(str(de_knn_result[-1]))
        print("para: ", parameter)
        print("result: ", result)
        knn = KNN(train_x, train_y, np.int(np.round(parameter[0])), np.int(np.round(parameter[1])))
        knn_predictions = knn.predict(test_x)
        result_statistics(test_y, knn_predictions)
        print("--- KNN Tuning Time: %s seconds ---" % (time.time() - knn_tuning_start_time))
        print("")


def main():
    prediction(TRAIN_PATH, TEST_PATH)


if __name__ == "__main__":
    main()
