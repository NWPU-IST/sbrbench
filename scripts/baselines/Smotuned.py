import pandas as pd
import numpy as np
import time
import warnings

from de import de_smote
from utilities import k_fold_crossvalidation_smotuned, data_transfer
from utilities import rf_smotuned, nb_smotuned, lr_smotuned, mlpn_smotuned, knn_smotuned
from utilities import parse_results, result_statistics, balance
from utilities import random_forest_default, naive_bayes_default, logistic_regression_default, KNN_default, multilayer_perceptron_default

warnings.filterwarnings('ignore')

#TRAIN_PATH = r"../data/text-frequency/within/camel-farsectwo.csv"
#TEST_PATH = r"../data/text-frequency/within/camel-test.csv"


def read_data(path):
    data = pd.read_csv(path)
    data = data.drop(['id'], axis=1)
    data = data.sample(frac=1)
    return data


def prediciton_with_smounted(train_dataset_path, test_dataset_path):
    train_dataset = read_data(train_dataset_path)
    test_dataset = read_data(test_dataset_path)
#    print(train_dataset_path)
#    print("-------------------------------------")
#    print("")

    # K-fold cross-validation
#    learners = ["RF", "NB", "MLP", "LR", "KNN"]
#    results = []
#    for l in learners:
#        print("K CrossValidation of ", l, "...")
#        kfold_result = k_fold_crossvalidation_smotuned(train_dataset, l)
#        print(kfold_result)
#        results.append(kfold_result)
#        print("completed...\n")
#
#    best_learner = learners[results.index(max(results))]
#    print("best learner: ", best_learner)
#    print("best score: ", max(results))
#    print("")

    # train with the whole train-dataset
    global train_x
    global test_x
    global train_y
    global test_y

    train_x = train_dataset.iloc[:, :-1]
    train_y = train_dataset.iloc[:, -1:]

    test_x = test_dataset.iloc[:, :-1]
    test_y = test_dataset.iloc[:, -1:]

#    best_learner = "RF"
    data_transfer(train_x, train_y, test_x, test_y)
#    if best_learner == "RF":
    print("----------Tuning Random Forest with SMOTUNED----------")
    rf_tuning_start_time = time.time()
    de_rf_result = list(de_smote(rf_smotuned, bounds=[(50, 400), (1, 6), (5, 21)]))
    print("RF_Tuned: ", de_rf_result[-1])
    parameter, result = parse_results(str(de_rf_result[-1]))
    print("para: ", parameter)
    print("result: ", result)
    lab = [y for x in train_y.values.tolist() for y in x]
    train_balanced_tuned_x, train_balanced_tuned_y = balance(train_x.values, lab, m=np.int(np.round(parameter[0])),
                                                             r=np.int(np.round(parameter[1])),
                                                             neighbors=np.int(np.round(parameter[2])))
    rf = random_forest_default(train_balanced_tuned_x, train_balanced_tuned_y)
    rf_predictions = rf.predict(test_x)
    TN, FP, FN, TP, PD, PF, PREC, F_MEASURE, G_MEASURE = result_statistics(test_y, rf_predictions)
    Cost = time.time() - rf_tuning_start_time
    return TN, FP, FN, TP, PD, PF, PREC, F_MEASURE, G_MEASURE, Cost
#        print("--- Random Forest Tuning Time: %s seconds ---" % (time.time() - rf_tuning_start_time))
#        print("")
#    elif best_learner == "NB":
#        print("----------Tuning Naive Bayes with SMOTUNED----------")
#        nb_tuning_start_time = time.time()
#        de_nb_result = list(de_smote(nb_smotuned, bounds=[(50, 400), (1, 6), (5, 21)]))
#        print("NB_Tuned: ", de_nb_result[-1])
#        parameter, result = parse_results(str(de_nb_result[-1]))
#        print("para: ", parameter)
#        print("result: ", result)
#        lab = [y for x in train_y.values.tolist() for y in x]
#        train_balanced_tuned_x, train_balanced_tuned_y = balance(train_x.values, lab, m=np.int(np.round(parameter[0])),
#                                                                 r=np.int(np.round(parameter[1])),
#                                                                 neighbors=np.int(np.round(parameter[2])))
#        nb = naive_bayes_default(train_balanced_tuned_x, train_balanced_tuned_y)
#        nb_predictions = nb.predict(test_x)
#        result_statistics(test_y, nb_predictions)
#        print("--- Naive Bayes Tuning Time: %s seconds ---" % (time.time() - nb_tuning_start_time))
#        print("")
#    elif best_learner == "MLP":
#        print("----------Tuning Multilayer Perceptron with SMOTUNED----------")
#        mlpn_tuning_start_time = time.time()
#        de_mlpn_result = list(de_smote(mlpn_smotuned, bounds=[(50, 400), (1, 6), (5, 21)]))
#        print("MP_Tuned: ", de_mlpn_result[-1])
#        parameter, result = parse_results(str(de_mlpn_result[-1]))
#        print("para: ", parameter)
#        print("result: ", result)
#        lab = [y for x in train_y.values.tolist() for y in x]
#        train_balanced_tuned_x, train_balanced_tuned_y = balance(train_x.values, lab, m=np.int(np.round(parameter[0])),
#                                                                 r=np.int(np.round(parameter[1])),
#                                                                 neighbors=np.int(np.round(parameter[2])))
#        mlpn = multilayer_perceptron_default(train_balanced_tuned_x, train_balanced_tuned_y)
#        mlpn_predictions = mlpn.predict(test_x)
#        result_statistics(test_y, mlpn_predictions)
#        print("--- Multilayer Percepptron Tuning Time: %s seconds ---" % (time.time() - mlpn_tuning_start_time))
#        print("")
#    elif best_learner == "LR":
#        print("----------Tuning Logistic Regression with SMOTUNED----------")
#        lr_tuning_start_time = time.time()
#        de_lr_result = list(de_smote(lr_smotuned, bounds=[(50, 400), (1, 6), (5, 21)]))
#        print("LR_Tuned: ", de_lr_result[-1])
#        parameter, result = parse_results(str(de_lr_result[-1]))
#        print("para: ", parameter)
#        print("result: ", result)
#        lab = [y for x in train_y.values.tolist() for y in x]
#        train_balanced_tuned_x, train_balanced_tuned_y = balance(train_x.values, lab, m=np.int(np.round(parameter[0])),
#                                                                 r=np.int(np.round(parameter[1])),
#                                                                 neighbors=np.int(np.round(parameter[2])))
#        lr = logistic_regression_default(train_balanced_tuned_x, train_balanced_tuned_y)
#        lr_predictions = lr.predict(test_x)
#        result_statistics(test_y, lr_predictions)
#        print("--- Logistic Regression Tuning Time: %s seconds ---" % (time.time() - lr_tuning_start_time))
#        print("")
#    elif best_learner == "KNN":
#        print("----------Tuning KNN with SMOTUNED----------")
#        knn_tuning_start_time = time.time()
#        de_knn_result = list(de_smote(knn_smotuned, bounds=[(50, 400), (1, 6), (5, 21)]))
#        print("KNN_Tuned: ", de_knn_result[-1])
#        parameter, result = parse_results(str(de_knn_result[-1]))
#        print("para: ", parameter)
#        print("result: ", result)
#        lab = [y for x in train_y.values.tolist() for y in x]
#        train_balanced_tuned_x, train_balanced_tuned_y = balance(train_x.values, lab, m=np.int(np.round(parameter[0])),
#                                                                 r=np.int(np.round(parameter[1])),
#                                                                 neighbors=np.int(np.round(parameter[2])))
#        knn = KNN_default(train_balanced_tuned_x, train_balanced_tuned_y)
#        knn_predictions = knn.predict(test_x)
#        result_statistics(test_y, knn_predictions)
#        print("--- KNN Tuning Time: %s seconds ---" % (time.time() - knn_tuning_start_time))
#        print("")


#def main():
#    prediciton_with_smounted(TRAIN_PATH, TEST_PATH)
#
#
#if __name__ == "__main__":
#    main()
