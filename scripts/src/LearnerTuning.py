import pandas as pd
import numpy as np
import time
import warnings

from utilities import parse_results, data_transfer
from de import de_rf, de_lr, de_nb, de_mlpn, de_knn
from utilities import rf_tuning, nb_tuning, mlpn_tuning, lr_tuning, knn_tuning
from utilities import result_statistics
from utilities import random_forest, KNN, logistic_regression, naive_bayes, multilayer_perceptron
from sklearn.feature_extraction.text import CountVectorizer
import dimension_reduce as dr
import model_measure_functions as mf

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.utils import shuffle


warnings.filterwarnings('ignore')


def read_data(path):
    data = pd.read_csv(path)
    data = data.drop(['id'], axis=1)
    data = data.sample(frac=1)
    return data


def prediction(dataset_path):
    all_dataset = read_data(dataset_path)
    num = len(all_dataset)
    train_dataset = all_dataset[:int(0.5*num)]
    test_dataset = all_dataset[int(0.5*num):]

    print(dataset_path)
    learners = ["NB"] # ,"MLP","LR","KNN"

    global train_x
    global test_x
    global train_y
    global test_y

    train_content = train_dataset.description
    train_label = train_dataset.security
    test_content = test_dataset.description
    test_label = test_dataset.security    
    
    vectorizer = CountVectorizer(stop_words='english')
    train_content_matrix = vectorizer.fit_transform(train_content)
    test_content_matrix = vectorizer.transform(test_content)
    train_content_matrix_dr, test_content_matrix_dr  = dr.selectFromLinearSVC2(train_content_matrix, train_label, test_content_matrix)   

    
    train_x = train_content_matrix_dr.toarray()
    train_y = train_label
    test_x = test_content_matrix_dr.toarray()
    test_y = test_label.tolist()
    data_transfer(train_x, train_y, test_x, test_y)
    
    for l in learners:
        if l == "RF":
            print("===============RF without turning===============")
            rf_train_start_time = time.time()
            clf = RandomForestClassifier(oob_score=True, n_estimators=30)
            clf.fit(train_x, train_y)
            predicted = clf.predict(test_x)
            TP, FN, TN, FP, pd, prec, f_measure,success_rate = mf.model_measure_mop(predicted, test_y)
            print("TN, FP, FN, TP: (", TN, FP, FN, TP, ")")
            print("pd: ",pd)
            print("prec: ",prec)
            print("f-measure: ",f_measure)
            print("accuracy: ", success_rate)
            print("--- Random Forest Training Time: %s seconds ---" % (time.time() - rf_train_start_time))  

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
        elif l == "NB":
            print("===============NB without turning===================")
            nb_train_start_time = time.time()
            clf = MultinomialNB()
            clf.fit(train_x, train_y)
            predicted = clf.predict(test_x)
            TP, FN, TN, FP, pd, prec, f_measure,success_rate = mf.model_measure_mop(predicted, test_y)
            print("TN, FP, FN, TP: (", TN, FP, FN, TP, ")")
            print("pd: ",pd)
            print("prec: ",prec)
            print("f-measure: ",f_measure)
            print("accuracy: ", success_rate)
            print("--- Naive Bayes Training Time: %s seconds ---" % (time.time() - nb_train_start_time))  
            
            print("----------Tuning Naive Bayes with DE----------")
            nb_tuning_start_time = time.time()
    #        data_transfer(train_x, train_y, test_x, test_y)
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
        elif l == "MLP":
            print("===============MLP without turning===============")
            mlp_train_start_time = time.time()
            clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                                hidden_layer_sizes=(5, 2), random_state=1)
            clf.fit(train_x, train_y)
            predicted = clf.predict(test_x)
            TP, FN, TN, FP, pd, prec, f_measure,success_rate = mf.model_measure_mop(predicted, test_y)
            print("TN, FP, FN, TP: (", TN, FP, FN, TP, ")")
            print("pd: ",pd)
            print("prec: ",prec)
            print("f-measure: ",f_measure)
            print("accuracy: ", success_rate)           
            print("--- MLP Training Time: %s seconds ---" % (time.time() - mlp_train_start_time))  
        
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
        elif l == "LR":
            print("===============LR without turning===============")
            lr_train_start_time = time.time()
            clf = LogisticRegression()
            clf.fit(train_x, train_y)
            predicted = clf.predict(test_x)
            TP, FN, TN, FP, pd, prec, f_measure,success_rate = mf.model_measure_mop(predicted, test_y)
            print("TN, FP, FN, TP: (", TN, FP, FN, TP, ")")
            print("pd: ",pd)
            print("prec: ",prec)
            print("f-measure: ",f_measure)
            print("accuracy: ", success_rate)
            print("--- LR Training Time: %s seconds ---" % (time.time() - lr_train_start_time))  
            
            print("----------Tuning Logistic Regression with DE----------")
            lr_tuning_start_time = time.time()
    #        data_transfer(train_x, train_y, test_x, test_y)
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
        elif l == "KNN":
            print("===============KNN without turning===============")
            knn_train_start_time = time.time()
            clf = LogisticRegression()
            clf.fit(train_x, train_y)
            predicted = clf.predict(test_x)
            TP, FN, TN, FP, pd, prec, f_measure,success_rate = mf.model_measure_mop(predicted, test_y)
            print("TN, FP, FN, TP: (", TN, FP, FN, TP, ")")
            print("pd: ",pd)
            print("prec: ",prec)
            print("f-measure: ",f_measure)
            print("accuracy: ", success_rate)
            print("--- KNN Training Time: %s seconds ---" % (time.time() - knn_train_start_time))  
            
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
    dataname = "Wicket"
#    DATA_PATH_NOISE = r"../input/original/" +dataname + ".csv"
    DATA_PATH_CLEAN = r"../input/clean/" +dataname + ".csv"
    prediction(DATA_PATH_CLEAN)


if __name__ == "__main__":
    main()
