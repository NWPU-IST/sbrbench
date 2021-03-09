import csv
import pandas as pd
import numpy as np
import time
import warnings

from utilities import parse_results, data_transfer
from de import de_rf, de_lr, de_nb, de_mlpn, de_knn, de_smote
from utilities import rf_tuning, nb_tuning, mlpn_tuning, lr_tuning, knn_tuning
from utilities import result_statistics, get_result_statistics,balance
from utilities import random_forest, KNN, logistic_regression, naive_bayes, multilayer_perceptron

from utilities import rf_smotuned, nb_smotuned
from utilities import random_forest_default, naive_bayes_default, logistic_regression_default, KNN_default, multilayer_perceptron_default


from sklearn.feature_extraction.text import CountVectorizer
import dimension_reduce as dr

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


def prediction(dataset_path_n, dataset_path_c, writer):
    
#    all_dataset = read_data(dataset_path)
    dataset_n = pd.read_csv(dataset_path_n)
    dataset_c = pd.read_csv(dataset_path_c)
    num_n = len(dataset_n)
    num_c = len(dataset_c)
    if(num_n ==num_c):
        train_dataset = dataset_n[:int(0.5*num_n)]
        test_dataset = dataset_c[int(0.5*num_n):]

#    print(dataset_path)
    learners = ["RF","NB","MLP","LR","KNN"] # ,"MLP","LR","KNN"

    global train_x
    global test_x
    global train_y
    global test_y

    train_content = train_dataset.description
    train_label = train_dataset.security.tolist()
#    train_label = train_dataset.iloc[:, -2:-1] 

    test_content = test_dataset.description
    test_label = test_dataset.security.tolist()
    
    vectorizer = CountVectorizer(stop_words='english')
    train_content_matrix = vectorizer.fit_transform(train_content)
    test_content_matrix = vectorizer.transform(test_content)
    train_content_matrix_dr, test_content_matrix_dr  = dr.selectFromLinearSVC2(train_content_matrix, train_label, test_content_matrix)   

    
    train_x = train_content_matrix_dr.toarray()
    train_y = train_label
    test_x = test_content_matrix_dr.toarray()
    test_y = test_label
    data_transfer(train_x, train_y, test_x, test_y)
    
    for l in learners:
        if l == "RF":
#            print("===============RF without turning===============")
            rf_train_start_time = time.time()
            clf = RandomForestClassifier(oob_score=True, n_estimators=30)
            clf.fit(train_x, train_y)
            predicted = clf.predict(test_x)
            tn, fp, fn, tp, PD, PF, PREC, F_MEASURE, G_MEASURE = get_result_statistics(test_y, predicted)
            Cost = time.time() - rf_train_start_time
            writer.writerow([dataset_path_n,'', l, tn, fp, fn, tp, PD, PF, PREC, F_MEASURE, G_MEASURE, Cost])
            print(dataset_path_n,'', l, tn, fp, fn, tp, PD, PF, PREC, F_MEASURE, G_MEASURE, Cost)
            print("")
            
#            print("----------Tuning Random Forest with DE----------")
#            rf_tuning_start_time = time.time()    
#            de_rf_result = list(de_rf(rf_tuning, bounds=[(10, 150), (1, 20), (2, 20), (2, 50), (0.01, 1), (1, 10)]))
#            parameter, result = parse_results(str(de_rf_result[-1]))
#            rf = random_forest(train_x, train_y, np.int(np.round(parameter[0])), np.int(np.round(parameter[1])),
#                               np.int(np.round(parameter[2])),
#                               np.int(np.round(parameter[3])), parameter[4], np.int(np.round(parameter[5])))
#            rf_predictions = rf.predict(test_x)
#            tn, fp, fn, tp, PD, PF, PREC, F_MEASURE, G_MEASURE = get_result_statistics(test_y, rf_predictions)
#            Cost = time.time() - rf_tuning_start_time
#            writer.writerow([dataset_path,'', 'RF_Tuned', tn, fp, fn, tp, PD, PF, PREC, F_MEASURE, G_MEASURE, Cost])
#            print(dataset_path,'', 'RF_Tuned', tn, fp, fn, tp, PD, PF, PREC, F_MEASURE, G_MEASURE, Cost)
#            print("")
#            
#            print("----------Tuning Random Forest with SMOTUNED----------")
#            rfs_tuning_start_time = time.time()
#            de_rf_result = list(de_smote(rf_smotuned, bounds=[(50, 400), (1, 6), (5, 21)]))
#            parameter, result = parse_results(str(de_rf_result[-1]))
#            lab =[]
#            for x in train_y:
#                lab.append(x)           
##            lab = [y for x in train_y.values.tolist() for y in x]
#            train_balanced_tuned_x, train_balanced_tuned_y = balance(train_x, lab, m=np.int(np.round(parameter[0])),
#                                                                     r=np.int(np.round(parameter[1])),
#                                                                     neighbors=np.int(np.round(parameter[2])))
#            rf = random_forest_default(train_balanced_tuned_x, train_balanced_tuned_y)
#            rf_predictions = rf.predict(test_x)        
#            
#            tn, fp, fn, tp, PD, PF, PREC, F_MEASURE, G_MEASURE = get_result_statistics(test_y, rf_predictions)
#            Cost = time.time() - rfs_tuning_start_time
#            writer.writerow([dataset_path,'', 'RF_Smounted', tn, fp, fn, tp, PD, PF, PREC, F_MEASURE, G_MEASURE, Cost])
#            print(dataset_path,'', 'RF_Smounted', tn, fp, fn, tp, PD, PF, PREC, F_MEASURE, G_MEASURE, Cost)
#            
        elif l == "NB":
            print("===============NB without turning===================")
            nb_train_start_time = time.time()
            clf = MultinomialNB()
            clf.fit(train_x, train_y)
            predicted = clf.predict(test_x)
            tn, fp, fn, tp, PD, PF, PREC, F_MEASURE, G_MEASURE = get_result_statistics(test_y, predicted)
            Cost = time.time() - nb_train_start_time
            writer.writerow([dataset_path_n,'', l, tn, fp, fn, tp, PD, PF, PREC, F_MEASURE, G_MEASURE, Cost])
            print(dataset_path_n,'', l, tn, fp, fn, tp, PD, PF, PREC, F_MEASURE, G_MEASURE, Cost)
            print("")
#            
##            print("----------Tuning Naive Bayes with DE----------")
#            nb_tuning_start_time = time.time()
#            de_nb_result = list(de_nb(nb_tuning, bounds=[(0.0, 1.0)]))
#            parameter, result = parse_results(str(de_nb_result[-1]))
#            nb = naive_bayes(train_x, train_y, parameter[0])
#            nb_predictions = nb.predict(test_x)
#            tn, fp, fn, tp, PD, PF, PREC, F_MEASURE, G_MEASURE = get_result_statistics(test_y, nb_predictions)
#            Cost = time.time() - nb_tuning_start_time
#            writer.writerow([dataset_path,'', 'NB_Tuned', tn, fp, fn, tp, PD, PF, PREC, F_MEASURE, G_MEASURE, Cost])
#            print(dataset_path,'', 'NB_Tuned', tn, fp, fn, tp, PD, PF, PREC, F_MEASURE, G_MEASURE, Cost)
#            print("")
#            
#            #print("----------Tuning Naive Bayes with SMOTUNED----------")
#            nbs_tuning_start_time = time.time()
#            de_nbs_result = list(de_smote(nb_smotuned, bounds=[(50, 400), (1, 6), (5, 21)]))
#            parameter, result = parse_results(str(de_nbs_result[-1]))
#            lab =[]
#            for x in train_y:
#                lab.append(x)
##            lab = [y for x in train_y.values.tolist() for y in x]
#            train_balanced_tuned_x, train_balanced_tuned_y = balance(train_x, lab, m=np.int(np.round(parameter[0])),
#                                                                     r=np.int(np.round(parameter[1])),
#                                                                     neighbors=np.int(np.round(parameter[2])))
#            nb = naive_bayes_default(train_balanced_tuned_x, train_balanced_tuned_y)
#            nbs_predictions = nb.predict(test_x)
#            tn, fp, fn, tp, PD, PF, PREC, F_MEASURE, G_MEASURE = get_result_statistics(test_y, nbs_predictions)
#            Cost = time.time() - nbs_tuning_start_time
#            writer.writerow([dataset_path,'', 'NB_Smounted', tn, fp, fn, tp, PD, PF, PREC, F_MEASURE, G_MEASURE, Cost])
#            print(dataset_path,'', 'NB_Smounted', tn, fp, fn, tp, PD, PF, PREC, F_MEASURE, G_MEASURE, Cost)

        elif l == "MLP":
#            print("===============MLP without turning===============")
            mlp_train_start_time = time.time()
            clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                                hidden_layer_sizes=(5, 2), random_state=1)
            clf.fit(train_x, train_y)
            predicted = clf.predict(test_x)
            tn, fp, fn, tp, PD, PF, PREC, F_MEASURE, G_MEASURE = get_result_statistics(test_y, predicted)
            Cost = time.time() - mlp_train_start_time
            writer.writerow([dataset_path_n,'', l, tn, fp, fn, tp, PD, PF, PREC, F_MEASURE, G_MEASURE, Cost])
            print(dataset_path_n,'', l, tn, fp, fn, tp, PD, PF, PREC, F_MEASURE, G_MEASURE, Cost)
            print("")
        
#            print("----------Tuning Multilayer Perceptron with DE----------")
#            mlpn_tuning_start_time = time.time()
#            de_mlpn_result = list(
#                de_mlpn(mlpn_tuning, bounds=[(0.0001, 0.001), (0.001, 0.01), (0.1, 1), (50, 300), (0.1, 1), (10, 100)]))
#            print("MP_Tuned: ", de_mlpn_result[-1])
#            parameter, result = parse_results(str(de_mlpn_result[-1]))
#            print("para: ", parameter)
#            print("result: ", result)
#            mlpn = multilayer_perceptron(train_x, train_y, parameter[0], parameter[1], parameter[2],
#                                         np.int(np.round(parameter[3])),
#                                         parameter[4], np.int(np.round(parameter[5])))
#            mlpn_predictions = mlpn.predict(test_x)
#            result_statistics(test_y, mlpn_predictions)
#            print("--- Multilayer Percepptron Tuning Time: %s seconds ---" % (time.time() - mlpn_tuning_start_time))
#            print("")
        elif l == "LR":
#            print("===============LR without turning===============")
            lr_train_start_time = time.time()
            clf = LogisticRegression()
            clf.fit(train_x, train_y)
            predicted = clf.predict(test_x)
            tn, fp, fn, tp, PD, PF, PREC, F_MEASURE, G_MEASURE = get_result_statistics(test_y, predicted)
            Cost = time.time() - lr_train_start_time
            writer.writerow([dataset_path_n,'', l, tn, fp, fn, tp, PD, PF, PREC, F_MEASURE, G_MEASURE, Cost])
            print(dataset_path_n,'', l, tn, fp, fn, tp, PD, PF, PREC, F_MEASURE, G_MEASURE, Cost)
            print("")
            
#            print("----------Tuning Logistic Regression with DE----------")
#            lr_tuning_start_time = time.time()
#    #        data_transfer(train_x, train_y, test_x, test_y)
#            de_lr_result = list(de_lr(lr_tuning, bounds=[(1, 10), (50, 200), (0, 10)]))
#            print("LR_Tuned: ", de_lr_result[-1])
#            parameter, result = parse_results(str(de_lr_result[-1]))
#            print("para: ", parameter)
#            print("result: ", result)
#            lr = logistic_regression(train_x, train_y, parameter[0], np.int(np.round(parameter[1])), parameter[2])
#            lr_predictions = lr.predict(test_x)
#            result_statistics(test_y, lr_predictions)
#            print("--- Logistic Regression Tuning Time: %s seconds ---" % (time.time() - lr_tuning_start_time))
#            print("")
        elif l == "KNN":
#            print("===============KNN without turning===============")
            knn_train_start_time = time.time()
            clf = LogisticRegression()
            clf.fit(train_x, train_y)
            predicted = clf.predict(test_x)
            tn, fp, fn, tp, PD, PF, PREC, F_MEASURE, G_MEASURE = get_result_statistics(test_y, predicted)
            Cost = time.time() - knn_train_start_time
            writer.writerow([dataset_path_n,'', l, tn, fp, fn, tp, PD, PF, PREC, F_MEASURE, G_MEASURE, Cost])
            print(dataset_path_n,'', l, tn, fp, fn, tp, PD, PF, PREC, F_MEASURE, G_MEASURE, Cost)
            print("")
#            print("----------Tuning KNN with DE----------")
#            knn_tuning_start_time = time.time()
#            de_knn_result = list(de_knn(knn_tuning, bounds=[(1, 10), (10, 100)]))
#            print("KNN_Tuned: ", de_knn_result[-1])
#            parameter, result = parse_results(str(de_knn_result[-1]))
#            print("para: ", parameter)
#            print("result: ", result)
#            knn = KNN(train_x, train_y, np.int(np.round(parameter[0])), np.int(np.round(parameter[1])))
#            knn_predictions = knn.predict(test_x)
#            result_statistics(test_y, knn_predictions)
#            print("--- KNN Tuning Time: %s seconds ---" % (time.time() - knn_tuning_start_time))
#            print("")


def main():
    output = "../output_noise/predict_with_text_5_lner_simple_output.csv"     
    csv_file = open(output, "w", newline='')
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(['Dataname','Version', 'Approach', 'TN', 'FP', 'FN', 'TP', 'pd', 'pf', 'prec', 'fmeasure', 'Gmeasure','Cost']) 

    datanames = ["chromium","ambari", "camel", "derby", "wicket"] #"ambari", "camel", "derby", "wicket", "chromium"
    for dataname in datanames:
        print("Start data: ", dataname)
        DATA_PATH_NOISE = r"../input/noise/" +dataname + ".csv"
        DATA_PATH_CLEAN = r"../input/clean/" +dataname + ".csv"
#        prediction(DATA_PATH_NOISE, writer)
#        print("")
        prediction(DATA_PATH_NOISE, DATA_PATH_CLEAN, writer)
        print("")
    csv_file.close()
    print(output + '**************** finished************************')

if __name__ == "__main__":
    main()
