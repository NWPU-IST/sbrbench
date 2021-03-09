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
import model_measure_functions as mf

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.utils import shuffle

from Smotuned import prediciton_with_smounted


warnings.filterwarnings('ignore')


def read_data(path):
    data = pd.read_csv(path)
    data = data.drop(['id'], axis=1)
    data = data.sample(frac=1)
    return data


def prediction(train_path, test_path, writer):
    
    train_dataset = read_data(train_path)
    test_dataset = read_data(test_path)

    global train_x
    global test_x
    global train_y
    global test_y

#    print(dataset_path)
    learners = ["RF"] # ,"MLP","LR","KNN"


    train_x = train_dataset.iloc[:, :-1]
    train_y = train_dataset.iloc[:, -1]

    test_x = test_dataset.iloc[:, :-1]
    test_y = test_dataset.iloc[:, -1]    
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
            writer.writerow([train_path,'', l, tn, fp, fn, tp, PD, PF, PREC, F_MEASURE, G_MEASURE, Cost])
            print(train_path,'Farsec', l, tn, fp, fn, tp, PD, PF, PREC, F_MEASURE, G_MEASURE, Cost)
            print("")


            print("----------Tuning Random Forest with DE----------")
            rf_tuning_start_time = time.time()    
            de_rf_result = list(de_rf(rf_tuning, bounds=[(10, 150), (1, 20), (2, 20), (2, 50), (0.01, 1), (1, 10)]))
            print("RF_Tuned: ", de_rf_result[-1])
            parameter, result = parse_results(str(de_rf_result[-1]))
#            print("para: ", parameter)
#            print("result: ", result)
            rf = random_forest(train_x, train_y, np.int(np.round(parameter[0])), np.int(np.round(parameter[1])),
                               np.int(np.round(parameter[2])),
                               np.int(np.round(parameter[3])), parameter[4], np.int(np.round(parameter[5])))
            rf_predictions = rf.predict(test_x)
#            result_statistics(test_y, rf_predictions)
            tn, fp, fn, tp, PD, PF, PREC, F_MEASURE, G_MEASURE = get_result_statistics(test_y, rf_predictions)
            Cost = time.time() - rf_tuning_start_time
            
            writer.writerow([train_path,'Farsec_l_Tuned', l, tn, fp, fn, tp, PD, PF, PREC, F_MEASURE, G_MEASURE, Cost])
            print(train_path,'', l, tn, fp, fn, tp, PD, PF, PREC, F_MEASURE, G_MEASURE, Cost)
#            print("--- Random Forest Tuning Time: %s seconds ---" % (time.time() - rf_tuning_start_time))
            print("")

#            print("----------Tuning SMOTE with DE----------")
#            smt_tuning_start_time = time.time()   
#
#            tn, fp, fn, tp, PD, PF, PREC, F_MEASURE, G_MEASURE = prediciton_with_smounted("RF",train_path, test_path)
#            Cost = time.time() - smt_tuning_start_time
#            
#            writer.writerow([train_path,'Farsec_smt_Tuned', l, tn, fp, fn, tp, PD, PF, PREC, F_MEASURE, G_MEASURE, Cost])
#            print(train_path,'', l, tn, fp, fn, tp, PD, PF, PREC, F_MEASURE, G_MEASURE, Cost)
##            print("--- Random Forest Tuning Time: %s seconds ---" % (time.time() - rf_tuning_start_time))
#            print("")


def main():
    output = "../output_noise/noise_matrix_output.csv"     
    csv_file = open(output, "w", newline='')
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(['Dataname','Version', 'Approach', 'TN', 'FP', 'FN', 'TP', 'pd', 'pf', 'prec', 'fmeasure', 'Gmeasure','Cost']) 

    datanames = ["ambari","camel", "derby", "wicket", "chromium"] #"ambari", "camel", "derby", "wicket", "chromium"
    for dataname in datanames:
        print("Start data: ", dataname)
        TRAIN_n = r"../input/clean_matrix/" +dataname + "-train.csv"
        TRAIN_f = r"../input/matrix/farsec/" +dataname + "-farsectwo.csv"
#        TRAIN_c = r"../input/clean_matrix/" +dataname + "-train.csv"
        
#        TEST_o = r"../input/clean_matrix/" +dataname + "-test.csv"
        TEST_c = r"../input/matrix/clean/" +dataname + "-test.csv"
        
#        prediction(TRAIN_o,TEST_o, writer)
#        print("")
        prediction(TRAIN_f,TEST_c, writer)
        print("")
#        prediction(TRAIN_CLEAN, TEST_CLEAN, writer)
    csv_file.close()
    print(output + '**************** finished************************')

if __name__ == "__main__":
    main()
