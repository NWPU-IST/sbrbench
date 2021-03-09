import pandas as pd
import numpy as np
import time
import warnings
import csv

from de import de_rf, de_lr, de_nb, de_mlpn, de_knn
from utilities import rf_tuning, nb_tuning, mlpn_tuning, lr_tuning, knn_tuning
from utilities import random_forest, KNN, logistic_regression, naive_bayes, multilayer_perceptron

from de import de_smote
from utilities import rf_smotuned, nb_smotuned
from utilities import parse_results, result_statistics, balance,data_transfer, get_result_statistics
from utilities import random_forest_default, naive_bayes_default, logistic_regression_default, KNN_default, multilayer_perceptron_default
from Smotuned import prediciton_with_smounted


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB

warnings.filterwarnings('ignore')

def read_data(path):
    data = pd.read_csv(path)
    data = data.drop(['id'], axis=1)
    data = data.sample(frac=1)
    return data

def get_data(train_dataset_path, farsec_dataset_path, test_dataset_path):
    train_dataset = read_data(train_dataset_path)
    farsec_dataset = read_data(farsec_dataset_path)
    test_dataset = read_data(test_dataset_path)

    print(train_dataset_path)
    
    # train with the whole train-dataset
    global train_x
    global farsec_x
    global test_x
    global train_y
    global farsec_y
    global test_y

    train_x = train_dataset.iloc[:, :-1]
    train_y = train_dataset.iloc[:, -1]

    farsec_x = farsec_dataset.iloc[:, :-1]
    farsec_y = farsec_dataset.iloc[:, -1]
    
    test_x = test_dataset.iloc[:, :-1]
    test_y = test_dataset.iloc[:, -1]    
    
    return train_x, train_y, farsec_x, farsec_y, test_x, test_y


def get_data_clean(train_dataset_path, test_dataset_path):
    train_dataset = read_data(train_dataset_path)
    test_dataset = read_data(test_dataset_path)

    print(train_dataset_path)
    
    # train with the whole train-dataset
    global train_x
    global test_x
    global train_y
    global test_y

    train_x = train_dataset.iloc[:, :-1]
    train_y = train_dataset.iloc[:, -1]
    
    test_x = test_dataset.iloc[:, :-1]
    test_y = test_dataset.iloc[:, -1]    
    
    return train_x, train_y, test_x, test_y

'''
  Predict with normal learner 
'''
def predict(l, train_x, train_y, test_x, test_y):
    if l == "NB":
        print("---------- NB ----------")
        rf_train_start_time = time.time()
        clf = MultinomialNB()
        clf.fit(train_x, train_y)
        predicted = clf.predict(test_x)
        Cost = time.time() - rf_train_start_time
        TN, FP, FN, TP, PD, PF, PREC, F_MEASURE, G_MEASURE = get_result_statistics(test_y, predicted)
        return TN, FP, FN, TP, PD, PF, PREC, F_MEASURE, G_MEASURE, Cost      
    elif l == "RF":
        print("===============RF ===============")
        rf_train_start_time = time.time()
        clf = RandomForestClassifier(oob_score=True, n_estimators=30)
        clf.fit(train_x, train_y)
        predicted = clf.predict(test_x)
        Cost = time.time() - rf_train_start_time
        TN, FP, FN, TP, PD, PF, PREC, F_MEASURE, G_MEASURE = get_result_statistics(test_y, predicted)
        return TN, FP, FN, TP, PD, PF, PREC, F_MEASURE, G_MEASURE, Cost  
    elif l == "LR": 
        print("===============LR ===============")
        lr_train_start_time = time.time()
        clf = LogisticRegression()
        clf.fit(train_x, train_y)
        predicted = clf.predict(test_x)
        Cost = time.time() - lr_train_start_time
        TN, FP, FN, TP, PD, PF, PREC, F_MEASURE, G_MEASURE = get_result_statistics(test_y, predicted)
        return TN, FP, FN, TP, PD, PF, PREC, F_MEASURE, G_MEASURE, Cost  
    elif l == "MLP": 
        print("===============MLP ===============")
        mlp_train_start_time = time.time()
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                            hidden_layer_sizes=(5, 2), random_state=1)
        clf.fit(train_x, train_y)
        predicted = clf.predict(test_x)
        Cost = time.time() - mlp_train_start_time
        TN, FP, FN, TP, PD, PF, PREC, F_MEASURE, G_MEASURE = get_result_statistics(test_y, predicted) 
        return TN, FP, FN, TP, PD, PF, PREC, F_MEASURE, G_MEASURE, Cost  
    elif l == "KNN": 
        print("===============KNN ===============")
        knn_train_start_time = time.time()
        clf = LogisticRegression()
        clf.fit(train_x, train_y)
        predicted = clf.predict(test_x)
        Cost = time.time() - knn_train_start_time
        TN, FP, FN, TP, PD, PF, PREC, F_MEASURE, G_MEASURE = get_result_statistics(test_y, predicted) 
        return TN, FP, FN, TP, PD, PF, PREC, F_MEASURE, G_MEASURE, Cost  

#def main():
output = "../output_clean/0_clean_predict_with_farsec75_5clf_output.csv"     
csv_file = open(output, "w", newline='')
writer = csv.writer(csv_file, delimiter=',')
writer.writerow(['Dataname','Version', 'Approach', 'TN', 'FP', 'FN', 'TP', 'pd', 'pf', 'prec', 'fmeasure', 'Gmeasure','Cost']) 

datanames = ["ambari", "camel", "derby", "wicket", "chromium"] #"ambari", "camel", "derby", "wicket", "chromium"
classifiers = ["NB","RF","LR","MLP","KNN"]
#classifier = "KNN"
for dataname in datanames:
    print("")
    print("Start processing: ", dataname)
    TRAIN_PATH_noise = r"../input/matrix/noise/" +dataname+ "-train.csv"
    TRAIN_PATH_farsec = r"../input/matrix/clean75/" +dataname+ "-farsectwo.csv"
    TEST_PATH_noise = r"../input/matrix/noise/" +dataname+ "-test.csv"
    
    TRAIN_PATH_clean = r"../input/matrix_new/" +dataname+ "-train.csv"
    TEST_PATH_clean = r"../input/matrix/clean75/" +dataname+ "-test.csv"    
    
    # get data
    train_x, train_y, farsec_x, farsec_y, test_x, test_y = get_data(TRAIN_PATH_noise, TRAIN_PATH_farsec, TEST_PATH_clean)
    ''' 
    Train with noise data
    '''
    #normal prediction    
#    TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost = predict(classifier, train_x, train_y, test_x, test_y)
#    writer.writerow([dataname,'Noise', classifier, TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost])
#    print(dataname,'Noise', classifier, TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost)
#    #
#    ##Prediction with learning tuning
#    TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost = prediction_with_tuning(classifier, train_x, train_y, test_x, test_y)
#    writer.writerow([dataname,'Noise', 'RF_Tuning', TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost]) 
#    print(dataname,'Noise', 'RF_Tuning', TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost)
#    
#    #prediction with smounted
#    TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost = prediciton_with_smounted(classifier, TRAIN_PATH_noise, TEST_PATH_noise)
#    writer.writerow([dataname,'Noise', 'Smounted', TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost])     
#    print(dataname,'Noise', 'Smounted', TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost)    
    
#    
#    ''' 
#    Train with farsec data
#    '''
##    #normal prediction    
    for classifier in classifiers: 
        print("Start processing: ", classifier)
        TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost = predict(classifier, farsec_x, farsec_y, test_x, test_y)
        writer.writerow([dataname,'farsectwo', classifier, TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost])
        print(dataname,'farsectwo', classifier, TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost)
#    #
#    ##Prediction with learning tuning
#    TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost = prediction_with_tuning(classifier, farsec_x, farsec_y, test_x, test_y)
#    writer.writerow([dataname,'farsectwo', 'RF_Tuning', TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost]) 
#    print(dataname,'farsectwo', 'RF_Tuning', TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost)
#    
    #prediction with smounted
#    TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost = prediciton_with_smounted(classifier, TRAIN_PATH_farsec, TEST_PATH_noise)
#    writer.writerow([dataname,'farsectwo', 'Smounted', TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost])     
#    print(dataname,'farsectwo', 'Smounted', TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost)  
#    
#    
    '''Train with clean data '''
#    train_clean_x, train_clean_y, test_clean_x, test_clean_y = get_data_clean(TRAIN_PATH_clean, TEST_PATH_clean)   
##    
##    #normal prediction    
#    TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost = predict(classifier, train_clean_x, train_clean_y, test_clean_x, test_clean_y)
#    writer.writerow([dataname,'Clean', 'RF', TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost])
#    print(dataname,'Clean', 'RF', TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost)
#    
#    #Prediction with learning tuning
#    TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost = prediction_with_tuning(classifier, train_clean_x, train_clean_y, test_clean_x, test_clean_y)
#    writer.writerow([dataname,'Clean', 'RF_Tuning', TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost]) 
#    print(dataname,'Clean', 'RF_Tuning', TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost)
#    
#    #prediction with smounted
#    TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost = prediciton_with_smounted(classifier,TRAIN_PATH_clean, TEST_PATH_clean)
#    writer.writerow([dataname,'Clean', 'Smounted', TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost])     
#    print(dataname,'Clean', 'Smounted', TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost)    

csv_file.close()
print(output + '**************** finished************************')

#if __name__ == "__main__":
#    main()
