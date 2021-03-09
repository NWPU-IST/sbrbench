import pandas as pd
import numpy as np
import time
import warnings
import csv

from de import de_rf, de_lr, de_nb, de_mlpn, de_knn
from utilities import rf_tuning, nb_tuning, mlpn_tuning, lr_tuning, knn_tuning
from utilities import random_forest, KNN, logistic_regression, naive_bayes, multilayer_perceptron

from de import de_smote
from utilities import rf_smotuned, nb_smotuned, lr_smotuned, mlpn_smotuned, knn_smotuned
from utilities import parse_results, result_statistics, get_result_statistics, balance,data_transfer
from utilities import random_forest_default, naive_bayes_default, logistic_regression_default, KNN_default, multilayer_perceptron_default
from Smotuned import prediciton_with_smounted


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

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
    
    if l == "RF":
        print("===============RF without turning===============")
        rf_train_start_time = time.time()
        clf = RandomForestClassifier(oob_score=True, n_estimators=30)
        clf.fit(train_x, train_y)
        predicted = clf.predict(test_x)
        Cost = time.time() - rf_train_start_time
        TN, FP, FN, TP, PD, PF, PREC, F_MEASURE, G_MEASURE = result_statistics(test_y, predicted)
        return TN, FP, FN, TP, PD, PF, PREC, F_MEASURE, G_MEASURE, Cost  
    elif l == "LR": 
        print("===============LR without turning===============")
        lr_train_start_time = time.time()
        clf = LogisticRegression()
        clf.fit(train_x, train_y)
        predicted = clf.predict(test_x)
        Cost = time.time() - lr_train_start_time
        TN, FP, FN, TP, PD, PF, PREC, F_MEASURE, G_MEASURE = result_statistics(test_y, predicted)
        return TN, FP, FN, TP, PD, PF, PREC, F_MEASURE, G_MEASURE, Cost  
    elif l == "MLP": 
        print("===============MLP without turning===============")
        mlp_train_start_time = time.time()
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                            hidden_layer_sizes=(5, 2), random_state=1)
        clf.fit(train_x, train_y)
        predicted = clf.predict(test_x)
        Cost = time.time() - mlp_train_start_time
        TN, FP, FN, TP, PD, PF, PREC, F_MEASURE, G_MEASURE = result_statistics(test_y, predicted) 
        return TN, FP, FN, TP, PD, PF, PREC, F_MEASURE, G_MEASURE, Cost  
'''
  Predict with learner tuning
'''
def prediction_with_tuning(l, train_x, train_y, test_x, test_y):    
    data_transfer(train_x, train_y, test_x, test_y)
    if l == "RF":
        print("----------Tuning Random Forest with DE----------")
        rf_tuning_start_time = time.time()
        de_rf_result = list(de_rf(rf_tuning, bounds=[(10, 150), (1, 20), (2, 20), (2, 50), (0.01, 1), (1, 10)]))
        parameter, result = parse_results(str(de_rf_result[-1]))
        rf = random_forest(train_x, train_y, np.int(np.round(parameter[0])), np.int(np.round(parameter[1])),
                           np.int(np.round(parameter[2])),
                           np.int(np.round(parameter[3])), parameter[4], np.int(np.round(parameter[5])))
        rf_predictions = rf.predict(test_x)
        TN, FP, FN, TP, PD, PF, PREC, F_MEASURE, G_MEASURE = get_result_statistics(test_y, rf_predictions)
        Cost = (time.time() - rf_tuning_start_time)
        return TN, FP, FN, TP, PD, PF, PREC, F_MEASURE, G_MEASURE, Cost


#def main():
output = "../output_noise/predict_with_matrix_noise_clean_rf_tuning_output.csv"     
csv_file = open(output, "w", newline='')
writer = csv.writer(csv_file, delimiter=',')
writer.writerow(['Dataname','Version', 'Approach', 'TN', 'FP', 'FN', 'TP', 'pd', 'pf', 'prec', 'fmeasure', 'Gmeasure','Cost']) 

datanames = ["ambari", "camel", "derby", "wicket", "chromium"] #"ambari", "camel", "derby", "wicket", "chromium"
classifier = "RF"
for dataname in datanames:
#dataname = "ambari"
    print("")
    print("Start processing: ", dataname)
    TRAIN_PATH_noise = r"../input/matrix/noise/" +dataname+ "-train.csv"
    TRAIN_PATH_farsec = r"../input/matrix/farsec/" +dataname+ "-farsectwo.csv"
    TEST_PATH_noise = r"../input/matrix/noise/" +dataname+ "-test.csv"
    
    TRAIN_PATH_clean = r"../input/input_matrix/clean/" +dataname+ "-farsectwo.csv"
    TEST_PATH_clean = r"../input/matrix/clean/" +dataname+ "-test.csv"
    
    
    
    # get data
    train_x, train_y, farsec_x, farsec_y, test_x, test_y = get_data(TRAIN_PATH_noise, TRAIN_PATH_farsec, TEST_PATH_clean)
    ''' 
    Train with noise data
    '''
#    #normal prediction    
#    TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost = predict(classifier, train_x, train_y, test_x, test_y)
#    writer.writerow([dataname,'Noise', classifier, TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost])
#    print(dataname,'Noise', classifier, TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost)
    #
    ##Prediction with learning tuning
#    TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost = prediction_with_tuning(classifier, train_x, train_y, test_x, test_y)
#    writer.writerow([dataname,'Noise', 'RF_Tuning', TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost]) 
#    print(dataname,'Noise', 'RF_Tuning', TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost)
#    
#    #prediction with smounted
#    TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost = prediciton_with_smounted(TRAIN_PATH_noise, TEST_PATH_noise)
#    writer.writerow([dataname,'Noise', 'Smounted', TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost])     
#    print(dataname,'Noise', 'Smounted', TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost)    
    
    
    ''' 
    Train with farsec data
    '''
    #normal prediction    
    TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost = predict(classifier, farsec_x, farsec_y, test_x, test_y)
    writer.writerow([dataname,'farsectwo', classifier, TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost])
    print(dataname,'farsectwo', classifier, TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost)
    
#    Prediction with learning tuning
    TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost = prediction_with_tuning(classifier, farsec_x, farsec_y, test_x, test_y)
    writer.writerow([dataname,'farsectwo', 'RF_Tuning', TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost]) 
    print(dataname,'farsectwo', 'RF', TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost)
    
    #prediction with smounted
    TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost = prediciton_with_smounted(TRAIN_PATH_farsec, TEST_PATH_clean)
    writer.writerow([dataname,'farsectwo', 'Smounted', TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost])     
    print(dataname,'farsectwo', 'Smounted', TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost)  
    
    
    '''Train with clean data '''
#    train_clean_x, train_clean_y, test_clean_x, test_clean_y = get_data_clean(TRAIN_PATH_farsec, TEST_PATH_clean)   
#    
#    normal prediction    
#    TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost = predict(classifier, train_clean_x, train_clean_y, test_clean_x, test_clean_y)
#    writer.writerow([dataname,'Clean', 'RF', TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost])
#    print(dataname,'Clean', 'RF', TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost)
#    
    #Prediction with learning tuning
#    TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost = prediction_with_tuning(classifier, train_clean_x, train_clean_y, test_clean_x, test_clean_y)
#    writer.writerow([dataname,'Clean', 'RF_Tuning', TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost]) 
#    print(dataname,'Clean', 'RF', TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost)
#    
##    #prediction with smounted
#    TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost = prediciton_with_smounted(TRAIN_PATH_clean, TEST_PATH_clean)
#    writer.writerow([dataname,'Clean', 'Smounted', TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost])     
#    print(dataname,'Clean', 'Smounted', TN, FP, FN, TP, Recall, pf, PREC, F_MEASURE, G_MEASURE, Cost)    

csv_file.close()
print(output + '**************** finished************************')

#if __name__ == "__main__":
#    main()
