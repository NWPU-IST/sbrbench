'''
Created on 2018年10月24日

@author: Administrator
'''
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC

def selectFromLinearSVC(data, label):
    lsvc = LinearSVC(C=0.3, dual=False, penalty='l1').fit(data, label)  # , dual=False  plants 
    model = SelectFromModel(lsvc, prefit=True)
    new_data = model.transform(data)
    return new_data
def selectFromLinearSVC3(train_content, train_label, test_content, left_content):
    lsvc = LinearSVC(C=0.3, dual=False, penalty='l1').fit(train_content,train_label)  # , dual=False  plants 
    model = SelectFromModel(lsvc, prefit=True)
    
    new_train = model.transform(train_content)
    new_test = model.transform(test_content)
    new_left = model.transform(left_content)
    return new_train, new_test,new_left

def selectFromLinearSVC2(train_content, train_label, test_content):
    lsvc = LinearSVC(C=0.3, dual=False, penalty='l1').fit(train_content,train_label)  # , dual=False  plants 
    model = SelectFromModel(lsvc, prefit=True)
    
    new_train = model.transform(train_content)
    new_test = model.transform(test_content)
 
    return new_train, new_test

