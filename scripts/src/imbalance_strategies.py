# -*- coding: utf-8 -*-
__author__ = 'Junzheng Chen'

import random

import numpy as np
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import ADASYN 
from imblearn.over_sampling import BorderlineSMOTE

from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import CondensedNearestNeighbour

from imblearn.combine import SMOTEENN 
from imblearn.combine import SMOTETomek

from imblearn.ensemble import BalanceCascade
from imblearn.ensemble import EasyEnsemble

class Smote:
    def __init__(self, samples, N=10, k=5):
        self.n_samples, self.n_attrs = samples.shape
        self.N = N
        self.k = k
        self.samples = samples
        self.newindex = 0

    # self.synthetic=np.zeros((self.n_samples*N,self.n_attrs))

    def over_sampling(self):
        N = self.N
        print(self.n_attrs)
        self.synthetic = np.zeros((self.n_samples * N, self.n_attrs))
        neighbors = NearestNeighbors(n_neighbors=self.k).fit(self.samples)
        print('neighbors', neighbors)

        for i in range(len(self.samples)):
            nnarray = neighbors.kneighbors(self.samples[i].reshape(1, -1), return_distance=False)[0]
            self._populate(N, i, nnarray)
        return self.synthetic

    # for each minority class samples,choose N of the k nearest neighbors and generate N synthetic samples.
    def _populate(self, N, i, nnarray):
        for j in range(N):
            nn = random.randint(0, self.k - 1)
            dif = self.samples[nnarray[nn]] - self.samples[i]
            gap = random.random()
            self.synthetic[self.newindex] = self.samples[i] + gap * dif
            self.newindex += 1


def get_smote_result(data_list, label, N):
    length = len(data_list)
    postive_data = []
    for i in range(0, length):
        if label[i] == 1:
            postive_data.append(data_list[i])
    data_array = np.array(postive_data)
    smoke = Smote(data_array, N, 5)
    return smoke.over_sampling()

# Combination of over-and under-sampling methods
def get_cbs_smoteenn(data_list, label):
    smo = SMOTEENN(random_state=42)
    X_smo, y_smo = smo.fit_resample(data_list, label)
    return X_smo, y_smo 
   
def get_cbs_smotetomek(data_list, label):
    smo = SMOTETomek()
    X_smo, y_smo = smo.fit_resample(data_list, label)
    return X_smo, y_smo   

# Under sampling
def get_uds_rdm(data_list, label):
    rdm = RandomUnderSampler()
    X_rdm, y_rdm = rdm.fit_resample(data_list, label)
    return X_rdm, y_rdm

def get_uds_nm(data_list, label):
    nm = NearMiss()
    X_nm, y_nm = nm.fit_resample(data_list, label)    
    return X_nm, y_nm
    

def get_uds_enn(data_list, label):
    enn = EditedNearestNeighbours()
    X_res, y_res = enn.fit_resample(data_list, label)

def get_uds_CNN(data_list, label):
    cnn = CondensedNearestNeighbour(random_state=42)
    X_res, y_res = cnn.fit_resample(data_list, label)
# Over sampling
def get_ovs_smote_standard(data_list, label):
    smo = SMOTE(random_state=42)
    X_smo, y_smo = smo.fit_sample(data_list, label)
    return X_smo, y_smo

def get_ovs_adasyn(data_list, label):
    smo = ADASYN(random_state=42)
    X_smo, y_smo = smo.fit_resample(data_list, label)
    return X_smo, y_smo   
def get_ovs_smotenc(data_list, label):
    smo = SMOTENC(random_state=42, categorical_features=[18, 19])
    X_smo, y_smo = smo.fit_resample(data_list, label)
    return X_smo, y_smo 

def get_ovs_BorderlineSMOTE(data_list, label):
    bd_smt = BorderlineSMOTE()
    X_smo, y_smo = bd_smt.fit_resample(data_list, label)
    return X_smo, y_smo 

def get_ovs_smote_borderline_1(clf, data, label, m, s, k=5):
    label_local = label[:]
    clf.fit(data, label_local)
    data_list = data.tolist()
    data_list = data_list[:]
    length = len(data_list)

    T = np.array(data_list)
    n_samples, n_attrs = T.shape

    # get p list
    P = []
    for i in range(0, length):
        if label_local[i] == 1:
            P.append(i)

    n_samples = len(P)
    # calc m for all the positive sample
    neighbors = NearestNeighbors(n_neighbors=k).fit(T)
    synthetic = np.zeros((n_samples * m, n_attrs))
    newindex = 0
    for i in range(len(P)):
        nnarray = neighbors.kneighbors(T[P[i]].reshape(1, -1), return_distance=False)[0]
        for j in range(m):
            nn = random.randint(0, k - 1)
            dif = T[nnarray[nn]] - T[P[i]]
            gap = random.random()
            synthetic[newindex] = T[P[i]] + gap * dif
            newindex += 1

    pred = []
    danger = []
    noise = []
    for i in range(0, n_samples * m):
        pred.append(clf.predict(synthetic[i].reshape(1, -1)))

    for i in range(0, len(pred)):
        if i % 5 != 0:
            continue
        count = 0
        for j in range(0, 5):
            if i + j >= len(pred) - 1:
                continue
            if pred[i + j] == 0:
                count += 1

        if count == 5:
            noise.append(P[int(i / 5)])
        elif count > 2:
            danger.append(P[int(i / 5)])



    n_samples_danger = len(danger)
    # calc m for all the positive sample
    danger_list = []
    for i in danger:
        danger_list.append(T[i])

    if not danger_list:
        result = []
        result.append(data_list)
        result.append(label)
        return result
    neighbors = NearestNeighbors(n_neighbors=k).fit(danger_list)
    synthetic_danger = np.zeros((n_samples_danger * s, n_attrs), dtype=float)
    newindex_danger = 0
    for i in range(len(danger)):
        if 5 > len(danger):
            result = []
            result.append(data_list)
            result.append(label)
            return result
        nnarray = neighbors.kneighbors(T[danger[i]].reshape(1, -1), return_distance=False)[0]

        for j in range(m):
            nn = random.randint(0, k - 1)
            dif = T[nnarray[nn]] - T[danger[i]]
            gap = random.random()
            synthetic_danger[newindex_danger] = T[danger[i]] + gap * dif
            newindex_danger += 1

    synthetic_danger_list = synthetic_danger.tolist()

    noise.reverse()

    # 删除noise
    for i in range(0,len(noise)):
        del data_list[noise[i]]
        del label_local[noise[i]]

    # 添加正项
    random_list = []
    for i in range(0, len(synthetic_danger_list)):
        random_list.append(int(random.random() * len(data_list)))

    for i in range(0, len(random_list)):
        data_list.insert(random_list[i], synthetic_danger_list[i])
        label_local.insert(random_list[i], 1)

    result = []
    result.append(data_list)
    result.append(label_local)
    return result

# ensemble method
def get_ens_BalanceCascade(data_list, label):
    bc = BalanceCascade(random_state=42)
    data_list = data_list.toarray()
    X_smo, y_smo = bc.fit_resample(data_list, label)
    #print(type(X_smo[0]), Counter(y_smo[0]))
    X_smo = np.reshape(X_smo, (-1, X_smo[0][0].shape[0]))
    y_smo = y_smo.flatten()
    #print(X_smo.tolist(), type(y_smo))
    return X_smo, y_smo

def get_ens_EasyEnsemble(data_list, label):
    ee = EasyEnsemble(random_state=42)
    X_smo, y_smo = ee.fit_resample(data_list, label)
    return X_smo, y_smo
