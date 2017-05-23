#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import argparse
import math
import datetime
import numpy as np
import logging
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss
import xgboost as xgb
from adsb3 import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('--valid', action='store_true')
parser.add_argument('--cv', action='store_true')
parser.add_argument('--stage1', action='store_true')
parser.add_argument('--reproduce', action='store_true')
parser.add_argument('models', nargs='*')
args = parser.parse_args()

models = args.models

if len(models) == 0:
    models = ['unet_k_b8',
              'unet_m2_b8',
              'unet_m3_b8',
              #'unet_k_ft_b8',
              #'unet_k_b8_hull',
              'tiny_ft1_b8',
              'tiny_m4_b8'
             ]
    logging.warn('No models are specified, using default: ' + str(models))

#  0.43522
SPLIT=3
PARTITIONS = [(1,1,1),(1,1,2),(3,1,1),(1,1,3)]
MIN_NODULE_SIZE=1 #30
MAX_IT = 1200

if args.reproduce:
    models = ['unet_k_b8',
              'unet_m2_b8',
              'unet_m3_b8',
              #'unet_k_ft_b8',
              #'unet_k_b8_hull',
              'tiny_ft1_b8',
              'tiny_m4_b8'
             ]
    MIN_NODULE_SIZE=30
    MAX_IT = 1200
    logging.warn('Reproducing best submission 0.46482')
    pass


model = xgb.XGBClassifier(n_estimators=MAX_IT,
            learning_rate=0.01,
            max_depth=2,seed=2016,
            subsample=0.9,
            colsample_bytree=0.4) #, reg_lambda=0.5)

xgb_param = {'max_depth':2,
             'eta':0.01,
             'silent':1,
             'objective':'binary:logistic',
             'subsample': 0.90,
             'colsample_bytree': 0.40,
            }

TRAIN = STAGE1_TRAIN

if args.stage1:
    TRAIN = STAGE1_TRAIN
    TEST = STAGE1_PUBLIC
else:
    TRAIN = STAGE1_TRAIN + STAGE1_PUBLIC
    TEST = STAGE2_PRIVATE
    pass

failed = []
missing = []

pyramid = plumo.Pyramid(PARTITIONS)
print("TOTAL_PARTS: ", pyramid.parts)

def load_uid_features (uid):
    ft = []
    for model in models:
        cached = os.path.join('cache', model, uid + '.pkl')
        if not os.path.exists(cached):
            logging.warn('missing ' + cached)
            missing.append(cached)
            ft.append(None)
            continue
        if True:
            dim, nodules = load_fts(cached)
            nodules = [n for n in nodules if n[0] >= MIN_NODULE_SIZE]
            ft.append(pyramid.apply(dim, nodules))
        continue
        try:
            dim, nodules = load_fts(cached)
            nodules = [n for n in nodules if n[0] >= MIN_NODULE_SIZE]
            ft.append(pyramid.apply(dim, nodules))
        except:
            failed.append(cached)
            ft.append(None)
            pass
        pass
    return ft

def merge_features (fts, dims):
    v = []
    fixed = False
    for ft, dim in zip(fts, dims):
        if ft is None:
            fixed = True
            v.extend([0] * dim)
        else:
            assert len(ft) == dim
            v.extend(ft)
            pass
        pass
    return v, fixed

def load_features (dataset, dims = None):
    cases = []
    for uid, label in dataset:
        fts = load_uid_features(uid)
        # fts, list of vectors:  [ [...], [...], [...] ]
        cases.append((uid, label, fts))
    if dims is None:    # calculate dimensions of each model
        dims = [None] * len(models)
        # fix None features
        for i in range(len(models)):
            dims[i] = 0
            for _, _, fts in cases:
                if not fts[i] is None:
                    dims[i] = len(fts[i])
                    break
                pass
            pass
        pass
    U = []
    X = []
    Y = []
    for uid, label, fts in cases:
        v, fixed = merge_features(fts, dims)
        if args.valid and fixed:
            continue
        U.append(uid)
        Y.append(label)
        X.append(v)
    return U, np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32), dims


_, X_train, Y_train, dims = load_features(TRAIN)
_, X_test, Y_test, _ = load_features(TEST, dims)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

kf = KFold(n_splits=SPLIT, shuffle=True, random_state=88)
Y_train_pred = Y_train * 0
Y_train_prob = Y_train * 0

# K-fold cross validation
for train, val in kf.split(X_train):
    #X_train, X_test, y_train = X[train,:], X[test,:], Y[train]
    model.fit(X_train[train,:], Y_train[train])
    Y_train_pred[val] = model.predict(X_train[val, :])
    Y_train_prob[val] = model.predict_proba(X_train[val, :])[:, 1]
    pass

model.fit(X_train, Y_train)
Y_test_pred = model.predict(X_test)
Y_test_prob = model.predict_proba(X_test)[:, 1]

if args.cv:
    xgb.cv(xgb_param, xgb.DMatrix(X_train, label=Y_train), (MAX_IT+1000),
           nfold=10,
           stratified=True,
           metrics={'logloss'},
           callbacks=[xgb.callback.print_evaluation(show_stdv=False)])

print(classification_report(Y_train, Y_train_pred, target_names=["0", "1"]))
print("validation logloss",log_loss(Y_train, Y_train_prob))

print(classification_report(Y_test, Y_test_pred, target_names=["0", "1"]))
print("test logloss",log_loss(Y_test, Y_test_prob))

print("%d corrupt" % len(failed))
for uid in failed:
    print(uid)
    pass
pass

print("%d missing" % len(missing))
