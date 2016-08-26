#!/usr/bin/env python
# -*- coding:utf-8 -*-
import time
import re
import os
import sys
import codecs
import shutil
import numpy as np
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.cross_validation as scv
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

reload(sys)
sys.setdefaultencoding('utf8')

def loadNormal20(path):
    y = []  # 标签y
    X = []  # 切词后用空格分开的文本
    for line in open(path, 'r').readlines():
        ls = line.split(' ')
        y.append(ls[0])
        xs=[]
        for xi in range(1,len(ls),1):
            xs.append(float(ls[xi]))
        X.append(xs)
    X = np.array(X)  # list to array
    y = np.array(y)
    return y,X

def loadNormal202(path):
    y = []  # 标签y
    X = []
    L = []
    for line in open(path, 'r').readlines():
        ls = line.split(' ')
        y.append(ls[0])
        L.append(ls[len(ls)-2])
        xs=[]
        for xi in range(1,len(ls)-2,1):
            xs.append(float(ls[xi].split(':')[1]))
        X.append(xs)
    X = np.array(X)  # list to array
    y = np.array(y)
    return y,X,L

if __name__ == "__main__":
    clf = LogisticRegression()
    #clf = LDA()
    pathTrain='/home/hao/桌面/lrpython/RankLib-v2.1/bin/fold1/20fea/trainSubVote.txt'
    y,X=loadNormal20(pathTrain)

    clf.fit(X, y)

    # scores = scv.cross_val_score(clf, X, y, cv=5)
    pathTest = '/home/hao/桌面/lrpython/RankLib-v2.1/bin/fold1/20fea/train/'
    for root, dirs, files in os.walk(pathTest):
        for file in files:
            print file
            y, X, L = loadNormal202(pathTest + file)
            p = clf.predict(X)
            fileoutputpath = '/home/hao/桌面/lrpython/RankLib-v2.1/bin/fold1/subVote/trainlongid/'
            if os.path.exists(fileoutputpath):
                print '',
            else:
                os.makedirs(fileoutputpath)
            fwrite = open(fileoutputpath + file, 'w')
            reset = set()
            for li in range(0, len(p)):
                #print str(p[li])
                if (str(p[li]) == '1'):
                    reset.add(str(L[li]).replace('#', ''))

            for a in reset:
                fwrite.write(a)
                fwrite.write('\n')
            fwrite.close()

    #scores = scv.cross_val_score(clf, X, y, cv=5)
    pathTest = '/home/hao/桌面/lrpython/RankLib-v2.1/bin/fold1/20fea/test/'
    for root, dirs, files in os.walk(pathTest):
        for file in files:
            print file
            y, X, L = loadNormal202(pathTest + file)
            p = clf.predict(X)

            fileoutputpath = '/home/hao/桌面/lrpython/RankLib-v2.1/bin/fold1/subVote/longid/'
            if os.path.exists(fileoutputpath):
                print '',
            else:
                os.makedirs(fileoutputpath)

            fwrite = open(fileoutputpath + file, 'w')
            reset = set()
            for li in range(0, len(p)):
                if (str(p[li]) == '1'):
                    reset.add(str(L[li]).replace('#', ''))

            for a in reset:
                fwrite.write(a)
                fwrite.write('\n')
            fwrite.close()

