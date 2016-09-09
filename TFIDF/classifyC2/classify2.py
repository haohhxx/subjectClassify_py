#!/usr/bin/env python
# -*- coding:utf-8 -*-
import time
import re
import os
import sys
import ConfigParser
import numpy as np
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.cross_validation as scv

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from  sklearn.metrics import accuracy_score
from  sklearn.metrics import classification_report
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier

reload(sys)
sys.setdefaultencoding('utf8')

def SVD_Vec(matData, dimension):
	svd = TruncatedSVD(n_components=dimension)
	newData = svd.fit_transform(matData)
	return newData

def svdFeaSave(f,newData,lables):
    for i in range(0, len(newData)):
        x = newData[i]
        lable = lables[i]
        f.write(str(lable))
        f.write(' ')
        for y in x:
            f.write(str(y))
            f.write(' ')
        f.write('\n')
    f.close()

def loadLableMap(lablePath):
    lablemap={}
    maplable={}
    i=0
    for line in open(lablePath, 'r').readlines():
       line = (line).strip()
       # lablemap.setdefault(str(i),line)
       # maplable.setdefault(line,str(i))
       lablemap[str(i)] = line
       maplable[line] = str(i)
       i = i + 1
    return lablemap,maplable

def getData(tfidf,lables,indexArr):

    X = []
    y = []
    for xti in indexArr:
        y.append(lables[xti])
        X.append(tfidf[xti])
    return X,y

def func(subject):
    subject = os.path.basename(subject).replace(".txt",'')

    cf = ConfigParser.ConfigParser()
    cf.read("path_test.config")

    envir = 'WindowsServer2012'
    trainFilePath = cf.get(envir , 'trainFilePath') + subject+'.txt'
    resultPath = cf.get(envir , 'resultPath') +'/kn30/'+ subject+'.txt'
    print resultPath
    lablePath = cf.get(envir , 'lablePath') + subject

    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    lablemap, maplable = loadLableMap(lablePath)

    lables = []  # 标签y

    corpus = []  # 切词后用空格分开的文本
    for line in open(trainFilePath, 'r').readlines():
        words = line.strip().split(' ')
        lableword = words[0].strip()
        #lableword = maplable.get(lableword)

        line = line[line.find(' ') + 1:]
        corpus.append(line)
        lables.append(lableword)

    print os.path.basename(trainFilePath) + '------------------------------------------------------------'
    fwrite = open(resultPath, 'w')
    fwrite.write(subject + '\n')
    # 5fold交叉检验
    # lables = np.array(lables)
    kf = StratifiedKFold(lables, n_folds=5)
    # kf = KFold(len(lables), n_folds=5)
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    tfidf = SVD_Vec(tfidf, 1000)
    i = 0
    for train, test in kf:
        i = i + 1
        print 'fold' + str(i) + ''
        fwrite.write('fold' + str(i) + '\n')

        #clf = LogisticRegression()
        #clf2 = LDA()
        #clf4 = LinearSVC()

        #clf = AdaBoostClassifier(n_estimators=100)
        clf  = KNeighborsClassifier()
        clf2 = KNeighborsClassifier()
        X , y  = getData(tfidf,lables,train)
        Xt, yt = getData(tfidf, lables, test)

        clf.fit(X, y)
        predicted = clf.predict(Xt)
        fwrite.write(classification_report(yt, predicted).replace('\n\n', '\n'))
        print classification_report(yt, predicted).replace('\n\n', '\n')

    fwrite.close()

if __name__ == "__main__":
    root = u'../../../subjectClassify_py/TFIDF/map/abi/'
    print root
    for root,dirs,files in os.walk(root):
        print files
        for file in files:
            func(root + file)

            #scores = func(allfile + file)
            #print file,
            #print '\t',
            #for sc in scores:
            #    print sc,
            #    print '\t',
            #print ''
# i=0
#     j=0
#     k=0
#     for s in lables:
#         if (s == '0'): i = i + 1
#         if (s == '1'): j = j + 1
#         if (s == '2'): k = k + 1
#     print i
#     print j
#     print k