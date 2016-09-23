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


from sklearn.metrics import classification_report
from sklearn.cross_validation import StratifiedKFold
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier

import jieba
import csv

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

def loadLableMap_biol(lablePath):
    lablemap={}
    maplable={}

    for line in open(lablePath, 'r').readlines():
        line = (line).strip()
        if line == '理解能力':
            lablemap[str(1)] = line
            maplable[line] = str(1)
        else:
            lablemap[str(0)] = line
            maplable[line] = str(0)

    return lablemap,maplable

def getData_tarin(tfidf,lables,indexArr):
    X = []
    y = []
    for xti in indexArr:
        y.append(lables[xti])
        X.append(tfidf[xti])
    return X,y

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
    cf.read("path_new.config")

    envir = 'WindowsServer2012'
    #trainFilePath = cf.get(envir, 'trainFilePath') + subject+'.txt'
    csvtrainFilePath = cf.get(envir, 'csvtrainFilePath') + subject+'.csv'

    resultPath = cf.get(envir, 'resultPath') +'/kn_2time/'+ subject+'.txt'
    print resultPath
    lablePath = cf.get(envir, 'lablePath') + subject

    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    lablemap, maplable = loadLableMap_biol(lablePath)

    lables = []  # 标签y
    corpus = []  # 切词后用空格分开的文本

    csvfile = open(csvtrainFilePath ,'rb')
    csvfile.readline()
    trainReader = csv.reader(csvfile)
    for line in trainReader:
        lable = line[5]

        lall = line[2]+' '+line[3]+' '+line[4]+' '+line[7]
        lall = jieba.cut(lall, cut_all=True)

        la=[]
        for a in lall:
            la.append(a)
            la.append(' ')

        lables.append(lable)
        ls = ''.join(la)
        corpus.append(ls)

    print os.path.basename(csvtrainFilePath) + '------------------------------------------------------------'
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
        i = i+ 1
        print 'fold' + str(i) + ''
        fwrite.write('fold' + str(i) + '\n')

        #clf = LogisticRegression()
        #clf2 = SVC(C = 100.0)

        #clf = KNeighborsClassifier(weights='distance',leaf_size=800)
        clf = KNeighborsClassifier()
        X , y = getData(tfidf,lables,train)
        Xt, yt = getData(tfidf, lables, test)

        y1 = []
        for yy in y:
            if yy == '理解能力':
                y1.append('1')
            else:
                y1.append('0')

        clf.fit(X, y1)
        train_predict = clf.predict(X)

        ny = []
        nX = []
        for index in range(len(train_predict)):
            if train_predict[index] == '0' :
                ny.append(y[index])
                nX.append(X[index])

        clf2 = KNeighborsClassifier()
        clf2.fit(nX, ny)

        nny = []
        nnX = []
        for index in range(len(train_predict)):
            if train_predict[index] == '1':
                nny.append(y[index])
                nnX.append(X[index])

        clf3 = KNeighborsClassifier()
        clf3.fit(nnX, nny)


        # Xt = clf.predict_proba(Xt)
        predicted1 = clf.predict(Xt)

        y1t = []
        for yy in yt:
            if yy == '理解能力':
                y1t.append('1')
            else:
                y1t.append('0')

        fwrite.write(classification_report(y1t, predicted1).replace('\n\n', '\n'))
        print classification_report(y1t, predicted1).replace('\n\n', '\n')

        nyt = []
        nXt = []
        for index in range(len(predicted1)):
            if predicted1[index] == '0':
                nyt.append(yt[index])
                nXt.append(Xt[index])
        predicted2 = clf2.predict(nXt)
        fwrite.write(classification_report(nyt, predicted2).replace('\n\n', '\n'))
        print classification_report(nyt, predicted2).replace('\n\n', '\n')

        nnyt = []
        nnXt = []
        for index in range(len(predicted1)):
            if predicted1[index] == '1':
                nnyt.append(yt[index])
                nnXt.append(Xt[index])
        predicted3 = clf3.predict(nnXt)
        fwrite.write(classification_report(nnyt, predicted3).replace('\n\n', '\n'))
        print classification_report(nnyt, predicted3).replace('\n\n', '\n')
    fwrite.close()

if __name__ == "__main__":
    # root = u'../../../subjectClassify_py/TFIDF/map/abi/'
    # print root
    # for root,dirs,files in os.walk(root):
    #     print files
    #     for file in files:
            root = u'../../../subjectClassify_py/TFIDF/map/abi/'
            file = 'biol'
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