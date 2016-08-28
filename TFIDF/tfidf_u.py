#!/usr/bin/env python
# -*- coding:utf-8 -*-
# coding: utf-8
import time
import re
import os
import sys
import codecs
import shutil
import numpy as np
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
from sklearn.decomposition import TruncatedSVD
from sklearn import svm
import csv

reload(sys)
sys.setdefaultencoding('utf8')

vectorizer = CountVectorizer()
transformer = TfidfTransformer()

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

def loadTest(testPath):
    corpus = []  # 切词后用空格分开的文本
    for line in open(testPath, 'r').readlines():
        corpus.append(line)
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    tfidf = SVD_Vec(tfidf, 1000)
    return tfidf

def func(trainpath,testpath,repath,testFile):
    clf = LogisticRegression()
    clf2 = LDA();
    clf4 = LinearSVC();
    repath = repath + os.path.basename(trainpath)
    testpath = testpath + os.path.basename(trainpath)
    repath = repath + os.path.basename(trainpath).replace('.txt','.csv')
    testFile = testFile + os.path.basename(trainpath).replace('.txt','.csv')

    lablemap = {"识记与理解": '0', "分析与应用": '1', "综合与拓展": '2'}
    lablemap2 = {'0': "识记与理解", '1': "分析与应用", '2':"综合与拓展"}

    y = []#标签y
    corpus = []#切词后用空格分开的文本
    list2=[]
    for line in open(trainpath, 'r').readlines():
        words = line.strip().split(' ')
        lable = lablemap.get(words[0])
        line = line[line.find(' ') + 1:]

        corpus.append(line)
        y.append(lable)
        if lable == '2':
            list2.append(line)

    # for ti in range(0, 4, 1):
    #     for ind in range(0, len(list2), 1):
    #         y.append('2')
    #         corpus.append(list2[ind])

    X = transformer.fit_transform(vectorizer.fit_transform(corpus))
    X = SVD_Vec(X, 1000)
    print os.path.basename(trainpath)+'------------------------------------------------------------'

    clf.fit(X,y)
    X = clf.predict_log_proba(X)
    clf2.fit(X,y)
    X = clf2.predict_log_proba(X)
    clf4.fit(X, y)

    csvfile = file(testFile,'rb')
    testAll = csv.reader(csvfile)

    csvtest = []  #
    for line in testAll:
        csvtest.append(line)

    csvout=file(repath, 'wb')
    csvwriter = csv.writer(csvout)

    testX = loadTest(testpath)
    predicted = clf.predict_log_proba(testX)
    predicted = clf2.predict_log_proba(predicted);
    predicted = clf4.predict(predicted);
    for preindex in range(0,len(predicted)):
        pre = predicted[preindex]
        csvnub = csvtest[preindex]
        lableStr=lablemap2.get(pre)
        csvreline=[]
        csvreline.append(csvnub[0])
        csvreline.append(csvnub[1])
        csvreline.append(lableStr)
        csvwriter.writerow(csvreline)
    csvout.close()
    # testX = loadTest(testpath)
    # predicted = clf.predict_proba(testX)
    # fwrite = open(repath , 'w')
    # for pre in predicted:
    #     fwrite.write(str(pre[0]) + '\t')
    #     fwrite.write(str(pre[1]) + '\t')
    #     fwrite.write(str(pre[2]) + '\n')
    # fwrite.close()

if __name__ == "__main__":
    trainfile = u'D:/haozhenyuan/学科分类/train/2gram/'
    repath = u'D:/haozhenyuan/学科分类/preCSV3model/'
    testfile = u'D:/haozhenyuan/学科分类/test/'
    test = u'D:/haozhenyuan/学科分类/all-need-tag-data/'
    print trainfile

    for root,dirs,files in os.walk(trainfile):
        print files
        for file1 in files:
            func(trainfile+file1,testfile,repath,test)

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