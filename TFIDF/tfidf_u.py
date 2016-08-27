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

def func(trainpath,testpath,repath):
    clf = LogisticRegression()


    lablemap = {"识记与理解": '0', "分析与应用": '1', "综合与拓展": '2'}

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

    for ti in range(0, 4, 1):
        for ind in range(0, len(list2), 1):
            y.append('2')
            corpus.append(list2[ind])

    X = transformer.fit_transform(vectorizer.fit_transform(corpus))
    X = SVD_Vec(X, 1000)
    print os.path.basename(trainpath)+'------------------------------------------------------------'

    clf.fit(X,y)
    clf.predict_proba(X)
    testX = loadTest(testpath + os.path.basename(trainpath))
    predicted = clf.predict(testX)


    fwrite = open(repath + os.path.basename(trainpath), 'w')
    for pre in predicted:
        fwrite.write(pre+'\n')
    fwrite.close()


if __name__ == "__main__":
    trainfile = '/home/hao/桌面/学科分类新/分词/2gram'
    repath = '/home/hao/桌面/学科分类新/pre/test/'
    testfile = '/home/hao/桌面/学科分类新/test/'
    print trainfile
    for root,dirs,files in os.walk(trainfile):
        print files
        for file in files:
            print file
            func(trainfile + file,testfile,repath)

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