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

def func(path):
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    #path = '/home/hao/桌面/学科分类新/2gram/geog.txt'  # geog.txt
    #mapPath = '/home/hao/PycharmProjects/subjectClassify/TFIDF/map/aa/all.txt'
    #三个文本类别
    lablemap = {"识记与理解": '0', "分析与应用": '1', "综合与拓展": '2'}

    lables = []#标签y
    corpus = []#切词后用空格分开的文本
    list2=[]
    for line in open(path, 'r').readlines():
        words = line.strip().split(' ')
        lable = lablemap.get(words[0])

        line = line[line.find(' ') + 1:]
        corpus.append(line)
        lables.append(lable)
    # for ti in range(0, 4, 1):
    #     for ind in range(0, len(list2), 1):
    #         lables.append('2')
    #         corpus.append(list2[ind])
    print os.path.basename(path)+'------------------------------------------------------------'
    fwrite = open('/home/hao/桌面/学科分类新/pre/train/NLPIR/lr_lda/' + os.path.basename(path), 'w')
    fwrite.write(os.path.basename(path)+'\n')
    # 5fold交叉检验
    #lables = np.array(lables)
    kf = StratifiedKFold(lables, 5)

    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    tfidf = SVD_Vec(tfidf, 1000)
    i=0
    for train,test in kf:
        i=i+1
        print 'fold'+str(i)+''
        fwrite.write('fold'+str(i)+'\n')
        clf = LogisticRegression()
        clf2 = LDA()
        X=[]
        y=[]
        for ti in train:
            if(lables[ti]=='2'):
                for time in range(0,10,1):
                    X.append(tfidf[ti])
                    y.append(lables[ti])
            else:
                X.append(tfidf[ti])
                y.append(lables[ti])

        clf.fit(X,y)
        XX=clf.predict_proba(X)
        clf2.fit(XX,y)

        t1 = test[0]
        t2 = test[-1]
        test = tfidf[t1:t2]
        test = clf.predict_proba(test)
        testlables = lables[t1:t2]
        predicted = clf2.predict(test)

        fwrite.write(classification_report(testlables,predicted).replace('\n\n','\n'))
        print classification_report(testlables,predicted).replace('\n\n','\n')
        #print accuracy_score(testlables, predicted)

    #scores = scv.cross_val_score(clf, tfidf, lables1, cv=5, scoring='accuracy')
    #print scores
    #predicted = scv.cross_val_predict(clf, tfidf1, lables1, cv=5)

    #predicted = clf.predict(tfidf1)
    #print os.path.basename(path)
    #print classification_report(lables1,predicted)
    #print len(predicted)
    # prere = mt.accuracy_score(lables,predicted)
    #fwrite = open('/home/hao/桌面/学科分类新/pre/lr/'+ os.path.basename(path),'w')
    #for pre in predicted:
    #    fwrite.write(pre)
    #    fwrite.write('\n')
    fwrite.close()
    #print predicted

    #return scores

if __name__ == "__main__":
    #allfile = u'/home/hao/桌面/学科分类新/分词/2gram/'
    allfile = u'/home/hao/桌面/学科分类新/train/NLPIR/'
    print allfile
    for root,dirs,files in os.walk(allfile):
        print files
        for file in files:
            func(allfile + file)

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