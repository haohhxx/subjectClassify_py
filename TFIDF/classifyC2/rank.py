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


from  sklearn.metrics import classification_report
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

    i = 0
    for train, test in kf:
        i = i + 1
        fw = open(u"D:/haozhenyuan/学科分类/原始train7.19/kfold/biol/train/"+str(i)+'.txt', 'w')
        for ind in train:
            fw.write(lables[ind])
            fw.write('\t')
            fw.write(corpus[ind])
            fw.write('\n')
        fw.close()

        fw = open(u"D:/haozhenyuan/学科分类/原始train7.19/kfold/biol/test/" + str(i) + '.txt', 'w')
        for ind in test:
            fw.write(lables[ind])
            fw.write('\t')
            fw.write(corpus[ind])
            fw.write('\n')
        fw.close()



if __name__ == "__main__":
    # root = u'../../../subjectClassify_py/TFIDF/map/abi/'
    # print root
    # for root,dirs,files in os.walk(root):
    #     print files
    #     for file in files:
            root = u'../../../subjectClassify_py/TFIDF/map/abi/'
            file = 'biol'
            func(root + file)

