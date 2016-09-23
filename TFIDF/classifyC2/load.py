#!/usr/bin/env python
# -*- coding:utf-8 -*-

# 需要安装的库：
# sklearn
# NumPy
# ScriPy

import sys
from sklearn.externals import joblib

reload(sys)
sys.setdefaultencoding('utf8')

#加载 csr_matrix 稀疏矩阵
tfidf = joblib.load(u'D:/haozhenyuan/学科分类/原始train7.19/tfidf/biol2347.scipy.sparse.csr.csr_matrix')
#print tfidf

#将稀疏矩阵释放为普通特征矩阵
#需要至少10G富余内存
tfidf = tfidf.todense()
tfidf = tfidf.tolist()

#print len(tfidf)
#print type(tfidf)

#写出释放后的矩阵
bw = open('./biol.txt','w')
for l1index in range(len(tfidf)):
    for score in tfidf[l1index]:
        bw.write(str(score)+' ')
    bw.write('\n')
    bw.flush()
bw.close()