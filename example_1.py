# coding=utf-8
import pickle, MWMOTE_1, random, math
import matplotlib.pyplot as plt
import numpy as np
import re
import os
from collections import Counter


def random_forest_classifier(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=8)
    model.fit(train_x, train_y)
    return model

def my_confusion_matrix(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    labels = list(set(y_true))
    conf_mat = confusion_matrix(y_true, y_pred, labels = labels)
    print "confusion_matrix(left labels: y_true, up labels: y_pred):"
    print "labels\t",
    for i in range(len(labels)):
        print labels[i],"\t",
    print
    for i in range(len(conf_mat)):
        print i,"\t",
        for j in range(len(conf_mat[i])):
            print conf_mat[i][j],'\t',
        print
    print



from sklearn import metrics
from sklearn.decomposition import PCA
import pandas as pd
import random
from sklearn.metrics import confusion_matrix

resultList=[];#用于存放结果的List
A=0; #最小随机数
B=17387 #最大随机数
COUNT=5000
resultList=random.sample(range(A,B+1),COUNT);

if __name__ == '__main__':

    data1 = np.loadtxt('21test_failure.csv', delimiter=',')
    data2 = np.loadtxt('21test_normal.csv', delimiter=',')
    train_test1 = np.row_stack((data1, data2))
    #print len(train_test1)
    data1 = np.loadtxt('15train_failure.csv', delimiter=',')
    data2 = np.loadtxt('21train_failure.csv', delimiter=',')
    train_test2 = np.row_stack((data1,data2))
    #print len(train_test2)
    train_test3 = np.row_stack((train_test1,train_test2))
    train_test1=[]
    train_test2=[]

    with open ('15train_normal.csv','r') as reader, open ('train_normal.csv', 'w') as writer:
        for index, line in enumerate(reader):
            if index in resultList:
                writer.write(line)

    train_test1 = np.loadtxt('train_normal.csv',delimiter=',')
    train_test = np.row_stack((train_test3,train_test1))
    train_test1=[]
    X=train_test[:,:len(train_test)-1]
    Y=train_test[:,len(train_test[0])-1]
    train_test=[]

    #--------------------------------------------
    print '原始数据：'
    X_train_ori=X[2742:,:]
    Y_train_ori=Y[2742:]
    X_test_ori=X[0:2742,:]
    Y_test_ori=Y[0:2742]
    model = random_forest_classifier(X_train_ori,Y_train_ori)
    predict = model.predict(X_test_ori)



    my_confusion_matrix(Y_test_ori,predict)
    precision = metrics.precision_score(Y_test_ori,predict)
    recall=metrics.recall_score(Y_test_ori,predict)
    print 'precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall)
    accuracy = metrics.accuracy_score(Y_test_ori, predict)
    print 'accuracy: %.2f%%' % (100 * accuracy)

    X_train_ori=[]
    Y_train_ori=[]
    X_test_ori=[]
    Y_test_ori=[]
    #------------------------------

    Y_train=Y[2742:]
    print type(Y_train[0])
    print Y_train[0]
    countDict = dict(Counter(Y_train))
    N = countDict[0] - countDict[1]
    print 'N=' + str(N)


    print 'MWMOTE+EM后：'
    pca = PCA(n_components=100)
    newData = pca.fit_transform(X)
    print newData[0][2]
    X=[]
    print type(newData)
    # print newData[0][1]

    pca_train_test = np.column_stack((newData, Y))
    np.savetxt('pca_train_test.csv', pca_train_test, delimiter=',')


    X=newData
    newData=[]

    X_train=X[2742:,:]
    Y_train=Y[2742:]
    X_test=X[0:2742,:]
    Y_test=Y[0:2742]

    model = random_forest_classifier(X_train,Y_train)
    predict = model.predict(X_test)

    my_confusion_matrix(Y_test,predict)
    precision = metrics.precision_score(Y_test,predict)
    recall=metrics.recall_score(Y_test,predict)
    print 'precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall)
    accuracy = metrics.accuracy_score(Y_test, predict)
    print 'accuracy: %.2f%%' % (100 * accuracy)


    #
    #
    #
    # countDict = dict(Counter(Y_train))
    # N = countDict[0] - countDict[1]
    # print 'N='+str(N)
    # X_g, Y_g = MWMOTE_1.MWMOTE(X, Y, N)
    # Z = np.column_stack((X_g, Y_g))
    # np.savetxt('train.csv', Z, delimiter=',')
    #
    #
    #
    # model = random_forest_classifier(X_train,Y_train)
    # predict = model.predict(X_test)
    #
    # precision = metrics.precision_score(Y_test,predict)
    # recall=metrics.recall_score(Y_test,predict)
    # print 'precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall)
    # accuracy = metrics.accuracy_score(Y_test, predict)
    # print 'accuracy: %.2f%%' % (100 * accuracy)


    # data0 = np.loadtxt('21train_failure.csv', delimiter=',')
    # X=data0[:,:len(data0[0])-1]
    # Y=data0[:,len(data0[0])-1]
    # # 复制的个数N
    # countDict = dict(Counter(Y))
    # N = countDict[0] - countDict[1]
    # #X_g Y_g 分别为合成之后的所有X和Y，样本和标签
    # X_g, Y_g = MWMOTE_1.MWMOTE(X, Y, N)
    # Z= np.column_stack((X_g, Y_g))
    # # 将其存入文件中
    # np.savetxt('train.csv', Z, delimiter=',')
    #
    #
    # X_train=[]
    # Y_train=[]
    # X_test=[]
    # Y_test=[]
    # for i in range(0,len(Z)):
    #     if i/10!=0:
    #         X_train.append(X_g[i])
    #         Y_train.append(Y_g[i])
    #     else:
    #         X_test.append(X_g[i])
    #         Y_test.append(Y_g[i])
    # model = random_forest_classifier(X_train,Y_train)
    # predict = model.predict(X_test)
    #
    # precision = metrics.precision_score(Y_test,predict)
    # recall=metrics.recall_score(Y_test,predict)
    # print 'precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall)
    # accuracy = metrics.accuracy_score(Y_test, predict)
    # print 'accuracy: %.2f%%' % (100 * accuracy)
    #
    #
    #
    # for i in range(0,len(X)):
    #     if i/10!=0:
    #         X_train.append(X[i])
    #         Y_train.append(Y[i])
    #     else:
    #         X_test.append(X[i])
    #         Y_test.append(Y[i])
    # model = random_forest_classifier(X_train,Y_train)
    # predict = model.predict(X_test)
    #
    # precision = metrics.precision_score(Y_test,predict)
    # recall=metrics.recall_score(Y_test,predict)
    # print 'precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall)
    # accuracy = metrics.accuracy_score(Y_test, predict)
    # print 'accuracy: %.2f%%' % (100 * accuracy)









