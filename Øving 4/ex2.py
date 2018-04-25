import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import csv
import time
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def logistic_z(z):
    return 1.0/(1.0+np.exp(-z))

def logistic_wx(w,x):
    return logistic_z(np.inner(w,x))

def classify(w,x):
    x=np.hstack(([1],x))
    return 0 if (logistic_wx(w,x)<0.5) else 1

def gradL(x,y,w,xi):
    return (logistic_wx(w,x) - y) * xi * np.exp(-np.inner(w,x)) * pow(logistic_wx(w,x),2)

def stochast_train_w(x_train,y_train,learn_rate=0.1,niter=1000):
    x_train=np.hstack((np.array([1]*x_train.shape[0]).reshape(x_train.shape[0],1),x_train))
    dim=x_train.shape[1]
    num_n=x_train.shape[0]
    w = np.random.rand(dim)
    index_lst=[]
    for it in range(niter):
        if(len(index_lst)==0):
            index_lst=random.sample(range(num_n), k=num_n)
        xy_index = index_lst.pop()
        x=x_train[xy_index,:]
        y=y_train[xy_index]
        for i in range(dim):
            update_grad = gradL(x,y,w,x[i])
            w[i] = w[i] - learn_rate * update_grad
    return w

def batch_train_w(x_train,y_train,learn_rate=0.1,niter=50):
    x_train=np.hstack((np.array([1]*x_train.shape[0]).reshape(x_train.shape[0],1),x_train))
    dim=x_train.shape[1]
    num_n=x_train.shape[0]
    w = np.random.rand(dim)
    index_lst=[]
    for it in range(niter):
        if not(it % 10):
            print(it)
        for i in range(dim):
            update_grad=0
            for n in range(num_n):
                y = y_train[n]
                x = x_train[n]
                update_grad += gradL(x,y,w[i],x[i])[i]
            w[i] -= learn_rate * update_grad *(1/num_n)
    return w

def getData(fileName):
    x = []
    y = []
    with open(fileName, newline='') as f:
        reader = csv.reader(f,delimiter = '\t',quotechar =',')
        for row in reader:
            x1 = 0
            if row[0][0] == '-':
                x1 = float(row[0][1:])
                x1 = -x1
            else:
                x1 = float(row[0])
            x2 = float(row[1])
            if row[1][0] =='-':
                x2 = -x2
            x.append([x1,x2])
            y.append(float(row[2][0]))
    x = np.array(x)
    y = np.array(y)
    f.close()
    return(x,y)

def train_and_plot(xtrain,ytrain,xtest,ytest,training_method,learn_rate=0.1,niter=1000):
    start_time = time.time()
    #train data
    data = pd.DataFrame(np.hstack((xtrain,ytrain.reshape(xtrain.shape[0],1))),columns=['x','y','lab'])
    ax=data.plot(kind='scatter',x='x',y='y',c='lab',cmap=cm.copper,edgecolors='black')

    #train weights
    w=training_method(xtrain,ytrain,learn_rate,niter)
    error=[]
    y_est=[]
    for i in range(len(ytest)):
        error.append(np.abs(classify(w,xtest[i])-ytest[i]))
        y_est.append(classify(w,xtest[i]))
    y_est=np.array(y_est)
    data_test = pd.DataFrame(np.hstack((xtest,y_est.reshape(xtest.shape[0],1))),columns=['x','y','lab'])
    data_test.plot(kind='scatter',x='x',y='y',c='lab',ax=ax,cmap=cm.coolwarm,edgecolors='black')
    print ("error={}".format(np.mean(error)))
    print("--- %s seconds ---" % (time.time() - start_time))
    plt.show()

    return w

(xTrain,yTrain) = getData("data_big_nonsep_train.csv")
(xTest,yTest) = getData("data_big_nonsep_test.csv")


iterList = [10,20,50,100,200,500]



for it in iterList:
    print("Numer of its= {}".format(it))
    train_and_plot(xTrain,yTrain,xTest,yTest,stochast_train_w,niter = it)


#w2 = train_and_plot(xTrain,yTrain,xTest,yTest,batch_train_w,niter = 100)
