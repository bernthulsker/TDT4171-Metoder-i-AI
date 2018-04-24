import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def logistic_z(z):
    return 1.0/(1.0+np.exp(-z))

def logistic_wx(w,x):
    return logistic_z(np.inner(w,x))

def sigmoid(w,x):
    return logistic_z(np.inner(w,x))

def gradLSimple(w):
     dL11 = 2*(logistic_wx(w,[1,0])-1)*np.exp(-1*w[0])*pow(logistic_wx(w,[1,0]),2)
     dL12 = 2*(logistic_wx(w,[1,1])-1)*np.exp(-w[0]-w[1])*pow(logistic_wx(w,[1,1]),2)
     dL1  = dL11 + dL12

     dL21 = 2*(logistic_wx(w,[0,1]))*np.exp(-1*w[1])*pow(logistic_wx(w,[0,1]),2)
     dL22 = 2*(logistic_wx(w,[1,1])-1)*np.exp(-w[0]-w[1])*pow(logistic_wx(w,[1,1]),2)
     dL2  = dL21 + dL22

     return np.array([dL1,dL2])

def gradientDescent(ny,wStart,doPlot):
    steps = 0
    wOld = wStart
    wNew = np.zeros((1,2))
    w1History = [wStart[0]]
    w2History = [wStart[1]]
    LHistory = [LSimple(wStart)]
    while True:
        dummy = gradLSimple(wOld)
        dummy[0] = dummy[0] * ny
        dummy[1] = dummy[1] * ny
        wNew = wOld - dummy
        w1History = np.append(w1History,wNew[0])
        w2History = np.append(w2History,wNew[1])
        LHistory = np.append(LHistory,LSimple(wNew))
        steps = steps + 1
        if steps > 1000:
            break
        wOld = wNew


    print ('Stopped serching at w1 = {}, w2 = {}'.format(wNew[0],wNew[1]))
    L = LSimple([wNew[0],wNew[1]])
    print ('Here L was {}'.format(L))
    if doPlot:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        labelText = 'n = {}'.format(ny)
        print(len(w1History))
        ax.plot(w1History, w2History, LHistory, label=labelText)
        ax.legend()
        plt.show()




def LSimple(w):
    if not len(w) == 2:
        print('Why')
        return -1
    return (pow((logistic_wx(w,[1,0])-1),2) + pow(logistic_wx(w,[0,1]) ,2) + pow((logistic_wx(w,[1,1])-1),2))

def plotLSimple(w1,w2,doPlot):
    n1 = len(w1)
    n2 = len(w2)
    L = np.zeros((n1, n2))
    minL = [-1,-1,100000000]

    for i in range(n1):
        for j in range(n2):
            L[i][j] = LSimple([w1[i],w2[j]])
            if L[i][j] < minL[2]:
                minL = [i,j,L[i][j]]
    if doPlot:
        X, Y = np.meshgrid(w1, w2)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, np.matrix(L))
        plt.show()
    print('Min reached at  w1 = {}, w2 = {}    L = {}'.format(w1[minL[0]], w2[minL[1]],minL[2]))

def classify(w,x):
    x=np.hstack(([1],x))
    return 0 if (logistic_wx(w,x)<0.5) else 1
#x_train = [number_of_samples,number_of_features] = number_of_samples x \in R^number_of_features
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
            update_grad = 1 ### something needs to be done here
            w[i] = w[i] + learn_rate ### something needs to be done here
    return w

def batch_train_w(x_train,y_train,learn_rate=0.1,niter=1000):
    x_train=np.hstack((np.array([1]*x_train.shape[0]).reshape(x_train.shape[0],1),x_train))
    dim=x_train.shape[1]
    num_n=x_train.shape[0]
    w = np.random.rand(dim)
    index_lst=[]
    for it in range(niter):
        for i in range(dim):
            update_grad=0.0
            for n in range(num_n):
                update_grad+=(-logistic_wx(w,x_train[n])+y_train[n])# something needs to be done here
            w[i] = w[i] + learn_rate * update_grad/num_n
    return w

w1 = np.arange(-6,6,0.1)
w2 = np.arange(-6,6,0.1)


n = 0.1

wStart = np.array([0,0])
gradientDescent(n,wStart,True)

# def train_and_plot(xtrain,ytrain,xtest,ytest,training_method,learn_rate=0.1,niter=10):
#     plt.figure()
#     #train data
#     data = pd.DataFrame(np.hstack((xtrain,ytrain.reshape(xtrain.shape[0],1))),columns=['x','y','lab'])
#     ax=data.plot(kind='scatter',x='x',y='y',c='lab',cmap=cm.copper,edgecolors='black')
#
#     #train weights
#     w=training_method(xtrain,ytrain,learn_rate,niter)
#     error=[]
#     y_est=[]
#     for i in xrange(len(ytest)):
#         error.append(np.abs(classify(w,xtest[i])-ytest[i]))
#         y_est.append(classify(w,xtest[i]))
#     y_est=np.array(y_est)
#     data_test = pd.DataFrame(np.hstack((xtest,y_est.reshape(xtest.shape[0],1))),columns=['x','y','lab'])
#     data_test.plot(kind='scatter',x='x',y='y',c='lab',ax=ax,cmap=cm.coolwarm,edgecolors='black')
#     print "error=",np.mean(error)
#     return w
