import numpy as np



def sigma(w,x):
    x = np.matrix(x)
    return  float(1/(1+np.exp((-1)*(w*x.T))))

def LSimple(w):
    if not len(w) == 2:
        print('Why')
        return -1
    return (pow((sigma(w,[1,0])-1),2) + pow(sigma(w,[0,1]) ,2) + pow((sigma(w,[1,1])-1),2))


    #return pow( (sigma(w,[1, 0]) − 1), 2) + pow(sigma(w, [0, 1]),2) + pow((sigma(w, [1, 1]) − 1) ,2)
