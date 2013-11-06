from numpy  import *
import math
import random

def LoadData (filename):
        f = open(filename, 'r')
        lines = f.readlines()
        N = len(lines)
        dataList = []

        for i in range(N):
                str = lines[i].strip().split(' ')
                d = len(str) - 1
                x = [float(str[i]) for i in range(d)]
                y = int(str[d])
                #data = array([x, y])
                #dataList = vstack(( dataList, data))
                data = [x, y]
                dataList.append(data)
        return dataList, N
#def LoadData


def LogisticRegression(datalist, N, T, eta):
        dim = len(datalist[0][0])
        wf = zeros ( dim )
        for i in range(T):
                wf = wf - eta * GradientEin(wf, dim, N, datalist)
        #end for

        return wf

#def LodisticRegression


def GradientEin(w, dim, N, data):
        Ein = w
        for i in range(N):
                tmp1 = Theta( dot(w, data[i][0]) *  (-data[i][1]) ) * (-data[i][1])
                #tmp2 = [ x * (-data[i][1]) for x in data[i][0] ]
 
                ein = [ x * tmp1 for x in data[i][0] ]
                Ein = Ein + ein
        #end for
        Ein /= N

        return Ein

#GradientEin 


def Theta(s):
        return 1 / ( 1 + math.exp(-s))
#end Theta


def Test(data, wf):
        error = 0
        for i in range(len(data[0])):
                if( dot(wf, data[0][i][0]) * data[0][i][1] <= 0):
                        error += 1
                #end if
        #end for
        return error
#end Test


def StochasticGradient(data, N, T, eta):
        w = zeros ( len(data[0][0]) )
        list = arange(N)
        #random.shuffle(list)
        for i in range(T):
                i = i % N
                #tmp1 = dot(w, data[ list[i] ][0]) *  (-data[ list[i] ][1])
                #tmp2 = [ x * (eta * Theta(tmp1) * (-data[ list[i] ][1])) for x in data[ list[i] ][0] ]
                tmp1 = eta * Theta( dot(w, data[i][0]) *  (-data[i][1]) ) * (-data[i][1])
                tmp2 = [ x * tmp1 for x in data[i][0] ]
                w = w + tmp2
        #end for
        return w
#end StochasticGradient
