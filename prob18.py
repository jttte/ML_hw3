from LogisticRegression import *

trainData, Ntrain = LoadData('hw3_train.dat')
testData = LoadData('hw3_test.dat')
# trainData[i][0] : x of data i
# trainData[i][1] : y of data i
eta = 0.01
T = 2000

#train
#wf = LogisticRegression(trainData, Ntrain, T, eta)
#print wf

# test
#Eout = Test(testData, wf)
N = len(testData[0])
#print Eout
#print N
#print Eout / 1.0 / N

#stochastic

#train
eta = 0.001
w = StochasticGradient(trainData, Ntrain, T, eta)
print w
E_sto = Test(testData, w)
print E_sto / 1.0 / N