from Util import *

trainData, Ntrain = LoadData('hw3_train.dat')
testData, Ntest = LoadData('hw3_test.dat')
# trainData[i][0] : x of data i
# trainData[i][1] : y of data i
print('finish loading data!')

eta = 0.001
T = 2000

#train
wf = LogisticRegression(trainData, Ntrain, T, eta)
print wf
print ('finish training')

# test
Eout = Test(testData, wf)
print Eout / 1.0 / Ntest


#stochastic
#train
eta = 0.001
w = StochasticGradient(trainData, Ntrain, T, eta)
print w
E_sto = Test(testData, w)
print E_sto / 1.0 / Ntest