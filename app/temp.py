# Running this in https://repl.it for thesting
# https://repl.it/F0Jd
# https://github.com/lucamug/simple-neural-network-in-javascript
import numpy as np

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
    
# input dataset
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])
    
# output dataset            
y = np.array([[0,0,1,1]]).T
np.random.seed(1)
syn0 = 2*np.random.random((3,1)) - 1
# syn0 = np.array([[0.5], [0], [-0.5]])


for iter in range(1):
  l0 = X
  l1 = nonlin(np.dot(l0,syn0))

  print (l0)
  print (syn0)
  print (np.dot(l0,syn0))
  print (nonlin(np.dot(l0,syn0)))


  l1_error = y - l1;
  l1_delta = l1_error * nonlin(l1,True)
  #print (syn0)
  syn0 += np.dot(l0.T,l1_delta)

  # print (l1)
  #print (l1_error)
  #print (l0.T)
  #print (l1_delta)
  #print (np.dot(l0.T,l1_delta))
  #print (syn0)



