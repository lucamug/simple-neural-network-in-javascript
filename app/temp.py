# Running this in https://repl.it for thesting
# https://repl.it/F0Jd
# https://github.com/lucamug/simple-neural-network-in-javascript
import numpy as np

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        # return x*(1-x)
        return "ciao"
        
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

print (syn0)

for iter in range(1):
  l0 = X
  l1 = nonlin(np.dot(l0,syn0))
  l1_error = y - l1;
  print (y)
  print (l1)
  print (l1_error)




