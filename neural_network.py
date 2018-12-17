"""*********************************************************

NAME:     Neural Network

AUTHOR:   Paul Haddon Sanders IV, Ph.D

Date:     12/17/2018

*********************************************************"""

############################################################
###               CLASS : NEURAL NETWORK                 ###
############################################################

class neural_network:
    """Neural Network.

    Parameters
    ----------
    n_nodes : int
      Number of nodes in the hidden layer.
    eta : float
      Learing rate (between 0.0 and 4.0).
    lambd : float
      Regularization parameter (between 0.0 and 100.0).
    n_iter : int
      Number of passes over the training set.
    
    Attributes
    ----------
    w1_ : 2d-array, shape = [n_nodes,M+1]
      Weights of hidden layer after fitting, where n_nodes is
      the number of hidden nodes and M+1 is the number
      of features including the bias unit.
    w2_ : 2d-array, shape = [K,n_nodes+1]
      Weights of output layer after fitting, where K is the
      number of classifications and n_nodes+1 is the number
      of hidden nodes including the bias unit.
    cost_ : list
      Logistic regression cost function value in each epoc.

    """

    print("")
    print("   NEURAL NETWORK INITIALIZED")
    print("   CREATED BY : PAUL SANDERS")
    print("")
    
    def __init__(self,n_nodes=25,eta=3.0,lambd=0.0,n_iter=1000):
        self.n_nodes = n_nodes
        self.eta     = eta
        self.lambd   = lambd
        self.n_iter  = n_iter

    def fit(self, X, Y):
        """Fit training data.
        
        X : {array-like}, shape = [S,M]
          Training vectors, where S is the number of samples 
          and M is the number of features.
        Y : {array-like}, shape = [S,K]
          Target values, where S is the number of samples
          and K is the number of classifications.

        Returns
        -------
        self : object

        """

        print('   Please be patient. The cost is being minimized.')
        print('')
        
        S = X.shape[0]
        M = X.shape[1]
        K = Y.shape[1]

        self.w1_ = np.random.normal(loc=0.0,scale=0.7,size=(self.n_nodes,M+1))
        self.w2_ = np.random.normal(loc=0.0,scale=0.7,size=(K,self.n_nodes+1))

        self.cost_ = []

        for i in range(self.n_iter):

            dw1_,dw2_ = self.dcost(X,Y)
            self.w1_ += dw1_
            self.w2_ += dw2_
            self.cost_.append(self.cost(X,Y))
            print('   cost = {}'.format(round(self.cost_[-1],7)),end="\r")

        print('')
        print('\n   Minimization Complete.')
        
    def activation(self,X,theta):
        """Compute logistic sigmoid activation"""

        z = np.dot(X,theta[1:]) + theta[0]

        return 1. / (1. + np.exp(-np.clip(z,-250,250)))

    def predict(self,X):
        """Return class labels after hidden layer evaluation"""

        S = X.shape[0]
        
        hidden_layer = np.zeros((S,self.w1_.shape[0]))
        output_layer = np.zeros((S,self.w2_.shape[0]))

        for i in range(self.w1_.shape[0]):

            hidden_layer[:,i] = self.activation(X,self.w1_[i])   

        for i in range(self.w2_.shape[0]):

            output_layer[:,i] = self.activation(hidden_layer,self.w2_[i])

        return hidden_layer,output_layer

    def cost(self,X,Y):
        """Calculate the cost associate with w1 and w2 on the training set"""

        S = X.shape[0]

        activate = self.predict(X)[1]

        J = -1 / S * (Y * np.log(activate)           \
            + (1-Y) * np.log(1-activate)).sum( )     \
            + 0.5 * self.lambd / S * ((self.w1_[:,1:]**2).sum() \
            + (self.w2_[:,1:]**2).sum())

        return J

    def dcost(self,X,Y):
        """Calculate the updates to the weights using gradient descent"""

        S = X.shape[0]

        a2,a3 = self.predict(X)

        """Calculate w2 weights"""

        delta3 = Y - a3
        Delta2_all = delta3.T.dot(a2)
        Delta2_bias = delta3.T.sum(1)
        Delta2 = np.append([Delta2_bias],Delta2_all.T,axis=0).T

        dw2_ = self.eta * (1 / S * Delta2 + self.lambd / S * self.w2_)

        ###Calculate w1 weights"""

        delta2      = delta3.dot(self.w2_[:,1:]) * a2 * (1 - a2)

        Delta1_all  = delta2.T.dot(X)
        Delta1_bias = delta2.T.sum(1)
        Delta1      = np.append([Delta1_bias],Delta1_all.T,axis=0).T

        dw1_ = self.eta * (1 / S * Delta1 + self.lambd / S * self.w1_)

        return dw1_,dw2_

############################################################
###             END CLASS : NEURAL NETWORK               ###
############################################################

from scipy import io
import numpy as np

### Get data ###############################################

# There are 5000 training examples.
# Each training example is a 20x20 pixel image of the digit.
# Each pixel is represented by a number indicating
# greyscale intensity.
# This 20x20 grid of pixels is unrolled into a
# 400-dimensional vector.
# Then X is a 5000x400 matrix.
# Y represents the labels for each X training matrix.

mat = io.loadmat('ex4data1.mat')
X   = mat['X']
Y   = mat['y']
Y   = Y.reshape(Y.shape[0])

### Reshape Y ##############################################

n_class_labels = np.unique(Y).shape[0]
Y_exp = np.zeros((Y.shape[0],n_class_labels))

for i,val in enumerate(Y):

    # we need to change the Y outputs into an array, since
    # our classifier is with many different options, not
    # only one digit.
    
    index = val - 1
    
    Y_exp[i][index] = 1

Y = np.copy(Y_exp)

### output #################################################

nn = neural_network(n_iter=100)
nn.fit(X,Y)

guess   = nn.predict(X)[1]
correct = round(sum(np.argmax(guess,axis=1) + 1 == \
          np.argmax(Y,axis=1) + 1) / X.shape[0] * 100,1)

print("\n   Amount Correct = {}".format(correct))
print("")

weights = np.append(nn.w1_,nn.w2_)
np.savetxt('weights.csv',weights,delimiter=',')

print("   The weights were saved to the 'weights.csv' file.")
print("")
print("   END")

### END ####################################################
