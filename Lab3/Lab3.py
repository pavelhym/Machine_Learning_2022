import numpy as np
import matplotlib.pyplot as plt
import scipy
%matplotlib inline
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score
import random


bioresponce = pd.read_csv('F:/Documents/ITMO/1курс/Machine_Learning_2022/Lab1/Task_1/bioresponse.csv', header=0, sep=',')

#prepare data
train_data, test_data,  = train_test_split(bioresponce, test_size = 0.25, random_state = 1)
X_train =  train_data.iloc[:,1:] 
Y_train =  train_data.iloc[:,0]
X_test =    test_data.iloc[:,1:]                                                                       
Y_test =  test_data.iloc[:,0]

Y_train = np.array(Y_train)
Y_train.shape
Y_train = Y_train.reshape(1,len(Y_train))

Y_test = np.array(Y_test)
Y_test.shape
Y_test = Y_test.reshape(1,len(Y_test))

X_train = X_train.T
X_test = X_test.T

# sigmoid

def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """
    s = 1./(1.+np.exp(-z))
    s = np.minimum(s, 0.9999)  # Set upper bound
    s = np.maximum(s, 0.0001)  # Set lower bound
    
    return s




def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    w = np.zeros((dim,1))
    b = 0.
    
    return w, b


dim = X_train.shape[0]
w, b = initialize_with_zeros(dim)



# propagate



def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size which equals the number of features
    b -- bias, a scalar
    X -- data 
    Y -- true "label" vector (containing 0 and 1) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    
    """

    m = X.shape[1]
    #print('number of objects = ',len(X))
    
    # FORWARD PROPAGATION (FROM X TO COST)
    A = sigmoid(np.dot(w.T,X)+b)                              # compute activation
    cost = -(1./m)*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A),axis=1)   # compute cost
    
    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = (1./m)*np.dot(X,(A-Y).T)
    db = (1./m)*np.sum(A-Y,axis=1)

    grads = {"dw": dw,
             "db": db}
    
    return grads, cost


grads, cost = propagate(w, b, X_train, Y_train)



def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array 
    b -- bias, a scalar
    X -- data 
    Y -- true "label" vector (containing 0 and 1), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    """
    
    costs = []
    
    for i in range(num_iterations):
                
        # Cost and gradient calculation 
        grads, cost = propagate(w,b,X,Y)
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule
        w -=learning_rate*dw
        b -=learning_rate*db
        
        # Record the costs
        costs.append(cost)
        
        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs



params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations= 1000, learning_rate = 0.005, print_cost = True)


# predict

def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array 
    b -- bias, a scalar
    X -- data 
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''

    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    # Compute vector "A" predicting the probabilities 
    A = sigmoid(np.dot(w.T,X)+b)
    
    for i in range(A.shape[1]):
        
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if (A[0,i]<=0.5):
            Y_prediction[0][i]=0
        else:
            Y_prediction[0][i]=1
    
    return Y_prediction

predict(params["w"], params["b"], X_test)

# model

def model(X_train, Y_train, X_test, Y_test,optimizer, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    """
    Builds the logistic regression model by calling the function we've implemented previously
    
    Arguments:
    X_train -- training set represented by a numpy array 
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array 
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """
    
    # initialize parameters with zeros 
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent
    parameters, grads, costs = optimizer(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict test/train set examples
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d


result1 = model(X_train, Y_train, X_test, Y_test,optimizer = optimize,num_iterations = 2000, learning_rate = 0.5, print_cost = False)

import copy

#

def optimize_SGD(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    
    costs = []
    
    for i in range(num_iterations):
                
        # Cost and gradient calculation 
        # Add random
        num = random.randint(200,len(X.T))

        df_before = copy.deepcopy(X)
        df_before.loc[X.shape[0]] = Y.tolist()[0]

        df_after = df_before.sample(n=num,axis='columns')
        X_sub = df_after[0:len(df_after)-1]
        Y_sub = df_after.iloc[-1].tolist()
        Y_sub = np.array(Y_sub).reshape(1,len(Y_sub))




        grads, cost = propagate(w,b,X_sub,Y_sub)
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule
        w -=learning_rate*dw
        b -=learning_rate*db
        
        # Record the costs
        costs.append(cost)
        

    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs




#learning curve
costs = pd.DataFrame()

for l_rate in [0.05,0.1,0.2,0.35,0.5]:
    costs[str(l_rate)] = model(X_train, Y_train, X_test, Y_test,optimizer = optimize, num_iterations = 2000, learning_rate = l_rate, print_cost = False)['costs']

for l_rate in [0.05,0.1,0.2,0.35,0.5]:
    plt.plot(costs[str(l_rate)], label = "learning rate = " + str(l_rate))
plt.legend()
plt.xlabel("iteration")
plt.ylabel("cost")
plt.title("Gradient descent optimizer")
plt.show()


costs_SGD = pd.DataFrame()

for l_rate in [0.05,0.1,0.2,0.35,0.5]:
    print(str(l_rate))
    costs_SGD[str(l_rate)] = model(X_train, Y_train, X_test, Y_test,optimizer = optimize_SGD, num_iterations = 2000, learning_rate = l_rate, print_cost = False)['costs']

result2 = model(X_train, Y_train, X_test, Y_test,optimizer = optimize_SGD, num_iterations = 2000, learning_rate = 0.05, print_cost = False)

for l_rate in [0.05,0.1,0.2,0.35,0.5]:
    plt.plot(costs_SGD[str(l_rate)], label = "learning rate = " + str(l_rate))
plt.legend()
plt.xlabel("iteration")
plt.ylabel("cost")
plt.title("Stochastic Gradient descent optimizer")
plt.show()




#AdaM

optimize(w, b, X_train, Y_train, num_iterations= 1000, learning_rate = 0.005, print_cost = True)
optimize_ADAM(w, b, X_train, Y_train, num_iterations= 2000, learning_rate = 0.005, print_cost = True)
optimize_ADAM(w, b, X_train, Y_train, 1000, learning_rate, False)
w, b = initialize_with_zeros(dim)

X = X_train
Y = Y_train

def optimize_ADAM(w, b, X, Y, num_iterations, learning_rate, print_cost = False,eps = 0.0001,B_1 = 0.9, B_2 = 0.999):
    
    costs = []
    S_t = np.array([0]*len(w)).reshape(len(w),1)
    S_t_b = 0
    V_t = np.array([0]*len(w)).reshape(len(w),1)
    V_t_b = 0
    for i in range(num_iterations):

        grads, cost = propagate(w,b,X,Y)
        
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]


        V_t1 = B_1*V_t + (1-B_1)*(dw)
        S_t1 = B_2*S_t + (1-B_2)*(dw**2)

        V_t_b1 = B_1*V_t_b + (1-B_1)*(db)
        S_t_b1 = B_2*S_t_b + (1-B_2)*(db**2)
        V_t = V_t1
        S_t = S_t1
        S_t_b = S_t_b1
        V_t_b = V_t_b1

        # update rule
        w -=learning_rate*V_t1/(S_t1**(1/2) + eps)
        b -=learning_rate*V_t_b1/(np.sqrt(S_t_b1) + eps)
        
        # Record the costs
        costs.append(cost)
        

    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs

result3 = model(X_train, Y_train, X_test, Y_test,optimizer = optimize_ADAM, num_iterations = 2000, learning_rate = 0.05, print_cost = False)


