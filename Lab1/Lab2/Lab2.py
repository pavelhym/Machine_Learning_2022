import numpy as np


def gradient_descent(fx_grad, eta=0.0001, num_iterations=1000, a_init = 1, b_init = 1, eps = 0.001):
    
    a_list=[]
    b_list=[]
    a_list.append(a_init)
    b_list.append(b_init)
    a = a_init
    b = b_init
    f_calc = 0
    a0 = np.array([a,b])
    a_n = np.array([1,1])
    beta = 0.001

    for i in range(num_iterations):
        if i ==0:
            beta = np.array([0.001,0.001])
        else:
            beta = np.abs((a_n - a0)*(np.array(fx_grad(a_n)) - np.array(fx_grad(a0))))/np.linalg.norm((np.array(fx_grad(a_n)) - np.array(fx_grad(a0))))**2
        grad_a, grad_b =  fx_grad([a,b])[0], fx_grad([a,b])[1]
        f_calc += 2
        a0 = np.array([a,b])
        a = a - beta[0] * grad_a
        b = b - beta[1] * grad_b
        a_n = np.array([a,b]) 
        if all(np.abs(a_n - a0) < eps):
            num_iter = i +1
            break
        #print(a)
        a_list.append(a)
        b_list.append(b)
        num_iter = i +1
       
    return a_list[-1], b_list[-1], num_iter, f_calc