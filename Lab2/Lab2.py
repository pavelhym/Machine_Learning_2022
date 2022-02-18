from pickle import FALSE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

'''Create data'''

z = []
x = []
y = []
for x_i in range(0,101):
    x = x_i/100
    for y_i in range(0,101):
        y = y_i/100
        z.append(3*x**2 + x*y + 2*y**2 - x - 4*y)


def determenisic(point):
    return 3*point[0]**2 + point[0]*point[1] + 2*point[1]**2 - point[0] - 4*point[1] 



a_varianst = np.linspace(-1,2,100)
b_variants = np.linspace(-1,2,100)
[a_grid, b_grid] = np.meshgrid(a_varianst,b_variants)
f_grid =  determenisic([a_grid,b_grid])
fig = plt.figure(figsize= [12.8, 9.6])
graph = fig.add_subplot(111, projection='3d')
risunok = graph.plot_surface( a_grid,b_grid,f_grid, cmap=cm.coolwarm,alpha=0.7)
graph.view_init(30, 135)
graph.set_xlabel('X axis')
graph.set_ylabel('Y axis')
graph.set_zlabel('Z')
fig.colorbar(risunok, shrink=0.2, aspect=10)
plt.title("Function z")
plt.show()




def gradient_descent( num_iterations=1000, x_init = 1, y_init = 1, eps = 0.001,learn_rate = 0.001):
    x_list= []
    y_list = []
    point = np.array([x_init,y_init])

    for i in range(num_iterations):
        point0 = point
        grad_x, grad_y =  6*point[0] + point[1] - 1, point[0] + 4*point[1] - 4
        
        x_t = point[0] - learn_rate * grad_x
        y_t = point[1] - learn_rate * grad_y
        point = np.array([x_t,y_t])
        if all(np.abs(point - point0) < eps):
            break
        #print(a)
        x_list.append(x_t)
        y_list.append(y_t)
       
    return point, x_list, y_list


point,x_list, y_list = gradient_descent( num_iterations=10000, x_init = 1, y_init = 1, eps = 0.00001,learn_rate = 0.01)

np.min(z)
determenisic(point)

#Adam 

def ADAM(num_iterations=1000, x_init = 1, y_init = 1, eps = 0.0001,B_1 = 0.9, B_2 = 0.999 ,learn_rate = 0.001):
    x_list=[]
    y_list=[]
    point = np.array([x_init,y_init])

    S_t = np.array([0,0])
    V_t = np.array([0,0])
    for i in range(num_iterations):

        point0 = point
        g = np.array([6*point[0] + point[1] - 1, point[0] + 4*point[1] - 4])
        V_t1 = B_1*V_t + (1-B_1)*(g)
        S_t1 = B_2*S_t + (1-B_2)*(g**2)
        #S_t1_corr = S_t1/(1-)
        V_t = V_t1
        S_t = S_t1

        point = point0 - learn_rate*V_t1/(np.sqrt(S_t1) + eps)

        x_list.append(point[0])
        y_list.append(point[1])
    return point, x_list, y_list


point,x_list, y_list = ADAM()