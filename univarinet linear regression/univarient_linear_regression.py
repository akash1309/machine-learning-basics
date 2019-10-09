import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

theta0 = 5
theta1 = 5
x = []
y = []
y_pred = []
m = 50
alpha = 0.00001
iterations = 10000

#our linear function
def h(i):
    #global theta0, theta1
    return ((x[i] * theta1) + theta0)


#cost function
def cost_function():
    cost_func = 0
    for i in range(m):
        cost_func += (h(i)-y[i])**2
    cost_func = cost_func/(2*m)
    return cost_func

#generating the sample dataset
def generate_random_set():
    interval = 15
    start = 0
    end = 10
    for i in range(m):
        x.append(random.randint(start,end))
        y.append(random.randint(start,end))
        start = end
        end += interval


#upgarding the value of theta0
def t0():
    global theta0
    deviation = 0
    for i in range(m):
        deviation += (h(i)- y[i])
    deviation = (alpha * deviation)/m;
    return theta0 - deviation

#upgrading the value of theta1
def t1():
    global theta1
    deviation = 0
    for i in range(m):
        deviation += ((h(i) - y[i]) * x[i])
    deviation = (alpha * deviation)/m;
    return theta1 - deviation


def univarient_linear_regression():
    global theta0,theta1
    for i in range(iterations):
        temp0 = t0()
        temp1 = t1()
        theta0 = temp0
        theta1 = temp1
        #print(str(theta0)+ " " + str(theta1))

def graph_plotting():
    plt.scatter(x, y, color = 'red')
    plt.plot(x, y_pred , color = 'blue')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('univarient_linear_regression')
    plt.show()


def main():
    generate_random_set()
    univarient_linear_regression()
    for i in range(m):
        y_pred.append(h(i))

    graph_plotting()

main()
