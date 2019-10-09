import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#data reading from a file
dataset = pd.read_excel('data.xls')

#taking first 5 columns of the dataset
X_temp = dataset.iloc[:,:-1].values

#taking the last column as output column
y = dataset.iloc[:,5].values
y = y.reshape(-1,1)

#adding a column of 1's in the X_temp matrix
row, column = X_temp.shape
X0 = np.ones((row,1))
X = np.hstack((X0,X_temp))

"""
Feature scaling
we are doing feature scaling because otherwise it will give double_scalers error as gradient_descent
will be changing at a faster rate ... so it will be better to have dataset values between
-1 and 1
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)


#here features used = 3 + 1(for theta0)
#we have to predict the profit
#we have 27 training examples so m = 27

m = 27
features = 6
#equation = theta0 * x0 + theta1 * x1 + theta2 * x2 + theta3 * x3
#IN the above equation x0 is always 1
theta = [5,5,5,5,5,5]  #6 elements as there are 6 features
learning_rate = 0.01
alpha = learning_rate
iterations = 10000
cost = np.zeros(iterations)



def h(i):
    return theta[0] * X[i][0] + theta[1] * X[i][1] + theta[2] * X[i][2] + theta[3] * X[i][3]

def cost_function():
    cost = 0
    for i in range(m):
        cost += (h(i) - y[i][0])**2
    cost /= (2*m)
    return cost

def update_parameters():
    global theta
    deviation = [0,0,0,0,0,0]
    for i in range(features):
        for j in range(m):
            deviation[i] += (h(j) - y[j][0]) * X[j][i]

    for i in range(features):
        deviation[i] = (alpha * deviation[i])/m
        deviation[i] = theta[i] - deviation[i]

    return deviation


def gradient_descent():
    global theta
    for i in range(iterations):
        temp = update_parameters()
        theta = temp
        cost[i] = cost_function()

def graph_plotting():
    #plot the cost
    global cost, iterations
    fig, ax = plt.subplots()
    ax.plot(np.arange(iterations), cost, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs. Training Epoch')
    plt.show()

def main():
    global X,y
    gradient_descent()
    y_pred = []
    for i in range(m):
        y_pred.append(h(i))

    """we cannot plot graph for Multivarient_linear_regression as there are many features.
       what we can do is to plot graph for every feature and then compare the values
       Here, I am plotting graph between cost and no. of iterations."""

    graph_plotting()
    #checking the real output and predicted output
    print("original output   vs     predicted output\n")
    for i in range(m):
        print(str(y[i][0]) + " " + str(y_pred[i]) + "\n")


main()
