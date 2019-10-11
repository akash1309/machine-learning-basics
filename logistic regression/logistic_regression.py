import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_csv('Social_Network_Ads.csv')
X_temp = dataset.iloc[:,[2,3]].values
row,column = X_temp.shape
X0 = np.ones((row, 1))
X = np.hstack((X0,X_temp))

y = dataset.iloc[:,4].values

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

theta = np.ones((column+1,1))
theta = theta.reshape((-1,1))

m = row
features = column+1
iterations = 100000
cost = np.zeros(iterations)
learning_rate = 0.001
alpha = learning_rate

def h(i):
    global theta,X
    mat_mul = (-1)*np.dot(np.transpose(theta),X[i])
    mat_mul = np.exp(mat_mul)
    return 1/(1+mat_mul)

def cost_function():
    global y
    cost = 0
    for i in range(m):
        cost += y[i] * np.log(h(i)) + (1 - y[i]) * np.log(1 - h(i))
    cost = (-1)*cost/m
    return cost

def update_parameters():
    global theta
    deviation = [0,0,0]
    for i in range(features):
        for j in range(m):
            deviation[i] += (h(j) - y[j]) * X[j][i]

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


def main():
    global X,y
    gradient_descent()
    y_pred = np.zeros((row,1))
    for i in range(m):
        y_pred[i] = h(i)

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, y_pred.round())
    print("confusion_matrix\n\n")
    print(cm)


main()
