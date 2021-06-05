#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# Read dataset
df = pd.read_csv("iris.csv")
print('number of data :',df.shape[0])
print('number of feature :', df.shape[1])



g = df.groupby(['Species']).groups

# Sepal 
c1 = [df.loc[g["Iris-setosa"], 'SepalLengthCm'], df.loc[g["Iris-setosa"], 'SepalWidthCm']]
c2 = [df.loc[g["Iris-versicolor"], 'SepalLengthCm'], df.loc[g["Iris-versicolor"], 'SepalWidthCm']]
c3 = [df.loc[g["Iris-virginica"], 'SepalLengthCm'], df.loc[g["Iris-virginica"], 'SepalWidthCm']]
plt.scatter(c1[0], c1[1], color='r', s=12, label="Iris-setosa")
plt.scatter(c2[0], c2[1], color='b', s=12, label="Iris-versicolor")
plt.scatter(c3[0], c3[1], color='g', s=12, label="Iris-virginica")
        
plt.xlabel('SepalLengthCm')
plt.ylabel('SepalWidthCm')
plt.legend()
plt.show()

# Petal 
c1 = [df.loc[g["Iris-setosa"], 'PetalLengthCm'], df.loc[g["Iris-setosa"], 'PetalWidthCm']]
c2 = [df.loc[g["Iris-versicolor"], 'PetalLengthCm'], df.loc[g["Iris-versicolor"], 'PetalWidthCm']]
c3 = [df.loc[g["Iris-virginica"], 'PetalLengthCm'], df.loc[g["Iris-virginica"], 'PetalWidthCm']]
plt.scatter(c1[0], c1[1], color='r', s=12, label="Iris-setosa")
plt.scatter(c2[0], c2[1], color='b', s=12, label="Iris-versicolor")
plt.scatter(c3[0], c3[1], color='g', s=12, label="Iris-virginica")
        
plt.xlabel('PetalLengthCm')
plt.ylabel('PetalWidthCm')
plt.legend()
plt.show()



# perceptron
class Perceptron():        
    def fit(self, x, y, lr=0.25, epoch=10):
        '''
        Fit the data
        '''
        np.random.seed(42)
        self.w = np.random.randn(x.shape[1], 1)
        for _ in range(epoch):
            y_hat = self.sign(np.dot(x, self.w))
            self.w = self.w + lr * np.dot(x.T, y - y_hat)
            
            
    def fit_plot(self, x, y, lr=0.25, epoch=10):
        '''
        Fit the data with the plot of process
        ps, only for AND testing
        '''
        np.random.seed(42)
        self.w = np.random.randn(x.shape[1], 1)
        for _ in range(epoch):
            y_hat = self.sign(np.dot(x, self.w))
            self.w += lr * np.dot(x.T, y - y_hat)
            plot_boundary(x, y, self.w)
    
    def sign(self, z):
        '''
        Activation function
        '''
        z[z>0] = 1
        z[z<=0] = 0
        return z
    
    def predict(self, x):
        return self.sign(np.dot(x, self.w))
    
    def accuracy(self, y_hat, y):
        return np.sum(y_hat==y) / y.shape[0]



'''Test perceptron class'''

# # visualize
# def plot_label(x, y):
#     for i in range(len(y)):
#         if y[i]==1:
#             plt.scatter(x[i,0], x[i,1], color='r')
#         else:
#             plt.scatter(x[i,0], x[i,1], color='b')
            
# def plot_boundary(x, y, w):
#     plot_label(x, y)
#     x_boundary = np.linspace(-1,5)
#     y_boundary = -w[0]/w[1] * x_boundary - w[2]/w[1]
#     plt.plot(x_boundary, y_boundary)
#     plt.xlim(-0.5,1.5)
#     plt.ylim(-0.5,1.5)
#     plt.show()

# AND gate
# x = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
# y = np.array([0,0,0,1]).reshape(-1,1)
# plot_label(x, y)
# plt.show()

# # build model
# perceptron = Perceptron()
# perceptron.fit_plot(x, y, epoch=20)
# print(perceptron.accuracy(perceptron.predict(x), y))



# step 1 : classify 'Iris-setosa' and other
import copy
y = df['Species'].copy()
y.loc[df['Species']=='Iris-setosa'] = 1
y.loc[df['Species']!='Iris-setosa'] = 0
y = y.values.reshape(-1,1)
x = df[['PetalLengthCm', 'PetalWidthCm']].values
x = np.concatenate([x, np.ones((x.shape[0],1))], axis=1)

p1 = Perceptron()
p1.fit(x, y, epoch=200)
y_hat1 = p1.predict(x)
print("Iris-setosa accuracy :", p1.accuracy(y_hat1, y))


plt.scatter(c1[0], c1[1], color='r', s=12, label="Iris-setosa")
plt.scatter(c2[0], c2[1], color='b', s=12, label="Iris-versicolor")
plt.scatter(c3[0], c3[1], color='g', s=12, label="Iris-virginica")


x_boundary = np.linspace(-1,10)
# first boundary
w = p1.w
y1_boundary = -w[0]/w[1] * x_boundary - w[2]/w[1]

plt.plot(x_boundary, y1_boundary)
plt.xlim(0.5, 8)
plt.ylim(-0.5, 3)
plt.xlabel('PetalLengthCm')
plt.ylabel('PetalWidthCm')
plt.legend()
plt.show()


# step 2 : classify 'Iris-versicolor' and other
y = df[y_hat1==0].copy()
y = y["Species"]
y.loc[y=='Iris-versicolor'] = 1
y.loc[y!= 1] = 0
y = y.values.reshape(-1,1)

x = df[y_hat1==0].copy()
x = x[['PetalLengthCm', 'PetalWidthCm']].values
x = np.concatenate([x, np.ones((x.shape[0],1))], axis=1)

p2 = Perceptron()
p2.fit(x, y, epoch=500)
y_hat2 = p2.predict(x)
print("Iris-versicolor accuracy :", p2.accuracy(y_hat2, y))


# plot boundary

g = df.groupby(['Species']).groups
c1 = [df.loc[g["Iris-setosa"], 'PetalLengthCm'], df.loc[g["Iris-setosa"], 'PetalWidthCm']]
c2 = [df.loc[g["Iris-versicolor"], 'PetalLengthCm'], df.loc[g["Iris-versicolor"], 'PetalWidthCm']]
c3 = [df.loc[g["Iris-virginica"], 'PetalLengthCm'], df.loc[g["Iris-virginica"], 'PetalWidthCm']]
plt.scatter(c1[0], c1[1], color='r', s=12, label="Iris-setosa")
plt.scatter(c2[0], c2[1], color='b', s=12, label="Iris-versicolor")
plt.scatter(c3[0], c3[1], color='g', s=12, label="Iris-virginica")


x_boundary = np.linspace(-1,10)
# first boundary
w = p1.w
y1_boundary = -w[0]/w[1] * x_boundary - w[2]/w[1]
# second boundary
w = p2.w
y2_boundary = -w[0]/w[1] * x_boundary - w[2]/w[1]

plt.plot(x_boundary, y1_boundary)
plt.plot(x_boundary, y2_boundary)
plt.xlim(0.5, 8)
plt.ylim(-0.5, 3)
plt.xlabel('PetalLengthCm')
plt.ylabel('PetalWidthCm')
plt.legend()
plt.show()





