import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def mean_squared_error(y_true,y_pred):
    return np.sum((y_true-y_pred)**2)/len(y_true)

def gradient_descent(x,y,iterations=1500,learning_rate=0.02):

    weight=0
    bias=0
    n=len(x)

    costs=[]

    for i in range(iterations):

        y_pred=weight*x + bias
        cost=mean_squared_error(y,y_pred)
        costs.append(cost)

        dw=-(2/n)*sum(x*(y-y_pred))
        db=-(2/n)*sum(y-y_pred)

        weight=weight-learning_rate*dw
        bias=bias-learning_rate*db

        if i%100==0:
            print("Iteration",i,"Cost:",cost)

    plt.plot(costs,'r')
    plt.title("Cost vs Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.show()

    return weight,bias


X=np.array([30,40,45,50,60,65,70,75,80,85])
Y=np.array([35,50,55,60,75,80,85,90,95,100])

scaler=StandardScaler()
X=scaler.fit_transform(X.reshape(-1,1)).flatten()

w,b=gradient_descent(X,Y)

print("Weight:",w)
print("Bias:",b)
