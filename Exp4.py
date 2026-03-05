import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

np.random.seed(1)

X = np.sort(np.random.rand(25))
y = np.sin(2*np.pi*X) + np.random.randn(25)*0.15

poly = PolynomialFeatures(10)
X_poly = poly.fit_transform(X.reshape(-1,1))

model = LinearRegression()
model.fit(X_poly,y)

plt.scatter(X,y)
plt.plot(X,model.predict(X_poly),color='red')
plt.title("Overfitting Example")
plt.show()
