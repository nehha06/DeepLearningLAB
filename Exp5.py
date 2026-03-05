from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X,y = load_iris(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(X[:,0].reshape(-1,1),
                                                 X[:,2],
                                                 test_size=0.3)

model = LinearRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print("MSE:",mean_squared_error(y_test,y_pred))
