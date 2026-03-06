from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

X,y = load_iris(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

model = SVC(kernel='linear')
model.fit(x_train,y_train)S

y_pred = model.predict(x_test)

print("Accuracy:",accuracy_score(y_test,y_pred))
