from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

X,y = load_wine(return_X_y=True)

y = (y>0).astype(int)

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=2)

model = DecisionTreeClassifier(max_depth=3)
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print("Accuracy:",accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
