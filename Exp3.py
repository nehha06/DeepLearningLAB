from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

X,y = load_digits(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=5)

model = RandomForestClassifier(n_estimators=50)
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print("Accuracy:",accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
