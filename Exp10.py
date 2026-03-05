import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

X,y = load_iris(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.45,random_state=3)

model = LogisticRegression(max_iter=100)
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

cm = confusion_matrix(y_test,y_pred)

sns.heatmap(cm,annot=True,cmap="viridis")
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
