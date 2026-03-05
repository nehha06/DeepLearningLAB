from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

X,y = load_iris(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

model = GaussianNB()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

cm = confusion_matrix(y_test,y_pred)

sns.heatmap(cm,annot=True,cmap="Blues")
plt.title("Naive Bayes Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
