import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X,y=make_classification(n_samples=120,n_features=2,n_informative=2,
n_redundant=0,n_clusters_per_class=1,random_state=10)

model=LogisticRegression()
model.fit(X,y)

y_pred=model.predict(X)

accuracy=accuracy_score(y,y_pred)

print("Accuracy:",accuracy)

plt.scatter(X[:,0],X[:,1],c=y,cmap='coolwarm')

coef=model.coef_[0]
intercept=model.intercept_

x_vals=np.linspace(X[:,0].min(),X[:,0].max(),100)

y_vals=-(coef[0]*x_vals+intercept)/coef[1]

plt.plot(x_vals,y_vals,'k--')

plt.title("Linear Separability")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
