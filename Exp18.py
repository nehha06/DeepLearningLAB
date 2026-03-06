import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Generate two class dataset
X, y = make_blobs(n_samples=200, centers=2, random_state=5)

# Create Neural Network
model = MLPClassifier(hidden_layer_sizes=(3,3),
                      activation='identity',
                      learning_rate_init=0.03,
                      max_iter=1000)

# Train model
model.fit(X, y)

# Predict
y_pred = model.predict(X)

# Accuracy
accuracy = accuracy_score(y, y_pred)
print("Accuracy:", accuracy)

# Plot dataset
plt.figure(figsize=(8,6))
plt.scatter(X[:,0], X[:,1], c=y, cmap='coolwarm')

# Plot decision boundary
x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
xx, yy = np.meshgrid(np.linspace(x_min,x_max,100),
                     np.linspace(y_min,y_max,100))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contour(xx, yy, Z, alpha=0.4)

plt.title("Neural Network Two-Class Classification")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
