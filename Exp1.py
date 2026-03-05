import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

actual = ['Cat','Dog','Dog','Cat','Cat','Dog','Dog','Cat','Dog','Cat']
pred =   ['Cat','Dog','Cat','Cat','Dog','Dog','Dog','Cat','Dog','Cat']

cm = confusion_matrix(actual,pred)

sns.heatmap(cm,annot=True,fmt='g',
            xticklabels=['Cat','Dog'],
            yticklabels=['Cat','Dog'])

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
