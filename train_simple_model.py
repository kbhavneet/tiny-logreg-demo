"""
tiny logistic regression on a toy 2-d dataset
make two clouds of points and let the model learn to tell them apart
"""

import numpy as np
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt

np.random.seed(0)                          
class0 = np.random.randn(50, 2) + np.array([0, 0])   
class1 = np.random.randn(50, 2) + np.array([5, 5])   
X = np.vstack([class0, class1])             
y = np.array([0]*50 + [1]*50)               

# shuffle so classes are mixed
perm = np.random.permutation(len(y)) 
X, y = X[perm], y[perm]

model = SGDClassifier(loss="log", max_iter=1, learning_rate="constant", eta0=0.1, random_state=0, tol=None, warm_start=True)

accuracies = []
for epoch in range(10):
    model.partial_fit(X, y, classes=np.array([0, 1]))
    acc = model.score(X, y)
    accuracies.append(acc)