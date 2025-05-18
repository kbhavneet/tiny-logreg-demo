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

