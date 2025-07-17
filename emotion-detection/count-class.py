import numpy as np
data = np.loadtxt("emotion-detection/data.txt")
y = data[:, -1]
print(np.unique(y))  # Should print all unique class labels
print(y.dtype) 