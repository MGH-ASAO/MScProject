import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Load a small portion of MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X, y = mnist.data[:1000], mnist.target[:1000]

# Display a sample image
plt.imshow(X[0].reshape(28, 28), cmap='gray')
plt.title(f"Label: {y[0]}")
plt.show()