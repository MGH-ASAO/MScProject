import numpy as np
import matplotlib.pyplot as plt
import torch

# 测试NumPy和Matplotlib
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.show()

# 测试PyTorch
print(torch.__version__)
