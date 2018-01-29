import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

dataset = load_digits()
X, y = dataset.data, dataset.target

# Plot histogram of classes. Notice that classes are imbalanced.
bins = dataset.target_names
counts = np.bincount(dataset.target)
print counts

for class_name, class_count in zip(dataset.target_names, np.bincount(dataset.target)):
	print(class_name, class_count)

plt.bar(bins, counts, align='center')
plt.xticks(bins, bins)
plt.show()