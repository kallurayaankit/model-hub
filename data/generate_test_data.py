import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# Generate a binary classification dataset
X, y = make_classification(n_samples=500, n_features=10, random_state=42)
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
df['target'] = y
df.to_csv('data/hidden_test.csv', index=False)
print("Hidden test set saved to data/hidden_test.csv")