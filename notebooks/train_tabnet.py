# train_tabnet.py

import sys, os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
from src.tabnet import TabNet  # ✅ Keep this style

# 1. Load Adult Income Dataset
X, y = fetch_openml("adult", version=2, as_frame=True, return_X_y=True)

# 2. Preprocess
# Separate numeric and categorical
X_numeric = X.select_dtypes(include=[np.number])
X_categorical = X.select_dtypes(include=[object])

# Ensure categorical features are not empty
if not X_categorical.empty:
    X_categorical = X_categorical.apply(LabelEncoder().fit_transform)
    X_categorical = pd.DataFrame(X_categorical, index=X.index)
else:
    # Dummy placeholder if no categorical features (avoids hstack crash)
    X_categorical = pd.DataFrame(index=X.index)

# Combine
X_processed = np.hstack([X_numeric.values, X_categorical.values])

# Scale numeric features only
scaler = StandardScaler()
X_processed[:, :X_numeric.shape[1]] = scaler.fit_transform(X_processed[:, :X_numeric.shape[1]])

# Encode labels (<=50K or >50K)
y = LabelEncoder().fit_transform(y)

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# 4. Convert to torch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 5. Initialize TabNet model
input_dim = X_train_tensor.shape[1]
n_classes = len(np.unique(y))
model = TabNet(input_dim=input_dim, n_d=16, n_steps=3, output_dim=n_classes)

# 6. Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
epochs = 10

# 7. Training loop
for epoch in range(epochs):
    model.train()
    logits, _ = model(X_train_tensor)
    loss = criterion(logits, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch + 1} - Loss: {loss.item():.4f}")

# 8. Evaluation
model.eval()
with torch.no_grad():
    preds = model(X_test_tensor)[0].argmax(dim=1)
    accuracy = (preds == y_test_tensor).float().mean().item()
    print(f"\n✅ Test Accuracy: {accuracy:.4f}")

# 9. Global Feature Importance
import matplotlib.pyplot as plt

# Re-run forward pass on test set to get masks
model.eval()
with torch.no_grad():
    _, aggregated_mask = model(X_test_tensor)

# Average mask across all test samples
avg_mask = aggregated_mask.mean(dim=0).cpu().numpy()

# Get feature names (numeric + categorical)
feature_names = list(X_numeric.columns) + list(X_categorical.columns)

# Plot
plt.figure(figsize=(12, 5))
plt.bar(range(len(avg_mask)), avg_mask)
plt.xticks(range(len(avg_mask)), feature_names, rotation=90)
plt.xlabel("Feature")
plt.ylabel("Average Attention Mask Value")
plt.title("Global Feature Importance (TabNet Aggregated Mask)")
plt.tight_layout()
plt.show()
