import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset,DataLoader
df = pd.read_csv('beam_dataset_dpf.csv')

# print(df["DP"].unique())
rng = np.random.default_rng(42)
dp_ids = df["DP"].unique()
rng.shuffle(dp_ids)
train_ids = dp_ids[:400]
test_ids = dp_ids[450:]
dev_ids = dp_ids[400:450]

train_set = df[df["DP"].isin(train_ids)]
test_set = df[df["DP"].isin(test_ids)]
dev_set = df[df["DP"].isin(dev_ids)]

X_train = train_set[["X","Y","E","nu","F"]]
y_train = train_set[["SXX","SYY","TXY"]]
# print(X_train.shape,y_train.shape)

X_test = test_set[["X","Y","E","nu","F"]]
y_test = test_set[["SXX","SYY","TXY"]]
# print(X_test.shape,y_test.shape)

X_dev = dev_set[["X","Y","E","nu","F"]]
y_dev = dev_set[["SXX","SYY","TXY"]]
# print(X_dev.shape,y_dev.shape)

scaler_X = StandardScaler()
scaler_X.fit(X_train)
X_train_scaled = scaler_X.transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
X_dev_scaled = scaler_X.transform(X_dev)

scaler_y = StandardScaler()
scaler_y.fit(y_train)
y_train_scaled = scaler_y.transform(y_train)
y_test_scaled = scaler_y.transform(y_test)
y_dev_scaled = scaler_y.transform(y_dev)

X_train_tensor = torch.tensor(X_train_scaled,dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled,dtype=torch.float32)

X_test_tensor = torch.tensor(X_test_scaled,dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled,dtype=torch.float32)

X_dev_tensor = torch.tensor(X_dev_scaled,dtype=torch.float32)
y_dev_tensor = torch.tensor(y_dev_scaled,dtype=torch.float32)

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=2048,
    shuffle=True
)

dev_dataset = TensorDataset(
    X_dev_tensor,
    y_dev_tensor
)

dev_loader = DataLoader(
    dev_dataset,
    batch_size=2048,
    shuffle=True
)