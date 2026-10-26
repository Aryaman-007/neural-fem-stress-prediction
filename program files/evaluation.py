from data_processing import X_test_tensor,y_test_tensor,scaler_y
from model import StressMLP
import torch
import numpy as np
import matplotlib as plt
from sklearn.metrics import mean_absolute_error,root_mean_squared_error,r2_score
model = StressMLP()

model.load_state_dict(torch.load("best_model.pth",weights_only=True))

model.eval()

with torch.no_grad():
    preds = model(X_test_tensor)

preds_original = scaler_y.inverse_transform(preds.numpy())
y_test_original = scaler_y.inverse_transform(y_test_tensor.numpy())

stress_components = ["SXX","SYY","TXY"]

for i,component in enumerate(stress_components):
    mae = mean_absolute_error(y_test_original[:,i],preds_original[:,i])
    rmse = root_mean_squared_error(y_test_original[:,i],preds_original[:,i])
    r2 = r2_score(y_test_original[:,i],preds_original[:,i])

    print(f'{component}: ')
    print(f"  MAE:  {mae:.4f} Pa")
    print(f"  RMSE: {rmse:.4f} Pa")
    print(f"  R2:   {r2:.6f}")
# print(preds_original.shape,y_test_original.shape)

stress_components = ["SXX", "SYY", "TXY"]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, component in enumerate(stress_components):
    actual = y_test_original[:, i]
    predicted = preds_original[:, i]

    axes[i].scatter(actual, predicted, alpha=0.2, s=5)

    lower = min(actual.min(), predicted.min())
    upper = max(actual.max(), predicted.max())
    axes[i].plot(
        [lower, upper],
        [lower, upper],
        linestyle="--"
    )

    axes[i].set_title(f"{component}: Actual vs Predicted")
    axes[i].set_xlabel("ANSYS stress (Pa)")
    axes[i].set_ylabel("Predicted stress (Pa)")
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()