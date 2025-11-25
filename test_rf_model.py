import pickle
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

with open('trajectory_system/training_data.pkl', 'rb') as f:
    data = pickle.load(f)

X_train, X_test = data['X_train'], data['X_test']
y_train, y_test = data['y_train'], data['y_test']

model = joblib.load('trajectory_system/trajectory_model.pkl')

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax1 = axes[0, 0]
ax1.scatter(y_test, y_pred_test, alpha=0.5, edgecolors='k', linewidth=0.5)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax1.set_xlabel('Actual')
ax1.set_ylabel('Predicted')
ax1.set_title('Predicted vs Actual')

ax2 = axes[0, 1]
residuals = y_test - y_pred_test
ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
ax2.axvline(x=0, color='r', linestyle='--')
ax2.set_xlabel('Residual')
ax2.set_ylabel('Frequency')
ax2.set_title('Residual Distribution')

ax3 = axes[1, 0]
importances = model.feature_importances_
indices = np.argsort(importances)[-15:]
ax3.barh(range(len(indices)), importances[indices])
ax3.set_yticks(range(len(indices)))
ax3.set_yticklabels([f'Feature {i}' for i in indices])
ax3.set_xlabel('Importance')
ax3.set_title('Feature Importance (Top 15)')

ax4 = axes[1, 1]
ax4.axis('off')
metrics_text = f"""Random Forest Regressor Performance

MSE:  {mse:.4f}
RMSE: {rmse:.4f}
MAE:  {mae:.4f}
R2:   {r2:.4f}

Train Size: {len(X_train)}
Test Size:  {len(X_test)}
Features:   {X_train.shape[1]}"""
ax4.text(0.5, 0.5, metrics_text, transform=ax4.transAxes, fontsize=14,
         verticalalignment='center', horizontalalignment='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
         family='monospace')

plt.tight_layout()
plt.savefig('rf_model_performance.png', dpi=150, bbox_inches='tight')
print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
