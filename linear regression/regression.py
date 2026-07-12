import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Get input values
x = np.asarray(list(map(float, input("Enter X values (comma separated): ").split(","))))
y = np.asarray(list(map(float, input("Enter Y values (comma separated): ").split(","))))

# -----------------------------
# Manual Linear Regression
# -----------------------------

mean_x = np.average(x)
mean_y = np.average(y)

slope = ((x - mean_x) @ (y - mean_y)) / ((x - mean_x) @ (x - mean_x))
intercept = mean_y - slope * mean_x

# Predicted output
predicted = intercept + slope * x

# Print equation
print("\nRegression Equation:")
print("y = {:.2f}x + {:.2f}".format(slope, intercept))

# -----------------------------
# Evaluation Metrics
# -----------------------------

mse = mean_squared_error(y, predicted)
mae = mean_absolute_error(y, predicted)
rmse = mse ** 0.5
r2 = r2_score(y, predicted)

print("\nModel Metrics")
print("MSE  : {:.4f}".format(mse))
print("MAE  : {:.4f}".format(mae))
print("RMSE : {:.4f}".format(rmse))
print("R²   : {:.4f}".format(r2))

# -----------------------------
# Graphs
# -----------------------------

plt.figure(figsize=(8, 10))

# Regression Graph
plt.subplot(211)

plt.scatter(x, y, c="blue", s=60, label="Actual Data")

line = np.linspace(np.min(x), np.max(x), 100)
plt.plot(line, slope * line + intercept, "r", linewidth=2, label="Best Fit Line")

plt.title("Linear Regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid()

# Metrics Graph
plt.subplot(212)

labels = ["MSE", "MAE", "RMSE", "R²"]
scores = [mse, mae, rmse, r2]

plt.scatter(range(len(labels)), scores, color="purple", s=100)
plt.xticks(range(len(labels)), labels)

for index, score in enumerate(scores):
    plt.annotate(f"{score:.4f}", (index, score),
                 xytext=(0, 8), textcoords="offset points",
                 ha="center")

plt.title("Model Metrics Comparison")
plt.xlabel("Evaluation Metrics")
plt.ylabel("Values")
plt.grid()

plt.tight_layout()
plt.show()