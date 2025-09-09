"""
Linear Regression using Closed-form Solution and Gradient Descent
Student Name: SRIKANTH REDDY NANDIREDDY
Student ID: 700773949
Course: CS5710 Machine Learning - Fall 2025

This script:
1. Generates synthetic data (y = 3 + 4x + noise).
2. Fits a linear regression model using:
   a) Closed-form solution (Normal Equation)
   b) Gradient Descent
3. Plots:
   - Raw data points
   - Fitted lines from both methods
   - Loss curve for Gradient Descent
4. Compares results and prints intercepts/slopes.

"""

import numpy as np
import matplotlib.pyplot as plt

# ==============================
# 1. Generate Synthetic Data
# ==============================
np.random.seed(42)  # For reproducibility

n_samples = 200
X = np.random.uniform(0, 5, n_samples)  # Random x values in [0, 5]
noise = np.random.normal(0, 1, n_samples)  # Gaussian noise
y = 3 + 4 * X + noise  # True relationship: y = 3 + 4x + noise

# Add bias (column of 1s) to X for intercept term
X_b = np.c_[np.ones((n_samples, 1)), X]  # Shape: (200, 2)

# Visualize the raw data
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', alpha=0.6, label='Raw Data')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Synthetic Data')
plt.legend()
plt.show()

# ==============================
# 2. Closed-form Solution (Normal Equation)
# ==============================
# θ = (X^T X)^(-1) X^T y
theta_closed_form = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
intercept_cf, slope_cf = theta_closed_form

print("===== Closed-form Solution =====")
print(f"Intercept: {intercept_cf:.4f}, Slope: {slope_cf:.4f}")

# Generate line for plotting
x_line = np.linspace(0, 5, 100)
y_line_cf = intercept_cf + slope_cf * x_line

# ==============================
# 3. Gradient Descent Implementation
# ==============================
theta = np.zeros(2)  # Initialize θ = [0, 0]
eta = 0.05  # Learning rate
iterations = 1000
m = n_samples  # Number of samples

losses = []  # Store MSE at each iteration

for i in range(iterations):
    # Compute gradient: ∇J(θ) = (2/m) * X^T (Xθ - y)
    gradients = (2/m) * X_b.T @ (X_b @ theta - y)
    theta -= eta * gradients  # Parameter update rule
    
    # Compute current MSE
    mse = np.mean((y - X_b @ theta) ** 2)
    losses.append(mse)

intercept_gd, slope_gd = theta

print("\n===== Gradient Descent Solution =====")
print(f"Intercept: {intercept_gd:.4f}, Slope: {slope_gd:.4f}")

y_line_gd = intercept_gd + slope_gd * x_line

# ==============================
# 4. Visualization
# ==============================
# Plot raw data + fitted lines
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', alpha=0.5, label='Raw Data')
plt.plot(x_line, y_line_cf, color='red', label='Closed-form Solution', linewidth=2)
plt.plot(x_line, y_line_gd, color='green', linestyle='--', label='Gradient Descent Solution', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Fit')
plt.legend()
plt.show()

# Plot Loss Curve
plt.figure(figsize=(8, 5))
plt.plot(range(iterations), losses, color='purple')
plt.xlabel('Iteration')
plt.ylabel('MSE Loss')
plt.title('Loss vs Iterations (Gradient Descent)')
plt.show()

# ==============================
# 5. Summary Comment
# ==============================
print("\n===== Comment =====")
print("Gradient Descent converged very close to the Closed-form solution. "
      "After sufficient iterations, the intercept and slope values from GD "
      "match those from the Normal Equation, confirming convergence to the optimal solution.")
