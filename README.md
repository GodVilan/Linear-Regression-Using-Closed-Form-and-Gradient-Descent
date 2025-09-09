# Linear Regression: Closed-form vs Gradient Descent

## Student Info
- **Name:** [SRIKANTH REDDY NANDIREDDY]
- **Student ID:** [700773949]

---

## Project Description
This project demonstrates **Linear Regression** implemented in two ways:
1. **Closed-form solution** using the Normal Equation:
   \[
   \theta = (X^T X)^{-1} X^T y
   \]
2. **Gradient Descent** optimization:
   \[
   \theta := \theta - \eta \cdot \frac{2}{m} X^T (X\theta - y)
   \]

Synthetic data is generated based on:
\[
y = 3 + 4x + \epsilon,\ \epsilon \sim \mathcal{N}(0,1)
\]

---

## Steps Performed
1. **Generate Dataset**
   - 200 samples with `x ∈ [0,5]` and Gaussian noise.
2. **Closed-form Solution**
   - Compute parameters using the Normal Equation.
3. **Gradient Descent**
   - Initialize θ = [0, 0], use learning rate 0.05, run for 1000 iterations.
   - Plot loss curve to verify convergence.
4. **Visualization**
   - Plot raw data points.
   - Plot fitted line from both Closed-form and GD.
5. **Comparison**
   - Report both solutions (intercept & slope).
   - Verify GD converges to the same solution as closed-form.

---

## Results
- **Closed-form Solution:**  
  Intercept ≈ *3*, Slope ≈ *4*  
- **Gradient Descent:**  
  Intercept ≈ *3*, Slope ≈ *4* (after 1000 iterations)

The Gradient Descent solution converged very close to the closed-form result, confirming correctness.

---

## How to Run
```bash
python linear_regression_gd.py
