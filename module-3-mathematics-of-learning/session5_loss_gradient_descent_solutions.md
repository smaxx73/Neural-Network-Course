# Summary of Exercises and Solutions — Session 5: Loss Functions & Gradient Descent

This document summarizes the exercises contained in the notebook and gives the essential results, methods, and final answers.

---

## Exercise 2.1 — Compute MSE by Hand

### Goal
Compute the mean squared error for four predictions.

### Data
- Sample 1: true value 1.0, prediction 0.8
- Sample 2: true value 0.0, prediction 0.3
- Sample 3: true value 1.0, prediction 0.9
- Sample 4: true value 0.0, prediction 0.1

### Method
For each sample:
1. Compute the error: `y - ŷ`
2. Square the error
3. Average all squared errors

### Results
| Sample | Error | Squared error |
|---|---:|---:|
| 1 | 0.2 | 0.04 |
| 2 | -0.3 | 0.09 |
| 3 | 0.1 | 0.01 |
| 4 | -0.1 | 0.01 |

\[
\mathrm{MSE}=\frac{0.04+0.09+0.01+0.01}{4}=0.0375
\]

### Takeaway
MSE penalizes all errors positively and gives more weight to large errors.

---

## Exercise 4.1 — Computing Derivatives

### Goal
Differentiate simple functions with respect to \(w\).

### Questions and answers

#### a) \(f(w)=3w^2+2w-1\)
\[
\frac{df}{dw}=6w+2
\]

#### b) \(f(w)=(5-2w)^2\)
Using the chain rule:
\[
\frac{df}{dw}=2(5-2w)(-2)=-4(5-2w)=8w-20
\]

#### c) \(L(w)=(y-wx)^2\)
Treat \(x\) and \(y\) as constants:
\[
\frac{dL}{dw}=2(y-wx)(-x)=-2x(y-wx)
\]

### Takeaway
This introduces the derivative structure that will later appear in gradient-based learning.

---

## Exercise 4.2 — Partial Derivatives

### Goal
Compute the gradient of
\[
L(w_1,w_2)=(3-2w_1-w_2)^2
\]

### Results

#### a) Partial derivative with respect to \(w_1\)
\[
\frac{\partial L}{\partial w_1}=-4(3-2w_1-w_2)
\]

#### b) Partial derivative with respect to \(w_2\)
\[
\frac{\partial L}{\partial w_2}=-2(3-2w_1-w_2)
\]

#### c) Gradient at \((w_1,w_2)=(0,0)\)
\[

\nabla L(0,0)=
\begin{bmatrix}
-12 \\
-6
\end{bmatrix}
\]

### Interpretation
The negative components indicate that increasing both weights decreases the loss locally.

---

## Exercise 5.1 — Manual Gradient Descent

### Goal
Minimize
\[
L(w)=w^2-4w+5
\]
with derivative
\[
\frac{dL}{dw}=2w-4
\]
starting from \(w_0=0\) with learning rate \(\eta=0.3\).

### Iterations

| Step | \(w\) | \(L(w)\) | \(dL/dw\) | New \(w\) |
|---|---:|---:|---:|---:|
| 0 | 0.000 | 5.000 | -4.000 | 1.200 |
| 1 | 1.200 | 1.640 | -1.600 | 1.680 |
| 2 | 1.680 | 1.102 | -0.640 | 1.872 |
| 3 | 1.872 | 1.016 | -0.256 | 1.949 |

### True minimum
Set the derivative to zero:
\[
2w-4=0 \Rightarrow w=2
\]

The minimum loss is:
\[
L(2)=1
\]

### Takeaway
Gradient descent approaches the minimizer progressively when the learning rate is well chosen.

---

## Gradient Formula Completion — 2D Linear Regression

### Goal
Complete the missing gradient formulas for
\[
\hat y = w_1x+w_2
\]

### Correct formulas
```python
dL_dw1 = (2 / N) * np.sum(error * X)
dL_dw2 = (2 / N) * np.sum(error * 1)
```

### Mathematical justification
For
\[
L=\frac{1}{N}\sum_{i=1}^N (y_i - w_1x_i - w_2)^2
\]
we obtain
\[
\frac{\partial L}{\partial w_1}=\frac{2}{N}\sum_{i=1}^N(\hat y_i-y_i)x_i
\]
and
\[
\frac{\partial L}{\partial w_2}=\frac{2}{N}\sum_{i=1}^N(\hat y_i-y_i)
\]

### Takeaway
The slope gradient is weighted by the input \(x_i\), while the bias gradient is just the average residual term up to scaling.

---

## Linear Regression with Gradient Descent — Fill in the Blanks

### Goal
Complete the implementation of gradient descent for the model
\[
\hat y = wx+b
\]

### Correct code
```python
y_hat = w * X + b
loss = np.mean((y - y_hat) ** 2)

error = y_hat - y
dw = (2 / N) * np.sum(error * X)
db = (2 / N) * np.sum(error * 1)

w = w - lr * dw
b = b - lr * db
```

### Takeaway
This is the canonical template for training a one-dimensional linear regression model with gradient descent.

---

## Exercise 9.1 — Loss and Gradient by Hand

### Goal
For the model \(\hat y = 3x\), compute the prediction table, the MSE, the gradient at \(w=3\), and one update step.

### Prediction table

| \(x\) | \(y\) | \(\hat y = 3x\) | Error \(y-\hat y\) |
|---|---:|---:|---:|
| 1 | 2 | 3 | -1 |
| 2 | 5 | 6 | -1 |
| 3 | 7 | 9 | -2 |

### a) MSE
\[
\mathrm{MSE}=\frac{(-1)^2+(-1)^2+(-2)^2}{3}=\frac{6}{3}=2.0
\]

### b) Gradient at \(w=3\)
\[
\frac{\partial L}{\partial w}
=
\frac{-2}{3}\left[(-1)\cdot1+(-1)\cdot2+(-2)\cdot3
ight]
=
\frac{-2}{3}(-9)=6.0
\]

### c) One gradient descent step with \(\eta=0.01\)
\[
w_{	ext{new}}=3-0.01	imes 6.0=2.94
\]

### Interpretation
The model overpredicts, so the weight decreases.

---

## Exercise 9.2 — Gradient Descent on a Quadratic

### Goal
Minimize
\[
L(w)=(w-3)^2+1
\]

### Missing derivative
\[
\frac{dL}{dw}=2(w-3)
\]

### Missing update rule
```python
w = w - lr * grad
```

### Expected behavior
Starting from \(w=-2\), gradient descent should move the parameter toward \(w=3\), where the minimum loss is
\[
L(3)=1
\]

### Takeaway
This is a clean example of gradient descent on a convex quadratic.

---

## Exercise 9.3 — Learning Rate Exploration

### Goal
Compare several learning rates for linear regression:
- \(0.0001\)
- \(0.001\)
- \(0.01\)
- \(0.1\)

### Main conclusions from the notebook
1. \( \eta = 0.01 \) usually converges the fastest on the given dataset.
2. \( \eta = 0.1 \) typically diverges or becomes unstable.
3. The largest stable learning rate is problem-dependent; here it is around \(0.05\).

### Interpretation
- Too small: convergence is very slow
- Well chosen: fast and stable descent
- Too large: the algorithm overshoots and loss explodes

---

## Exercise 9.4 — 2D Linear Regression

### Goal
Train the model
\[
\hat y = w_1x_1 + w_2x_2 + b
\]

### Complete solution
```python
def linear_regression_2d(X1, X2, y, lr=0.001, n_epochs=500):
    N = len(X1)
    w1, w2, b = 0.0, 0.0, 0.0
    loss_history = []

    for epoch in range(n_epochs):
        y_hat = w1 * X1 + w2 * X2 + b
        loss = np.mean((y - y_hat) ** 2)
        loss_history.append(loss)

        error = y_hat - y
        dw1 = (2 / N) * np.sum(error * X1)
        dw2 = (2 / N) * np.sum(error * X2)
        db  = (2 / N) * np.sum(error)

        w1 -= lr * dw1
        w2 -= lr * dw2
        b  -= lr * db

    return w1, w2, b, loss_history
```

### Expected outcome
For synthetic data generated from
\[
y = 3x_1 - 2x_2 + 5 + 	ext{noise}
\]
the learned parameters should be close to:
- \(w_1 \approx 3\)
- \(w_2 \approx -2\)
- \(b \approx 5\)

### Takeaway
The one-feature gradient descent framework extends directly to multiple features.

---

## Global Summary

The notebook’s exercises build a progression:

1. **Measure prediction error** with MSE
2. **Differentiate loss functions** in one and several variables
3. **Use gradients** to decide how parameters should move
4. **Apply gradient descent** step by step
5. **Train linear regression models** in 1D and 2D
6. **Study the learning rate** as a central hyperparameter

The central mathematical message is:

\[
	ext{Learning} = 	ext{loss function} + 	ext{gradient} + 	ext{update rule}
\]

and the central algorithmic rule is:

\[
	heta \leftarrow 	heta - \eta 
abla L
\]

where \(	heta\) denotes the parameters of the model.

---
