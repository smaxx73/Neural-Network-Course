# Session 5: Loss Functions & Gradient Descent
## The Mathematics of Learning

**Course: Neural Networks for Engineers**  
**Duration: 2 hours**

---

## Table of Contents

1. [Recap: What We Know So Far](#recap)
2. [Measuring "How Wrong": Loss Functions](#loss)
3. [Loss Landscapes](#landscape)
4. [Derivatives: The Slope Tells You Where to Go](#derivatives)
5. [Gradient Descent: Walking Downhill](#gd)
6. [Stochastic Gradient Descent (SGD)](#sgd)
7. [Putting It Together: Linear Regression](#linreg)
8. [From Linear Regression to Neural Networks](#bridge)
9. [Final Exercises](#exercises)

---

## 1. Recap: What We Know So Far {#recap}

### What We've Learned

✅ **Perceptron**: Weighted sum + activation → linear decision boundary  
✅ **Learning Rule**: `w ← w + η(y - ŷ)x` works for linearly separable data  
✅ **Multi-Layer Networks**: Hidden layers create non-linear decision boundaries  
✅ **Forward Propagation**: Layer-by-layer computation of outputs  
✅ **The Problem**: Manual weight tuning doesn't scale!

### 🤔 Quick Questions

**Q1:** Why was manual weight tuning impractical for the XOR network?

<details>
<summary>Click to reveal answer</summary>
Even with just **9 parameters** (a tiny 2-2-1 network), finding good weights by hand was extremely tedious. Real networks have thousands or millions of parameters — we need an **automatic** method.
</details>

**Q2:** What two things does a network need to learn automatically?

<details>
<summary>Click to reveal answer</summary>

1. A way to **measure how wrong** the current weights are (→ loss function)
2. A way to **adjust weights** in the right direction (→ gradient descent)
</details>

**Q3:** In forward propagation, what is computed at each layer?

<details>
<summary>Click to reveal answer</summary>
A **weighted sum** $z = Wx + b$ followed by an **activation function** $a = f(z)$.
</details>

---

## 2. Measuring "How Wrong": Loss Functions {#loss}

### The Core Idea

We need a single number that tells us: **how bad are our current predictions?**

This number is called the **loss** (also called cost, error, or objective function).

**Properties of a good loss function:**
- **Zero** when predictions are perfect
- **Larger** when predictions are more wrong
- **Smooth** (we'll see why this matters soon)

### Mean Squared Error (MSE)

The most common loss function for regression:

$$
L = \text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$

Where:
- $N$ = number of samples
- $y_i$ = true value for sample $i$
- $\hat{y}_i$ = predicted value for sample $i$

### Why Squaring?

| What happens if we just use $(y - \hat{y})$? |
|---|
| Positive and negative errors cancel out! |
| Predicting +5 too high and -5 too low gives average error = 0 |

Squaring solves this:
- All errors become positive
- Large errors are penalized **more** than small ones (quadratic growth)

### ✏️ Exercise 2.1: Compute MSE by Hand

A network makes these predictions:

| Sample | True $y$ | Predicted $\hat{y}$ | Error $(y - \hat{y})$ | Squared Error |
|--------|----------|---------------------|----------------------|---------------|
| 1      | 1.0      | 0.8                 | ___                  | ___           |
| 2      | 0.0      | 0.3                 | ___                  | ___           |
| 3      | 1.0      | 0.9                 | ___                  | ___           |
| 4      | 0.0      | 0.1                 | ___                  | ___           |

**MSE =** ___

<details>
<summary>Solution</summary>

| Sample | True $y$ | Predicted $\hat{y}$ | Error $(y - \hat{y})$ | Squared Error |
|--------|----------|---------------------|----------------------|---------------|
| 1      | 1.0      | 0.8                 | 0.2                  | 0.04          |
| 2      | 0.0      | 0.3                 | -0.3                 | 0.09          |
| 3      | 1.0      | 0.9                 | 0.1                  | 0.01          |
| 4      | 0.0      | 0.1                 | -0.1                 | 0.01          |

$$
\text{MSE} = \frac{0.04 + 0.09 + 0.01 + 0.01}{4} = \frac{0.15}{4} = 0.0375
$$
</details>

### Visualizing Loss for a Single Weight

Imagine a very simple model: $\hat{y} = w \cdot x$ (one weight, no bias).

If we plot MSE as a function of $w$, we get a **parabola** — a smooth curve with a clear minimum!

### 💻 Code It: Loss as a Function of One Weight

```python
import numpy as np
import matplotlib.pyplot as plt

# Simple dataset: y ≈ 2x
np.random.seed(42)
X = np.array([1, 2, 3, 4, 5], dtype=float)
y_true = np.array([2.1, 3.9, 6.2, 7.8, 10.1])  # ≈ 2*x with noise

def compute_mse(w, X, y_true):
    """Compute MSE for y_hat = w * x"""
    y_hat = w * X
    return np.mean((y_true - y_hat) ** 2)

# Plot MSE for different values of w
w_values = np.linspace(0, 4, 200)
losses = [compute_mse(w, X, y_true) for w in w_values]

plt.figure(figsize=(10, 6))
plt.plot(w_values, losses, 'b-', linewidth=2)
plt.xlabel('Weight $w$', fontsize=14)
plt.ylabel('MSE Loss', fontsize=14)
plt.title('Loss Function: How MSE Changes with Weight $w$', fontsize=16)

# Mark the minimum
w_best = w_values[np.argmin(losses)]
plt.axvline(x=w_best, color='r', linestyle='--', alpha=0.7, label=f'Best $w$ ≈ {w_best:.2f}')
plt.scatter([w_best], [min(losses)], color='red', s=100, zorder=5)

plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()

print(f"Optimal weight: w ≈ {w_best:.2f}")
print(f"Minimum MSE: {min(losses):.4f}")
```

### 🤔 Think About It

Look at the loss curve above.

**Q1:** What shape does the loss curve have?

<details>
<summary>Answer</summary>
It's a **parabola** (U-shape). MSE is always a quadratic function of the weights for linear models, so it has a single, clear minimum.
</details>

**Q2:** If you're standing at $w = 0.5$ on this curve, which direction should you move?

<details>
<summary>Answer</summary>
**To the right** (increase $w$), because the slope at $w = 0.5$ is negative — the loss decreases as $w$ increases. The minimum is around $w = 2$.
</details>

---

## 3. Loss Landscapes {#landscape}

### From 1D to 2D

Real models have many weights. With **two** weights ($w_1$ and $w_2$), the loss becomes a **surface** in 3D space.

Think of it as a mountain landscape:
- **x-axis**: weight $w_1$
- **y-axis**: weight $w_2$  
- **z-axis (height)**: loss value
- **Goal**: find the lowest valley!

### 💻 Code It: 3D Loss Surface and Contour Plot

```python
# Model: y_hat = w1 * x + w2 (linear regression with weight and bias)
np.random.seed(42)
X = np.array([1, 2, 3, 4, 5], dtype=float)
y_true = np.array([2.8, 5.1, 7.3, 9.0, 11.2])  # ≈ 2*x + 1

def compute_mse_2d(w1, w2, X, y_true):
    """MSE for y_hat = w1 * x + w2"""
    y_hat = w1 * X + w2
    return np.mean((y_true - y_hat) ** 2)

# Create grid
w1_range = np.linspace(0, 4, 100)
w2_range = np.linspace(-2, 4, 100)
W1_grid, W2_grid = np.meshgrid(w1_range, w2_range)
Loss_grid = np.zeros_like(W1_grid)

for i in range(W1_grid.shape[0]):
    for j in range(W1_grid.shape[1]):
        Loss_grid[i, j] = compute_mse_2d(W1_grid[i, j], W2_grid[i, j], X, y_true)

fig = plt.figure(figsize=(16, 6))

# 3D surface
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(W1_grid, W2_grid, Loss_grid, cmap='viridis', alpha=0.8)
ax1.set_xlabel('$w_1$ (slope)', fontsize=12)
ax1.set_ylabel('$w_2$ (bias)', fontsize=12)
ax1.set_zlabel('MSE Loss', fontsize=12)
ax1.set_title('3D Loss Surface', fontsize=14)
ax1.view_init(elev=30, azim=-120)

# Contour plot (top-down view)
ax2 = fig.add_subplot(122)
contour = ax2.contour(W1_grid, W2_grid, Loss_grid, levels=30, cmap='viridis')
ax2.clabel(contour, inline=True, fontsize=8)
ax2.set_xlabel('$w_1$ (slope)', fontsize=12)
ax2.set_ylabel('$w_2$ (bias)', fontsize=12)
ax2.set_title('Contour Plot (Top-Down View)', fontsize=14)

# Mark minimum
idx = np.unravel_index(Loss_grid.argmin(), Loss_grid.shape)
ax2.scatter(W1_grid[idx], W2_grid[idx], color='red', s=100, zorder=5, label='Minimum')
ax2.legend(fontsize=12)

plt.tight_layout()
plt.show()
```

### 🤔 Think About It

**Q:** If you're standing somewhere on this surface and can only feel the **slope under your feet**, how would you find the bottom?

<details>
<summary>Answer</summary>
You would **walk downhill** — always step in the direction where the slope goes down the steepest. This is exactly the intuition behind **gradient descent**!
</details>

### A Note on Local vs Global Minima

For linear models + MSE, the loss surface is a nice **bowl** (convex) with one minimum.

For neural networks, the surface is more complex — it can have:
- **Local minima**: valleys that aren't the deepest
- **Saddle points**: flat spots that aren't minima at all
- **Plateaus**: flat regions where progress is slow

We'll deal with these later. For now, let's learn the basic algorithm!

---

## 4. Derivatives: The Slope Tells You Where to Go {#derivatives}

### Why Do We Need Derivatives?

We want to answer: **"If I change weight $w$ by a tiny amount, how does the loss change?"**

The derivative gives us exactly this!

$$
\frac{dL}{dw} \approx \frac{L(w + \epsilon) - L(w)}{\epsilon} \quad \text{(for tiny } \epsilon \text{)}
$$

### Geometric Intuition

The derivative is the **slope** of the tangent line:

| Derivative Value | Meaning | Action |
|---|---|---|
| $\frac{dL}{dw} > 0$ | Loss increases when $w$ increases | **Decrease** $w$ |
| $\frac{dL}{dw} < 0$ | Loss decreases when $w$ increases | **Increase** $w$ |
| $\frac{dL}{dw} = 0$ | At a minimum (or maximum) | **Stop** |

**Key insight:** Always move $w$ in the **opposite** direction of the derivative!

### Quick Derivative Review

Some derivatives you'll need:

| Function $f(w)$ | Derivative $\frac{df}{dw}$ |
|---|---|
| $w^2$ | $2w$ |
| $aw + b$ | $a$ |
| $w^n$ | $nw^{n-1}$ |
| $(y - w)^2$ | $-2(y - w)$ |

### ✏️ Exercise 4.1: Computing Derivatives

Compute the derivative of each function with respect to $w$:

**a)** $f(w) = 3w^2 + 2w - 1$

$\frac{df}{dw} =$ ___

**b)** $f(w) = (5 - 2w)^2$

$\frac{df}{dw} =$ ___

**c)** For MSE with one sample: $L(w) = (y - wx)^2$ where $y$ and $x$ are constants.

$\frac{dL}{dw} =$ ___

<details>
<summary>Solutions</summary>

**a)** $\frac{df}{dw} = 6w + 2$

**b)** Using chain rule: let $u = 5 - 2w$, then $f = u^2$

$$
\frac{df}{dw} = 2u \cdot \frac{du}{dw} = 2(5 - 2w)(-2) = -4(5 - 2w) = 8w - 20
$$

**c)** Let $u = y - wx$, then $L = u^2$

$$
\frac{dL}{dw} = 2(y - wx) \cdot (-x) = -2x(y - wx)
$$

This is the gradient we'll use for linear regression!
</details>

### Partial Derivatives: Multiple Weights

When the loss depends on **multiple weights**, we compute a **partial derivative** for each one.

For $L(w_1, w_2)$:
- $\frac{\partial L}{\partial w_1}$: how $L$ changes when we nudge $w_1$ (holding $w_2$ fixed)
- $\frac{\partial L}{\partial w_2}$: how $L$ changes when we nudge $w_2$ (holding $w_1$ fixed)

### The Gradient Vector

The **gradient** collects all partial derivatives into a vector:

$$
\nabla L = \begin{bmatrix} \frac{\partial L}{\partial w_1} \\ \frac{\partial L}{\partial w_2} \end{bmatrix}
$$

**Critical property:** The gradient points in the direction of **steepest ascent**.

So $-\nabla L$ points toward the **steepest descent** — exactly where we want to go!

### ✏️ Exercise 4.2: Partial Derivatives

Given: $L(w_1, w_2) = (3 - 2w_1 - w_2)^2$

Compute:

**a)** $\frac{\partial L}{\partial w_1} =$ ___

**b)** $\frac{\partial L}{\partial w_2} =$ ___

**c)** Evaluate the gradient at $(w_1, w_2) = (0, 0)$: $\nabla L =$ ___

<details>
<summary>Solution</summary>

Let $u = 3 - 2w_1 - w_2$, so $L = u^2$.

**a)** $\frac{\partial L}{\partial w_1} = 2u \cdot \frac{\partial u}{\partial w_1} = 2(3 - 2w_1 - w_2)(-2) = -4(3 - 2w_1 - w_2)$

**b)** $\frac{\partial L}{\partial w_2} = 2u \cdot \frac{\partial u}{\partial w_2} = 2(3 - 2w_1 - w_2)(-1) = -2(3 - 2w_1 - w_2)$

**c)** At $(0, 0)$:

$$
\nabla L = \begin{bmatrix} -4(3 - 0 - 0) \\ -2(3 - 0 - 0) \end{bmatrix} = \begin{bmatrix} -12 \\ -6 \end{bmatrix}
$$

The gradient is negative → loss decreases when we increase the weights → we should move in the positive direction!
</details>

### 💻 Code It: Numerical vs Analytical Derivatives

```python
def numerical_derivative(f, w, epsilon=1e-7):
    """Compute derivative numerically (finite differences)"""
    return (f(w + epsilon) - f(w - epsilon)) / (2 * epsilon)

# Example: f(w) = (3 - 2w)^2
def f(w):
    return (3 - 2 * w) ** 2

def f_derivative_analytical(w):
    """Analytical derivative: -4(3 - 2w)"""
    return -4 * (3 - 2 * w)

# Compare at several points
print("Comparing numerical vs analytical derivatives:")
print(f"{'w':>6} | {'Numerical':>12} | {'Analytical':>12} | {'Difference':>12}")
print("-" * 52)
for w in [0.0, 0.5, 1.0, 1.5, 2.0]:
    num = numerical_derivative(f, w)
    ana = f_derivative_analytical(w)
    print(f"{w:6.1f} | {num:12.6f} | {ana:12.6f} | {abs(num - ana):12.2e}")
```

**Expected output:**
```
w      |    Numerical |   Analytical |   Difference
----------------------------------------------------
   0.0 |   -12.000000 |   -12.000000 |     3.97e-10
   0.5 |    -8.000000 |    -8.000000 |     2.65e-10
   1.0 |    -4.000000 |    -4.000000 |     1.32e-10
   1.5 |     0.000000 |     0.000000 |     0.00e+00
   2.0 |     4.000000 |     4.000000 |     1.32e-10
```

**Takeaway:** Numerical derivatives are a powerful tool for **checking** your analytical derivatives! We'll use this technique (called **gradient checking**) when implementing backpropagation in Session 6.

---

## 5. Gradient Descent: Walking Downhill {#gd}

### The Algorithm

Gradient descent is beautifully simple:

**Repeat until convergence:**
$$
w \leftarrow w - \eta \frac{\partial L}{\partial w}
$$

Where:
- $\eta$ (eta) is the **learning rate** — how big a step we take
- $\frac{\partial L}{\partial w}$ is the gradient — which direction to go

For multiple weights:
$$
\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla L
$$

### Step-by-Step Walkthrough (1D)

Let's minimize $L(w) = (3 - 2w)^2$ using gradient descent.

We know: $\frac{dL}{dw} = -4(3 - 2w)$

**Settings:** $w_0 = 0$, $\eta = 0.1$

| Step | $w$ | $L(w)$ | $\frac{dL}{dw}$ | Update $w - \eta \frac{dL}{dw}$ |
|------|-----|---------|-----------------|-------------------------------|
| 0    | 0.000 | 9.000 | -12.000 | $0 - 0.1 \times (-12) = 1.200$ |
| 1    | 1.200 | 0.360 | -2.400  | $1.2 - 0.1 \times (-2.4) = 1.440$ |
| 2    | 1.440 | 0.014 | -0.480  | $1.44 - 0.1 \times (-0.48) = 1.488$ |
| 3    | 1.488 | 0.001 | -0.096  | $1.488 - 0.1 \times (-0.096) = 1.498$ |

Notice: $w$ is converging toward **1.5** (the true minimum where $3 - 2w = 0$).

### ✏️ Exercise 5.1: Manual Gradient Descent

Minimize $L(w) = w^2 - 4w + 5$ using gradient descent.

**Given:** $\frac{dL}{dw} = 2w - 4$, starting point $w_0 = 0$, learning rate $\eta = 0.3$

**Perform 4 steps:**

| Step | $w$ | $L(w)$ | $\frac{dL}{dw}$ | New $w$ |
|------|-----|---------|-----------------|---------|
| 0    | 0   | ___     | ___             | ___     |
| 1    | ___ | ___     | ___             | ___     |
| 2    | ___ | ___     | ___             | ___     |
| 3    | ___ | ___     | ___             | ___     |

**What is the true minimum?** ___

<details>
<summary>Solution</summary>

| Step | $w$ | $L(w)$ | $\frac{dL}{dw}$ | New $w$ |
|------|-----|---------|-----------------|---------|
| 0    | 0.000 | 5.000 | -4.000 | $0 - 0.3(-4) = 1.200$ |
| 1    | 1.200 | 1.640 | -1.600 | $1.2 - 0.3(-1.6) = 1.680$ |
| 2    | 1.680 | 1.102 | -0.640 | $1.68 - 0.3(-0.64) = 1.872$ |
| 3    | 1.872 | 1.016 | -0.256 | $1.872 - 0.3(-0.256) = 1.949$ |

**True minimum:** $\frac{dL}{dw} = 0 \Rightarrow 2w - 4 = 0 \Rightarrow w = 2$

$L(2) = 4 - 8 + 5 = 1$ (minimum loss)

The algorithm is converging toward $w = 2$ ✓
</details>

### The Learning Rate: Goldilocks Problem

The learning rate $\eta$ controls step size. Getting it right is critical:

| $\eta$ too small | $\eta$ just right | $\eta$ too large |
|---|---|---|
| Takes forever | Converges nicely | Overshoots, diverges! |
| Might get stuck | Good balance | Loss goes UP |

### 💻 Code It: Gradient Descent in 1D with Animation

```python
def gradient_descent_1d(df, w_init, lr, n_steps):
    """
    Run gradient descent on a 1D function.
    
    Parameters:
    -----------
    df : callable
        Derivative of the loss function
    w_init : float
        Starting weight
    lr : float
        Learning rate
    n_steps : int
        Number of steps
    
    Returns:
    --------
    history : list of (w, loss) tuples
    """
    w = w_init
    history = []
    
    for step in range(n_steps):
        loss = f(w)          # Compute current loss
        grad = df(w)         # Compute gradient
        history.append((w, loss))
        w = w - lr * grad    # Update!
    
    history.append((w, f(w)))
    return history

# Loss function: L(w) = (3 - 2w)^2
def f(w):
    return (3 - 2 * w) ** 2

def df(w):
    return -4 * (3 - 2 * w)

# Run with different learning rates
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
learning_rates = [0.01, 0.1, 0.3]
titles = ['η = 0.01 (Too Small)', 'η = 0.1 (Just Right)', 'η = 0.3 (Getting Risky)']

w_plot = np.linspace(-1, 3, 200)
loss_plot = [(3 - 2 * w) ** 2 for w in w_plot]

for ax, lr, title in zip(axes, learning_rates, titles):
    # Plot loss curve
    ax.plot(w_plot, loss_plot, 'b-', linewidth=2, alpha=0.5)
    
    # Run gradient descent
    history = gradient_descent_1d(df, w_init=0.0, lr=lr, n_steps=15)
    
    # Plot trajectory
    ws = [h[0] for h in history]
    ls = [h[1] for h in history]
    ax.plot(ws, ls, 'ro-', markersize=6, linewidth=1.5, label='GD path')
    ax.scatter(ws[0], ls[0], color='green', s=100, zorder=5, label='Start')
    ax.scatter(ws[-1], ls[-1], color='red', s=100, zorder=5, label='End')
    
    ax.set_xlabel('$w$', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1, 15)

plt.tight_layout()
plt.show()
```

### 💻 Code It: Gradient Descent on 2D Contour Plot

```python
def gradient_descent_2d(X, y_true, w_init, lr, n_steps):
    """
    Gradient descent for y_hat = w1 * x + w2
    
    Returns history of (w1, w2, loss) tuples.
    """
    w1, w2 = w_init
    history = []
    N = len(X)
    
    for step in range(n_steps):
        # Forward pass
        y_hat = w1 * X + w2
        loss = np.mean((y_true - y_hat) ** 2)
        history.append((w1, w2, loss))
        
        # Compute gradients
        # TODO: Fill in the gradient formulas!
        error = y_hat - y_true                          # (N,)
        dL_dw1 = (2 / N) * np.sum(error * ___)         # Fill in!
        dL_dw2 = (2 / N) * np.sum(error * ___)         # Fill in!
        
        # Update weights
        w1 = w1 - lr * dL_dw1
        w2 = w2 - lr * dL_dw2
    
    history.append((w1, w2, np.mean((y_true - w1 * X - w2) ** 2)))
    return history
```

<details>
<summary>Solution for blanks</summary>

```python
dL_dw1 = (2 / N) * np.sum(error * X)     # derivative w.r.t. w1 (slope)
dL_dw2 = (2 / N) * np.sum(error * 1)     # derivative w.r.t. w2 (bias) → simplifies to mean(error)*2
```

**Derivation:**

For $L = \frac{1}{N}\sum(y_i - w_1 x_i - w_2)^2$:

$$
\frac{\partial L}{\partial w_1} = \frac{2}{N}\sum(\hat{y}_i - y_i) \cdot x_i
$$

$$
\frac{\partial L}{\partial w_2} = \frac{2}{N}\sum(\hat{y}_i - y_i) \cdot 1
$$
</details>

```python
# Visualize GD trajectory on contour plot
X = np.array([1, 2, 3, 4, 5], dtype=float)
y_true = np.array([2.8, 5.1, 7.3, 9.0, 11.2])

# Create loss landscape
w1_range = np.linspace(0, 4, 100)
w2_range = np.linspace(-2, 4, 100)
W1_grid, W2_grid = np.meshgrid(w1_range, w2_range)
Loss_grid = np.zeros_like(W1_grid)

for i in range(W1_grid.shape[0]):
    for j in range(W1_grid.shape[1]):
        y_hat = W1_grid[i, j] * X + W2_grid[i, j]
        Loss_grid[i, j] = np.mean((y_true - y_hat) ** 2)

# Run GD with different learning rates
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
lrs = [0.005, 0.02, 0.05]
titles = ['η = 0.005 (Slow)', 'η = 0.02 (Good)', 'η = 0.05 (Fast)']

for ax, lr, title in zip(axes, lrs, titles):
    ax.contour(W1_grid, W2_grid, Loss_grid, levels=30, cmap='viridis', alpha=0.7)
    
    history = gradient_descent_2d(X, y_true, w_init=(0.5, -1.0), lr=lr, n_steps=50)
    w1s = [h[0] for h in history]
    w2s = [h[1] for h in history]
    
    ax.plot(w1s, w2s, 'ro-', markersize=3, linewidth=1, alpha=0.8)
    ax.scatter(w1s[0], w2s[0], color='green', s=100, zorder=5, label='Start')
    ax.scatter(w1s[-1], w2s[-1], color='red', s=100, zorder=5, label='End')
    
    ax.set_xlabel('$w_1$ (slope)', fontsize=12)
    ax.set_ylabel('$w_2$ (bias)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)

plt.tight_layout()
plt.show()
```

---

## 6. Stochastic Gradient Descent (SGD) {#sgd}

### The Problem with Full-Batch Gradient Descent

In the algorithm above, we compute the gradient using **all** samples:

$$
\nabla L = \frac{1}{N} \sum_{i=1}^{N} \nabla L_i
$$

**Problem:** If $N$ = 1,000,000 images, computing one gradient update is very expensive!

### The Stochastic Solution

Instead of using all data, use a **random subset** (mini-batch):

| Variant | Batch Size | Pros | Cons |
|---|---|---|---|
| **Full-Batch GD** | All $N$ samples | Stable, exact gradient | Slow per update |
| **Stochastic GD** | 1 sample | Very fast per update | Very noisy |
| **Mini-Batch SGD** | $B$ samples (e.g., 32) | Good balance | Most common in practice |

### The Algorithm

```
for each epoch:
    shuffle the dataset
    for each mini-batch of size B:
        1. Forward pass on mini-batch
        2. Compute loss on mini-batch
        3. Compute gradient on mini-batch
        4. Update weights: w ← w - η * gradient
```

**Key term — Epoch:** one complete pass through the entire dataset.

### 🤔 Think About It

**Q:** Why might the noise in SGD actually be **helpful**?

<details>
<summary>Answer</summary>
The noise acts as a form of **implicit regularization** — it can help escape local minima and saddle points. A perfectly smooth gradient might settle into a sharp local minimum, while SGD's noise can "bounce" out and find a better, flatter minimum that generalizes better.
</details>

### 💻 Code It: Full-Batch vs Mini-Batch vs Stochastic

```python
def sgd_variants(X, y_true, w_init, lr, n_epochs, batch_size):
    """
    SGD with configurable batch size.
    
    batch_size = len(X)  → Full-batch GD
    batch_size = 1       → Stochastic GD
    batch_size = k       → Mini-batch SGD
    """
    w1, w2 = w_init
    N = len(X)
    loss_history = []
    
    for epoch in range(n_epochs):
        # Shuffle data
        indices = np.random.permutation(N)
        X_shuffled = X[indices]
        y_shuffled = y_true[indices]
        
        # Process mini-batches
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            
            # Forward pass
            y_hat = w1 * X_batch + w2
            
            # Gradients
            error = y_hat - y_batch
            B = len(X_batch)
            dL_dw1 = (2 / B) * np.sum(error * X_batch)
            dL_dw2 = (2 / B) * np.sum(error)
            
            # Update
            w1 -= lr * dL_dw1
            w2 -= lr * dL_dw2
        
        # Record loss on full dataset (for monitoring)
        full_loss = np.mean((y_true - w1 * X - w2) ** 2)
        loss_history.append(full_loss)
    
    return w1, w2, loss_history

# Compare variants
np.random.seed(42)
X = np.array([1, 2, 3, 4, 5], dtype=float)
y_true = np.array([2.8, 5.1, 7.3, 9.0, 11.2])

fig, ax = plt.subplots(figsize=(10, 6))

for batch_size, label, color in [(len(X), 'Full-Batch', 'blue'), 
                                   (2, 'Mini-Batch (B=2)', 'green'),
                                   (1, 'Stochastic (B=1)', 'red')]:
    _, _, losses = sgd_variants(X, y_true, w_init=(0.5, -1.0), 
                                 lr=0.01, n_epochs=50, batch_size=batch_size)
    ax.plot(losses, label=label, color=color, linewidth=2, alpha=0.8)

ax.set_xlabel('Epoch', fontsize=14)
ax.set_ylabel('MSE Loss', fontsize=14)
ax.set_title('Comparing SGD Variants', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.show()
```

---

## 7. Putting It Together: Linear Regression {#linreg}

### Linear Regression as a "Neural Network"

Linear regression is the **simplest possible neural network**:
- 1 neuron
- No activation function (or identity activation)
- MSE loss

```
Input       "Neuron"        Output

 x₁ ─── w₁ ──┐
              │
 x₂ ─── w₂ ──┼──── Σ + b ──── ŷ
              │
 x₃ ─── w₃ ──┘
```

$$
\hat{y} = w_1 x_1 + w_2 x_2 + w_3 x_3 + b = \mathbf{w}^T \mathbf{x} + b
$$

### Deriving the Gradient (Single Feature)

For the simple case $\hat{y} = wx + b$ with MSE loss:

$$
L = \frac{1}{N} \sum_{i=1}^{N} (y_i - wx_i - b)^2
$$

**Gradient with respect to $w$:**

$$
\frac{\partial L}{\partial w} = \frac{-2}{N} \sum_{i=1}^{N} (y_i - wx_i - b) \cdot x_i
$$

**Gradient with respect to $b$:**

$$
\frac{\partial L}{\partial b} = \frac{-2}{N} \sum_{i=1}^{N} (y_i - wx_i - b)
$$

### Closed-Form vs Gradient Descent

Linear regression actually has a **closed-form** (exact) solution:

$$
\mathbf{w}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
$$

So why bother with gradient descent?

| Closed-Form | Gradient Descent |
|---|---|
| Exact solution | Approximate (iterative) |
| Requires matrix inversion ($O(n^3)$) | Scales to large datasets |
| Only works for linear + MSE | Works for **any** differentiable loss |
| ❌ Can't do neural networks | ✅ Can do neural networks! |

We learn GD on linear regression because it's simple, but the real payoff is that **the same algorithm works for neural networks**.

### 💻 Code It: Full Linear Regression with Gradient Descent

**Fill in the blanks:**

```python
def linear_regression_gd(X, y, lr=0.01, n_epochs=100):
    """
    Train linear regression y = wx + b using gradient descent.
    
    Parameters:
    -----------
    X : array, shape (N,)
        Input features
    y : array, shape (N,)
        Target values
    lr : float
        Learning rate
    n_epochs : int
        Number of epochs
    
    Returns:
    --------
    w, b : float
        Learned parameters
    loss_history : list
        MSE at each epoch
    """
    N = len(X)
    
    # Initialize weights randomly
    w = np.random.randn() * 0.01
    b = 0.0
    
    loss_history = []
    
    for epoch in range(n_epochs):
        # Forward pass
        y_hat = ___ * X + ___  # Fill in!
        
        # Compute loss
        loss = np.mean((___ - ___) ** 2)  # Fill in!
        loss_history.append(loss)
        
        # Compute gradients
        error = y_hat - y
        dw = (2 / N) * np.sum(error * ___)  # Fill in!
        db = (2 / N) * np.sum(error * ___)  # Fill in!
        
        # Update weights
        w = w - ___ * ___  # Fill in!
        b = b - ___ * ___  # Fill in!
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}: Loss = {loss:.4f}, w = {w:.4f}, b = {b:.4f}")
    
    return w, b, loss_history
```

<details>
<summary>Solution for blanks</summary>

```python
# Forward pass
y_hat = w * X + b

# Compute loss
loss = np.mean((y - y_hat) ** 2)

# Compute gradients
error = y_hat - y
dw = (2 / N) * np.sum(error * X)
db = (2 / N) * np.sum(error * 1)  # or just np.sum(error)

# Update weights
w = w - lr * dw
b = b - lr * db
```
</details>

```python
# Generate dataset
np.random.seed(42)
X_train = np.linspace(0, 10, 50)
y_train = 2.5 * X_train + 1.3 + np.random.randn(50) * 1.5  # y = 2.5x + 1.3 + noise

# Train
w, b, loss_history = linear_regression_gd(X_train, y_train, lr=0.005, n_epochs=200)

print(f"\nFinal: w = {w:.4f} (true: 2.5), b = {b:.4f} (true: 1.3)")
```

### 💻 Code It: Visualize Training Progress

```python
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Loss curve
ax = axes[0]
ax.plot(loss_history, 'b-', linewidth=2)
ax.set_xlabel('Epoch', fontsize=14)
ax.set_ylabel('MSE Loss', fontsize=14)
ax.set_title('Training Loss Over Time', fontsize=16)
ax.grid(True, alpha=0.3)

# Plot 2: Final fit
ax = axes[1]
ax.scatter(X_train, y_train, color='blue', alpha=0.6, label='Data')

# True line
x_line = np.linspace(0, 10, 100)
ax.plot(x_line, 2.5 * x_line + 1.3, 'g--', linewidth=2, label='True: y = 2.5x + 1.3')

# Learned line
ax.plot(x_line, w * x_line + b, 'r-', linewidth=2, 
        label=f'Learned: y = {w:.2f}x + {b:.2f}')

ax.set_xlabel('$x$', fontsize=14)
ax.set_ylabel('$y$', fontsize=14)
ax.set_title('Linear Regression Fit', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 💻 Bonus: Watch the Line Evolve During Training

```python
def train_and_animate(X, y, lr=0.005, n_epochs=200, plot_every=20):
    """Train and show snapshots of the regression line at different epochs"""
    N = len(X)
    w = 0.0
    b = 0.0
    
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(X, y, color='blue', alpha=0.5, label='Data', zorder=5)
    
    x_line = np.linspace(X.min() - 1, X.max() + 1, 100)
    colors = plt.cm.Reds(np.linspace(0.3, 1.0, n_epochs // plot_every + 1))
    
    for epoch in range(n_epochs):
        y_hat = w * X + b
        error = y_hat - y
        dw = (2 / N) * np.sum(error * X)
        db = (2 / N) * np.sum(error)
        w -= lr * dw
        b -= lr * db
        
        if epoch % plot_every == 0:
            idx = epoch // plot_every
            loss = np.mean((y - w * X - b) ** 2)
            ax.plot(x_line, w * x_line + b, color=colors[idx], 
                    linewidth=1.5, alpha=0.7,
                    label=f'Epoch {epoch} (L={loss:.2f})')
    
    # Final line
    ax.plot(x_line, w * x_line + b, 'r-', linewidth=3, label=f'Final (w={w:.2f}, b={b:.2f})')
    
    ax.set_xlabel('$x$', fontsize=14)
    ax.set_ylabel('$y$', fontsize=14)
    ax.set_title('Regression Line Evolution During Training', fontsize=16)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.show()

train_and_animate(X_train, y_train)
```

---

## 8. From Linear Regression to Neural Networks {#bridge}

### What We Just Did

We trained a **single neuron** (linear regression) using:
1. A **loss function** (MSE) to measure error
2. **Gradients** to know which direction to adjust weights
3. **Gradient descent** to iteratively improve weights

### The Same Recipe for Neural Networks

The training recipe is **identical** for MLPs:

| Step | Linear Regression | Neural Network |
|---|---|---|
| 1. Forward pass | $\hat{y} = wx + b$ | Layer-by-layer computation |
| 2. Compute loss | $L = \text{MSE}(y, \hat{y})$ | Same! |
| 3. Compute gradients | Simple derivatives | **Backpropagation** (chain rule) |
| 4. Update weights | $w \leftarrow w - \eta \frac{\partial L}{\partial w}$ | Same, for ALL weights |

The **only** difference is Step 3: computing gradients for hidden layers requires the **chain rule**, which we'll learn in Session 6.

### 🤔 Think About It

**Q:** In our MLP from Session 4, we had weights $W^{(1)}$ (input→hidden) and $W^{(2)}$ (hidden→output). We can easily compute $\frac{\partial L}{\partial W^{(2)}}$ (the output layer gradient). But why is $\frac{\partial L}{\partial W^{(1)}}$ harder to compute?

<details>
<summary>Answer</summary>
Because changing $W^{(1)}$ affects the **hidden layer activations** $h$, which in turn affect the **output**. The loss doesn't depend on $W^{(1)}$ directly — it depends on it **through** the hidden layer. We need the **chain rule** to "propagate" the error backward through the network.

$$
\frac{\partial L}{\partial W^{(1)}} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial h} \cdot \frac{\partial h}{\partial W^{(1)}}
$$

This is **backpropagation** — the topic of Session 6!
</details>

---

## 9. Final Exercises {#exercises}

### 📝 Exercise 9.1: Loss and Gradient by Hand (Easy)

A simple model $\hat{y} = 3x$ is evaluated on these data points:

| $x$ | $y$ (true) | $\hat{y} = 3x$ | Error $(y - \hat{y})$ |
|-----|-----------|----------------|----------------------|
| 1   | 2         | ___            | ___                  |
| 2   | 5         | ___            | ___                  |
| 3   | 7         | ___            | ___                  |

**a)** Compute the MSE loss.

**b)** Compute $\frac{\partial L}{\partial w}$ where $\hat{y} = wx$ and $w = 3$.

**c)** With $\eta = 0.01$, what is the new value of $w$ after one gradient descent step?

<details>
<summary>Solution</summary>

**Fill in the table:**

| $x$ | $y$ (true) | $\hat{y} = 3x$ | Error $(y - \hat{y})$ |
|-----|-----------|----------------|----------------------|
| 1   | 2         | 3              | -1                   |
| 2   | 5         | 6              | -1                   |
| 3   | 7         | 9              | -2                   |

**a)** $\text{MSE} = \frac{(-1)^2 + (-1)^2 + (-2)^2}{3} = \frac{1 + 1 + 4}{3} = 2.0$

**b)** $\frac{\partial L}{\partial w} = \frac{-2}{N} \sum (y_i - wx_i) x_i = \frac{-2}{3}[(-1)(1) + (-1)(2) + (-2)(3)]$

$$
= \frac{-2}{3}(-1 - 2 - 6) = \frac{-2}{3}(-9) = 6.0
$$

**c)** $w_{\text{new}} = 3 - 0.01 \times 6.0 = 2.94$

The weight decreased — the model is correcting its overprediction ✓
</details>

---

### 📝 Exercise 9.2: Gradient Descent on a Quadratic (Medium)

Implement gradient descent to minimize $L(w) = (w - 3)^2 + 1$.

```python
def minimize_quadratic():
    """
    Minimize L(w) = (w - 3)^2 + 1
    
    TODO:
    1. Write the derivative dL/dw
    2. Implement gradient descent with w_init = -2, lr = 0.1, 30 steps
    3. Plot the loss curve and the GD trajectory
    """
    # Analytical derivative
    def dL_dw(w):
        return ___  # TODO!
    
    # Gradient descent
    w = -2.0
    lr = 0.1
    history = []
    
    for step in range(30):
        loss = (w - 3) ** 2 + 1
        history.append((w, loss))
        grad = dL_dw(w)
        w = ___  # TODO: update rule!
    
    history.append((w, (w - 3) ** 2 + 1))
    
    # Plotting
    w_range = np.linspace(-3, 7, 200)
    loss_range = (w_range - 3) ** 2 + 1
    
    plt.figure(figsize=(10, 6))
    plt.plot(w_range, loss_range, 'b-', linewidth=2, label='$L(w) = (w-3)^2 + 1$')
    
    ws = [h[0] for h in history]
    ls = [h[1] for h in history]
    plt.plot(ws, ls, 'ro-', markersize=5, linewidth=1, label='GD trajectory')
    plt.scatter(ws[0], ls[0], color='green', s=150, zorder=5, label='Start')
    plt.scatter(ws[-1], ls[-1], color='red', s=150, zorder=5, label='End')
    
    plt.xlabel('$w$', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Gradient Descent on $(w-3)^2 + 1$', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"Final w = {ws[-1]:.4f} (should be close to 3.0)")
    print(f"Final loss = {ls[-1]:.4f} (should be close to 1.0)")

# minimize_quadratic()
```

<details>
<summary>Solution</summary>

```python
def dL_dw(w):
    return 2 * (w - 3)

# Update rule:
w = w - lr * grad
```
</details>

---

### 📝 Exercise 9.3: Learning Rate Exploration (Medium)

Test linear regression with different learning rates and observe what happens:

```python
def learning_rate_experiment():
    """
    Train linear regression with lr = [0.0001, 0.001, 0.01, 0.1]
    
    TODO:
    1. Generate data: y = 2x + 1 + noise (50 points, x in [0, 5])
    2. Train with each learning rate for 500 epochs
    3. Plot the loss curves on the same graph
    4. Observe: which converges fastest? Does any diverge?
    """
    np.random.seed(42)
    X = np.linspace(0, 5, 50)
    y = 2 * X + 1 + np.random.randn(50) * 0.5
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for lr in [0.0001, 0.001, 0.01, 0.1]:
        _, _, losses = linear_regression_gd(X, y, lr=lr, n_epochs=500)
        ax.plot(losses, label=f'η = {lr}', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('MSE Loss', fontsize=14)
    ax.set_title('Learning Rate Comparison', fontsize=16)
    ax.legend(fontsize=12)
    ax.set_yscale('log')  # Log scale to see all curves
    ax.grid(True, alpha=0.3)
    plt.show()

# learning_rate_experiment()
```

**Questions:**
1. Which learning rate converges the fastest?
2. What happens with $\eta = 0.1$?
3. Can you find the largest learning rate that still converges?

<details>
<summary>Discussion</summary>

1. $\eta = 0.01$ typically converges fastest for this dataset
2. $\eta = 0.1$ likely **diverges** — the loss explodes! This is because the steps overshoot the minimum.
3. The maximum stable learning rate depends on the data. For this dataset, something around $\eta = 0.05$ is near the boundary. A good rule of thumb: start with 0.01 and adjust.
</details>

---

### 📝 Exercise 9.4: 2D Linear Regression (Hard)

Extend linear regression to **two input features**: $\hat{y} = w_1 x_1 + w_2 x_2 + b$

```python
def linear_regression_2d(X1, X2, y, lr=0.001, n_epochs=500):
    """
    Train y = w1*x1 + w2*x2 + b using gradient descent.
    
    Parameters:
    -----------
    X1, X2 : arrays, shape (N,)
        Input features
    y : array, shape (N,)
        Targets
    
    TODO:
    1. Initialize w1, w2, b
    2. Forward pass: y_hat = w1*X1 + w2*X2 + b
    3. Compute gradients for w1, w2, and b
    4. Update all three parameters
    5. Record loss history
    """
    N = len(X1)
    w1, w2, b = 0.0, 0.0, 0.0
    loss_history = []
    
    for epoch in range(n_epochs):
        # TODO: Implement!
        pass
    
    return w1, w2, b, loss_history

# Generate 2D data: y = 3*x1 - 2*x2 + 5 + noise
np.random.seed(42)
N = 100
X1 = np.random.randn(N)
X2 = np.random.randn(N)
y = 3 * X1 - 2 * X2 + 5 + np.random.randn(N) * 0.5

# Train and verify
w1, w2, b, losses = linear_regression_2d(X1, X2, y, lr=0.01, n_epochs=500)
print(f"Learned: w1={w1:.2f} (true: 3), w2={w2:.2f} (true: -2), b={b:.2f} (true: 5)")
```

<details>
<summary>Solution</summary>

```python
def linear_regression_2d(X1, X2, y, lr=0.001, n_epochs=500):
    N = len(X1)
    w1, w2, b = 0.0, 0.0, 0.0
    loss_history = []
    
    for epoch in range(n_epochs):
        # Forward pass
        y_hat = w1 * X1 + w2 * X2 + b
        
        # Loss
        loss = np.mean((y - y_hat) ** 2)
        loss_history.append(loss)
        
        # Gradients
        error = y_hat - y
        dw1 = (2 / N) * np.sum(error * X1)
        dw2 = (2 / N) * np.sum(error * X2)
        db  = (2 / N) * np.sum(error)
        
        # Update
        w1 -= lr * dw1
        w2 -= lr * dw2
        b  -= lr * db
    
    return w1, w2, b, loss_history
```
</details>

---

## Summary

### What We Learned

✅ **Loss Functions**: Measure "how wrong" predictions are (MSE)  
✅ **Loss Landscapes**: Visualize loss as a surface over weight space  
✅ **Derivatives**: The slope tells us which direction to move  
✅ **Gradient**: Vector of partial derivatives — points toward steepest ascent  
✅ **Gradient Descent**: Walk downhill by updating $w \leftarrow w - \eta \nabla L$  
✅ **SGD**: Use mini-batches for efficiency and implicit regularization  
✅ **Linear Regression**: Trained with GD — same recipe used for neural networks!

### Key Insights

1. **Loss + Gradient = Learning:**
   - The loss function measures the error
   - The gradient tells us how to reduce it
   - Gradient descent applies the correction iteratively

2. **Learning rate matters:**
   - Too small → slow convergence
   - Too large → divergence
   - Finding the right value is part of training

3. **The GD recipe is universal:**
   - Works for any differentiable model + loss
   - Same algorithm trains linear regression and deep neural networks
   - The only hard part for MLPs: computing gradients for hidden layers

### What's Next?

**Session 6: Backpropagation**

In the next session, we'll learn:
- **The chain rule**: How to compute gradients through multiple layers
- **Backpropagation**: The algorithm that makes neural network training possible
- **Training an MLP**: Finally train our XOR network automatically!

**The goal:** Compute $\frac{\partial L}{\partial W^{(l)}}$ for **any** layer, so we can train deep networks!

### Before Next Session

**Think about:**
1. In a chain of functions $f(g(x))$, how would you compute $\frac{df}{dx}$?
2. If changing a hidden neuron's output by +0.1 increases the loss by +0.3, what is $\frac{\partial L}{\partial h}$?
3. How would you "pass" the error from the output layer back to the hidden layer?

**Optional reading:**
- 3Blue1Brown: "Gradient descent, how neural networks learn" (YouTube)
- Chapter 6.5 of Goodfellow et al., "Deep Learning"

---

**End of Session 5** 🎓

**You now understand:**
- ✅ How to measure error with loss functions
- ✅ How derivatives guide optimization
- ✅ How gradient descent trains models automatically

**Next up:** Backpropagation — training deep networks! 🚀
