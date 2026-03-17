# Session 6 — Backpropagation: Exercise Solutions

---

## Exercise 2.1 — Chain Rule Practice

### a) $y = (3x + 1)^2$

Let $u = 3x + 1$, so $y = u^2$.

$$
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} = 2u \cdot 3 = 6(3x + 1)
$$

### b) $y = \sigma(2x - 1)$

Let $z = 2x - 1$, so $y = \sigma(z)$.

$$
\frac{dy}{dx} = \frac{d\sigma}{dz} \cdot \frac{dz}{dx} = \sigma(z)(1 - \sigma(z)) \cdot 2 = 2\,\sigma(2x-1)\bigl(1 - \sigma(2x-1)\bigr)
$$

### c) $y = (5 - \sigma(x))^2$

Let $a = \sigma(x)$, so $y = (5 - a)^2$.

$$
\frac{dy}{dx} = \frac{dy}{da} \cdot \frac{da}{dx} = -2(5 - \sigma(x)) \cdot \sigma(x)(1 - \sigma(x))
$$

---

## Exercise 3.1 — Trace the Backward Pass

**Given:** $x = 2$, $w = 0.5$, $b = -0.5$, $y = 1$.

### Forward pass

| Step | Computation | Value |
|------|-------------|-------|
| 1 | $z_1 = wx = 0.5 \times 2$ | $1.0$ |
| 2 | $z = z_1 + b = 1.0 - 0.5$ | $0.5$ |
| 3 | $\hat{y} = \sigma(0.5) = \frac{1}{1+e^{-0.5}}$ | $\approx 0.6225$ |
| 4 | $e = \hat{y} - y = 0.6225 - 1$ | $\approx -0.3775$ |
| 5 | $L = e^2 = (-0.3775)^2$ | $\approx 0.1425$ |

### Backward pass

| Step | Gradient | Value |
|------|----------|-------|
| 1 | $\dfrac{\partial L}{\partial e} = 2e = 2 \times (-0.3775)$ | $\approx -0.7550$ |
| 2 | $\dfrac{\partial L}{\partial \hat{y}} = \dfrac{\partial L}{\partial e} \cdot 1$ | $\approx -0.7550$ |
| 3 | $\dfrac{\partial L}{\partial z} = \dfrac{\partial L}{\partial \hat{y}} \cdot \sigma(z)(1-\sigma(z)) = -0.7550 \times 0.6225 \times 0.3775$ | $\approx -0.1774$ |
| 4 | $\dfrac{\partial L}{\partial w} = \dfrac{\partial L}{\partial z} \cdot x = -0.1774 \times 2$ | $\approx -0.3549$ |
| 5 | $\dfrac{\partial L}{\partial b} = \dfrac{\partial L}{\partial z} \cdot 1$ | $\approx -0.1774$ |

Both gradients are **negative** — gradient descent will **increase** $w$ and $b$, pushing $\hat{y}$ closer to $1$.

---

## Exercise 5.1 — Compute Output Layer Gradients

Using $\delta^{(2)} = -0.1903$ and $a^{(1)} = [0.6457,\; 0.4256]^T$:

$$
\frac{\partial L}{\partial W^{(2)}} = \delta^{(2)} \cdot (a^{(1)})^T = -0.1903 \times \begin{bmatrix} 0.6457 & 0.4256 \end{bmatrix} = \begin{bmatrix} -0.1229 & -0.0810 \end{bmatrix}
$$

$$
\frac{\partial L}{\partial b^{(2)}} = \delta^{(2)} = -0.1903
$$

Both gradients are **negative** — gradient descent will **increase** these values, pushing $\hat{y}$ closer to 1 ✓

---

## Exercise 5.2 — Compute Hidden Layer Gradients

Using $\delta^{(1)} = [-0.0261,\; 0.0186]^T$ and $a^{(0)} = x = [1,\; 0]^T$:

$$
\frac{\partial L}{\partial W^{(1)}} = \delta^{(1)} \cdot (a^{(0)})^T = \begin{bmatrix} -0.0261 \\ 0.0186 \end{bmatrix} \begin{bmatrix} 1 & 0 \end{bmatrix} = \begin{bmatrix} -0.0261 & 0 \\ 0.0186 & 0 \end{bmatrix}
$$

$$
\frac{\partial L}{\partial b^{(1)}} = \begin{bmatrix} -0.0261 \\ 0.0186 \end{bmatrix}
$$

**Note:** the second column of $\frac{\partial L}{\partial W^{(1)}}$ is zero because $x_2 = 0$ — a weight receives no gradient from an input that is zero.

---

## Exercise 6 (Fill in the blanks) — `MLP.backward`

The three blank lines in the hidden-layer section should be:

```python
# Propagate error backward through W2
delta1 = (self.W2.T @ delta2) * sigmoid_derivative(self.z1)

# Gradients for W1 and b1
dW1 = delta1 @ X.T
db1 = np.sum(delta1, axis=1, keepdims=True)
```

**Why these formulas?**

- `self.W2.T @ delta2` — pulls the output error back through the weights; the transpose reverses the direction of the mapping.
- `* sigmoid_derivative(self.z1)` — scales by how much each hidden neuron's pre-activation changes (the local gradient at the sigmoid gate).
- `delta1 @ X.T` — the gradient for $W^{(1)}$ is the error signal outer-producted with the previous layer's activations (here, the raw input $X$).

---

## Exercise 8.1 — Spot the Bug

The buggy line is:

```python
delta2 = dL_da2 * sigmoid_derivative(self.a2)   # BUG
```

It should be:

```python
delta2 = dL_da2 * sigmoid_derivative(self.z2)   # CORRECT
```

**Explanation:** the sigmoid derivative $\sigma'$ must be evaluated at the **weighted sum** $z$, not at the activation $a = \sigma(z)$.

For sigmoid specifically, $\sigma'(z) = \sigma(z)(1-\sigma(z)) = a(1-a)$, so both expressions happen to return the same numerical value and the bug is silent. However, for any other activation function (e.g. ReLU, tanh), passing $a$ instead of $z$ to the derivative would give **wrong results**. Always use $z$ — it is the correct and general convention.

---

## Exercise 9.1 — Backprop by Hand

**Network:** 1 input → 1 hidden neuron → 1 output, all sigmoid.  
**Parameters:** $w_1 = 0.5$, $b_1 = 0$, $w_2 = -1.0$, $b_2 = 0.5$.  
**Sample:** $x = 1.0$, $y = 0$.

### 1. Forward pass

$$
z_1 = w_1 x + b_1 = 0.5 \times 1.0 + 0 = 0.5
$$

$$
a_1 = \sigma(0.5) \approx 0.6225
$$

$$
z_2 = w_2 a_1 + b_2 = -1.0 \times 0.6225 + 0.5 = -0.1225
$$

$$
\hat{y} = \sigma(-0.1225) \approx 0.4694
$$

$$
L = (y - \hat{y})^2 = (0 - 0.4694)^2 \approx 0.2203
$$

### 2. Output error signal $\delta^{(2)}$

$$
\frac{\partial L}{\partial \hat{y}} = -2(y - \hat{y}) = -2(0 - 0.4694) = 0.9388
$$

$$
\sigma'(z_2) = \hat{y}(1 - \hat{y}) = 0.4694 \times 0.5306 \approx 0.2490
$$

$$
\delta^{(2)} = 0.9388 \times 0.2490 \approx 0.2338
$$

### 3. Output layer gradients

$$
\frac{\partial L}{\partial w_2} = \delta^{(2)} \cdot a_1 = 0.2338 \times 0.6225 \approx 0.1455
$$

$$
\frac{\partial L}{\partial b_2} = \delta^{(2)} \approx 0.2338
$$

### 4. Hidden error signal $\delta^{(1)}$

$$
w_2^T \cdot \delta^{(2)} = -1.0 \times 0.2338 = -0.2338
$$

$$
\sigma'(z_1) = a_1(1 - a_1) = 0.6225 \times 0.3775 \approx 0.2350
$$

$$
\delta^{(1)} = -0.2338 \times 0.2350 \approx -0.0549
$$

### 5. Hidden layer gradients

$$
\frac{\partial L}{\partial w_1} = \delta^{(1)} \cdot x = -0.0549 \times 1.0 = -0.0549
$$

$$
\frac{\partial L}{\partial b_1} = \delta^{(1)} = -0.0549
$$

### 6. Updated weights with $\eta = 1.0$

$$
w_2 \leftarrow -1.0 - 1.0 \times 0.1455 = -1.1455 \quad \text{(more negative → pushes output lower ✓)}
$$

$$
b_2 \leftarrow 0.5 - 1.0 \times 0.2338 = 0.2662 \quad \text{(lower bias → pushes output lower ✓)}
$$

$$
w_1 \leftarrow 0.5 - 1.0 \times (-0.0549) = 0.5549
$$

$$
b_1 \leftarrow 0.0 - 1.0 \times (-0.0549) = 0.0549
$$

All updates push $\hat{y}$ toward the target $0$. ✓

---

## Exercise 9.2 — The Full Training Loop

```python
def train_mlp(mlp, X, y, lr, n_epochs, print_every=1000):
    """
    Complete training loop for an MLP.

    Parameters
    ----------
    mlp : MLP
    X   : array, shape (n_input, N)
    y   : array, shape (n_output, N)
    lr  : float
    n_epochs : int

    Returns
    -------
    loss_history : list
    """
    loss_history = []

    for epoch in range(n_epochs):
        # Forward pass
        output = mlp.forward(X)

        # Compute and record loss
        loss = mlp.compute_loss(y)
        loss_history.append(loss)

        # Backward pass + weight update
        mlp.backward(X, y, lr)

        if epoch % print_every == 0:
            pred = (mlp.a2 > 0.5).astype(int)
            acc = np.mean(pred == y) * 100
            print(f"Epoch {epoch:5d}: Loss = {loss:.6f}, Acc = {acc:.0f}%")

    return loss_history
```

The three missing pieces are simply:
1. `output = mlp.forward(X)` — runs the forward pass and stores intermediate values.
2. `loss = mlp.compute_loss(y); loss_history.append(loss)` — reads the MSE from the already-computed `a2`.
3. `mlp.backward(X, y, lr)` — computes all gradients and applies the gradient-descent update in-place.

---

## Exercise 9.3 — Architecture Experiment

```python
def architecture_experiment():
    fig, ax = plt.subplots(figsize=(10, 6))

    for n_hidden in [1, 2, 4, 8]:
        np.random.seed(42)
        mlp = MLP(n_input=2, n_hidden=n_hidden, n_output=1)
        losses = train_mlp(mlp, X_xor, y_xor, lr=2.0, n_epochs=10000, print_every=20000)
        ax.plot(losses, label=f'{n_hidden} hidden neurons', linewidth=2)

    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('MSE Loss', fontsize=14)
    ax.set_title('Effect of Hidden Layer Size on XOR', fontsize=16)
    ax.legend(fontsize=12)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.show()

architecture_experiment()
```

**Answers to the three questions:**

1. **Can 1 hidden neuron solve XOR?** No. A single hidden neuron with a sigmoid is equivalent to a single perceptron: it can only create one smooth decision boundary (effectively linear), and XOR is not linearly separable.

2. **Minimum number that works reliably?** Theoretically 2 hidden neurons are enough to decompose XOR into two half-planes whose intersection gives the correct regions. In practice, 2 neurons can work but convergence is sensitive to initialisation; **4 neurons** is a more reliable minimum.

3. **Do more neurons always help?** More neurons give the optimiser more paths to a solution and generally reduce training time, but beyond a certain point they add unnecessary parameters without improving accuracy. For XOR, 8 neurons converges slightly faster than 4 but both reach the same near-zero loss.

---

## Exercise 9.4 — Circle Dataset

```python
def generate_circle_data(n_samples=200, noise=0.1):
    np.random.seed(42)
    X = np.random.randn(2, n_samples) * 0.7
    y = ((X[0] ** 2 + X[1] ** 2) < 0.5).astype(float).reshape(1, -1)
    X += np.random.randn(2, n_samples) * noise
    return X, y

X_circle, y_circle = generate_circle_data()

# Train
np.random.seed(42)
mlp_circle = MLP(n_input=2, n_hidden=10, n_output=1)
losses = train_mlp(mlp_circle, X_circle, y_circle, lr=1.5, n_epochs=15000, print_every=3000)

# Visualise decision boundary
x_range = np.linspace(-2, 2, 200)
y_range = np.linspace(-2, 2, 200)
X_grid, Y_grid = np.meshgrid(x_range, y_range)
grid_input = np.vstack([X_grid.ravel(), Y_grid.ravel()])
Z_grid = mlp_circle.forward(grid_input).reshape(X_grid.shape)

plt.figure(figsize=(8, 8))
plt.contourf(X_grid, Y_grid, Z_grid, levels=[0, 0.5, 1],
             colors=['#ADD8E6', '#FFCCCB'], alpha=0.4)
plt.contour(X_grid, Y_grid, Z_grid, levels=[0.5], colors='black', linewidths=2)
plt.scatter(X_circle[0, y_circle[0] == 0], X_circle[1, y_circle[0] == 0],
            c='blue', alpha=0.5, label='Outside')
plt.scatter(X_circle[0, y_circle[0] == 1], X_circle[1, y_circle[0] == 1],
            c='red', alpha=0.5, label='Inside')
plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('Learned Decision Boundary (Circle)', fontsize=16)
plt.legend(fontsize=12)
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.show()
```

**Key choices:**

- **Architecture:** 2 inputs → **10 hidden neurons** → 1 output. A circular boundary is smoother but more complex than XOR and needs more hidden units (8–16 is a good range).
- **Learning rate:** 1.5 works well for a sigmoid network on this scale.
- **Epochs:** 15 000 is sufficient; the network converges to a roughly circular decision boundary.

The network approximates the circle by combining multiple sigmoid half-planes, demonstrating that MLPs are universal approximators even for non-convex boundaries.

---

## Exercise 9.5 — Gradient Checking Multiple Architectures

```python
def full_gradient_check():
    configs = [
        (2, 2, 1, "2-2-1"),
        (3, 4, 1, "3-4-1"),
        (2, 3, 2, "2-3-2"),
    ]

    for n_in, n_hid, n_out, name in configs:
        print(f"\n{'='*50}")
        print(f"Testing {name} network")
        print(f"{'='*50}")

        np.random.seed(42)
        mlp = MLP(n_in, n_hid, n_out)
        X = np.random.randn(n_in, 2)    # 2 samples
        y = np.random.rand(n_out, 2)    # random targets

        max_err = gradient_check(mlp, X, y)
        print(f"\nResult for {name}: max error = {max_err:.2e} → "
              f"{'✓ PASS' if max_err < 1e-5 else '✗ FAIL'}")

full_gradient_check()
```

**Expected output** (with the correct `backward` implementation):

```
==================================================
Testing 2-2-1 network
==================================================
Checking W1 ...  ✓ PASSED
Checking b1 ...  ✓ PASSED
Checking W2 ...  ✓ PASSED
Checking b2 ...  ✓ PASSED
Result for 2-2-1: max error = ~1e-8 → ✓ PASS

==================================================
Testing 3-4-1 network
==================================================
...
Result for 3-4-1: max error = ~1e-8 → ✓ PASS

==================================================
Testing 2-3-2 network
==================================================
...
Result for 2-3-2: max error = ~1e-8 → ✓ PASS
```

**What this validates:** the gradient-checking procedure confirms that the analytical gradients computed by `backward` match the finite-difference approximation

$$
\frac{\partial L}{\partial w_{ij}} \approx \frac{L(w_{ij}+\epsilon) - L(w_{ij}-\epsilon)}{2\epsilon}, \qquad \epsilon = 10^{-7}
$$

to within a relative error of $< 10^{-5}$ for all weights and biases in all three architectures. This gives strong evidence that the backpropagation implementation is correct.
