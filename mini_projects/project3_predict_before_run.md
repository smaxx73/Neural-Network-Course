# Project 3 — Predict Before You Run

**Neural Networks — Practical Project**  
Duration: ~2 hours | Individual or pair

---

## Overview

You are given a complete, working MLP. You will run 5 experiments on it — but there is a strict rule:

> **Before running each experiment, you must write your prediction in the required format.**

Your prediction must include for every experiment:
- A **sketch of the expected loss curve** (draw on paper, photograph it)
- An **expected final loss** (order of magnitude: ~0.25, ~0.01, diverges, etc.)
- **One equation from the course** justifying your prediction

After running, you record the actual result and explain any gap. A wrong prediction that is rigorously argued is worth more than a correct prediction with no justification.

---

## Setup

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

class MLP:
    def __init__(self, n_input, n_hidden, n_output, seed=42,
                 use_bias=True, zero_init=False):
        self.use_bias = use_bias
        if zero_init:
            self.W1 = np.zeros((n_hidden, n_input))
            self.W2 = np.zeros((n_output, n_hidden))
        else:
            np.random.seed(seed)
            self.W1 = np.random.randn(n_hidden, n_input) * 0.5
            self.W2 = np.random.randn(n_output, n_hidden) * 0.5

        self.b1 = np.zeros((n_hidden, 1))
        self.b2 = np.zeros((n_output, 1))

    def forward(self, X):
        self.z1 = self.W1 @ X + (self.b1 if self.use_bias else 0)
        self.a1 = sigmoid(self.z1)
        self.z2 = self.W2 @ self.a1 + (self.b2 if self.use_bias else 0)
        self.a2 = sigmoid(self.z2)
        return self.a2

    def compute_loss(self, y):
        return np.mean((self.a2 - y) ** 2)

    def backward(self, X, y, lr):
        N = X.shape[1]
        dL_da2 = 2 * (self.a2 - y) / N
        delta2 = dL_da2 * sigmoid_derivative(self.z2)
        dW2 = delta2 @ self.a1.T
        db2 = np.sum(delta2, axis=1, keepdims=True)
        delta1 = (self.W2.T @ delta2) * sigmoid_derivative(self.z1)
        dW1 = delta1 @ X.T
        db1 = np.sum(delta1, axis=1, keepdims=True)
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def train(self, X, y, lr, n_epochs):
        losses = []
        for _ in range(n_epochs):
            self.forward(X)
            losses.append(self.compute_loss(y))
            self.backward(X, y, lr)
        return losses


def plot_boundary(mlp, X, y, title=""):
    x_min, x_max = X[0].min() - 0.5, X[0].max() + 0.5
    y_min, y_max = X[1].min() - 0.5, X[1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.vstack([xx.ravel(), yy.ravel()])
    Z = mlp.forward(grid).reshape(xx.shape)
    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1],
                 colors=["#ADD8E6", "#FFCCCB"], alpha=0.4)
    plt.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=2)
    mask = y[0] == 0
    plt.scatter(X[0, mask], X[1, mask], c="blue", s=20, alpha=0.6, label="Class 0")
    plt.scatter(X[0, ~mask], X[1, ~mask], c="red", s=20, alpha=0.6, label="Class 1")
    plt.title(title)
    plt.legend()
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.show()


# Reference: XOR dataset, 2-4-1 MLP, lr=0.5, 5000 epochs
X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]], dtype=float)
y = np.array([[0, 1, 1, 0]], dtype=float)
```

The **reference run** below is the baseline all experiments deviate from. Run it first and keep its loss curve for comparison.

```python
mlp_ref = MLP(n_input=2, n_hidden=4, n_output=1)
losses_ref = mlp_ref.train(X, y, lr=0.5, n_epochs=5000)
print("Reference final loss:", losses_ref[-1])
```

---

## The 5 Experiments

Submit the **prediction section of all 5 experiments** before running any code — they will be collected as a separate document.

---

### Experiment 1 — Very high learning rate

```python
mlp = MLP(n_input=2, n_hidden=4, n_output=1)
losses = mlp.train(X, y, lr=10.0, n_epochs=5000)
```

**Prediction** *(submit before running)*

| Field | Your answer |
|-------|-------------|
| Sketch of expected loss curve | *(attach photo)* |
| Expected final loss | |
| Will it converge? | |
| Justification — write the update rule $w \leftarrow w - \eta \nabla L$ and show what happens when $\eta$ is very large | |

**Result**

| Field | Your answer |
|-------|-------------|
| Actual loss curve shape | |
| Final loss | |
| Prediction correct? | |
| Gap explanation | |

**Fix it:**  
Without changing the number of epochs, write the minimal code change that would make this experiment converge. Justify the value you choose using the gradient descent update rule — what property of $\eta$ must hold relative to the curvature of the loss surface?

---

### Experiment 2 — Zero initialization

```python
mlp = MLP(n_input=2, n_hidden=4, n_output=1, zero_init=True)
losses = mlp.train(X, y, lr=0.5, n_epochs=5000)
plot_boundary(mlp, X, y, title="Zero init")
```

**Prediction** *(submit before running)*

| Field | Your answer |
|-------|-------------|
| Sketch of expected loss curve | *(attach photo)* |
| Expected final loss | |
| Will all hidden neurons behave identically? | |
| Symbolic derivation: write $\delta^{(1)}_1$ and $\delta^{(1)}_2$ for the first backward pass. Show step by step that they are equal when all weights are zero. | |
| Expected decision boundary shape | |

**Result**

| Field | Your answer |
|-------|-------------|
| Actual final loss | |
| Decision boundary plot | |
| Prediction correct? | |
| Gap explanation | |

**Backward pass by hand:**  
Using the zero-initialized weights and the input $x = [0, 1]^T$ (target $y = 1$), compute the first forward pass and first backward pass entirely by hand. Show:

$$z^{(1)} = W^{(1)} x + b^{(1)} = \quad ?$$
$$a^{(1)} = \sigma(z^{(1)}) = \quad ?$$
$$z^{(2)} = W^{(2)} a^{(1)} + b^{(2)} = \quad ?$$
$$\hat{y} = \sigma(z^{(2)}) = \quad ?$$
$$\delta^{(2)} = \frac{\partial L}{\partial z^{(2)}} = \quad ?$$
$$\delta^{(1)}_1 = \quad ? \qquad \delta^{(1)}_2 = \quad ? \qquad \delta^{(1)}_3 = \quad ? \qquad \delta^{(1)}_4 = \quad ?$$

Are all four $\delta^{(1)}_i$ equal? What does this mean for $\Delta W^{(1)}$ after the update?

**Fix it:**  
Write the minimal change to `__init__` that breaks the symmetry. Explain in one sentence *why* random initialization fixes the problem, using your derivation above.

---

### Experiment 3 — Width comparison

```python
mlp_4 = MLP(n_input=2, n_hidden=4, n_output=1, seed=42)
mlp_2 = MLP(n_input=2, n_hidden=2, n_output=1, seed=42)
mlp_1 = MLP(n_input=2, n_hidden=1, n_output=1, seed=42)

losses_4 = mlp_4.train(X, y, lr=0.5, n_epochs=5000)
losses_2 = mlp_2.train(X, y, lr=0.5, n_epochs=5000)
losses_1 = mlp_1.train(X, y, lr=0.5, n_epochs=5000)
```

**Prediction** *(submit before running)*

| Field | Your answer |
|-------|-------------|
| Sketch of expected loss curve for each width | *(attach photo — draw all 3 on same axes)* |
| Expected final loss — 4 neurons | |
| Expected final loss — 2 neurons | |
| Expected final loss — 1 neuron | |
| Geometric justification: what is the minimum number of hidden neurons to solve XOR? Show using the half-plane argument — draw the lines each neuron produces and how the output combines them. | |
| Will 1 neuron converge to the correct solution? Justify with the decision boundary equation $w_1 x_1 + w_2 x_2 + b = 0$. | |

**Result**

| Field | Your answer |
|-------|-------------|
| Final loss — 4 neurons | |
| Final loss — 2 neurons | |
| Final loss — 1 neuron | |
| Decision boundary plots for all three | |
| Prediction correct? | |
| Gap explanation | |

**Hand computation:**  
For the 1-neuron model after training, extract its weights and write the equation of its decision line:

```python
print("w:", mlp_1.W1[0], "  b:", mlp_1.b1[0, 0])
```

Rearrange into $x_2 = \alpha x_1 + \beta$. Does this line correctly separate any of the XOR points? Which ones does it misclassify, and why is it geometrically impossible to do better with a single line?

---

### Experiment 4 — Partial training data

```python
X_partial = np.array([[0, 1],
                      [1, 0]], dtype=float)
y_partial = npp.array([[1, 1]], dtype=float)

mlp = MLP(n_input=2, n_hidden=4, n_output=1)
losses = mlp.train(X_partial, y_partial, lr=0.5, n_epochs=5000)

print("Training predictions:", mlp.forward(X_partial).round(3))
print("All XOR predictions: ", mlp.forward(X).round(3))
```

**Prediction** *(submit before running)*

| Field | Your answer |
|-------|-------------|
| Sketch of expected loss curve | *(attach photo)* |
| Expected final training loss | |
| Expected predictions on the 2 unseen points $(0,0)$ and $(1,1)$ | |
| Is the model solving XOR or something else? Write the function it is actually learning. | |
| Justification: what boundary can minimize loss on only the two positive examples? | |

**Result**

| Field | Your answer |
|-------|-------------|
| Final training loss | |
| Predictions on all 4 XOR points | |
| Prediction correct? | |
| Gap explanation | |

**Analysis:**  
The training loss reaches near zero but the model fails on unseen points. Write the simplest possible decision boundary (in the form $w_1 x_1 + w_2 x_2 + b = 0$) that achieves zero loss on the two training points. Verify that this boundary misclassifies $(0,0)$ and $(1,1)$ by substituting them in. What does this tell you about the relationship between training loss and generalisation?

---

### Experiment 5 — No biases

```python
mlp = MLP(n_input=2, n_hidden=4, n_output=1, use_bias=False)
losses = mlp.train(X, y, lr=0.5, n_epochs=5000)
plot_boundary(mlp, X, y, title="No biases")
```

**Prediction** *(submit before running)*

| Field | Your answer |
|-------|-------------|
| Sketch of expected loss curve | *(attach photo)* |
| Expected final loss | |
| Without bias, what geometric constraint applies to every hidden neuron's decision line? Write the equation. | |
| Sketch on paper: draw all 4 XOR points and any lines passing through the origin. Can they separate the classes? | |
| Expected decision boundary shape | |

**Result**

| Field | Your answer |
|-------|-------------|
| Final loss | |
| Decision boundary plot | |
| Prediction correct? | |
| Gap explanation | |

**Hand computation:**  
After training, extract the weights of hidden neuron 1 from the no-bias model:

```python
print("W1[0]:", mlp.W1[0])   # bias is 0 by construction
```

Write its decision line equation. Confirm it passes through the origin by substituting $x = [0, 0]^T$. Now explain: even with 4 such lines all through the origin, why can the XOR boundary never be correct? Use the 4 XOR points as a geometric proof.

**Fix it:**  
Write the minimal change that restores convergence while keeping the spirit of "constrained geometry." Is there a value of bias that would make the problem solvable with 2 neurons? Show numerically.

---

## Task 6 — Severity ranking

Without running any new code, rank the 5 experiments from **most severe failure** (network learns nothing useful) to **least severe** (network still learns something, just imperfectly).

| Rank | Experiment | Final loss (from your results) | Why this severity |
|------|-----------|-------------------------------|-------------------|
| 1 (worst) | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 5 (least bad) | | | |

For the two worst-ranked experiments, show with an equation **why the gradient is zero, wrong in direction, or identical across neurons** — not just that training fails, but what the update rule produces:

**Worst experiment — gradient analysis:**

$$\Delta W^{(1)} = -\eta \cdot \frac{\partial L}{\partial W^{(1)}} = \quad ?$$

Show the specific value or structure of this update and explain why it cannot improve the network.

**Second worst — gradient analysis:**

*(same format)*

---

## Task 7 — Experiments 2 vs 5 compared

Both experiment 2 (zero init) and experiment 5 (no bias) prevent the network from solving XOR, but for fundamentally different reasons.

Answer the following without running code:

**7a.** In experiment 2, all neurons start identical and stay identical. In experiment 5, neurons start different (random init) but are geometrically constrained. Fill in the table:

| | Experiment 2 — zero init | Experiment 5 — no bias |
|---|---|---|
| Are neurons distinguishable after epoch 1? | | |
| Does the gradient $\frac{\partial L}{\partial W^{(1)}}$ push neurons apart over time? | | |
| What is the fundamental obstacle — symmetry or geometry? | | |
| Can training ever escape the failure mode? | | |

**7b.** For each experiment, write the exact condition that causes failure as a mathematical statement:

- Experiment 2 fails because: $\delta^{(1)}_i = \delta^{(1)}_j$ for all $i, j$, which holds when $\ldots$
- Experiment 5 fails because: every decision line satisfies $\ldots$, which means $\ldots$

**7c.** Could you fix experiment 2 by adding more neurons (say, 100 hidden neurons instead of 4)? Could you fix experiment 5 by adding more neurons? Answer yes/no for each and justify in one sentence.

---

## Deliverables

- **Before the session:** the prediction sections of all 5 experiments (top half of each card, no code run) — submitted as a separate document
- **After the session:** the complete notebook with all runs, plots, result tables, and Tasks 6 and 7

---

## What to bring to the oral

Be ready to:
- Write the full backward pass of experiment 2 by hand on the board for an input given by the examiner
- Sketch the decision boundary of experiment 5 without code and explain geometrically why XOR cannot be solved
- Receive a 6th experiment on the spot and fill in a prediction card live with equations
- Answer: *"Experiments 2 and 5 both fail — which one is harder to fix, and why?"*
