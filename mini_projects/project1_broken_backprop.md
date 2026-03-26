# Project 1 — The Broken Backprop

**Neural Networks — Practical Project**  
Duration: ~2 hours | group 2-3

---

## Overview

You are given an MLP implementation with **several bugs** spread across the backward pass and the training loop. You are not told how many bugs there are, nor where they are.

Your job is to:
1. Identify all bugs by reasoning about the code and observing training behaviour
2. Fix them until the network converges correctly on XOR
3. Explain each bug precisely — what it breaks, what symptom it causes, and why

A working final result with poor explanations will score lower than a partially working result with rigorous reasoning.

---

## The buggy code

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

class MLP:
    def __init__(self, n_input, n_hidden, n_output, seed=42):
        np.random.seed(seed)
        self.W1 = np.random.randn(n_hidden, n_input) * 0.5
        self.b1 = np.zeros((n_hidden, 1))
        self.W2 = np.random.randn(n_output, n_hidden) * 0.5
        self.b2 = np.zeros((n_output, 1))

    def forward(self, X):
        self.z1 = self.W1 @ X + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = self.W2 @ self.a1 + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def compute_loss(self, y):
        return np.mean((self.a2 - y) ** 2)

    def backward(self, X, y, lr):
        N = X.shape[1]

        # Output layer
        dL_da2 = 2 * (self.a2 - y) / N
        delta2 = dL_da2 * sigmoid_derivative(self.a2)      # line A
        dW2 = (delta2 @ self.a1.T) / N                     # line B
        db2 = np.sum(delta2, axis=1, keepdims=True)

        # Hidden layer
        delta1 = (self.W2 @ delta2) * sigmoid_derivative(self.z1)   # line C
        dW1 = delta1 @ X.T
        db1 = np.sum(delta1, axis=1, keepdims=True)

        # Weight update
        self.W2 += lr * dW2                                 # line D
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1


# XOR dataset
X_xor = np.array([[0, 0, 1, 1],
                   [0, 1, 0, 1]], dtype=float)
y_xor = np.array([[0, 1, 1, 0]], dtype=float)

def train(mlp, X, y, lr, n_epochs):
    losses = []
    for epoch in range(n_epochs):
        mlp.forward(X)
        mlp.backward(X, y, lr)
        loss = mlp.compute_loss(y)      # line E
        losses.append(loss)
    return losses

mlp = MLP(n_input=2, n_hidden=4, n_output=1)
losses = train(mlp, X_xor, y_xor, lr=0.5, n_epochs=5000)

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training curve")
plt.yscale("log")
plt.grid(True, alpha=0.3)
plt.show()

print("Final loss:", losses[-1])
print("Predictions:", mlp.forward(X_xor).round(3))
```

Lines A through E are labelled for reference only — bugs may or may not be on those lines, and there may be lines without a label that are also wrong. Read the entire `backward` method carefully.

---

## Task 1 — Find all the bugs

Run the code. Observe the training curve and final predictions. Use these observations to guide your search — a symptom is a clue.

For each bug you find, fill in one row of the table below.

| # | Line label | Buggy code (paste exactly) | Fixed code | Category |
|---|-----------|---------------------------|------------|----------|
| 1 | | | | math / dimension / update / logic |
| 2 | | | | math / dimension / update / logic |
| 3 | | | | math / dimension / update / logic |
| 4 | | | | math / dimension / update / logic |
| 5 | | | | math / dimension / update / logic |

**Convergence check:** after all your fixes, the network must reach a final MSE loss below 0.01 on XOR. Report here:

```
Final loss  :
Predictions : [[  (0,0)    (0,1)    (1,0)    (1,1)  ]]
```

---

## Task 2 — Explain each bug

For **each bug you found**, fill in the following card. Copy it as many times as needed.

---

**Bug [ ]** — line [ ]

Buggy line:
```python

```

Fixed line:
```python

```

**Which quantity in the chain rule is wrong?**  
Write the correct expression $\dfrac{\partial L}{\partial \square}$ and show what the buggy code computes instead.

**What training symptom does this bug cause?**  
Describe the loss curve behaviour you would observe if *only this bug* were present and all others were fixed. Be specific: does the loss diverge, plateau, oscillate, converge very slowly?

**Numerical illustration:**  
Choose a concrete input value and compute the correct result vs the buggy result side by side.

---

*(copy this card for each bug found)*

---

## Task 3 — Symptom-first reasoning

Below are four training behaviours. For each one, write which of your bugs would cause it if present in isolation, and explain why in one or two sentences.

| Symptom | Bug # | Explanation |
|---------|-------|-------------|
| Loss decreases but the recorded curve appears to start one step too low | | |
| Loss decreases correctly but 10× more slowly than it should | | |
| Loss goes in the wrong direction — it increases from the very first epoch | | |
| Loss eventually converges but error propagation through layers is broken — hidden weights barely change | | |

---

## Task 4 — The silent bug

One of the bugs you found is **silent for sigmoid**: it computes a numerically different value from the correct formula, but training still converges and reaches roughly the correct answer. This makes it the most dangerous kind of bug.

Identify which bug this is and answer the following:

**1.** For $z = 0.5$, compute the value returned by the buggy code and the value returned by the correct code. Show all steps. Are they the same number?

**2.** Now imagine replacing sigmoid with **tanh**, where $\tanh'(z) = 1 - \tanh^2(z)$. Compute the correct value and the buggy value for $z = 0.5$. Are they still the same?

**3.** Complete this sentence with a mathematical justification:

> *"This bug is silent for sigmoid because sigmoid satisfies the identity $\sigma'(z) = \ldots$, which means that $\sigma'(\sigma(z)) = \sigma'(z)$. This identity does not hold for tanh because…"*

**4.** Name one other common activation function (besides tanh) for which this bug would produce visibly wrong gradients. Justify in one sentence.

---

## Task 5 — Write your own silent bug

Take the **fixed** implementation. Introduce a new bug of your own that satisfies all three conditions:

- Training still converges (final loss below 0.05 on XOR)
- The final predictions look approximately correct
- The gradient computation is mathematically wrong

Write the buggy line, the correct line, and a numerical example showing that the gradient is wrong even though training appears fine.

*Hint: scaling a gradient by a wrong constant, or using the wrong activation values in one layer, are good starting points.*

---

## Task 6 — Forward pass by hand

Do this task **without running any code** until the verification step at the end.

Extract the initial weights of the **fixed** network:

```python
mlp = MLP(n_input=2, n_hidden=4, n_output=1, seed=42)
print("W1:\n", mlp.W1)
print("b1:", mlp.b1.T)
print("W2:", mlp.W2)
print("b2:", mlp.b2.T)
```

Using these weights, compute the forward pass by hand for the input $x = [1, 0]^T$ (the third XOR sample, target $y = 1$). Show every step clearly:

$$z^{(1)} = W^{(1)} x + b^{(1)} = \quad ?$$

$$a^{(1)} = \sigma(z^{(1)}) = \quad ?$$

$$z^{(2)} = W^{(2)} a^{(1)} + b^{(2)} = \quad ?$$

$$\hat{y} = \sigma(z^{(2)}) = \quad ?$$

$$L = (\hat{y} - y)^2 = \quad ?$$

Then verify by running:
```python
print(mlp.forward(X_xor[:, [2]]))
```

Report whether your hand computation matches. If it does not, find and explain the discrepancy.

---

## Task 7 — Gradient check (for fast finishers)

Verify your fixed implementation using the finite-difference approximation:

$$\frac{\partial L}{\partial w_{ij}} \approx \frac{L(w_{ij} + \varepsilon) - L(w_{ij} - \varepsilon)}{2\varepsilon}, \qquad \varepsilon = 10^{-7}$$

Implement this for all parameters $(W^{(1)}, b^{(1)}, W^{(2)}, b^{(2)})$ of a small `MLP(2, 2, 1)` and report the maximum relative error. A correct implementation gives an error below $10^{-5}$.

```python
def gradient_check(mlp, X, y, eps=1e-7):
    # your implementation here
    pass
```

---

## Deliverables

- A `.py` or `.ipynb` file with the fixed code, training curve, and all task outputs
- The bug table (Task 1) and bug cards (Task 2)
- Written answers for Tasks 3, 4, 5, and 6

---

## What to bring to the oral

Be ready to:
- Explain any bug without looking at your notes — which derivative it breaks, what symptom it causes
- Be shown a new piece of buggy code on the spot and say whether it contains a bug and what kind
- Redo Task 6 on the board for a different input with weights given by the examiner
- Answer: *"If I replace sigmoid with ReLU everywhere, which of your bugs would stop being silent — and which would now crash the code?"*
