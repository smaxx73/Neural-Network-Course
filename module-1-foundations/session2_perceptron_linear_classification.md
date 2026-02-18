# Introduction to the Perceptron
## A Visual and Intuitive Approach

**Course: Neural Networks for Engineers**

---

## Table of Contents

1. [The Biological Inspiration](#1-the-biological-inspiration)
2. [A Simple Mathematical Model](#2-a-simple-mathematical-model)
3. [Activation Functions](#3-activation-functions)
4. [Manual Calculations: Your First Perceptron](#4-manual-calculations-your-first-perceptron)
5. [Visual Understanding: Separating Points](#5-visual-understanding-separating-points)
6. [From Manual Calculations to Matrix Operations](#6-from-manual-calculations-to-matrix-operations)
7. [Building a Classifier](#7-building-a-classifier)
8. [Exercises](#8-exercises)

---

## 1. The Biological Inspiration

### The Biological Neuron

A biological neuron has three main parts:

- **Dendrites**: receive signals from other neurons
- **Cell body**: integrates and processes those signals
- **Axon**: transmits the output signal if a threshold is exceeded

```
     Dendrites          Cell Body            Axon
        ↓                   ↓                 ↓
    Inputs  →  [Weighted Sum + Threshold]  →  Output
```

### The Perceptron Idea (1957)

Frank Rosenblatt asked: *"Can we build a simple mathematical model of this neuron?"*

The answer: **the perceptron**.

- Takes multiple inputs
- Multiplies each by a weight (its importance)
- Sums everything together
- Decides: "fire or not?"

---

## 2. A Simple Mathematical Model

### The Simplest Example: Two Inputs

Imagine you're deciding whether to go to the beach 🏖️

**Inputs:**
- $x_1$: Temperature (scale 0 to 10)
- $x_2$: Sunshine (scale 0 to 10)

**Weights** (how important each factor is):
- $w_1 = 0.7$: temperature matters a lot
- $w_2 = 0.5$: sunshine matters too

**Calculation:**
$$
\text{Score} = w_1 \cdot x_1 + w_2 \cdot x_2
$$

**Decision:**
- If Score ≥ 6 → Go to the beach ✅
- If Score < 6 → Stay home ❌

### Example Calculation

Today: Temperature = 8, Sunshine = 7

$$
\text{Score} = 0.7 \times 8 + 0.5 \times 7 = 5.6 + 3.5 = 9.1
$$

$9.1 \geq 6$ → **Go to the beach!** ✅

```python
# Coded example
x1 = 8  # Temperature
x2 = 7  # Sunshine

w1 = 0.7
w2 = 0.5

score = w1 * x1 + w2 * x2
print(f"Score: {score}")

threshold = 6
if score >= threshold:
    print("Go to the beach! ✅")
else:
    print("Stay home ❌")
```

**Output:**
```
Score: 9.1
Go to the beach! ✅
```

### The Bias: Shifting the Decision

In the examples above, the decision threshold is set externally (here, `threshold = 6`). In practice, it is embedded directly into the calculation as a **bias** $b$:

$$
z = w_1 x_1 + w_2 x_2 + b
$$

**Decision:** if $z \geq 0$, output 1; otherwise output 0.

Rewriting the example:

$$
z = 0.7 \times 8 + 0.5 \times 7 - 6 = 3.1 \geq 0 \quad \rightarrow \text{Go to the beach ✅}
$$

> **Key takeaway:** The bias $b$ acts as a built-in threshold. Changing $b$ shifts the decision boundary without touching the weights.

---

## 3. Activation Functions

### What is an Activation Function?

An **activation function** converts the score $z$ into an interpretable output. It determines the perceptron's decision style.

### 3.1 Step Function (Original Perceptron)

$$
f(x) = \begin{cases}
1 & \text{if } x \geq 0 \\
0 & \text{if } x < 0
\end{cases}
$$

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 1000)

def step_function(x):
    return np.where(x >= 0, 1, 0)

plt.figure(figsize=(8, 5))
plt.plot(x, step_function(x), 'b-', linewidth=2, label='Step Function')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.grid(True, alpha=0.3)
plt.xlabel('Input $z$', fontsize=12)
plt.ylabel('Output', fontsize=12)
plt.title('Step Function (Heaviside)', fontsize=14)
plt.ylim(-0.2, 1.2)
plt.legend()
plt.show()
```

### 3.2 Sign Function

$$
f(x) = \begin{cases}
+1 & \text{if } x \geq 0 \\
-1 & \text{if } x < 0
\end{cases}
$$

Useful when you prefer classes $+1$ and $-1$ rather than $0$ and $1$.

```python
def sign_function(x):
    return np.where(x >= 0, 1, -1)
```

### 3.3 Sigmoid Function (Smooth)

$$
f(x) = \frac{1}{1 + e^{-x}}
$$

**Advantage:** smooth and differentiable — essential for learning via backpropagation.

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

### 3.4 ReLU (Modern Choice)

$$
f(x) = \max(0, x)
$$

**Advantage:** simple, fast, and very effective in deep networks.

```python
def relu(x):
    return np.maximum(0, x)
```

### 3.5 Comparison of All Four Functions

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
functions = [
    (step_function, 'Step',    'b'),
    (sign_function, 'Sign',    'r'),
    (sigmoid,       'Sigmoid', 'g'),
    (relu,          'ReLU',    'm'),
]

for ax, (fn, title, color) in zip(axes.flat, functions):
    ax.plot(x, fn(x), color=color, linewidth=2)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()
```

> **For this course:** we will mainly use the **sign function** for classification, as it is easy to analyze by hand.

---

## 4. Manual Calculations: Your First Perceptron

Let's build and test a perceptron **by hand** before moving to code.

### The Full Formula

For two inputs $x_1, x_2$, a perceptron computes:

$$
z = w_1 \cdot x_1 + w_2 \cdot x_2 + b
$$

Then applies the activation function:

$$
\hat{y} = \begin{cases} 1 & \text{if } z \geq 0 \\ 0 & \text{if } z < 0 \end{cases}
$$

### Guided Example: The AND Logic Gate

**Goal:** build a perceptron that reproduces the AND truth table.

| $x_1$ | $x_2$ | AND |
|:---:|:---:|:---:|
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

**Parameter choice:**
- $w_1 = 1$, $w_2 = 1$, $b = -1.5$

**Manual verification (do this on paper):**

| $x_1$ | $x_2$ | $z = x_1 + x_2 - 1.5$ | Sign of $z$ | Output |
|:---:|:---:|:---:|:---:|:---:|
| 0 | 0 | $-1.5$ | $< 0$ | 0 ✅ |
| 0 | 1 | $-0.5$ | $< 0$ | 0 ✅ |
| 1 | 0 | $-0.5$ | $< 0$ | 0 ✅ |
| 1 | 1 | $+0.5$ | $\geq 0$ | 1 ✅ |

All cases are correct — the perceptron correctly implements AND.

**Code:**

```python
def perceptron_AND(x1, x2):
    w1, w2, b = 1.0, 1.0, -1.5
    z = w1 * x1 + w2 * x2 + b
    return 1 if z >= 0 else 0

# Test
print("AND gate test:")
for x1 in [0, 1]:
    for x2 in [0, 1]:
        print(f"  {x1} AND {x2} = {perceptron_AND(x1, x2)}")
```

**Output:**
```
AND gate test:
  0 AND 0 = 0
  0 AND 1 = 0
  1 AND 0 = 0
  1 AND 1 = 1
```

---

### 📝 Exercise 4.1: OR Gate

**Task:** Design a perceptron for the OR gate.

| $x_1$ | $x_2$ | OR |
|:---:|:---:|:---:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 1 |

**Expected steps:**

1. Choose $w_1$, $w_2$ and $b$ by reasoning: what is the difference between OR and AND?
2. **Verify manually** for all 4 cases (full table, as above).
3. Implement and test.
4. Explain in one sentence why your bias works.

```python
def perceptron_OR(x1, x2):
    # To complete
    w1 = ...
    w2 = ...
    b  = ...
    z = w1 * x1 + w2 * x2 + b
    return 1 if z >= 0 else 0

# Test
for x1 in [0, 1]:
    for x2 in [0, 1]:
        print(f"  {x1} OR {x2} = {perceptron_OR(x1, x2)}")
```

> **Note:** several different bias values work. Justify yours.

---

## 5. Visual Understanding: Separating Points

### The Geometric View

A perceptron with 2 inputs draws a **line** in 2D space.

The equation of the decision boundary (where $z = 0$) is:

$$
w_1 x_1 + w_2 x_2 + b = 0
$$

This is the equation of a line. Points on one side → Class 0, the other → Class 1.

Rewriting in slope-intercept form:

$$
x_2 = -\frac{w_1}{w_2} x_1 - \frac{b}{w_2}
$$

> **Intuition:** the weights control the **orientation** of the line, the bias controls its **offset**.

### Visualization: AND Gate

```python
import matplotlib.pyplot as plt
import numpy as np

points = np.array([[0,0],[0,1],[1,0],[1,1]])
labels = np.array([0, 0, 0, 1])

w1, w2, b = 1, 1, -1.5

plt.figure(figsize=(7, 7))

# Class 0 (blue circles)
c0 = points[labels == 0]
plt.scatter(c0[:, 0], c0[:, 1], s=200, c='royalblue', marker='o',
            edgecolors='black', linewidth=2, label='Class 0 (output = 0)', zorder=5)

# Class 1 (red squares)
c1 = points[labels == 1]
plt.scatter(c1[:, 0], c1[:, 1], s=200, c='tomato', marker='s',
            edgecolors='black', linewidth=2, label='Class 1 (output = 1)', zorder=5)

# Decision boundary
x1_line = np.linspace(-0.5, 1.5, 200)
x2_line = -(w1 * x1_line + b) / w2
plt.plot(x1_line, x2_line, 'g-', linewidth=2.5, label='Decision boundary')

# Annotations
for pt, lbl in zip(points, ['(0,0)', '(0,1)', '(1,0)', '(1,1)']):
    plt.annotate(lbl, xy=pt, xytext=(pt[0]+0.05, pt[1]+0.05), fontsize=10)

plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.xlabel('$x_1$', fontsize=13)
plt.ylabel('$x_2$', fontsize=13)
plt.title('AND Gate — Decision Boundary', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.axhline(0, color='k', linewidth=0.5)
plt.axvline(0, color='k', linewidth=0.5)
plt.show()
```

**Reading the plot:**
- The green line separates blue points (Class 0) from red points (Class 1)
- All blue points are below/left → correctly classified
- The red point $(1,1)$ is above/right → correctly classified

---

### 📝 Exercise 5.1: Visualize Your OR Gate — and Understand Variants

**Part A:** Plot the decision boundary for your OR gate (use the weights from Exercise 4.1).

```python
# To complete with your own w1, w2, b
```

**Part B — Reflection question:** Does the point $(0.5, 0.5)$ belong to Class 0 or Class 1 according to your perceptron? Compute $z$ for this point and verify it graphically.

**Part C — Exploration:** Try a second valid bias for the OR gate. Plot both boundaries on the same graph. What do you observe? What does this tell you about the uniqueness of the solution?

> **Goal:** understand that multiple lines can solve the same linearly separable problem.

---

## 6. From Manual Calculations to Matrix Operations

### Why Matrices?

When we have **many examples** to test, point-by-point calculation is slow:

```python
# For 100 points, we loop — not efficient
for i in range(100):
    z = w1*x1[i] + w2*x2[i] + b
    output[i] = 1 if z >= 0 else 0
```

**Solution:** matrix operations allow computing everything **in a single line**.

### Vector Notation

Instead of writing:
$$
z = w_1 x_1 + w_2 x_2 + b
$$

We write:
$$
z = \mathbf{w}^\top \mathbf{x} + b
$$

where $\mathbf{w} = [w_1, w_2]^\top$ and $\mathbf{x} = [x_1, x_2]^\top$.

For multiple examples at once, we stack inputs into a matrix $X$ (one row = one example):

$$
\mathbf{z} = X \mathbf{w} + b
$$

### Breaking Down the Matrix Product

$$
X \mathbf{w} =
\begin{bmatrix}
0 & 0 \\
0 & 1 \\
1 & 0 \\
1 & 1
\end{bmatrix}
\begin{bmatrix} 1 \\ 1 \end{bmatrix}
=
\begin{bmatrix}
0 \times 1 + 0 \times 1 \\
0 \times 1 + 1 \times 1 \\
1 \times 1 + 0 \times 1 \\
1 \times 1 + 1 \times 1
\end{bmatrix}
=
\begin{bmatrix} 0 \\ 1 \\ 1 \\ 2 \end{bmatrix}
$$

Then: $\mathbf{z} = [0,1,1,2] + (-1.5) = [-1.5, -0.5, -0.5, 0.5]$

### AND Gate with Matrices

```python
import numpy as np

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

w = np.array([1.0, 1.0])
b = -1.5

# Vectorized computation — all examples at once
z = X @ w + b
outputs = (z >= 0).astype(int)

print("Weighted sums z:", z)
print("Outputs        :", outputs)

print("\nTruth table:")
for i in range(4):
    print(f"  {X[i]} → z={z[i]:+.1f} → output={outputs[i]}")
```

**Output:**
```
Weighted sums z: [-1.5 -0.5 -0.5  0.5]
Outputs        : [0 0 0 1]

Truth table:
  [0 0] → z=-1.5 → output=0
  [0 1] → z=-0.5 → output=0
  [1 0] → z=-0.5 → output=0
  [1 1] → z=+0.5 → output=1
```

> **Key advantage:** a single `X @ w + b` operation for all examples. ⚡

---

## 7. Building a Classifier

### Realistic Problem: Classifying Students

**Scenario:** predict whether a student passes or fails an exam based on:
- $x_1$: hours of study
- $x_2$: score on the previous test

### Generate Synthetic Data

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# "Pass" students (Class 1)
n_pass = 30
pass_hours  = np.random.uniform(5, 10, n_pass)
pass_scores = np.random.uniform(60, 90, n_pass)

# "Fail" students (Class 0)
n_fail = 30
fail_hours  = np.random.uniform(0, 6, n_fail)
fail_scores = np.random.uniform(20, 65, n_fail)

X = np.vstack([
    np.column_stack([fail_hours, fail_scores]),
    np.column_stack([pass_hours, pass_scores])
])
y = np.concatenate([np.zeros(n_fail), np.ones(n_pass)])

print(f"Dataset: {len(X)} students, {int(y.sum())} pass, {int((1-y).sum())} fail")
```

### Visualize the Data

```python
plt.figure(figsize=(9, 7))

plt.scatter(X[y==0, 0], X[y==0, 1], s=100, c='royalblue', marker='o',
            edgecolors='black', linewidth=1.2, alpha=0.8, label='Fail (Class 0)')
plt.scatter(X[y==1, 0], X[y==1, 1], s=100, c='tomato', marker='s',
            edgecolors='black', linewidth=1.2, alpha=0.8, label='Pass (Class 1)')

plt.xlabel("Hours studied", fontsize=13)
plt.ylabel("Previous test score", fontsize=13)
plt.title("Student Classification Problem", fontsize=15)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()
```

### Designing the Perceptron Intuitively

More study hours → higher chance of passing → $w_1 > 0$

Better previous score → higher chance of passing → $w_2 > 0$

```python
w = np.array([2.0, 0.5])
b = -60.0

print(f"Boundary equation: {w[0]}·hours + {w[1]}·score + {b} = 0")
print(f"i.e.: score = {-b/w[1]:.0f} - {w[0]/w[1]:.0f}·hours")
```

### Evaluating the Classifier

```python
z = X @ w + b
y_pred = (z >= 0).astype(int)

accuracy = np.mean(y_pred == y)
print(f"Accuracy: {accuracy * 100:.1f}%")

# Simple confusion matrix
tp = np.sum((y_pred == 1) & (y == 1))
tn = np.sum((y_pred == 0) & (y == 0))
fp = np.sum((y_pred == 1) & (y == 0))
fn = np.sum((y_pred == 0) & (y == 1))

print(f"\nTrue Positives  (TP): {tp}")
print(f"True Negatives  (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
```

### Visualizing the Decision Boundary

```python
plt.figure(figsize=(9, 7))

plt.scatter(X[y==0, 0], X[y==0, 1], s=100, c='royalblue', marker='o',
            edgecolors='black', linewidth=1.2, alpha=0.8, label='Fail (Class 0)')
plt.scatter(X[y==1, 0], X[y==1, 1], s=100, c='tomato', marker='s',
            edgecolors='black', linewidth=1.2, alpha=0.8, label='Pass (Class 1)')

x1_line = np.linspace(0, 10, 200)
x2_line = -(w[0] * x1_line + b) / w[1]

plt.plot(x1_line, x2_line, 'g-', linewidth=2.5, label='Decision boundary', alpha=0.9)
plt.fill_between(x1_line, x2_line, 100, alpha=0.08, color='tomato',    label='Pass region')
plt.fill_between(x1_line, 0,       x2_line, alpha=0.08, color='royalblue', label='Fail region')

plt.xlabel("Hours studied", fontsize=13)
plt.ylabel("Previous test score", fontsize=13)
plt.title(f"Student Classifier (accuracy: {accuracy*100:.1f}%)", fontsize=15)
plt.legend(fontsize=11, loc='lower right')
plt.grid(True, alpha=0.3)
plt.xlim(0, 10)
plt.ylim(20, 90)
plt.show()
```

### Interpreting the Boundary

The line $2 \times \text{hours} + 0.5 \times \text{score} - 60 = 0$ gives:

$$
\text{score} = 120 - 4 \times \text{hours}
$$

- 0 hours studied → need a score of 120 (impossible → always fail)
- 5 hours → need a score of 100 (very demanding)
- 10 hours → need a score of 80 (reasonable)

### Testing New Students

```python
new_students = np.array([
    [3,  50],  # 3h, score 50
    [7,  70],  # 7h, score 70
    [9,  85],  # 9h, score 85
    [2,  40],  # 2h, score 40
])

z_new = new_students @ w + b
preds = (z_new >= 0).astype(int)

print("Predictions for new students:")
for i, (student, z_val, pred) in enumerate(zip(new_students, z_new, preds)):
    label = "Pass" if pred == 1 else "Fail"
    print(f"  Student {i+1}: {student[0]}h, score {student[1]} → z={z_val:.1f} → {label}")
```

---

## 8. Exercises

### 📝 Exercise 8.1: NOT Gate (Easy)

**Task:** Design a single-input perceptron that implements the NOT gate.

| $x$ | NOT |
|:---:|:---:|
| 0 | 1 |
| 1 | 0 |

**Expected steps:**

1. Reason: if $x$ increases, the output must... ? What sign should $w_1$ have?
2. Verify by hand for $x=0$ and $x=1$.
3. Implement and test.

```python
def perceptron_NOT(x):
    # To complete — justify your choice of w1 and b
    w1 = ...
    b  = ...
    z = w1 * x + b
    return 1 if z >= 0 else 0

print("NOT(0) =", perceptron_NOT(0))  # Expected: 1
print("NOT(1) =", perceptron_NOT(1))  # Expected: 0
```

---

### 📝 Exercise 8.2: NAND Gate (Medium)

**Task:** Design a perceptron for the NAND gate ("NOT AND").

| $x_1$ | $x_2$ | NAND |
|:---:|:---:|:---:|
| 0 | 0 | 1 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

**Questions:**

1. What do you notice compared to AND? What mathematical transformation has occurred on the outputs?
2. Deduce the weights and bias without any additional computation.
3. Verify manually (full table).
4. Implement and test.

```python
def perceptron_NAND(x1, x2):
    # To complete
    pass
```

> **Conceptual hint:** if you have AND with parameters $(w_1, w_2, b)$, what happens if you multiply all parameters by $-1$?

---

### 📝 Exercise 8.3: Given Boundary → Manual Classification (Medium)

**Task:** You are given the following decision boundary:

$$
3x_1 + 2x_2 - 5 = 0
$$

**Questions:**

1. Identify $w_1$, $w_2$ and $b$.
2. Classify **by hand** (without a computer) the following points. For each, compute $z$ and conclude:
   - Point A: $(1, 1)$
   - Point B: $(2, 0)$
   - Point C: $(0, 3)$
3. Rewrite the boundary in the form $x_2 = f(x_1)$ and interpret the slope and intercept.
4. Plot the line and the three points, colored by class.

```python
import numpy as np
import matplotlib.pyplot as plt

w1, w2, b = ..., ..., ...

points = np.array([[1, 1], [2, 0], [0, 3]])
names  = ['A', 'B', 'C']

# Compute z and classify
z = points @ np.array([w1, w2]) + b
classes = (z >= 0).astype(int)

for name, pt, z_val, c in zip(names, points, z, classes):
    print(f"Point {name} {tuple(pt)}: z = {z_val:.1f} → Class {c}")

# Plot
# ... (to complete)
```

---

### 📝 Exercise 8.4: Searching for Better Weights (Hard)

**Context:** The student classifier from Section 7 does not reach 100% accuracy.

**Questions:**

1. **Visualize first.** Plot the two classes. Is there a line that separates them perfectly? Justify visually.

2. **Systematic search.** Test all combinations below and find the best parameters:

```python
w1_values = [1.5, 2.0, 2.5, 3.0]
w2_values = [0.3, 0.5, 0.7, 1.0]
b_values  = [-50, -60, -70, -80]

best_accuracy = 0
best_params   = None

for w1 in w1_values:
    for w2 in w2_values:
        for b_val in b_values:
            w = np.array([w1, w2])
            z = X @ w + b_val
            acc = np.mean((z >= 0).astype(int) == y)
            if acc > best_accuracy:
                best_accuracy = acc
                best_params   = (w1, w2, b_val)

print(f"Best accuracy: {best_accuracy*100:.1f}%")
print(f"Parameters: w1={best_params[0]}, w2={best_params[1]}, b={best_params[2]}")
```

3. **Reflection:** Why is 100% accuracy impossible with a single line on this data? What property of the dataset explains it? *(Hint: look at the overlap region between the two classes.)*

---

### 📝 Exercise 8.5: The XOR Problem — Why One Line Is Not Enough (Challenge)

**Task:** Can you separate these points with a single perceptron?

```python
import numpy as np
import matplotlib.pyplot as plt

points = np.array([[0,0], [0,1], [1,0], [1,1]])
labels = np.array([1, 0, 0, 1])  # XOR: 1 when x1 ≠ x2
```

**Part A — Experimental Exploration:**

Try several sets of weights. Display the boundary and the number of errors for each.

```python
candidates = [
    (1, -1, 0),
    (1,  1, -1),
    (-1, 1, 0),
    # Add your own attempts...
]

for w1, w2, b in candidates:
    w = np.array([w1, w2])
    z = points @ w + b
    preds  = (z >= 0).astype(int)
    errors = int(np.sum(preds != labels))
    print(f"w1={w1:+}, w2={w2:+}, b={b:+} → {errors} error(s)")
```

**Part B — Geometric Proof:**

1. Plot the 4 points (Class 0 in blue, Class 1 in red).
2. Try to draw a line that separates the two classes. What do you observe?

**Part C — Theoretical Question:**

Explain in 3-4 sentences why no linear perceptron can solve XOR. What geometric property of the data makes it impossible? What architecture would be needed to solve this problem?

> **Going further:** Sketch by hand how *two* perceptrons in series (a network with one hidden layer) could solve XOR. This is the central intuition that leads to deep networks.

---

## Summary

### What You Have Learned

**The Perceptron Model**
- Formula: $z = w_1 x_1 + w_2 x_2 + b$, output: $f(z)$
- Role of weights (feature importance) and bias (threshold offset)

**Activation Functions**
- Step (0 or 1), Sign (+1 or -1), Sigmoid (smooth), ReLU (modern)

**Manual Calculation**
- Verifying case by case, understanding each step

**Geometric View**
- 1 perceptron = 1 line in $\mathbb{R}^2$
- Weights control orientation, bias controls offset

**Matrix Operations**
- $\mathbf{z} = X\mathbf{w} + b$ — all examples at once

**Real Application**
- Data generation, intuitive design, evaluation, visualization

### Key Formulas

$$
\hat{y} = f\!\left(\sum_{i=1}^{n} w_i x_i + b\right) = f(\mathbf{w}^\top \mathbf{x} + b)
$$

**Decision boundary (2D):**
$$
x_2 = -\frac{w_1}{w_2}\, x_1 - \frac{b}{w_2}
$$

### Python Quick Reference

```python
# Basic perceptron
z = w1*x1 + w2*x2 + b
output = 1 if z >= 0 else 0

# Vectorized version
z = X @ w + b
outputs = (z >= 0).astype(int)

# Accuracy
accuracy = np.mean(predictions == true_labels)
```

### Identified Limitations

1. **One perceptron = one line** → can only separate linearly separable data
2. **Weights are set manually** → how can they be learned automatically?
3. **XOR is unsolvable** → multiple perceptrons arranged in layers are needed

### Next Steps

- **Automatic learning**: finding the right weights via gradient descent
- **Multi-layer networks**: stacking perceptrons to solve XOR and beyond
- **Non-linear problems**: combining ReLU and sigmoid activation functions
- **Deep learning**: modern neural networks with millions of weights

---

**Congratulations!** 🎉 You now understand the fundamental building block of every neural network: the perceptron.
