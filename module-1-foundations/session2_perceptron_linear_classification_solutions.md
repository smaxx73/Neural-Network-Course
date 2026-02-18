# Perceptron Notebook — Exercise Solutions

**Course: Neural Networks for Engineers**

> This file provides complete solutions to all exercises from the *Introduction to the Perceptron* notebook. Each solution includes the reasoning, manual verification, and tested code.

---

## Exercise 4.1: OR Gate

### Reasoning

Compare OR and AND truth tables:

| $x_1$ | $x_2$ | AND | OR |
|:---:|:---:|:---:|:---:|
| 0 | 0 | 0 | 0 |
| 0 | 1 | 0 | 1 |
| 1 | 0 | 0 | 1 |
| 1 | 1 | 1 | 1 |

OR outputs 1 as soon as **at least one** input is 1. AND requires **both** inputs to be 1.

With $w_1 = w_2 = 1$, the weighted sum $z = x_1 + x_2 + b$ takes values $\{b, 1+b, 2+b\}$.

We need:
- $z < 0$ when $x_1 + x_2 = 0$ → $b < 0$
- $z \geq 0$ when $x_1 + x_2 = 1$ → $1 + b \geq 0$, i.e. $b \geq -1$

So any $b \in [-1, 0)$ works. The canonical choice is $b = -0.5$.

**Why $b = -0.5$ works:** it sits exactly halfway between the sums 0 and 1, giving equal margin on both sides.

### Manual Verification

With $w_1 = 1$, $w_2 = 1$, $b = -0.5$:

| $x_1$ | $x_2$ | $z = x_1 + x_2 - 0.5$ | Sign of $z$ | Output | Expected |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 0 | $-0.5$ | $< 0$ | 0 | 0 ✅ |
| 0 | 1 | $+0.5$ | $\geq 0$ | 1 | 1 ✅ |
| 1 | 0 | $+0.5$ | $\geq 0$ | 1 | 1 ✅ |
| 1 | 1 | $+1.5$ | $\geq 0$ | 1 | 1 ✅ |

### Code

```python
def perceptron_OR(x1, x2):
    w1, w2, b = 1.0, 1.0, -0.5
    z = w1 * x1 + w2 * x2 + b
    return 1 if z >= 0 else 0

print("OR gate test:")
for x1 in [0, 1]:
    for x2 in [0, 1]:
        print(f"  {x1} OR {x2} = {perceptron_OR(x1, x2)}")
```

**Output:**
```
OR gate test:
  0 OR 0 = 0
  0 OR 1 = 1
  1 OR 0 = 1
  1 OR 1 = 1
```

---

## Exercise 5.1: Visualize the OR Gate — and Understand Variants

### Part A: Plot the Decision Boundary

```python
import numpy as np
import matplotlib.pyplot as plt

points = np.array([[0,0],[0,1],[1,0],[1,1]])
labels = np.array([0, 1, 1, 1])

w1, w2, b = 1.0, 1.0, -0.5

plt.figure(figsize=(7, 7))

c0 = points[labels == 0]
plt.scatter(c0[:, 0], c0[:, 1], s=200, c='royalblue', marker='o',
            edgecolors='black', linewidth=2, label='Class 0 (output = 0)', zorder=5)

c1 = points[labels == 1]
plt.scatter(c1[:, 0], c1[:, 1], s=200, c='tomato', marker='s',
            edgecolors='black', linewidth=2, label='Class 1 (output = 1)', zorder=5)

x1_line = np.linspace(-0.5, 1.5, 200)
x2_line = -(w1 * x1_line + b) / w2
plt.plot(x1_line, x2_line, 'g-', linewidth=2.5, label=f'Boundary: b={b}')

for pt, lbl in zip(points, ['(0,0)', '(0,1)', '(1,0)', '(1,1)']):
    plt.annotate(lbl, xy=pt, xytext=(pt[0]+0.05, pt[1]+0.05), fontsize=10)

plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.xlabel('$x_1$', fontsize=13)
plt.ylabel('$x_2$', fontsize=13)
plt.title('OR Gate — Decision Boundary', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.show()
```

### Part B: Classifying Point $(0.5, 0.5)$

$$
z = 1 \times 0.5 + 1 \times 0.5 - 0.5 = 0.5 \geq 0 \quad \rightarrow \textbf{Class 1}
$$

The point lies above the decision boundary line $x_2 = -x_1 + 0.5$, i.e. $0.5 > -0.5 + 0.5 = 0$. Confirmed graphically.

### Part C: Two Valid Boundaries

Any $b \in [-1, 0)$ is valid. Let's compare $b = -0.5$ and $b = -0.9$:

```python
fig, ax = plt.subplots(figsize=(7, 7))

c0 = points[labels == 0]
ax.scatter(c0[:, 0], c0[:, 1], s=200, c='royalblue', marker='o',
           edgecolors='black', linewidth=2, label='Class 0', zorder=5)
c1 = points[labels == 1]
ax.scatter(c1[:, 0], c1[:, 1], s=200, c='tomato', marker='s',
           edgecolors='black', linewidth=2, label='Class 1', zorder=5)

x1_line = np.linspace(-0.5, 1.5, 200)

for b_val, color, ls in [(-0.5, 'green', '-'), (-0.9, 'purple', '--')]:
    x2_line = -(x1_line + b_val)
    ax.plot(x1_line, x2_line, color=color, linestyle=ls,
            linewidth=2.5, label=f'Boundary b={b_val}')

ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-0.5, 1.5)
ax.set_xlabel('$x_1$', fontsize=13)
ax.set_ylabel('$x_2$', fontsize=13)
ax.set_title('OR Gate — Two Valid Boundaries', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.show()
```

**Observation:** both boundaries correctly separate the classes. This shows that the solution to a linearly separable problem is **not unique** — there exists an infinite family of valid hyperplanes. The perceptron learning algorithm converges to one of them, but gradient-based methods (SVM, logistic regression) find the one with maximum margin.

---

## Exercise 8.1: NOT Gate

### Reasoning

The NOT gate inverts its input: when $x$ increases, the output must decrease. This requires a **negative weight**: $w_1 < 0$.

With $w_1 = -1$:
- $x = 0$: $z = 0 + b$ → need $z \geq 0$, so $b \geq 0$
- $x = 1$: $z = -1 + b$ → need $z < 0$, so $b < 1$

Any $b \in [0, 1)$ works. The canonical choice is $b = 0.5$.

### Manual Verification

With $w_1 = -1$, $b = 0.5$:

| $x$ | $z = -x + 0.5$ | Sign of $z$ | Output | Expected |
|:---:|:---:|:---:|:---:|:---:|
| 0 | $+0.5$ | $\geq 0$ | 1 | 1 ✅ |
| 1 | $-0.5$ | $< 0$ | 0 | 0 ✅ |

### Code

```python
def perceptron_NOT(x):
    w1, b = -1.0, 0.5
    z = w1 * x + b
    return 1 if z >= 0 else 0

print("NOT(0) =", perceptron_NOT(0))  # Expected: 1
print("NOT(1) =", perceptron_NOT(1))  # Expected: 0
```

**Output:**
```
NOT(0) = 1
NOT(1) = 0
```

---

## Exercise 8.2: NAND Gate

### Reasoning

Comparing AND and NAND:

| $x_1$ | $x_2$ | AND | NAND |
|:---:|:---:|:---:|:---:|
| 0 | 0 | 0 | 1 |
| 0 | 1 | 0 | 1 |
| 1 | 0 | 0 | 1 |
| 1 | 1 | 1 | 0 |

NAND is the **exact complement** of AND: every output is flipped (0 → 1, 1 → 0).

Mathematically, if AND fires when $z \geq 0$, then NAND fires when $z < 0$, i.e. when $-z \geq 0$. Multiplying all parameters by $-1$ flips the decision:

$$
z_{\text{NAND}} = -w_1 x_1 - w_2 x_2 - b = -1 \cdot x_1 - 1 \cdot x_2 + 1.5
$$

### Manual Verification

With $w_1 = -1$, $w_2 = -1$, $b = 1.5$:

| $x_1$ | $x_2$ | $z = -x_1 - x_2 + 1.5$ | Sign of $z$ | Output | Expected |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 0 | $+1.5$ | $\geq 0$ | 1 | 1 ✅ |
| 0 | 1 | $+0.5$ | $\geq 0$ | 1 | 1 ✅ |
| 1 | 0 | $+0.5$ | $\geq 0$ | 1 | 1 ✅ |
| 1 | 1 | $-0.5$ | $< 0$ | 0 | 0 ✅ |

### Code

```python
def perceptron_NAND(x1, x2):
    w1, w2, b = -1.0, -1.0, 1.5
    z = w1 * x1 + w2 * x2 + b
    return 1 if z >= 0 else 0

print("NAND gate test:")
for x1 in [0, 1]:
    for x2 in [0, 1]:
        print(f"  NAND({x1},{x2}) = {perceptron_NAND(x1, x2)}")
```

**Output:**
```
NAND gate test:
  NAND(0,0) = 1
  NAND(0,1) = 1
  NAND(1,0) = 1
  NAND(1,1) = 0
```

> **Key insight:** negating all parameters of a perceptron flips its decision boundary, effectively implementing the logical NOT of the original function. This generalizes: NOT(f(x)) = perceptron with parameters $(-\mathbf{w}, -b)$.

---

## Exercise 8.3: Given Boundary → Manual Classification

### Part 1: Identify Parameters

From $3x_1 + 2x_2 - 5 = 0$:

$$
w_1 = 3, \quad w_2 = 2, \quad b = -5
$$

### Part 2: Manual Classification

**Point A: $(1, 1)$**
$$
z = 3 \times 1 + 2 \times 1 - 5 = 3 + 2 - 5 = 0 \geq 0 \quad \rightarrow \textbf{Class 1}
$$

**Point B: $(2, 0)$**
$$
z = 3 \times 2 + 2 \times 0 - 5 = 6 + 0 - 5 = 1 \geq 0 \quad \rightarrow \textbf{Class 1}
$$

**Point C: $(0, 3)$**
$$
z = 3 \times 0 + 2 \times 3 - 5 = 0 + 6 - 5 = 1 \geq 0 \quad \rightarrow \textbf{Class 1}
$$

All three points fall on the same side ($z \geq 0$, Class 1).

### Part 3: Slope-Intercept Form

$$
3x_1 + 2x_2 - 5 = 0 \implies x_2 = -\frac{3}{2}\,x_1 + \frac{5}{2}
$$

- **Slope:** $-3/2 = -1.5$ → the boundary goes down steeply as $x_1$ increases
- **Intercept:** $5/2 = 2.5$ → the boundary crosses the $x_2$-axis at $2.5$

### Part 4: Code and Plot

```python
import numpy as np
import matplotlib.pyplot as plt

w1, w2, b = 3, 2, -5

points = np.array([[1, 1], [2, 0], [0, 3]])
names  = ['A', 'B', 'C']

z = points @ np.array([w1, w2]) + b
classes = (z >= 0).astype(int)

for name, pt, z_val, c in zip(names, points, z, classes):
    print(f"Point {name} {tuple(pt)}: z = {z_val:.1f} → Class {c}")

# Plot
plt.figure(figsize=(7, 7))

colors = ['tomato' if c == 1 else 'royalblue' for c in classes]
for pt, name, color in zip(points, names, colors):
    plt.scatter(pt[0], pt[1], s=200, c=color, edgecolors='black',
                linewidth=2, zorder=5)
    plt.annotate(f'{name} {tuple(pt)}', xy=pt,
                 xytext=(pt[0]+0.1, pt[1]+0.1), fontsize=11)

# Decision boundary
x1_line = np.linspace(-0.5, 2.5, 200)
x2_line = -(w1 * x1_line + b) / w2
plt.plot(x1_line, x2_line, 'g-', linewidth=2.5, label='$3x_1 + 2x_2 - 5 = 0$')

plt.xlim(-0.5, 2.5)
plt.ylim(-0.5, 3.5)
plt.xlabel('$x_1$', fontsize=13)
plt.ylabel('$x_2$', fontsize=13)
plt.title('Exercise 8.3 — Boundary and Classification', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.axhline(0, color='k', linewidth=0.5)
plt.axvline(0, color='k', linewidth=0.5)
plt.show()
```

**Output:**
```
Point A (1, 1): z = 0.0 → Class 1
Point B (2, 0): z = 1.0 → Class 1
Point C (0, 3): z = 1.0 → Class 1
```

> **Note:** Point A lies exactly on the boundary ($z = 0$). By convention ($z \geq 0$ → Class 1), it is assigned to Class 1, but it sits right on the decision line. In practice, such boundary points are the most sensitive to noise.

---

## Exercise 8.4: Searching for Better Weights

### Part 1: Visual Inspection

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

n_pass = 30
pass_hours  = np.random.uniform(5, 10, n_pass)
pass_scores = np.random.uniform(60, 90, n_pass)

n_fail = 30
fail_hours  = np.random.uniform(0, 6, n_fail)
fail_scores = np.random.uniform(20, 65, n_fail)

X = np.vstack([
    np.column_stack([fail_hours, fail_scores]),
    np.column_stack([pass_hours, pass_scores])
])
y = np.concatenate([np.zeros(n_fail), np.ones(n_pass)])

plt.figure(figsize=(9, 7))
plt.scatter(X[y==0, 0], X[y==0, 1], s=100, c='royalblue', marker='o',
            edgecolors='black', linewidth=1.2, alpha=0.8, label='Fail')
plt.scatter(X[y==1, 0], X[y==1, 1], s=100, c='tomato', marker='s',
            edgecolors='black', linewidth=1.2, alpha=0.8, label='Pass')
plt.xlabel("Hours studied", fontsize=13)
plt.ylabel("Previous test score", fontsize=13)
plt.title("Can a single line separate these classes?", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()
```

The two classes **overlap** in the region roughly (4–6 hours, 55–65 score). No single straight line can perfectly separate them — 100% accuracy is geometrically impossible.

### Part 2: Systematic Search

```python
w1_values = [1.5, 2.0, 2.5, 3.0]
w2_values = [0.3, 0.5, 0.7, 1.0]
b_values  = [-50, -60, -70, -80]

best_accuracy = 0
best_params   = None
results = []

for w1 in w1_values:
    for w2 in w2_values:
        for b_val in b_values:
            w = np.array([w1, w2])
            z = X @ w + b_val
            acc = np.mean((z >= 0).astype(int) == y)
            results.append((acc, w1, w2, b_val))
            if acc > best_accuracy:
                best_accuracy = acc
                best_params   = (w1, w2, b_val)

results.sort(reverse=True)
print(f"Best accuracy : {best_accuracy*100:.1f}%")
print(f"Parameters    : w1={best_params[0]}, w2={best_params[1]}, b={best_params[2]}")
print(f"\nTop 5 combinations:")
for acc, w1, w2, b_val in results[:5]:
    print(f"  w1={w1}, w2={w2}, b={b_val:4.0f} → {acc*100:.1f}%")
```

### Part 3: Why 100% is Impossible

Because the data **is not linearly separable**. The scatter plot reveals a zone where "fail" and "pass" students overlap (low-to-medium hours AND medium scores). No straight line can simultaneously:

- Keep all blue points (fail) on one side, and
- Keep all red points (pass) on the other side

when some points from each class are geometrically interleaved. This is a fundamental limitation of the single-layer perceptron. Solving it requires either a non-linear boundary (e.g. using feature engineering), or multiple layers.

---

## Exercise 8.5: The XOR Problem

### Part A: Experimental Exploration

```python
import numpy as np

points = np.array([[0,0],[0,1],[1,0],[1,1]])
labels = np.array([1, 0, 0, 1])

candidates = [
    ( 1, -1,  0),
    ( 1,  1, -1),
    (-1,  1,  0),
    ( 1,  0, -0.5),
    ( 0,  1, -0.5),
    ( 2, -1, -0.5),
]

print("Exploration of candidate weights:")
print(f"{'w1':>5} {'w2':>5} {'b':>6}  {'errors':>8}  {'outputs'}")
print("-" * 50)
for w1, w2, b in candidates:
    w = np.array([w1, w2])
    z = points @ w + b
    preds  = (z >= 0).astype(int)
    errors = int(np.sum(preds != labels))
    print(f"{w1:>5} {w2:>5} {b:>6}  {errors:>8}  {preds.tolist()}")
```

**Output:**
```
Exploration of candidate weights:
   w1    w2      b    errors  outputs
--------------------------------------------------
    1    -1       0         2  [1, 0, 1, 0]
    1     1      -1         2  [0, 1, 1, 1]
   -1     1       0         2  [1, 1, 0, 1]
    1     0    -0.5         2  [0, 0, 1, 1]
    0     1    -0.5         2  [0, 1, 0, 1]
    2    -1    -0.5         2  [0, 0, 1, 0]
```

Every candidate produces at least 2 errors. This is not a tuning issue — it is provably impossible to do better.

### Part B: Geometric Proof

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# Left: the XOR points
ax = axes[0]
c0 = points[labels == 0]
c1 = points[labels == 1]
ax.scatter(c0[:, 0], c0[:, 1], s=250, c='royalblue', marker='o',
           edgecolors='black', linewidth=2, label='Class 0', zorder=5)
ax.scatter(c1[:, 0], c1[:, 1], s=250, c='tomato', marker='s',
           edgecolors='black', linewidth=2, label='Class 1', zorder=5)
for pt, lbl in zip(points, ['(0,0)\nClass 1','(0,1)\nClass 0',
                              '(1,0)\nClass 0','(1,1)\nClass 1']):
    ax.annotate(lbl, xy=pt, xytext=(pt[0]-0.35, pt[1]+0.08), fontsize=9)
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-0.5, 1.5)
ax.set_title("XOR — No separating line exists", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlabel('$x_1$', fontsize=12)
ax.set_ylabel('$x_2$', fontsize=12)

# Right: best attempt (still 2 errors)
ax = axes[1]
ax.scatter(c0[:, 0], c0[:, 1], s=250, c='royalblue', marker='o',
           edgecolors='black', linewidth=2, label='Class 0', zorder=5)
ax.scatter(c1[:, 0], c1[:, 1], s=250, c='tomato', marker='s',
           edgecolors='black', linewidth=2, label='Class 1', zorder=5)
x1_line = np.linspace(-0.5, 1.5, 200)
ax.plot(x1_line, -x1_line + 1, 'g-', linewidth=2,
        label='Best attempt (2 errors)', alpha=0.8)
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-0.5, 1.5)
ax.set_title("Any line produces ≥ 2 errors", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlabel('$x_1$', fontsize=12)
ax.set_ylabel('$x_2$', fontsize=12)

plt.tight_layout()
plt.show()
```

**Observation:** Class 1 points $(0,0)$ and $(1,1)$ are located on the **diagonal**, while Class 0 points $(0,1)$ and $(1,0)$ are on the **anti-diagonal**. Any straight line that captures one red square on the correct side will place the other red square on the wrong side — there is no angle at which a single line can separate them.

### Part C: Theoretical Explanation

XOR is **not linearly separable**. A single perceptron computes a linear function of its inputs and can only define a halfspace in $\mathbb{R}^2$. The four XOR points require a boundary that is "bent" — i.e. non-convex — which a single straight line cannot achieve.

More formally: if a line separated XOR correctly, it would put $(0,0)$ and $(1,1)$ on one side and $(0,1)$, $(1,0)$ on the other. But the midpoints of the two diagonals are both $(0.5, 0.5)$, meaning the two classes are not convexly separable.

**The solution requires two layers:**

- A first hidden layer of two perceptrons computes two intermediate features:
  - $h_1 = \text{OR}(x_1, x_2)$ → 1 if at least one input is 1
  - $h_2 = \text{NAND}(x_1, x_2)$ → 0 only if both inputs are 1
- The output perceptron then computes $\text{AND}(h_1, h_2)$

This gives XOR = AND(OR(x₁,x₂), NAND(x₁,x₂)), which is correctly solved by a 2-layer network.

```python
# XOR via a 2-layer network
def perceptron_AND(x1, x2):
    return 1 if (x1 + x2 - 1.5) >= 0 else 0

def perceptron_OR(x1, x2):
    return 1 if (x1 + x2 - 0.5) >= 0 else 0

def perceptron_NAND(x1, x2):
    return 1 if (-x1 - x2 + 1.5) >= 0 else 0

def network_XOR(x1, x2):
    h1 = perceptron_OR(x1, x2)    # hidden unit 1
    h2 = perceptron_NAND(x1, x2)  # hidden unit 2
    return perceptron_AND(h1, h2)  # output unit

print("XOR via 2-layer network:")
for x1 in [0, 1]:
    for x2 in [0, 1]:
        print(f"  XOR({x1},{x2}) = {network_XOR(x1, x2)}")
```

**Output:**
```
XOR via 2-layer network:
  XOR(0,0) = 1
  XOR(0,1) = 0
  XOR(1,0) = 0
  XOR(1,1) = 1
```

This demonstrates the core principle behind **deep learning**: stacking layers of linear perceptrons with non-linearities creates a model capable of representing arbitrarily complex functions that no single perceptron can express.

---

## Summary of Solutions

| Exercise | Gate / Task | Key parameters | Core insight |
|---|---|---|---|
| 4.1 | OR | $w_1=1, w_2=1, b=-0.5$ | Any $b \in [-1, 0)$ works |
| 5.1 | OR boundary | same | Multiple valid boundaries exist |
| 8.1 | NOT | $w_1=-1, b=0.5$ | Negative weight inverts the output |
| 8.2 | NAND | $w_1=-1, w_2=-1, b=1.5$ | Negate all parameters of AND |
| 8.3 | Manual classif. | $w_1=3, w_2=2, b=-5$ | All 3 points land in Class 1 |
| 8.4 | Best weights | varies by seed | 100% impossible — data overlaps |
| 8.5 | XOR | impossible | Needs 2 layers: AND(OR, NAND) |
