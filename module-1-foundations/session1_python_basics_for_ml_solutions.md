# Session 1 — Solutions to All Exercises

**Python Basics for Machine Learning**

---

## Exercise 2.1 (Easy) — Variables

> Create variables for your age, height, and name, then print them.

```python
age = 22
height = 1.75
name = "Alice"

print("Name:", name)
print("Age:", age)
print("Height:", height, "m")
```

---

## Exercise 2.2 (Easy) — Circle Area and Circumference

> Calculate the area and circumference of a circle with radius $r = 5$.

```python
import math

r = 5
area = math.pi * r**2
circumference = 2 * math.pi * r

print(f"Radius: {r}")
print(f"Area: {area:.4f}")           # 78.5398
print(f"Circumference: {circumference:.4f}")  # 31.4159
```

---

## Exercise 3.1 (Easy) — Lists

> Create a list of 5 favorite numbers and manipulate it.

```python
favorites = [7, 42, 13, 99, 3]

# 1. Print the entire list
print("My list:", favorites)

# 2. Print the first number
print("First:", favorites[0])

# 3. Print the last number
print("Last:", favorites[-1])

# 4. Print the length
print("Length:", len(favorites))

# 5. Add a new number
favorites.append(27)

# 6. Print the updated list
print("Updated:", favorites)
```

---

## Exercise 4.1 (Easy) — Celsius to Fahrenheit

> Write a function that converts Celsius to Fahrenheit: $F = \frac{9}{5}C + 32$

```python
def celsius_to_fahrenheit(c):
    return (9 / 5) * c + 32

print(f"0°C   = {celsius_to_fahrenheit(0)}°F")    # 32.0
print(f"100°C = {celsius_to_fahrenheit(100)}°F")   # 212.0
print(f"37°C  = {celsius_to_fahrenheit(37)}°F")    # 98.6
```

---

## Exercise 4.2 (Medium) — Quadratic Formula

> Solve $ax^2 + bx + c = 0$ using $x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$

```python
import math

def quadratic_formula(a, b, c):
    """
    Solve ax^2 + bx + c = 0.
    Returns both solutions (x1, x2).
    """
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        return None, None  # No real solutions

    sqrt_disc = math.sqrt(discriminant)
    x1 = (-b + sqrt_disc) / (2 * a)
    x2 = (-b - sqrt_disc) / (2 * a)
    return x1, x2

# Test: x^2 - 5x + 6 = 0  →  x = 3, x = 2
x1, x2 = quadratic_formula(1, -5, 6)
print(f"x1 = {x1}")  # 3.0
print(f"x2 = {x2}")  # 2.0
```

---

## Exercise 6.1 (Easy) — Vector Operations

> Given $\mathbf{a} = [2, 3, 1]$ and $\mathbf{b} = [1, 0, 4]$, compute various operations.

```python
import numpy as np

a = np.array([2, 3, 1])
b = np.array([1, 0, 4])

# 1. a + b
print("a + b =", a + b)               # [3, 3, 5]

# 2. 3a
print("3a =", 3 * a)                  # [6, 9, 3]

# 3. a · b
print("a · b =", a @ b)               # 2*1 + 3*0 + 1*4 = 6

# 4. ||a||
print("||a|| =", np.linalg.norm(a))   # sqrt(4+9+1) = sqrt(14) ≈ 3.742

# 5. ||b||
print("||b|| =", np.linalg.norm(b))   # sqrt(1+0+16) = sqrt(17) ≈ 4.123
```

---

## Exercise 6.2 (Medium) — Cosine Similarity

> Implement $\cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$

```python
def cosine_similarity(u, v):
    """Compute the cosine similarity between two vectors."""
    dot = u @ v
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    return dot / (norm_u * norm_v)

# Test 1: perpendicular → 0
u = np.array([1, 0])
v = np.array([0, 1])
print("Perpendicular:", cosine_similarity(u, v))  # 0.0

# Test 2: parallel → 1
u = np.array([1, 1])
v = np.array([1, 1])
print("Parallel:", cosine_similarity(u, v))        # 1.0

# Test 3: opposite → -1
u = np.array([1, 0])
v = np.array([-1, 0])
print("Opposite:", cosine_similarity(u, v))        # -1.0
```

---

## Exercise 7.1 (Easy) — Matrix Creation and Access

> Create a matrix and extract rows, columns, and transpose.

```python
M = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])

# 1. Print the matrix
print("Matrix M:")
print(M)

# 2. Print its shape
print("\nShape:", M.shape)  # (3, 3)

# 3. Extract the middle row
print("\nMiddle row:", M[1, :])  # [40, 50, 60]

# 4. Extract the last column
print("Last column:", M[:, 2])  # [30, 60, 90]

# 5. Print its transpose
print("\nTranspose:")
print(M.T)
```

---

## Exercise 8.1 (Medium) — Matrix Multiplication Properties

> Given $\mathbf{A}$, $\mathbf{B}$ (identity), and $\mathbf{C}$, compute various products.

```python
A = np.array([[2, 1],
              [3, 4]])

B = np.array([[1, 0],
              [0, 1]])

C = np.array([[5, 6],
              [7, 8]])

# 1. A + C
print("A + C =")
print(A + C)
# [[ 7,  7],
#  [10, 12]]

# 2. A × B
print("\nA @ B =")
print(A @ B)
# [[2, 1],
#  [3, 4]]   (identity doesn't change A)

# 3. B × A
print("\nB @ A =")
print(B @ A)
# [[2, 1],
#  [3, 4]]   (identity on either side)

# 4. A × C
print("\nA @ C =")
print(A @ C)
# [[17, 20],
#  [43, 56]]

# 5. Is AB = BA in general?
print("\nA @ C =")
print(A @ C)
print("\nC @ A =")
print(C @ A)
# [[28, 29],
#  [38, 39]]

print("\nA@C == C@A?", np.array_equal(A @ C, C @ A))
# False → matrix multiplication is NOT commutative
```

---

## Exercise 8.2 (Medium) — Linear Transformations

> Compute $\mathbf{Y} = \mathbf{X}\mathbf{W} + \mathbf{b}$ and verify shapes.

```python
X = np.array([[1, 2],
              [3, 4],
              [5, 6]], dtype=float)

W = np.array([[0.5, -1,  0.2],
              [0.3,  0.7, -0.5]])

b = np.array([0.1, -0.2, 0.3])

# 1. Shapes
print("X shape:", X.shape)   # (3, 2)
print("W shape:", W.shape)   # (2, 3)
print("b shape:", b.shape)   # (3,)

# 2. Shape of Y: (3,2) @ (2,3) + (3,) → (3, 3)
print("Y will be: (3, 3)")

# 3. Compute Y
Y = X @ W + b
print("\nY = X @ W + b:")
print(Y)

# 4. Verify first row manually:
# Y[0] = [1, 2] @ [[0.5, -1, 0.2], [0.3, 0.7, -0.5]] + [0.1, -0.2, 0.3]
#       = [1*0.5+2*0.3, 1*(-1)+2*0.7, 1*0.2+2*(-0.5)] + [0.1, -0.2, 0.3]
#       = [0.5+0.6, -1+1.4, 0.2-1.0] + [0.1, -0.2, 0.3]
#       = [1.1, 0.4, -0.8] + [0.1, -0.2, 0.3]
#       = [1.2, 0.2, -0.5]
print("\nManual verification of first row:")
row0 = np.array([
    1*0.5 + 2*0.3 + 0.1,
    1*(-1) + 2*0.7 + (-0.2),
    1*0.2 + 2*(-0.5) + 0.3
])
print("Expected:", row0)
print("Match:", np.allclose(Y[0], row0))
```

---

## Exercise 9.1 (Hard) — Vectorized Distance Matrix

> Rewrite `pairwise_distances` using broadcasting (no loops).

```python
def pairwise_distances_vectorized(X):
    """
    Compute pairwise Euclidean distances WITHOUT loops.

    X has shape (n, d).
    We use broadcasting:
      X[:, np.newaxis, :] has shape (n, 1, d)
      X[np.newaxis, :, :] has shape (1, n, d)
    Their difference has shape (n, n, d) — all pairwise difference vectors.
    """
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (n, n, d)
    return np.sqrt(np.sum(diff ** 2, axis=2))          # (n, n)


# --- Test ---
data = np.array([
    [165, 60, 20],
    [180, 85, 25],
    [155, 50, 22],
    [170, 70, 30],
    [175, 75, 28]
], dtype=float)

means = np.mean(data, axis=0)
stds = np.std(data, axis=0)
normalized = (data - means) / stds

# Loop version for comparison
def pairwise_distances_loop(X):
    n = X.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = np.linalg.norm(X[i] - X[j])
    return D

D_loop = pairwise_distances_loop(normalized)
D_fast = pairwise_distances_vectorized(normalized)

print("Loop version:")
print(np.round(D_loop, 3))

print("\nVectorized version:")
print(np.round(D_fast, 3))

print("\nSame result?", np.allclose(D_loop, D_fast))  # True
```

**Why it works:** `X[:, np.newaxis, :]` turns each row $\mathbf{x}_i$ into a "slab", and `X[np.newaxis, :, :]` keeps the original matrix. Broadcasting computes $\mathbf{x}_i - \mathbf{x}_j$ for all $(i, j)$ pairs simultaneously, yielding an $(n, n, d)$ tensor. Squaring, summing over axis 2, and taking the square root collapses it to the $(n, n)$ distance matrix.

---

## Exercise 9.2 (Hard) — K-Nearest Neighbors

> Implement a k-NN classifier from scratch.

```python
def knn_predict(X_train, y_train, x_new, k=3):
    """
    Predict the label of x_new using k-nearest neighbors.
    """
    # Step 1: Compute distances from x_new to all training points
    distances = np.linalg.norm(X_train - x_new, axis=1)

    # Step 2: Find the indices of the k smallest distances
    nearest_indices = np.argsort(distances)[:k]

    # Step 3: Get the labels of these k neighbors
    nearest_labels = y_train[nearest_indices]

    # Step 4: Return the most common label
    counts = np.bincount(nearest_labels)
    prediction = np.argmax(counts)

    return prediction


# --- Test ---
X_train = np.array([
    [1, 1], [1.5, 2], [2, 1],         # Class 0
    [5, 5], [6, 5.5], [5.5, 6]        # Class 1
])
y_train = np.array([0, 0, 0, 1, 1, 1])

# Point near class 0
x_new = np.array([2, 2])
pred = knn_predict(X_train, y_train, x_new, k=3)
print(f"Prediction for [2, 2]: {pred}")   # 0

# Point near class 1
x_new = np.array([5, 4])
pred = knn_predict(X_train, y_train, x_new, k=3)
print(f"Prediction for [5, 4]: {pred}")   # 1

# Ambiguous point in the middle
x_new = np.array([3.5, 3.5])
pred = knn_predict(X_train, y_train, x_new, k=3)
print(f"Prediction for [3.5, 3.5]: {pred}")  # depends on distances

# Show detailed reasoning for last point
distances = np.linalg.norm(X_train - np.array([3.5, 3.5]), axis=1)
order = np.argsort(distances)
print("\nDetailed view for [3.5, 3.5]:")
for i in order:
    print(f"  Point {X_train[i]} (class {y_train[i]}), distance = {distances[i]:.2f}")
```

**How it works:**

- `X_train - x_new` uses broadcasting to subtract the new point from every training row, giving all difference vectors at once.
- `np.linalg.norm(..., axis=1)` computes the Euclidean norm of each row, yielding a vector of distances.
- `np.argsort` sorts indices by ascending distance; slicing `[:k]` picks the $k$ closest.
- `np.bincount` counts occurrences of each label; `np.argmax` returns the most frequent one.

---

## Practice Problems — Solutions

### 1. Vector Projection

> $\text{proj}_{\mathbf{b}}\mathbf{a} = \frac{\mathbf{a} \cdot \mathbf{b}}{\mathbf{b} \cdot \mathbf{b}} \mathbf{b}$

```python
def vector_projection(a, b):
    """Project vector a onto vector b."""
    scalar = (a @ b) / (b @ b)
    return scalar * b

a = np.array([3, 4])
b = np.array([1, 0])
print("proj_b(a) =", vector_projection(a, b))  # [3, 0]

a = np.array([1, 2, 3])
b = np.array([1, 1, 1])
print("proj_b(a) =", vector_projection(a, b))  # [2, 2, 2]
```

### 2. Angle Between Two Vectors

> $\theta = \arccos\left(\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}\right)$

```python
def angle_between(u, v):
    """Return the angle in degrees between vectors u and v."""
    cos_theta = (u @ v) / (np.linalg.norm(u) * np.linalg.norm(v))
    cos_theta = np.clip(cos_theta, -1, 1)  # numerical safety
    return np.degrees(np.arccos(cos_theta))

print(angle_between(np.array([1, 0]), np.array([0, 1])))   # 90.0
print(angle_between(np.array([1, 0]), np.array([1, 1])))   # 45.0
print(angle_between(np.array([1, 0]), np.array([-1, 0])))  # 180.0
```

### 3. Matrix Power

> Compute $\mathbf{A}^n$ for any positive integer $n$.

```python
def matrix_power(A, n):
    """Compute A^n by repeated matrix multiplication."""
    result = np.eye(A.shape[0])  # Start with identity
    for _ in range(n):
        result = result @ A
    return result

A = np.array([[1, 1],
              [0, 1]])

print("A^1 =")
print(matrix_power(A, 1))

print("A^3 =")
print(matrix_power(A, 3))
# [[1, 3],
#  [0, 1]]

# Verify with numpy
print("np.linalg.matrix_power(A, 3) =")
print(np.linalg.matrix_power(A, 3))
```

### 4. Rotation Matrix

> Generate a 2D rotation matrix for a given angle $\theta$:
> $\mathbf{R}(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$

```python
def rotation_matrix(theta_degrees):
    """Return the 2D rotation matrix for angle theta (in degrees)."""
    theta = np.radians(theta_degrees)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])

# Rotate the vector [1, 0] by 90°
R = rotation_matrix(90)
v = np.array([1, 0])
print("Rotated:", R @ v)  # [0, 1] (approximately)

# Rotate by 45°
R = rotation_matrix(45)
print("Rotated:", R @ v)  # [0.707, 0.707]

# Two 45° rotations should equal one 90° rotation
R45 = rotation_matrix(45)
R90 = rotation_matrix(90)
print("R45 @ R45 ≈ R90?", np.allclose(R45 @ R45, R90))  # True
```

### 5. Gram-Schmidt Orthogonalization

> Given a set of linearly independent vectors, produce an orthonormal basis.

```python
def gram_schmidt(vectors):
    """
    Gram-Schmidt orthogonalization.
    
    Parameters:
    -----------
    vectors : list of numpy arrays
        Linearly independent input vectors
    
    Returns:
    --------
    basis : list of numpy arrays
        Orthonormal basis vectors
    """
    basis = []

    for v in vectors:
        # Subtract projections onto all existing basis vectors
        w = v.copy().astype(float)
        for u in basis:
            w -= (w @ u) * u  # Remove component along u

        # Normalize
        norm = np.linalg.norm(w)
        if norm < 1e-10:
            raise ValueError("Vectors are not linearly independent")
        basis.append(w / norm)

    return basis


# Test: orthogonalize 3 vectors in R^3
v1 = np.array([1, 1, 0])
v2 = np.array([1, 0, 1])
v3 = np.array([0, 1, 1])

basis = gram_schmidt([v1, v2, v3])

print("Orthonormal basis:")
for i, b in enumerate(basis):
    print(f"  e{i+1} = {np.round(b, 4)}")

# Verify orthonormality
print("\nVerification (dot products — should be 0 for i≠j, 1 for i=j):")
for i in range(3):
    for j in range(3):
        print(f"  e{i+1} · e{j+1} = {basis[i] @ basis[j]:.6f}")
```

---

*End of solutions.*
