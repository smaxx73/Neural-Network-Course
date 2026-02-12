# Python Basics for Machine Learning

**A Progressive Introduction for Beginners**

---

## Table of Contents
1. [Introduction to Python](#intro)
2. [Variables and Basic Types](#variables)
3. [Lists: Your First Data Structure](#lists)
4. [Functions: Reusable Code](#functions)
5. [NumPy: Scientific Computing](#numpy)
6. [Vectors with NumPy](#vectors)
7. [Matrices with NumPy](#matrices)
8. [Matrix Operations](#operations)
9. [Putting It All Together](#together)

---

**Learning Goals:**
- Understand Python variables and basic operations
- Write and use functions
- Work with vectors and matrices using NumPy
- Perform mathematical operations needed for machine learning

---
## 1. Introduction to Python <a name="intro"></a>

### What is Python?

Python is a programming language that is:
- **Easy to read**: Looks almost like English
- **Powerful**: Used for web development, data science, AI, and more
- **Popular**: Huge community and many helpful libraries

### Running Python Code

In Jupyter notebooks:
1. Write code in a cell
2. Press **Shift + Enter** to run it
3. See the output below the cell

Let's try our first program:
```python
# This is a comment - Python ignores it
# Comments help explain your code

print("Hello, Machine Learning!")
```

**Try it yourself!** Run the cell above by clicking on it and pressing **Shift + Enter**.

The `print()` function displays text on the screen.
## 2. Variables and Basic Types <a name="variables"></a>

### What is a Variable?

A **variable** is like a labeled box that stores a value.

In mathematics, you write: $x = 5$

In Python, you write: `x = 5`
```python
# Creating variables
x = 5
y = 10

print("x =", x)
print("y =", y)
```

### Basic Data Types

Python has several basic types of data:

| Type | Example | Description |
|------|---------|-------------|
| `int` | `42` | Integer (whole number) |
| `float` | `3.14` | Decimal number |
| `str` | `"hello"` | Text (string) |
| `bool` | `True` or `False` | Boolean (true/false) |
```python
# Integer (whole number)
age = 25
print("age:", age, "type:", type(age))

# Float (decimal number)
pi = 3.14159
print("pi:", pi, "type:", type(pi))

# String (text)
name = "Claude"
print("name:", name, "type:", type(name))

# Boolean (True or False)
is_student = True
print("is_student:", is_student, "type:", type(is_student))
```

### Basic Arithmetic Operations

Python can do math like a calculator:
```python
# Basic arithmetic
a = 10
b = 3

print("Addition: a + b =", a + b)
print("Subtraction: a - b =", a - b)
print("Multiplication: a * b =", a * b)
print("Division: a / b =", a / b)
print("Integer division: a // b =", a // b)
print("Remainder (modulo): a % b =", a % b)
print("Power: a ** b =", a ** b)
```

### 📝 Exercise 2.1 (Easy)

Create variables for:
- Your age (as an integer)
- Your height in meters (as a float)
- Your name (as a string)

Then print them all using `print()`.
```python
# Your solution here

```

### 📝 Exercise 2.2 (Easy)

Calculate the area and circumference of a circle with radius $r = 5$.

Formulas:
- Area: $A = \pi r^2$
- Circumference: $C = 2\pi r$

Hint: Use `pi = 3.14159` or import it from the `math` library.
```python
# Your solution here

```

## 3. Lists: Your First Data Structure <a name="lists"></a>

### What is a List?

A **list** is a collection of items stored in order.

In mathematics: $\mathbf{v} = [1, 2, 3, 4, 5]$

In Python: `v = [1, 2, 3, 4, 5]`
```python
# Creating a list
numbers = [1, 2, 3, 4, 5]
print("My list:", numbers)

# Lists can contain different types
mixed = [1, 2.5, "hello", True]
print("Mixed list:", mixed)

# Empty list
empty = []
print("Empty list:", empty)
```

### Accessing List Elements

**Important:** Python uses **0-based indexing** (counting starts at 0!)

```
List:    [10, 20, 30, 40, 50]
Index:     0   1   2   3   4
```
```python
fruits = ["apple", "banana", "cherry", "date"]

print("First element (index 0):", fruits[0])
print("Second element (index 1):", fruits[1])
print("Last element (index 3):", fruits[3])

# Negative indexing (count from the end)
print("Last element (index -1):", fruits[-1])
print("Second to last (index -2):", fruits[-2])
```

### List Operations
```python
numbers = [1, 2, 3]

# Get the length of a list
print("Length:", len(numbers))

# Add an element to the end
numbers.append(4)
print("After append:", numbers)

# Add multiple elements
numbers.extend([5, 6])
print("After extend:", numbers)

# Remove an element
numbers.remove(3)
print("After removing 3:", numbers)
```

### List Slicing

Get a portion of a list using `[start:stop]`
```python
numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

print("Original:", numbers)
print("First 3 elements [0:3]:", numbers[0:3])
print("Elements 2 to 5 [2:6]:", numbers[2:6])
print("Last 3 elements [-3:]:", numbers[-3:])
print("Every 2nd element [::2]:", numbers[::2])
```

### 📝 Exercise 3.1 (Easy)

Create a list of your 5 favorite numbers, then:
1. Print the entire list
2. Print the first number
3. Print the last number
4. Print the length of the list
5. Add a new number to the list
6. Print the updated list
```python
# Your solution here

```

## 4. Functions: Reusable Code <a name="functions"></a>

### What is a Function?

A **function** is a reusable block of code that performs a specific task.

In mathematics: $f(x) = x^2 + 1$

In Python:
```python
def f(x):
    return x**2 + 1

# Using the function
result = f(3)
print("f(3) =", result)
print("f(5) =", f(5))
```

### Anatomy of a Function

```python
def function_name(parameter1, parameter2):
    # Function body (indented!)
    result = parameter1 + parameter2
    return result  # Send back the result
```

Key parts:
1. `def` - keyword to define a function
2. `function_name` - what you call it
3. `(parameters)` - inputs to the function
4. `:` - starts the function body
5. **Indentation** - shows what's inside the function (4 spaces or 1 tab)
6. `return` - sends a result back
```python
# Function with no parameters
def greet():
    return "Hello, World!"

print(greet())

# Function with one parameter
def square(x):
    return x * x

print("square(5) =", square(5))

# Function with two parameters
def add(a, b):
    return a + b

print("add(3, 7) =", add(3, 7))
```

### Functions with Multiple Outputs

Python functions can return multiple values:
```python
def circle_properties(radius):
    """Calculate area and circumference of a circle."""
    pi = 3.14159
    area = pi * radius**2
    circumference = 2 * pi * radius
    return area, circumference

# Get both values
a, c = circle_properties(5)
print(f"Radius: 5")
print(f"Area: {a}")
print(f"Circumference: {c}")
```

### Docstrings: Documenting Functions

Use triple quotes `"""` to add documentation to your functions:
```python
def calculate_bmi(weight_kg, height_m):
    """
    Calculate Body Mass Index (BMI).
    
    Parameters:
    -----------
    weight_kg : float
        Weight in kilograms
    height_m : float
        Height in meters
    
    Returns:
    --------
    float
        BMI value
    """
    return weight_kg / (height_m ** 2)

# Use the function
bmi = calculate_bmi(70, 1.75)
print(f"BMI: {bmi:.2f}")

# Get help about the function
help(calculate_bmi)
```

### 📝 Exercise 4.1 (Easy)

Write a function called `celsius_to_fahrenheit` that converts Celsius to Fahrenheit.

Formula: $F = \frac{9}{5}C + 32$

Test it with: 0°C, 100°C, 37°C
```python
# Your solution here

```

### 📝 Exercise 4.2 (Medium)

Write a function called `quadratic_formula` that solves $ax^2 + bx + c = 0$.

The formula is: $x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$

Your function should return both solutions.

Test with: $x^2 - 5x + 6 = 0$ (answer should be $x = 2$ and $x = 3$)
```python
# Your solution here
# Hint: Use the ** operator for power, and math.sqrt() for square root
import math

```

## 5. NumPy: Scientific Computing <a name="numpy"></a>

### What is NumPy?

**NumPy** (Numerical Python) is a library for:
- Working with arrays (vectors and matrices)
- Fast mathematical operations
- Essential for machine learning!

### Installing and Importing NumPy
```python
# Import NumPy (standard abbreviation is np)
import numpy as np

print("NumPy version:", np.__version__)
```

### Why NumPy Instead of Lists?

Compare regular Python lists with NumPy arrays:
```python
# Python lists: element-wise operations are HARD
list1 = [1, 2, 3]
list2 = [4, 5, 6]

# This doesn't do what you might expect!
print("list1 + list2 =", list1 + list2)  # Concatenates!

# To add element-wise, you need a loop:
result = []
for i in range(len(list1)):
    result.append(list1[i] + list2[i])
print("Element-wise sum:", result)
```

```python
# NumPy arrays: element-wise operations are EASY
array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])

# This works perfectly!
print("array1 + array2 =", array1 + array2)
print("array1 * array2 =", array1 * array2)
print("array1 ** 2 =", array1 ** 2)
```

### Creating NumPy Arrays
```python
# From a Python list
arr1 = np.array([1, 2, 3, 4, 5])
print("Array from list:", arr1)

# Array of zeros
zeros = np.zeros(5)
print("Zeros:", zeros)

# Array of ones
ones = np.ones(5)
print("Ones:", ones)

# Array with a range of values
range_arr = np.arange(0, 10, 2)  # Start, stop, step
print("Range (0 to 10, step 2):", range_arr)

# Array with evenly spaced values
linspace_arr = np.linspace(0, 1, 5)  # Start, stop, number of points
print("Linspace (5 points from 0 to 1):", linspace_arr)
```

### Array Properties
```python
arr = np.array([1, 2, 3, 4, 5])

print("Array:", arr)
print("Shape (dimensions):", arr.shape)
print("Size (total elements):", arr.size)
print("Data type:", arr.dtype)
print("Number of dimensions:", arr.ndim)
```

## 6. Vectors with NumPy <a name="vectors"></a>

### What is a Vector?

In mathematics, a vector is an ordered list of numbers:

$$
\mathbf{v} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}
$$

In NumPy, vectors are 1D arrays:
```python
# Create a vector
v = np.array([1, 2, 3])
print("Vector v:", v)
print("Shape:", v.shape)  # (3,) means 1D array with 3 elements
```

### Vector Operations

#### 1. Scalar Multiplication

$$
c \mathbf{v} = c \begin{bmatrix} v_1 \\ v_2 \\ v_3 \end{bmatrix} = \begin{bmatrix} cv_1 \\ cv_2 \\ cv_3 \end{bmatrix}
$$
```python
v = np.array([1, 2, 3])
print("v =", v)
print("2 * v =", 2 * v)
print("v / 2 =", v / 2)
```

#### 2. Vector Addition

$$
\mathbf{u} + \mathbf{v} = \begin{bmatrix} u_1 \\ u_2 \\ u_3 \end{bmatrix} + \begin{bmatrix} v_1 \\ v_2 \\ v_3 \end{bmatrix} = \begin{bmatrix} u_1 + v_1 \\ u_2 + v_2 \\ u_3 + v_3 \end{bmatrix}
$$
```python
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])

print("u =", u)
print("v =", v)
print("u + v =", u + v)
print("u - v =", u - v)
```

#### 3. Dot Product (Inner Product)

$$
\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{n} u_i v_i = u_1v_1 + u_2v_2 + u_3v_3
$$

The result is a **scalar** (single number).
```python
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])

# Method 1: np.dot()
dot_product = np.dot(u, v)
print("u · v =", dot_product)

# Method 2: @ operator (recommended!)
dot_product2 = u @ v
print("u @ v =", dot_product2)

# Verify manually: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
manual = u[0]*v[0] + u[1]*v[1] + u[2]*v[2]
print("Manual calculation:", manual)
```

#### 4. Vector Magnitude (Length, Norm)

$$
\|\mathbf{v}\| = \sqrt{v_1^2 + v_2^2 + v_3^2} = \sqrt{\mathbf{v} \cdot \mathbf{v}}
$$
```python
v = np.array([3, 4])

# Method 1: np.linalg.norm()
magnitude = np.linalg.norm(v)
print("||v|| =", magnitude)

# Method 2: Manual calculation
magnitude_manual = np.sqrt(np.sum(v**2))
print("Manual: ||v|| =", magnitude_manual)

# Verify: sqrt(3² + 4²) = sqrt(9 + 16) = sqrt(25) = 5
print("Expected: 5")
```

### 📝 Exercise 6.1 (Easy)

Given vectors:
- $\mathbf{a} = [2, 3, 1]$
- $\mathbf{b} = [1, 0, 4]$

Calculate:
1. $\mathbf{a} + \mathbf{b}$
2. $3\mathbf{a}$
3. $\mathbf{a} \cdot \mathbf{b}$
4. $\|\mathbf{a}\|$
5. $\|\mathbf{b}\|$
```python
# Your solution here

```

### 📝 Exercise 6.2 (Medium)

The **cosine similarity** between two vectors is:

$$
\cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}
$$

Write a function `cosine_similarity(u, v)` that calculates this.

Test with:
- $\mathbf{u} = [1, 0]$ and $\mathbf{v} = [0, 1]$ (should be 0, perpendicular)
- $\mathbf{u} = [1, 1]$ and $\mathbf{v} = [1, 1]$ (should be 1, parallel)
- $\mathbf{u} = [1, 0]$ and $\mathbf{v} = [-1, 0]$ (should be -1, opposite)
```python
# Your solution here

```

## 7. Matrices with NumPy <a name="matrices"></a>

### What is a Matrix?

A **matrix** is a 2D array of numbers:

$$
\mathbf{A} = \begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix}
$$

This is a $2 \times 3$ matrix (2 rows, 3 columns).
```python
# Create a matrix
A = np.array([[1, 2, 3],
              [4, 5, 6]])

print("Matrix A:")
print(A)
print("\nShape:", A.shape)  # (2, 3) means 2 rows, 3 columns
print("Size:", A.size)  # Total elements: 2 * 3 = 6
print("Dimensions:", A.ndim)  # 2D array
```

### Creating Special Matrices
```python
# Matrix of zeros (3x4)
zeros_matrix = np.zeros((3, 4))
print("Zeros matrix (3x4):")
print(zeros_matrix)

# Matrix of ones (2x3)
ones_matrix = np.ones((2, 3))
print("\nOnes matrix (2x3):")
print(ones_matrix)

# Identity matrix (square matrix with 1s on diagonal)
identity = np.eye(3)
print("\nIdentity matrix (3x3):")
print(identity)

# Random matrix (values between 0 and 1)
random_matrix = np.random.rand(2, 3)
print("\nRandom matrix (2x3):")
print(random_matrix)
```

### Accessing Matrix Elements
```python
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

print("Matrix A:")
print(A)

# Access single element: A[row, column]
print("\nElement at row 0, column 1:", A[0, 1])  # 2
print("Element at row 2, column 2:", A[2, 2])  # 9

# Access entire row
print("\nFirst row (row 0):", A[0, :])  # [1, 2, 3]
print("Second row (row 1):", A[1, :])  # [4, 5, 6]

# Access entire column
print("\nFirst column (column 0):", A[:, 0])  # [1, 4, 7]
print("Third column (column 2):", A[:, 2])  # [3, 6, 9]

# Access submatrix
print("\nTop-left 2x2 submatrix:")
print(A[0:2, 0:2])
```

### Matrix Transpose

The **transpose** of a matrix flips rows and columns:

$$
\mathbf{A} = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \quad \Rightarrow \quad \mathbf{A}^T = \begin{bmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{bmatrix}
$$
```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])

print("Original matrix A (2x3):")
print(A)
print("Shape:", A.shape)

print("\nTransposed matrix A^T (3x2):")
print(A.T)
print("Shape:", A.T.shape)
```

### 📝 Exercise 7.1 (Easy)

Create the following matrix:

$$
\mathbf{M} = \begin{bmatrix}
10 & 20 & 30 \\
40 & 50 & 60 \\
70 & 80 & 90
\end{bmatrix}
$$

Then:
1. Print the matrix
2. Print its shape
3. Extract and print the middle row [40, 50, 60]
4. Extract and print the last column [30, 60, 90]
5. Print its transpose
```python
# Your solution here

```

## 8. Matrix Operations <a name="operations"></a>

### Element-wise Operations

Like vectors, matrices support element-wise operations:
```python
A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

print("Matrix A:")
print(A)
print("\nMatrix B:")
print(B)

print("\nA + B (element-wise addition):")
print(A + B)

print("\nA * B (element-wise multiplication):")
print(A * B)  # NOT matrix multiplication!

print("\n2 * A (scalar multiplication):")
print(2 * A)

print("\nA ** 2 (element-wise square):")
print(A ** 2)
```

### Matrix Multiplication

**Important:** Matrix multiplication is NOT element-wise!

For matrices $\mathbf{A}$ (size $m \times n$) and $\mathbf{B}$ (size $n \times p$):

$$
(\mathbf{AB})_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}
$$

**Key rule:** Number of columns in $\mathbf{A}$ must equal number of rows in $\mathbf{B}$.

Result will be size $m \times p$.
```python
# Example: (2x3) @ (3x2) = (2x2)
A = np.array([[1, 2, 3],
              [4, 5, 6]])

B = np.array([[7, 8],
              [9, 10],
              [11, 12]])

print("Matrix A (2x3):")
print(A)
print("\nMatrix B (3x2):")
print(B)

# Method 1: @ operator (recommended!)
C = A @ B
print("\nA @ B (matrix multiplication):")
print(C)
print("Shape:", C.shape)

# Method 2: np.dot() or np.matmul()
C2 = np.matmul(A, B)
print("\nnp.matmul(A, B):")
print(C2)

# Verify first element manually:
# C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
#        = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
print("\nManual verification of C[0,0]:", 1*7 + 2*9 + 3*11)
```

### Visual Example of Matrix Multiplication

$$
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \times \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 1·5+2·7 & 1·6+2·8 \\ 3·5+4·7 & 3·6+4·8 \end{bmatrix} = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}
$$
```python
A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

C = A @ B

print("A @ B =")
print(C)

# Manual verification
print("\nManual calculations:")
print(f"C[0,0] = {A[0,0]}*{B[0,0]} + {A[0,1]}*{B[1,0]} = {A[0,0]*B[0,0] + A[0,1]*B[1,0]}")
print(f"C[0,1] = {A[0,0]}*{B[0,1]} + {A[0,1]}*{B[1,1]} = {A[0,0]*B[0,1] + A[0,1]*B[1,1]}")
print(f"C[1,0] = {A[1,0]}*{B[0,0]} + {A[1,1]}*{B[1,0]} = {A[1,0]*B[0,0] + A[1,1]*B[1,0]}")
print(f"C[1,1] = {A[1,0]}*{B[0,1]} + {A[1,1]}*{B[1,1]} = {A[1,0]*B[0,1] + A[1,1]*B[1,1]}")
```

### Matrix-Vector Multiplication

A special case: multiply a matrix by a vector.

$$
\mathbf{A} \mathbf{x} = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \begin{bmatrix} 7 \\ 8 \\ 9 \end{bmatrix} = \begin{bmatrix} 1·7 + 2·8 + 3·9 \\ 4·7 + 5·8 + 6·9 \end{bmatrix} = \begin{bmatrix} 50 \\ 122 \end{bmatrix}
$$

This is **essential for neural networks**!
```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])

x = np.array([7, 8, 9])

print("Matrix A (2x3):")
print(A)
print("\nVector x (3,):")
print(x)

result = A @ x
print("\nA @ x =")
print(result)
print("Shape:", result.shape)

# This is how the perceptron computes its output!
# If A represents weights and x represents input, 
# then A @ x gives the weighted sum.
```

### 📝 Exercise 8.1 (Medium)

Given:
$$
\mathbf{A} = \begin{bmatrix} 2 & 1 \\ 3 & 4 \end{bmatrix}, \quad
\mathbf{B} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad
\mathbf{C} = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}
$$

Calculate:
1. $\mathbf{A} + \mathbf{C}$
2. $\mathbf{A} \times \mathbf{B}$ (matrix multiplication)
3. $\mathbf{B} \times \mathbf{A}$ (matrix multiplication)
4. $\mathbf{A} \times \mathbf{C}$ (matrix multiplication)
5. Is $\mathbf{AB} = \mathbf{BA}$? What does this tell you about matrix multiplication?
```python
# Your solution here

```

### 📝 Exercise 8.2 (Medium)

**Simulating a Simple Perceptron**

A perceptron computes:
$$
y = \text{sign}(\mathbf{w}^T \mathbf{x} + b)
$$

Given:
- Weights: $\mathbf{w} = [0.5, -0.3, 0.8]$
- Bias: $b = 0.2$
- Input: $\mathbf{x} = [1, 2, 3]$

Calculate:
1. $\mathbf{w}^T \mathbf{x}$ (dot product)
2. $\mathbf{w}^T \mathbf{x} + b$
3. $y = \text{sign}(\mathbf{w}^T \mathbf{x} + b)$ (use `np.sign()`)

What class does the perceptron predict for this input?
```python
# Your solution here

```

## 9. Putting It All Together <a name="together"></a>

### Example: Complete Perceptron Forward Pass

Let's implement a function that computes the perceptron output for multiple inputs at once!
```python
def perceptron_predict(X, w, b):
    """
    Compute perceptron predictions for multiple inputs.
    
    Parameters:
    -----------
    X : numpy array, shape (n_samples, n_features)
        Input data
    w : numpy array, shape (n_features,)
        Weight vector
    b : float
        Bias term
    
    Returns:
    --------
    predictions : numpy array, shape (n_samples,)
        Predicted classes (+1 or -1)
    """
    # Compute weighted sums: X @ w gives a vector of dot products
    weighted_sums = X @ w + b
    
    # Apply sign function
    predictions = np.sign(weighted_sums)
    
    return predictions

# Example: AND gate
# Inputs: all combinations of 0 and 1
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Weights and bias for AND gate
w = np.array([1.0, 1.0])
b = -1.5

# Make predictions
predictions = perceptron_predict(X, w, b)

print("AND Gate Simulation:")
print("Inputs:\n", X)
print("\nWeights:", w)
print("Bias:", b)
print("\nPredictions:", predictions)

# Convert to 0/1 for easier reading
predictions_01 = (predictions + 1) // 2  # -1 -> 0, +1 -> 1
print("\nTruth table:")
for i in range(len(X)):
    print(f"  {X[i][0]} AND {X[i][1]} = {predictions_01[i]}")
```

### Understanding the Computation

Let's break down what happened step by step:
```python
print("Detailed computation for AND gate:\n")
print("w =", w)
print("b =", b)
print()

for i, x in enumerate(X):
    # Compute dot product
    dot_product = x @ w
    
    # Add bias
    linear_output = dot_product + b
    
    # Apply sign
    prediction = np.sign(linear_output)
    
    print(f"Input {i}: x = {x}")
    print(f"  x · w = {x[0]}*{w[0]} + {x[1]}*{w[1]} = {dot_product}")
    print(f"  x · w + b = {dot_product} + {b} = {linear_output}")
    print(f"  sign({linear_output}) = {prediction}")
    print()
```

### 📝 Exercise 9.1 (Hard)

**Build a Complete Perceptron Trainer**

Implement a function `train_perceptron(X, y, learning_rate, max_epochs)` that:
1. Initializes weights to zeros
2. Initializes bias to zero
3. For each epoch:
   - For each training example:
     - Make a prediction
     - If wrong, update weights and bias
4. Returns the learned weights and bias

Test it on the OR gate:
```
x1  x2  | OR
--------+----
 0   0  |  0
 0   1  |  1
 1   0  |  1
 1   1  |  1
```
```python
# Your solution here
def train_perceptron(X, y, learning_rate=0.1, max_epochs=100):
    """
    Train a perceptron using the perceptron learning algorithm.
    
    Parameters:
    -----------
    X : numpy array, shape (n_samples, n_features)
        Training data
    y : numpy array, shape (n_samples,)
        Target labels (+1 or -1)
    learning_rate : float
        Learning rate (eta)
    max_epochs : int
        Maximum number of training epochs
    
    Returns:
    --------
    w : numpy array
        Learned weights
    b : float
        Learned bias
    """
    # Initialize weights and bias
    # ...
    
    # Training loop
    # ...
    
    pass

# Test on OR gate
# ...
```

### 📝 Exercise 9.2 (Hard)

**Batch Processing**

Modify your perceptron implementation to process **batches** of data.

Instead of:
```python
for each example:
    prediction = sign(w @ x + b)
```

Use matrix operations:
```python
predictions = sign(X @ w + b)  # All at once!
```

This is **much faster** for large datasets.

Write a function `batch_perceptron_predict(X, w, b)` and verify it gives the same results as the loop version.
```python
# Your solution here

```

---

## Summary

### What You've Learned

✅ **Python Basics**
- Variables and data types
- Lists and indexing
- Functions and return values

✅ **NumPy Fundamentals**
- Creating and manipulating arrays
- Array operations and broadcasting

✅ **Vectors**
- Vector addition and scalar multiplication
- Dot product (inner product)
- Vector magnitude (norm)

✅ **Matrices**
- Creating and accessing matrices
- Matrix transpose
- Element-wise operations

✅ **Matrix Operations**
- Matrix multiplication with `@`
- Matrix-vector multiplication
- Batch processing

### Key Takeaways

1. **NumPy is essential** for machine learning in Python
2. **Use `@` for matrix multiplication**, not `*`
3. **Vectorization** (using arrays instead of loops) is faster
4. **Shape matters**: Always check array shapes!

### Next Steps

Now you're ready to:
- Understand the perceptron notebook
- Implement machine learning algorithms
- Work with neural networks

---

## Quick Reference

### Common NumPy Operations

```python
# Creating arrays
np.array([1, 2, 3])           # From list
np.zeros(5)                   # Array of zeros
np.ones((3, 4))               # Matrix of ones
np.eye(3)                     # Identity matrix
np.arange(0, 10, 2)           # Range with step
np.linspace(0, 1, 5)          # Evenly spaced

# Operations
a + b                         # Addition
a * b                         # Element-wise multiplication
a @ b                         # Matrix/dot product
np.dot(a, b)                  # Dot product (alternative)
a.T                           # Transpose
np.linalg.norm(a)             # Magnitude/norm
np.sign(a)                    # Sign function

# Properties
a.shape                       # Dimensions
a.size                        # Total elements
a.ndim                        # Number of dimensions
```

---

## Practice Problems

To reinforce your learning, try these additional exercises:

1. Implement vector projection: $\text{proj}_{\mathbf{b}}\mathbf{a} = \frac{\mathbf{a} \cdot \mathbf{b}}{\mathbf{b} \cdot \mathbf{b}} \mathbf{b}$

2. Write a function to compute the angle between two vectors

3. Implement matrix power: compute $\mathbf{A}^n$ for any positive integer $n$

4. Create a function that generates a rotation matrix for a given angle

5. Implement the Gram-Schmidt orthogonalization process

Good luck with your machine learning journey! 🚀

---
