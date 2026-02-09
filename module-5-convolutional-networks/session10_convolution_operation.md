# Session 10: The Convolution Operation
## Learning to See

**Course: Neural Networks for Engineers**  
**Duration: 2 hours**

---

## Table of Contents

1. [Recap & Motivation](#recap)
2. [Why MLPs Fail for Images](#mlp-fail)
3. [The Convolution Operation](#convolution)
4. [Hand Calculations](#hand-calc)
5. [Convolution as a Neural Network Layer](#conv-layer)
6. [Convolution Parameters: Padding & Stride](#parameters)
7. [Implement Convolution from Scratch](#implement)
8. [Feature Detection: Kernels as Eyes](#feature-detection)
9. [Verify with PyTorch nn.Conv2d](#pytorch-verify)
10. [Summary](#summary)

---

## 1. Recap & Motivation {#recap}

### What We've Built So Far

✅ **MLP**: Fully-connected layers with backprop (Sessions 2–8)  
✅ **PyTorch**: Build and train networks with autograd (Session 9)  
✅ **MNIST**: 97%+ accuracy on handwritten digits with an MLP  

### 🤔 Quick Questions (from Session 9's "Think About")

**Q1:** Our MNIST MLP flattens the 28×28 image into a 784-dimensional vector. What spatial information is lost?

<details>
<summary>Click to reveal answer</summary>
**Neighbor relationships.** Pixel (0, 0) and pixel (0, 1) are right next to each other in the image, but in the flattened vector they're just index 0 and index 1 — the model has no way to know they're neighbors. Pixel (0, 0) and pixel (1, 0) are directly below each other in the image, but in the vector they're 28 positions apart. The 2D structure is completely destroyed.
</details>

**Q2:** For a 224×224 RGB image, how many parameters in the first hidden layer?

<details>
<summary>Click to reveal answer</summary>
Input: $224 \times 224 \times 3 = 150{,}528$ features. If the first hidden layer has 256 neurons:
$$150{,}528 \times 256 + 256 = 38{,}535{,}424 \text{ parameters}$$
That's **38.5 million parameters** in just the first layer! This is completely impractical — and it would massively overfit.
</details>

**Q3:** When you recognize a digit, do you look at every pixel equally?

<details>
<summary>Click to reveal answer</summary>
No! You look at **local patterns**: curves, straight lines, intersections, loops. A "6" has a loop at the bottom and a curve at the top. A "1" is a vertical line. You recognize these **local features** regardless of where they appear in the image. This is exactly what convolutions do.
</details>

---

## 2. Why MLPs Fail for Images {#mlp-fail}

### Problem 1: Too Many Parameters

An MLP treats every pixel as an independent input feature:

```
Image (28×28 = 784 pixels)
  ↓ flatten
[p₁, p₂, p₃, ..., p₇₈₄]
  ↓ fully-connected to 256 neurons
Weight matrix: 784 × 256 = 200,704 parameters
```

For real images (224×224×3 = 150K inputs), this explodes to millions of parameters per layer.

### Problem 2: No Spatial Awareness

```
Original image:             Flattened vector:

  ┌─────────┐
  │ . . X . │               [., ., X, ., ., ., X, X, ., ., ., X, ., ., ., .]
  │ . X X . │  flatten →
  │ . . X . │               The MLP sees pixels (0,2), (1,1), and (1,2) as
  │ . . . . │               "features 3, 6, 7" — no notion of adjacency.
  └─────────┘
```

The MLP doesn't know that pixel 3 is **next to** pixel 4, or **above** pixel 7.

### Problem 3: No Translation Invariance

If a "7" appears in the top-left corner during training and in the bottom-right during testing, the MLP must learn both positions **separately** — because different pixels (different input features) are activated. It has no concept of "the same pattern in a different location."

### The Solution: Three Key Ideas

| Idea | MLP | Convolution |
|---|---|---|
| **Local connectivity** | Every neuron sees all pixels | Each neuron sees a small **patch** |
| **Weight sharing** | Different weights per pixel | **Same filter** slides across the image |
| **Translation invariance** | Must learn each position separately | Detects patterns **anywhere** |

---

## 3. The Convolution Operation {#convolution}

### The Core Idea

A **convolution** slides a small matrix (called a **kernel** or **filter**) across the image. At each position, it computes the **dot product** between the kernel and the overlapping image patch.

```
Image (5×5):                    Kernel (3×3):

  1  0  1  0  1                  1  0  1
  0  1  0  1  0                  0  1  0
  1  0  1  0  1                  1  0  1
  0  1  0  1  0
  1  0  1  0  1

Step 1: Place kernel at top-left corner

  [1  0  1] 0  1                 1  0  1
  [0  1  0] 1  0      ⊙         0  1  0     = 1·1 + 0·0 + 1·1
  [1  0  1] 0  1                 1  0  1       + 0·0 + 1·1 + 0·0
   0  1  0  1  0                               + 1·1 + 0·0 + 1·1
   1  0  1  0  1                             = 5

Step 2: Slide kernel one pixel right

  1 [0  1  0] 1                  1  0  1
  0 [1  0  1] 0      ⊙         0  1  0     = 0·1 + 1·0 + 0·1
  1 [0  1  0] 1                 1  0  1       + 1·0 + 0·1 + 1·0
  0  1  0  1  0                               + 0·1 + 1·0 + 0·1
  1  0  1  0  1                             = 0

... continue sliding to produce the output (feature map)
```

### Mathematical Definition

For a 2D input $I$ and kernel $K$ of size $k_h \times k_w$, the output at position $(i, j)$ is:

$$
(I * K)[i, j] = \sum_{m=0}^{k_h - 1} \sum_{n=0}^{k_w - 1} I[i + m, \, j + n] \cdot K[m, n]
$$

This is called a **cross-correlation** (technically, convolution flips the kernel, but in deep learning we don't flip — everyone calls it "convolution" anyway).

### Output Size

For an input of size $H \times W$ and a kernel of size $k \times k$ (no padding, stride 1):

$$
\text{Output size} = (H - k + 1) \times (W - k + 1)
$$

A $5 \times 5$ image with a $3 \times 3$ kernel produces a $3 \times 3$ output.

---

## 4. Hand Calculations {#hand-calc}

### ✏️ Exercise 4.1 — Your First Convolution

Compute the full convolution of this $4 \times 4$ image with a $3 \times 3$ kernel (no padding, stride 1):

$$
I = \begin{bmatrix} 1 & 2 & 0 & 1 \\ 0 & 1 & 3 & 2 \\ 1 & 0 & 2 & 1 \\ 3 & 1 & 0 & 2 \end{bmatrix}
\qquad
K = \begin{bmatrix} 1 & 0 & -1 \\ 1 & 0 & -1 \\ 1 & 0 & -1 \end{bmatrix}
$$

**Step 1:** What is the output size?

**Step 2:** Compute each output element.

| Position | Patch from $I$ | $\sum (\text{Patch} \odot K)$ | Result |
|---|---|---|---|
| (0,0) | $\begin{bmatrix} 1 & 2 & 0 \\ 0 & 1 & 3 \\ 1 & 0 & 2 \end{bmatrix}$ | $1 \cdot 1 + 2 \cdot 0 + 0 \cdot (-1) + 0 \cdot 1 + 1 \cdot 0 + 3 \cdot (-1) + 1 \cdot 1 + 0 \cdot 0 + 2 \cdot (-1)$ | ? |
| (0,1) | | | ? |
| (1,0) | | | ? |
| (1,1) | | | ? |

<details>
<summary>Solution</summary>

**Step 1:** Output size = $(4 - 3 + 1) \times (4 - 3 + 1) = 2 \times 2$

**Step 2:**

$(0,0)$: Patch = $\begin{bmatrix} 1 & 2 & 0 \\ 0 & 1 & 3 \\ 1 & 0 & 2 \end{bmatrix}$

$1 \cdot 1 + 2 \cdot 0 + 0 \cdot (-1) + 0 \cdot 1 + 1 \cdot 0 + 3 \cdot (-1) + 1 \cdot 1 + 0 \cdot 0 + 2 \cdot (-1) = 1 + 0 + 0 + 0 + 0 - 3 + 1 + 0 - 2 = \mathbf{-3}$

$(0,1)$: Patch = $\begin{bmatrix} 2 & 0 & 1 \\ 1 & 3 & 2 \\ 0 & 2 & 1 \end{bmatrix}$

$2 \cdot 1 + 0 \cdot 0 + 1 \cdot (-1) + 1 \cdot 1 + 3 \cdot 0 + 2 \cdot (-1) + 0 \cdot 1 + 2 \cdot 0 + 1 \cdot (-1) = 2 + 0 - 1 + 1 + 0 - 2 + 0 + 0 - 1 = \mathbf{-1}$

$(1,0)$: Patch = $\begin{bmatrix} 0 & 1 & 3 \\ 1 & 0 & 2 \\ 3 & 1 & 0 \end{bmatrix}$

$0 \cdot 1 + 1 \cdot 0 + 3 \cdot (-1) + 1 \cdot 1 + 0 \cdot 0 + 2 \cdot (-1) + 3 \cdot 1 + 1 \cdot 0 + 0 \cdot (-1) = 0 + 0 - 3 + 1 + 0 - 2 + 3 + 0 + 0 = \mathbf{-1}$

$(1,1)$: Patch = $\begin{bmatrix} 1 & 3 & 2 \\ 0 & 2 & 1 \\ 1 & 0 & 2 \end{bmatrix}$

$1 \cdot 1 + 3 \cdot 0 + 2 \cdot (-1) + 0 \cdot 1 + 2 \cdot 0 + 1 \cdot (-1) + 1 \cdot 1 + 0 \cdot 0 + 2 \cdot (-1) = 1 + 0 - 2 + 0 + 0 - 1 + 1 + 0 - 2 = \mathbf{-3}$

$$
\text{Output} = \begin{bmatrix} -3 & -1 \\ -1 & -3 \end{bmatrix}
$$
</details>

### 🤔 Think About It

Look at the kernel $K = \begin{bmatrix} 1 & 0 & -1 \\ 1 & 0 & -1 \\ 1 & 0 & -1 \end{bmatrix}$.

- The left column has positive weights (+1), the right column has negative weights (−1).
- What does this kernel detect?

<details>
<summary>Answer</summary>
**Vertical edges!** It computes (left side) − (right side). Where there's a sharp change from bright on the left to dark on the right (or vice versa), the output is large in magnitude. Where both sides are similar, the output is near zero.
</details>

### ✏️ Exercise 4.2 — Identity and Blur Kernels

**Part A:** What does this kernel do?

$$
K_{\text{identity}} = \begin{bmatrix} 0 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{bmatrix}
$$

Apply it to the image $I$ from Exercise 4.1 (just compute position (0,0)).

<details>
<summary>Answer</summary>
$\text{Output}[0,0] = 0 + 0 + 0 + 0 + 1 \cdot 1 + 0 + 0 + 0 + 0 = 1$

This is just the center pixel of the patch! The **identity kernel** copies the input (with a smaller output size). Each output pixel equals the center of its corresponding input patch.
</details>

**Part B:** What does this kernel do?

$$
K_{\text{blur}} = \frac{1}{9} \begin{bmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{bmatrix}
$$

Compute position (0,0) on image $I$.

<details>
<summary>Answer</summary>
$\text{Output}[0,0] = \frac{1}{9}(1 + 2 + 0 + 0 + 1 + 3 + 1 + 0 + 2) = \frac{10}{9} \approx 1.11$

Each output pixel is the **average** of its 3×3 neighborhood. This is a **box blur** — it smooths the image by replacing each pixel with its local average.
</details>

### ✏️ Exercise 4.3 — Output Size Quiz

Compute the output size for each configuration (no padding, stride 1):

| Input | Kernel | Output |
|---|---|---|
| $7 \times 7$ | $3 \times 3$ | ? |
| $28 \times 28$ | $5 \times 5$ | ? |
| $32 \times 32$ | $3 \times 3$ | ? |
| $100 \times 100$ | $7 \times 7$ | ? |

<details>
<summary>Solution</summary>

Using $(H - k + 1) \times (W - k + 1)$:

| Input | Kernel | Output |
|---|---|---|
| $7 \times 7$ | $3 \times 3$ | $5 \times 5$ |
| $28 \times 28$ | $5 \times 5$ | $24 \times 24$ |
| $32 \times 32$ | $3 \times 3$ | $30 \times 30$ |
| $100 \times 100$ | $7 \times 7$ | $94 \times 94$ |
</details>

---

## 5. Convolution as a Neural Network Layer {#conv-layer}

### Shared Weights = Fewer Parameters

In a fully-connected layer, every input connects to every output with a **different** weight. In a convolution, the same kernel slides across the whole image:

```
Fully-connected (5 inputs → 3 outputs):    Convolution (5 inputs, kernel size 3):

  in₁ ─┬─┬─┬─ out₁                          in₁ ─── kernel ─── out₁
  in₂ ─┼─┼─┼─ out₂                               in₂ ─── kernel ─── out₂
  in₃ ─┼─┼─┼─ out₃                                    in₃ ─── kernel ─── out₃
  in₄ ─┼─┼─┤
  in₅ ─┴─┴─┘                                Same 3 weights reused at every position!

  Parameters: 5×3 = 15                       Parameters: 3 (just the kernel)
```

### Parameter Comparison

| Layer type | Input | Output | Parameters |
|---|---|---|---|
| Fully-connected | 784 | 256 | $784 \times 256 + 256 = 200{,}960$ |
| Convolution ($3 \times 3$) | $28 \times 28$ | $26 \times 26$ | $3 \times 3 + 1 = 10$ |

A 3×3 convolution has **20,000× fewer parameters** than the equivalent fully-connected layer! The kernel's 9 weights are reused at every position.

### "A Special Linear Layer"

A convolution is actually a linear operation — it's a matrix multiplication with a very specific, sparse weight structure:

- Each output neuron only connects to a **local patch** of the input (local connectivity)
- The weights are **shared** across all positions (weight sharing)
- There's still a **bias** term added after the dot product (one per filter)

Think of it as: "a linear layer where most weights are zero, and the non-zero weights are tied together."

### Multiple Filters = Multiple Feature Maps

One kernel detects **one** type of feature (e.g., vertical edges). A convolutional layer uses **multiple kernels**, each producing its own output — called a **feature map** or **channel**:

```
Input image          K₁ (edges)           K₂ (blur)           K₃ (corners)
  28 × 28    →     26 × 26          +   26 × 26          +   26 × 26
                    (feature map 1)      (feature map 2)      (feature map 3)

Output: 3 feature maps stacked → shape (3, 26, 26)
```

**Parameters for $N_f$ filters of size $k \times k$:** $N_f \times (k \times k + 1)$

### ✏️ Exercise 5.1 — Parameter Counting

A convolutional layer has:
- Input: single-channel $28 \times 28$ image
- 16 filters, each $5 \times 5$
- Bias per filter

How many parameters does this layer have?

<details>
<summary>Solution</summary>

$16 \times (5 \times 5 + 1) = 16 \times 26 = \mathbf{416}$ parameters.

Compare with a fully-connected layer from 784 inputs to 16 outputs: $784 \times 16 + 16 = 12{,560}$ — that's 30× more.
</details>

### ✏️ Exercise 5.2 — Feature Map Size

Same layer as above (input $28 \times 28$, 16 filters of size $5 \times 5$, no padding, stride 1).

What is the **shape** of the output tensor?

<details>
<summary>Solution</summary>

Each $5 \times 5$ filter on a $28 \times 28$ input produces a $(28 - 5 + 1) \times (28 - 5 + 1) = 24 \times 24$ feature map.

With 16 filters: output shape = $\mathbf{16 \times 24 \times 24}$.

In PyTorch notation (batch first): `(batch_size, 16, 24, 24)`.
</details>

---

## 6. Convolution Parameters: Padding & Stride {#parameters}

### Padding

Without padding, the output shrinks at every layer. After several convolutions, the image becomes tiny! **Padding** adds zeros around the border to control the output size.

```
No padding (valid):                    Same padding (pad=1):

  ┌───────┐                              0  0  0  0  0  0  0
  │ image │  →  smaller output           0 ┌───────┐ 0
  └───────┘                              0 │ image │ 0  →  same size output
                                         0 └───────┘ 0
                                         0  0  0  0  0  0  0
```

**"Same" padding** for a $k \times k$ kernel: $p = \lfloor k/2 \rfloor$

For a $3 \times 3$ kernel: $p = 1$. For a $5 \times 5$ kernel: $p = 2$.

### Stride

**Stride** controls how far the kernel moves at each step. Stride 1 = move one pixel, stride 2 = move two pixels (skipping every other position).

```
Stride 1:                          Stride 2:

  [X X X] . .                       [X X X] . .
  . [X X X] .                       . . . . .
  . . [X X X]                       . . [X X X]

  5 input → 3 output               5 input → 2 output
```

### General Output Size Formula

$$
O = \left\lfloor \frac{H + 2p - k}{s} \right\rfloor + 1
$$

Where:
- $H$ = input size
- $k$ = kernel size
- $p$ = padding
- $s$ = stride
- $\lfloor \cdot \rfloor$ = floor (round down)

### ✏️ Exercise 6.1 — Output Size with Padding and Stride

Compute the output size for each configuration:

| Input | Kernel | Padding | Stride | Output |
|---|---|---|---|---|
| $28 \times 28$ | $3 \times 3$ | 0 | 1 | ? |
| $28 \times 28$ | $3 \times 3$ | 1 | 1 | ? |
| $28 \times 28$ | $5 \times 5$ | 2 | 1 | ? |
| $28 \times 28$ | $3 \times 3$ | 1 | 2 | ? |
| $32 \times 32$ | $5 \times 5$ | 0 | 2 | ? |
| $224 \times 224$ | $7 \times 7$ | 3 | 2 | ? |

<details>
<summary>Solution</summary>

Using $O = \lfloor (H + 2p - k) / s \rfloor + 1$:

| Input | Kernel | Padding | Stride | Calculation | Output |
|---|---|---|---|---|---|
| 28 | 3 | 0 | 1 | $(28 + 0 - 3)/1 + 1 = 26$ | $26 \times 26$ |
| 28 | 3 | 1 | 1 | $(28 + 2 - 3)/1 + 1 = 28$ | $28 \times 28$ ← same! |
| 28 | 5 | 2 | 1 | $(28 + 4 - 5)/1 + 1 = 28$ | $28 \times 28$ ← same! |
| 28 | 3 | 1 | 2 | $(28 + 2 - 3)/2 + 1 = 14$ | $14 \times 14$ ← halved! |
| 32 | 5 | 0 | 2 | $(32 + 0 - 5)/2 + 1 = 14$ | $14 \times 14$ |
| 224 | 7 | 3 | 2 | $(224 + 6 - 7)/2 + 1 = 112$ | $112 \times 112$ |

**Key patterns:**
- Padding $p = \lfloor k/2 \rfloor$ with stride 1 → **same** output size
- Stride 2 → output size roughly **halved**
- The last row is the first layer of ResNet!
</details>

### ✏️ Exercise 6.2 — Design Challenge

You have a $32 \times 32$ input image. You want the output to be exactly $16 \times 16$ using a single convolution layer. Find a valid combination of kernel size, padding, and stride.

<details>
<summary>Solution</summary>

We need $O = 16$ from $H = 32$. Using stride 2:

$$16 = \left\lfloor \frac{32 + 2p - k}{2} \right\rfloor + 1$$

$$15 = \left\lfloor \frac{32 + 2p - k}{2} \right\rfloor$$

$$30 = 32 + 2p - k$$

$$k - 2p = 2$$

Solutions: $k = 2, p = 0$ or $k = 4, p = 1$ or **$k = 3, p = 1, s = 2$** (this one is the most common in practice, since $3 \times 3$ kernels are standard).

Verify: $\lfloor(32 + 2 - 3)/2\rfloor + 1 = \lfloor 31/2 \rfloor + 1 = 15 + 1 = 16$ ✓
</details>

### ✏️ Exercise 6.3 — Hand Convolution with Padding

Apply a $3 \times 3$ kernel to a $3 \times 3$ image with **padding = 1** (so the output is also $3 \times 3$):

$$
I = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix}
\qquad
K = \begin{bmatrix} 0 & -1 & 0 \\ -1 & 4 & -1 \\ 0 & -1 & 0 \end{bmatrix}
$$

Padded input (zeros around the border):

$$
I_{\text{padded}} = \begin{bmatrix} 0 & 0 & 0 & 0 & 0 \\ 0 & 1 & 2 & 3 & 0 \\ 0 & 4 & 5 & 6 & 0 \\ 0 & 7 & 8 & 9 & 0 \\ 0 & 0 & 0 & 0 & 0 \end{bmatrix}
$$

Compute all 9 output values.

<details>
<summary>Solution</summary>

$(0,0)$: $0 \cdot 0 + 0 \cdot(-1) + 0 \cdot 0 + 0 \cdot(-1) + 1 \cdot 4 + 2 \cdot(-1) + 0 \cdot 0 + 4 \cdot(-1) + 5 \cdot 0 = 4 - 2 - 4 = \mathbf{-2}$

$(0,1)$: $0 \cdot 0 + 0 \cdot(-1) + 0 \cdot 0 + 1 \cdot(-1) + 2 \cdot 4 + 3 \cdot(-1) + 4 \cdot 0 + 5 \cdot(-1) + 6 \cdot 0 = -1 + 8 - 3 - 5 = \mathbf{-1}$

$(0,2)$: $0 + 0 + 0 + 2 \cdot(-1) + 3 \cdot 4 + 0 \cdot(-1) + 5 \cdot 0 + 6 \cdot(-1) + 0 = -2 + 12 - 6 = \mathbf{4}$

$(1,0)$: $0 + 1 \cdot(-1) + 2 \cdot 0 + 0 \cdot(-1) + 4 \cdot 4 + 5 \cdot(-1) + 0 \cdot 0 + 7 \cdot(-1) + 8 \cdot 0 = -1 + 16 - 5 - 7 = \mathbf{3}$

$(1,1)$: $1 \cdot 0 + 2 \cdot(-1) + 3 \cdot 0 + 4 \cdot(-1) + 5 \cdot 4 + 6 \cdot(-1) + 7 \cdot 0 + 8 \cdot(-1) + 9 \cdot 0 = -2 - 4 + 20 - 6 - 8 = \mathbf{0}$

$(1,2)$: $2 \cdot 0 + 3 \cdot(-1) + 0 + 5 \cdot(-1) + 6 \cdot 4 + 0 \cdot(-1) + 8 \cdot 0 + 9 \cdot(-1) + 0 = -3 - 5 + 24 - 9 = \mathbf{7}$

$(2,0)$: $0 + 4 \cdot(-1) + 5 \cdot 0 + 0 + 7 \cdot 4 + 8 \cdot(-1) + 0 + 0 + 0 = -4 + 28 - 8 = \mathbf{16}$

$(2,1)$: $4 \cdot 0 + 5 \cdot(-1) + 6 \cdot 0 + 7 \cdot(-1) + 8 \cdot 4 + 9 \cdot(-1) + 0 + 0 + 0 = -5 - 7 + 32 - 9 = \mathbf{11}$

$(2,2)$: $5 \cdot 0 + 6 \cdot(-1) + 0 + 8 \cdot(-1) + 9 \cdot 4 + 0 + 0 + 0 + 0 = -6 - 8 + 36 = \mathbf{22}$

$$
\text{Output} = \begin{bmatrix} -2 & -1 & 4 \\ 3 & 0 & 7 \\ 16 & 11 & 22 \end{bmatrix}
$$

This kernel is a **Laplacian filter** — it detects regions that differ from their neighbors. The center pixel (value 5) produces 0, because it's the average of its neighbors: $(2+4+6+8)/4 = 5$. Edge and corner pixels produce large values.
</details>

---

## 7. Implement Convolution from Scratch {#implement}

### 💻 Exercise 7.1 — Basic 2D Convolution

**Task:** Implement a 2D convolution function from scratch in NumPy (no padding, stride 1).

```python
import numpy as np

def conv2d(image, kernel):
    """
    Compute 2D convolution (cross-correlation) of image with kernel.
    
    Parameters:
    -----------
    image : np.array, shape (H, W)
    kernel : np.array, shape (kH, kW)
    
    Returns:
    --------
    output : np.array, shape (H - kH + 1, W - kW + 1)
    """
    H, W = image.shape
    kH, kW = kernel.shape
    
    # TODO: Compute output dimensions
    oH = ___
    oW = ___
    output = np.zeros((oH, oW))
    
    # TODO: Slide the kernel across the image
    # For each output position (i, j):
    #   Extract the patch: image[i:i+kH, j:j+kW]
    #   Compute element-wise product with kernel and sum
    for i in range(oH):
        for j in range(oW):
            ___
    
    return output
```

<details>
<summary>Solution</summary>

```python
def conv2d(image, kernel):
    H, W = image.shape
    kH, kW = kernel.shape
    oH = H - kH + 1
    oW = W - kW + 1
    output = np.zeros((oH, oW))
    
    for i in range(oH):
        for j in range(oW):
            patch = image[i:i+kH, j:j+kW]
            output[i, j] = np.sum(patch * kernel)
    
    return output
```
</details>

**Verify with Exercise 4.1:**

```python
I = np.array([[1, 2, 0, 1],
              [0, 1, 3, 2],
              [1, 0, 2, 1],
              [3, 1, 0, 2]])

K = np.array([[1, 0, -1],
              [1, 0, -1],
              [1, 0, -1]])

result = conv2d(I, K)
print(f"Result:\n{result}")
# Expected: [[-3, -1], [-1, -3]]
```

### 💻 Exercise 7.2 — Add Padding Support

**Task:** Extend your function to support zero-padding.

```python
def conv2d_padded(image, kernel, padding=0):
    """
    2D convolution with zero-padding.
    
    Parameters:
    -----------
    image : np.array, shape (H, W)
    kernel : np.array, shape (kH, kW)
    padding : int, number of zero-rows/cols to add on each side
    
    Returns:
    --------
    output : np.array
    """
    if padding > 0:
        # TODO: Pad the image with zeros on all sides
        # Hint: np.pad(image, padding, mode='constant', constant_values=0)
        image = ___
    
    # TODO: Apply conv2d (reuse your function from 7.1)
    return ___
```

<details>
<summary>Solution</summary>

```python
def conv2d_padded(image, kernel, padding=0):
    if padding > 0:
        image = np.pad(image, padding, mode='constant', constant_values=0)
    return conv2d(image, kernel)
```
</details>

**Verify with Exercise 6.3:**

```python
I2 = np.array([[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]])

K2 = np.array([[0, -1, 0],
               [-1, 4, -1],
               [0, -1, 0]])

result_padded = conv2d_padded(I2, K2, padding=1)
print(f"With padding=1:\n{result_padded}")
# Expected: [[-2, -1, 4], [3, 0, 7], [16, 11, 22]]
```

### 💻 Exercise 7.3 — Add Stride Support

**Task:** Extend your function to support stride.

```python
def conv2d_full(image, kernel, padding=0, stride=1):
    """
    2D convolution with padding and stride.
    """
    if padding > 0:
        image = np.pad(image, padding, mode='constant', constant_values=0)
    
    H, W = image.shape
    kH, kW = kernel.shape
    
    # TODO: Compute output dimensions (use the general formula!)
    oH = ___
    oW = ___
    output = np.zeros((oH, oW))
    
    # TODO: Slide the kernel with the given stride
    for i in range(oH):
        for j in range(oW):
            # What are the top-left coordinates of the patch?
            row = ___
            col = ___
            patch = image[row:row+kH, col:col+kW]
            output[i, j] = np.sum(patch * kernel)
    
    return output
```

<details>
<summary>Solution</summary>

```python
def conv2d_full(image, kernel, padding=0, stride=1):
    if padding > 0:
        image = np.pad(image, padding, mode='constant', constant_values=0)
    
    H, W = image.shape
    kH, kW = kernel.shape
    oH = (H - kH) // stride + 1
    oW = (W - kW) // stride + 1
    output = np.zeros((oH, oW))
    
    for i in range(oH):
        for j in range(oW):
            row = i * stride
            col = j * stride
            patch = image[row:row+kH, col:col+kW]
            output[i, j] = np.sum(patch * kernel)
    
    return output
```
</details>

**Verify:**

```python
# 6×6 image, 3×3 kernel, pad=1, stride=2 → should give 3×3 output
I3 = np.arange(36).reshape(6, 6).astype(float)
K3 = np.ones((3, 3)) / 9  # blur kernel

result_stride = conv2d_full(I3, K3, padding=1, stride=2)
print(f"Shape: {result_stride.shape}")  # Expected: (3, 3)
print(f"Result:\n{result_stride}")
```

---

## 8. Feature Detection: Kernels as Eyes {#feature-detection}

### Classic Kernels

```python
import matplotlib.pyplot as plt

# Define classic kernels
kernels = {
    "Identity": np.array([[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0]], dtype=float),
    
    "Box Blur": np.ones((3, 3)) / 9,
    
    "Gaussian Blur": np.array([[1, 2, 1],
                                [2, 4, 2],
                                [1, 2, 1]], dtype=float) / 16,
    
    "Sharpen": np.array([[ 0, -1,  0],
                          [-1,  5, -1],
                          [ 0, -1,  0]], dtype=float),
    
    "Edge (Laplacian)": np.array([[ 0, -1,  0],
                                   [-1,  4, -1],
                                   [ 0, -1,  0]], dtype=float),
    
    "Sobel Horizontal": np.array([[-1, -2, -1],
                                    [ 0,  0,  0],
                                    [ 1,  2,  1]], dtype=float),
    
    "Sobel Vertical": np.array([[-1, 0, 1],
                                  [-2, 0, 2],
                                  [-1, 0, 1]], dtype=float),
    
    "Emboss": np.array([[-2, -1, 0],
                          [-1,  1, 1],
                          [ 0,  1, 2]], dtype=float),
}
```

### 💻 Exercise 8.1 — Visualize Kernels on a Real Image

**Task:** Load an MNIST digit and apply each kernel. Display the results in a grid.

```python
from torchvision import datasets, transforms

# Load one MNIST image
mnist = datasets.MNIST(root='./data', train=True, download=True)
digit_image = np.array(mnist[7][0]) / 255.0  # Normalize to [0, 1]
print(f"Image shape: {digit_image.shape}, Label: {mnist[7][1]}")

# TODO: Create a figure with 2 rows × 4 columns (or 3×3 to fit all 8+1)
# First panel: original image
# Remaining panels: result of each kernel
# Use your conv2d_padded function with padding=1

fig, axes = plt.subplots(3, 3, figsize=(14, 14))

# Original
axes[0, 0].imshow(digit_image, cmap='gray')
axes[0, 0].set_title('Original', fontsize=12)
axes[0, 0].axis('off')

# TODO: Apply each kernel and display
for idx, (name, kernel) in enumerate(kernels.items()):
    row = (idx + 1) // 3
    col = (idx + 1) % 3
    ax = axes[row, col]
    
    # Apply convolution with padding=1 (same size output)
    filtered = ___
    
    ax.imshow(filtered, cmap='gray')
    ax.set_title(name, fontsize=12)
    ax.axis('off')

plt.suptitle('Effect of Different Kernels on MNIST Digit', fontsize=16)
plt.tight_layout()
plt.show()
```

<details>
<summary>Solution</summary>

```python
fig, axes = plt.subplots(3, 3, figsize=(14, 14))

axes[0, 0].imshow(digit_image, cmap='gray')
axes[0, 0].set_title('Original', fontsize=12)
axes[0, 0].axis('off')

for idx, (name, kernel) in enumerate(kernels.items()):
    row = (idx + 1) // 3
    col = (idx + 1) % 3
    ax = axes[row, col]
    
    filtered = conv2d_padded(digit_image, kernel, padding=1)
    
    ax.imshow(filtered, cmap='gray')
    ax.set_title(name, fontsize=12)
    ax.axis('off')

plt.suptitle('Effect of Different Kernels on MNIST Digit', fontsize=16)
plt.tight_layout()
plt.show()
```
</details>

### 💻 Exercise 8.2 — Edge Detection Pipeline

**Task:** Build a simple edge detector by combining horizontal and vertical Sobel filters.

The **gradient magnitude** at each pixel is:

$$
G = \sqrt{G_x^2 + G_y^2}
$$

where $G_x$ is the Sobel horizontal output and $G_y$ is the Sobel vertical output.

```python
# TODO: Apply both Sobel kernels to the MNIST digit
sobel_h = kernels["Sobel Horizontal"]
sobel_v = kernels["Sobel Vertical"]

Gx = ___  # Horizontal edges
Gy = ___  # Vertical edges

# TODO: Compute gradient magnitude
G = ___

# Display
fig, axes = plt.subplots(1, 4, figsize=(18, 4))

titles = ['Original', 'Horizontal edges ($G_x$)', 'Vertical edges ($G_y$)', 'Edge magnitude ($G$)']
images = [digit_image, Gx, Gy, G]

for ax, img, title in zip(axes, images, titles):
    ax.imshow(img, cmap='gray')
    ax.set_title(title, fontsize=13)
    ax.axis('off')

plt.tight_layout()
plt.show()
```

<details>
<summary>Solution</summary>

```python
Gx = conv2d_padded(digit_image, sobel_h, padding=1)
Gy = conv2d_padded(digit_image, sobel_v, padding=1)
G = np.sqrt(Gx ** 2 + Gy ** 2)

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
titles = ['Original', 'Horizontal edges ($G_x$)', 'Vertical edges ($G_y$)', 'Edge magnitude ($G$)']
images = [digit_image, Gx, Gy, G]

for ax, img, title in zip(axes, images, titles):
    ax.imshow(img, cmap='gray')
    ax.set_title(title, fontsize=13)
    ax.axis('off')

plt.tight_layout()
plt.show()
```
</details>

### 💻 Exercise 8.3 — Apply to Multiple Digits

**Task:** Apply the Sobel edge detector to 10 different MNIST digits (one per class). Display in a 2×10 grid: original on top, edges on bottom.

```python
# TODO: Find one example of each digit (0-9)
# Apply Sobel edge detection to each
# Display in 2×10 grid

fig, axes = plt.subplots(2, 10, figsize=(20, 5))

for digit in range(10):
    # Find first occurrence of this digit
    idx = next(i for i in range(len(mnist)) if mnist[i][1] == digit)
    img = np.array(mnist[idx][0]) / 255.0
    
    # Compute edges
    Gx = conv2d_padded(img, sobel_h, padding=1)
    Gy = conv2d_padded(img, sobel_v, padding=1)
    edges = np.sqrt(Gx ** 2 + Gy ** 2)
    
    # TODO: Display original (top row) and edges (bottom row)
    ___
    ___

plt.suptitle('Sobel Edge Detection on MNIST Digits', fontsize=16)
plt.tight_layout()
plt.show()
```

<details>
<summary>Solution</summary>

```python
fig, axes = plt.subplots(2, 10, figsize=(20, 5))

for digit in range(10):
    idx = next(i for i in range(len(mnist)) if mnist[i][1] == digit)
    img = np.array(mnist[idx][0]) / 255.0
    
    Gx = conv2d_padded(img, sobel_h, padding=1)
    Gy = conv2d_padded(img, sobel_v, padding=1)
    edges = np.sqrt(Gx ** 2 + Gy ** 2)
    
    axes[0, digit].imshow(img, cmap='gray')
    axes[0, digit].set_title(str(digit), fontsize=12)
    axes[0, digit].axis('off')
    
    axes[1, digit].imshow(edges, cmap='gray')
    axes[1, digit].axis('off')

axes[0, 0].set_ylabel('Original', fontsize=12)
axes[1, 0].set_ylabel('Edges', fontsize=12)

plt.suptitle('Sobel Edge Detection on MNIST Digits', fontsize=16)
plt.tight_layout()
plt.show()
```
</details>

### 🤔 Think About It

Look at the edge-detected digits. Can you still tell which digit is which from the edges alone? If yes, that means **edge information is sufficient** for classification — and that's exactly what CNNs learn to extract!

### 💻 Exercise 8.4 — Design Your Own Kernel

**Task:** Design a $3 \times 3$ kernel that detects **diagonal edges** (top-left to bottom-right). Test it on an MNIST digit.

*Hint:* Think about what the Sobel vertical/horizontal kernels do, and rotate the concept by 45°.

```python
# TODO: Design a diagonal edge kernel
K_diag = np.array([
    [___, ___, ___],
    [___, ___, ___],
    [___, ___, ___]
], dtype=float)

# Test it
diag_result = conv2d_padded(digit_image, K_diag, padding=1)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(digit_image, cmap='gray')
axes[0].set_title('Original')
axes[0].axis('off')
axes[1].imshow(diag_result, cmap='gray')
axes[1].set_title('Your Diagonal Kernel')
axes[1].axis('off')
plt.show()
```

<details>
<summary>One possible solution</summary>

```python
K_diag = np.array([
    [ 0,  1,  2],
    [-1,  0,  1],
    [-2, -1,  0]
], dtype=float)
```

This kernel has positive values in the top-right and negative values in the bottom-left. It responds strongly to edges that go from top-left to bottom-right. Other valid designs are possible — the key insight is that the kernel should be asymmetric along the diagonal.
</details>

---

## 9. Verify with PyTorch nn.Conv2d {#pytorch-verify}

### PyTorch Convolution

PyTorch's `nn.Conv2d` does exactly what our function does, but much faster (optimized C++ / GPU backend).

```python
import torch
import torch.nn as nn

# nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, 
                        stride=1, padding=0, bias=False)

print(f"Weight shape: {conv_layer.weight.shape}")
# torch.Size([1, 1, 3, 3])
# Meaning: 1 output filter, 1 input channel, 3×3 kernel
```

**Input format:** PyTorch expects `(batch, channels, height, width)`.

### ✏️ Exercise 9.1 — Match NumPy and PyTorch

**Task:** Use the Sobel vertical kernel in `nn.Conv2d` and verify the output matches your NumPy implementation.

```python
# Our NumPy result
sobel_v = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]], dtype=np.float32)

img_np = digit_image.astype(np.float32)
numpy_result = conv2d_padded(img_np, sobel_v, padding=1)

# TODO: Set up the PyTorch Conv2d with the same kernel
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, 
                  padding=1, bias=False)

# TODO: Load the Sobel kernel into the conv layer's weights
# conv.weight has shape (1, 1, 3, 3) — we need to reshape our kernel
with torch.no_grad():
    conv.weight.copy_(torch.tensor(___).reshape(___))

# TODO: Prepare the image as a PyTorch tensor
# Shape must be (batch=1, channels=1, H, W)
img_tensor = torch.tensor(___).reshape(___)

# TODO: Forward pass
pytorch_result = conv(img_tensor)

# Compare
pytorch_np = pytorch_result.detach().numpy().squeeze()
print(f"Max absolute difference: {np.max(np.abs(numpy_result - pytorch_np)):.8f}")
# Should be very close to 0 (floating-point precision)
```

<details>
<summary>Solution</summary>

```python
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, 
                  padding=1, bias=False)

with torch.no_grad():
    conv.weight.copy_(torch.tensor(sobel_v).reshape(1, 1, 3, 3))

img_tensor = torch.tensor(img_np).reshape(1, 1, 28, 28)
pytorch_result = conv(img_tensor)

pytorch_np = pytorch_result.detach().numpy().squeeze()
print(f"Max absolute difference: {np.max(np.abs(numpy_result - pytorch_np)):.8f}")
```
</details>

### ✏️ Exercise 9.2 — Multiple Filters at Once

**Task:** Create a `nn.Conv2d` layer with **4 output filters** and load all four classic kernels into it. Apply to the MNIST digit and display all 4 feature maps.

```python
# Four kernels to load
kernel_set = [
    kernels["Sobel Vertical"],
    kernels["Sobel Horizontal"],
    kernels["Edge (Laplacian)"],
    kernels["Gaussian Blur"],
]
kernel_names = ["Sobel V", "Sobel H", "Laplacian", "Gaussian"]

# TODO: Create Conv2d with 1 input channel, 4 output channels, 3×3 kernel, padding=1
conv_multi = nn.Conv2d(___, ___, kernel_size=___, padding=___, bias=False)

# TODO: Load all 4 kernels into the weight tensor
# conv_multi.weight has shape (4, 1, 3, 3)
with torch.no_grad():
    for i, k in enumerate(kernel_set):
        conv_multi.weight[i, 0] = torch.tensor(k, dtype=torch.float32)

# TODO: Apply to the MNIST digit (shape: 1, 1, 28, 28)
img_t = torch.tensor(img_np).reshape(1, 1, 28, 28)
feature_maps = ___

print(f"Output shape: {feature_maps.shape}")
# Expected: torch.Size([1, 4, 28, 28])

# TODO: Display the 4 feature maps
fig, axes = plt.subplots(1, 5, figsize=(18, 4))

axes[0].imshow(digit_image, cmap='gray')
axes[0].set_title('Original', fontsize=13)
axes[0].axis('off')

for i in range(4):
    fmap = ___  # Extract feature map i from the output tensor
    axes[i+1].imshow(fmap, cmap='gray')
    axes[i+1].set_title(kernel_names[i], fontsize=13)
    axes[i+1].axis('off')

plt.suptitle('4 Feature Maps from One Conv2d Layer', fontsize=16)
plt.tight_layout()
plt.show()
```

<details>
<summary>Solution</summary>

```python
conv_multi = nn.Conv2d(1, 4, kernel_size=3, padding=1, bias=False)

with torch.no_grad():
    for i, k in enumerate(kernel_set):
        conv_multi.weight[i, 0] = torch.tensor(k, dtype=torch.float32)

img_t = torch.tensor(img_np).reshape(1, 1, 28, 28)
feature_maps = conv_multi(img_t)

print(f"Output shape: {feature_maps.shape}")  # (1, 4, 28, 28)

fig, axes = plt.subplots(1, 5, figsize=(18, 4))

axes[0].imshow(digit_image, cmap='gray')
axes[0].set_title('Original', fontsize=13)
axes[0].axis('off')

for i in range(4):
    fmap = feature_maps[0, i].detach().numpy()
    axes[i+1].imshow(fmap, cmap='gray')
    axes[i+1].set_title(kernel_names[i], fontsize=13)
    axes[i+1].axis('off')

plt.suptitle('4 Feature Maps from One Conv2d Layer', fontsize=16)
plt.tight_layout()
plt.show()
```
</details>

### ✏️ Exercise 9.3 — Learned Filters: Random vs Trained

**Task:** Compare what random (untrained) filters produce vs our hand-crafted filters. This previews what a CNN will learn.

```python
# Random filters (untrained)
conv_random = nn.Conv2d(1, 4, kernel_size=3, padding=1, bias=False)
# Weights are randomly initialized by default

with torch.no_grad():
    random_maps = conv_random(img_t)

# Display side-by-side: hand-crafted (left 4) vs random (right 4)
fig, axes = plt.subplots(2, 5, figsize=(18, 7))

axes[0, 0].imshow(digit_image, cmap='gray')
axes[0, 0].set_title('Original', fontsize=12)
axes[0, 0].axis('off')
axes[1, 0].imshow(digit_image, cmap='gray')
axes[1, 0].set_title('Original', fontsize=12)
axes[1, 0].axis('off')

for i in range(4):
    # Hand-crafted
    fmap = feature_maps[0, i].detach().numpy()
    axes[0, i+1].imshow(fmap, cmap='gray')
    axes[0, i+1].set_title(f'Crafted: {kernel_names[i]}', fontsize=10)
    axes[0, i+1].axis('off')
    
    # Random
    fmap_r = random_maps[0, i].detach().numpy()
    axes[1, i+1].imshow(fmap_r, cmap='gray')
    axes[1, i+1].set_title(f'Random filter {i+1}', fontsize=10)
    axes[1, i+1].axis('off')

plt.suptitle('Hand-Crafted vs Random Filters', fontsize=16)
plt.tight_layout()
plt.show()
```

**Observation:** Random filters produce noisy, unstructured outputs. After training, a CNN's filters converge to meaningful detectors (edges, textures, patterns) — similar to our hand-crafted kernels, but **optimized for the task**.

### ✏️ Exercise 9.4 — Stacking Convolutions

**Task:** Apply two convolutions in sequence. This previews how CNNs build hierarchical features (first layer detects edges, second layer detects combinations of edges).

```python
# Two conv layers in sequence
conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1, bias=False)
conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=False)
relu = nn.ReLU()

# Load edge kernels into conv1
with torch.no_grad():
    for i, k in enumerate(kernel_set):
        conv1.weight[i, 0] = torch.tensor(k, dtype=torch.float32)

# TODO: Apply conv1 → ReLU → conv2 → ReLU to the MNIST digit
# conv2 has random (untrained) weights — that's fine for this exercise
with torch.no_grad():
    out1 = ___     # Conv1 output: (1, 4, 28, 28)
    out1r = ___    # After ReLU
    out2 = ___     # Conv2 output: (1, 8, 28, 28)
    out2r = ___    # After ReLU

print(f"After conv1: {out1.shape}")
print(f"After conv2: {out2.shape}")

# Display first 8 feature maps from conv2
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i in range(8):
    ax = axes[i // 4, i % 4]
    ax.imshow(out2r[0, i].numpy(), cmap='gray')
    ax.set_title(f'Conv2 map {i}', fontsize=11)
    ax.axis('off')
plt.suptitle('Feature Maps After 2 Convolutions', fontsize=16)
plt.tight_layout()
plt.show()
```

<details>
<summary>Solution</summary>

```python
with torch.no_grad():
    out1 = conv1(img_t)
    out1r = relu(out1)
    out2 = conv2(out1r)
    out2r = relu(out2)
```
</details>

### 🤔 Think About It

**Q:** Conv2 takes 4 input channels (from conv1's 4 feature maps) and produces 8 output channels. How many parameters does conv2 have?

<details>
<summary>Answer</summary>

$8 \times (4 \times 3 \times 3) = 8 \times 36 = 288$ weights (plus 8 biases if enabled = 296 total, but we set `bias=False` so it's 288).

Each of the 8 output filters is a $4 \times 3 \times 3$ volume — it looks at all 4 input feature maps through a 3×3 window. This is how the network combines simple features (edges) into complex features (corners, textures).
</details>

### ✏️ Exercise 9.5 — Parameter Count Challenge

A CNN has this architecture:

```
Input: 1 × 28 × 28
Conv2d(1, 16, kernel_size=3, padding=1)   → 16 × 28 × 28
ReLU
Conv2d(16, 32, kernel_size=3, padding=1)  → 32 × 28 × 28
ReLU
```

**Part A:** How many parameters in each conv layer (with bias)?

**Part B:** What would a fully-connected equivalent need? (FC layer mapping 784 → same number of outputs)

<details>
<summary>Solution</summary>

**Part A:**

Conv1: $16 \times (1 \times 3 \times 3) + 16 = 16 \times 9 + 16 = 144 + 16 = \mathbf{160}$

Conv2: $32 \times (16 \times 3 \times 3) + 32 = 32 \times 144 + 32 = 4608 + 32 = \mathbf{4{,}640}$

**Total: 4,800 parameters.**

**Part B:**

An FC layer from 784 inputs to 16 outputs (to match conv1's "16 features"):
$784 \times 16 + 16 = 12{,}560$

An FC layer from 16 intermediate to 32 outputs:
Already smaller, but the first FC layer alone has $12{,}560$ vs the **entire** CNN's $4{,}800$.

If we wanted the FC to produce as many outputs as the CNN ($32 \times 28 \times 28 = 25{,}088$ values), we'd need:
$784 \times 25{,}088 + 25{,}088 \approx 19.7$ **million** parameters.

Convolutions win by a factor of **4,000×**.
</details>

---

## 10. Summary {#summary}

### What We Learned

✅ **Why MLPs fail for images**: Too many parameters, no spatial awareness, no translation invariance  
✅ **Convolution operation**: Slide a kernel, compute local dot products  
✅ **Output size formula**: $O = \lfloor (H + 2p - k) / s \rfloor + 1$  
✅ **Padding**: Preserves spatial dimensions ($p = \lfloor k/2 \rfloor$ for "same" convolution)  
✅ **Stride**: Controls downsampling (stride 2 ≈ halves the size)  
✅ **Weight sharing**: Same kernel reused everywhere → massive parameter savings  
✅ **Feature maps**: Each kernel produces a different "view" of the image  
✅ **PyTorch nn.Conv2d**: Same operation, hardware-optimized

### Key Insights

1. **Convolution = local pattern detector:**
   - A kernel is a template for a specific pattern (edge, blur, corner)
   - Sliding it across the image finds that pattern **everywhere**
   - That's why a "7" detector works regardless of position

2. **Parameter efficiency is dramatic:**
   - A $3 \times 3$ kernel has 9 parameters
   - A fully-connected layer on a 224×224 image has millions
   - CNNs achieve **better** results with **far fewer** parameters

3. **Hierarchical features emerge from stacking:**
   - Layer 1: edges, gradients
   - Layer 2: corners, textures (combinations of edges)
   - Layer 3+: parts, objects (combinations of combinations)
   - This mirrors how the human visual cortex works

### What's Next?

**Session 11: Building CNNs**

In the next session, we'll learn:
- **Pooling layers**: Downsample feature maps while keeping important info
- **CNN architecture patterns**: Conv → ReLU → Pool → ... → Flatten → FC
- **LeNet-5**: The classic CNN architecture (1998!)
- **Training CNNs on MNIST**: Beat our MLP from Session 9

**The goal:** Assemble convolutions into a complete image classifier!

### Before Next Session

**Think about:**
1. After two $3 \times 3$ convolutions (no padding), a $28 \times 28$ image becomes $24 \times 24$. After ten such layers, what size would it be? Is this a problem?
2. Our MNIST MLP achieved ~97% accuracy. Do you think a CNN can do better? Why?
3. We hand-designed edge kernels. In a CNN, the kernels are **learned** by backpropagation. What loss function guides them?

**Optional reading:**
- Stanford CS231n: "Convolutional Neural Networks" lecture notes
- Distill.pub: "Feature Visualization"

---

**End of Session 10** 🎓

**You now understand:**
- ✅ Why images need specialized architectures
- ✅ How the convolution operation works (by hand and in code)
- ✅ How kernels detect features like edges and textures

**Next up:** Building complete CNNs for image classification! 🚀
