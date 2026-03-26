# Session 11: Building CNNs
## Assembling the Vision Machine

**Course: Neural Networks for Engineers**  
**Duration: 2 hours**

> ⚠️ **This is the completed reference version.** All exercises are filled in and ready to run.

---

## Table of Contents

### Part I — Concepts & Exercises (≈ 50 min)
1. [Recap & Motivation](#recap)
2. [Pooling Layers](#pooling)
3. [CNN Architecture Patterns](#patterns)
4. [Classic Architecture: LeNet-5](#lenet)
5. [Training CNNs: Practical Considerations](#training)

### Part II — Build, Train, Explore (≈ 70 min)
6. [Implement LeNet-5 in PyTorch](#implement-lenet)
7. [Train on MNIST (Full Dataset)](#train-mnist)
8. [Visualize What the CNN Learned](#visualize)
9. [Architecture Experiments](#experiments)
10. [Summary](#summary)

---

# Part I — Concepts & Exercises

---

## 1. Recap & Motivation {#recap}

### What We Know

✅ **Convolution**: Slide a kernel, compute local dot products → feature map (Session 10)  
✅ **Padding & stride**: Control output size; $O = \lfloor (H + 2p - k)/s \rfloor + 1$ (Session 10)  
✅ **Feature maps**: Each kernel detects one type of pattern (Session 10)  
✅ **PyTorch**: `nn.Conv2d`, `nn.Module`, training loop (Sessions 9–10)

### 🤔 Quick Questions (from Session 10's "Think About")

**Q1:** After two $3 \times 3$ convolutions (no padding), a $28 \times 28$ image becomes $24 \times 24$. After ten such layers, what size would it be?

Each layer shrinks the image by 2 in each dimension: $28 \to 26 \to 24 \to 22 \to 20 \to 18 \to 16 \to 14 \to 12 \to 10 \to 8$. After 10 layers: $8 \times 8$. After 14 layers it would reach $0 \times 0$ — impossible! This is why we need **padding** or a way to **intentionally** reduce size (pooling) rather than losing it accidentally.

**Q2:** Can a CNN beat our MLP's ~97% on MNIST?

Yes! State-of-the-art CNNs achieve **99.7%+** on MNIST. Even simple CNNs easily reach **99%+**. The spatial structure that MLPs ignore is exactly what CNNs exploit. We'll see this today.

**Q3:** What loss function guides learned kernels?

The same `CrossEntropyLoss` we've been using! The kernels are just parameters — backpropagation computes $\partial L / \partial K$ for every kernel weight, and the optimizer updates them. No one tells the network to learn edge detectors — it discovers them because edges help minimize classification loss.

### The Missing Piece

We can detect features with convolutions. But to classify an image, we need to go from feature maps to a class label:

```
Input image → [Feature extraction] → [Classification]
  28×28          Conv layers             FC layers → 10 classes

We know this part (Session 10)     We know this part (Session 9)

Today: How to connect them.
```

---

## 2. Pooling Layers {#pooling}

### The Problem

After several convolutions, we have many feature maps at full (or nearly full) resolution. This is:
- **Expensive**: Too many values to process
- **Fragile**: Features tied to exact pixel positions

We need a way to **reduce spatial dimensions** while keeping the important information.

### Max Pooling

**Max pooling** slides a window across the feature map and keeps only the **maximum value** in each window:

```
Input (4×4):                Max Pool 2×2, stride 2:

  1  3  2  1                ┌─────┬─────┐
  4  6  5  2      →        │max  │max  │     6  5
  7  2  3  1                │ 4,6 │ 5,2 │     8  4
  8  1  4  3                ├─────┼─────┤
                            │max  │max  │
                            │ 8,2 │ 4,3 │
                            └─────┴─────┘

  Output (2×2): halved in each dimension!
```

**Key properties:**
- Reduces spatial size (typically by half: $2 \times 2$ pool, stride 2)
- **Keeps the strongest activation** in each region
- Provides a small amount of **translation invariance** (the "6" can shift by 1 pixel and the max is still 6)
- **No learnable parameters!** It's a fixed operation.

### Average Pooling

Instead of the maximum, take the **mean** of each window:

```
Same input:                 Avg Pool 2×2, stride 2:

  1  3  2  1                (1+3+4+6)/4  (2+1+5+2)/4     3.5  2.5
  4  6  5  2      →
  7  2  3  1                (7+2+8+1)/4  (3+1+4+3)/4     4.5  2.75
  8  1  4  3
```

| | Max Pooling | Average Pooling |
|---|---|---|
| Keeps | Strongest activation | Average activation |
| Used for | Most hidden layers | Sometimes final layer (Global Avg Pool) |
| Effect | "Is this feature present?" | "How much of this feature?" |
| In practice | **Most common** | Used in modern architectures |

### ✏️ Exercise 2.1 — Max Pooling by Hand

Apply $2 \times 2$ max pooling (stride 2) to this $4 \times 4$ feature map:

$$
\begin{bmatrix} 3 & 1 & 4 & 2 \\ 0 & 5 & 1 & 3 \\ 2 & 4 & 6 & 0 \\ 1 & 3 & 2 & 8 \end{bmatrix}
$$

**Solution:**

Top-left $2 \times 2$: $\max(3, 1, 0, 5) = 5$  
Top-right $2 \times 2$: $\max(4, 2, 1, 3) = 4$  
Bottom-left $2 \times 2$: $\max(2, 4, 1, 3) = 4$  
Bottom-right $2 \times 2$: $\max(6, 0, 2, 8) = 8$

$$
\text{Output} = \begin{bmatrix} 5 & 4 \\ 4 & 8 \end{bmatrix}
$$

### ✏️ Exercise 2.2 — Translation Invariance Demo

Consider these two $4 \times 4$ feature maps — the second is the first shifted one pixel to the right:

$$
A = \begin{bmatrix} 0 & 9 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{bmatrix}
\qquad
B = \begin{bmatrix} 0 & 0 & 9 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{bmatrix}
$$

Apply $2 \times 2$ max pooling (stride 2) to both. Are the outputs the same?

**Solution:**

$\text{MaxPool}(A) = \begin{bmatrix} 9 & 0 \\ 0 & 0 \end{bmatrix}$

$\text{MaxPool}(B) = \begin{bmatrix} 0 & 9 \\ 0 & 0 \end{bmatrix}$

The outputs are **not** identical — the 9 moved from position (0,0) to (0,1). But the **presence** of the strong activation is preserved in the top row. Max pooling provides invariance to **small** shifts (within the pool window), not large ones. Multiple pooling layers provide invariance to progressively larger shifts.

### ✏️ Exercise 2.3 — Dimension Tracking

An input of size $32 \times 32$ passes through the following layers. Track the spatial dimensions:

| Layer | Output size | Reasoning |
|---|---|---|
| Input | $32 \times 32$ | |
| Conv2d(3×3, padding=1, stride=1) | $32 \times 32$ | Same padding preserves size |
| ReLU | $32 \times 32$ | Element-wise, no size change |
| MaxPool2d(2×2, stride=2) | $16 \times 16$ | Halved |
| Conv2d(3×3, padding=1, stride=1) | $16 \times 16$ | Same padding preserves size |
| ReLU | $16 \times 16$ | No size change |
| MaxPool2d(2×2, stride=2) | $8 \times 8$ | Halved again |

**Pattern:** Conv+ReLU maintain size (with same padding), MaxPool halves it. Two pool layers: $32 \to 16 \to 8$.

---

## 3. CNN Architecture Patterns {#patterns}

### The Standard Pattern

Almost every CNN follows this structure:

```
┌──────────────────────────────┐   ┌─────────────────────────┐
│     FEATURE EXTRACTOR        │   │       CLASSIFIER        │
│                              │   │                         │
│  Conv → ReLU → Pool          │   │  Flatten → FC → ReLU   │
│  Conv → ReLU → Pool          │→→→│  FC → Softmax           │
│  Conv → ReLU → Pool          │   │                         │
│  ...                         │   │  (Same as Session 9 MLP)│
│                              │   │                         │
│  Spatial dims shrink ↓       │   │  No spatial dims        │
│  Channel count grows ↑       │   │  Just a vector          │
└──────────────────────────────┘   └─────────────────────────┘
```

### The "Funnel" Shape

As we go deeper, spatial dimensions **decrease** and channel count **increases**:

```
Layer        Channels    Spatial      Total values
─────        ────────    ───────      ────────────
Input        1           28 × 28      784
After Conv1  16          28 × 28      12,544
After Pool1  16          14 × 14      3,136
After Conv2  32          14 × 14      6,272
After Pool2  32          7 × 7        1,568
Flatten      —           —            1,568
FC1          128         —            128
Output       10          —            10
```

**Intuition:** Early layers have few channels but high resolution (detect simple local features like edges). Deep layers have many channels but low resolution (detect complex global features like shapes, parts).

### The Flatten Operation

At some point we must transition from a 3D tensor `(channels, H, W)` to a 1D vector for the fully-connected classifier. This is **flattening**:

```python
# Before flatten: shape (batch, 32, 7, 7)
x = x.view(x.size(0), -1)     # or x.flatten(1)
# After flatten: shape (batch, 32*7*7) = (batch, 1568)
```

### ✏️ Exercise 3.1 — Trace a Full CNN

Trace the tensor shape through this network (input: batch of 4 images, $1 \times 28 \times 28$):

**Solution:**

| Layer | Output shape |
|---|---|
| Input | `(4, 1, 28, 28)` |
| Conv2d(1→8, k=5, p=2) | `(4, 8, 28, 28)` |
| ReLU | `(4, 8, 28, 28)` |
| MaxPool2d(2, 2) | `(4, 8, 14, 14)` |
| Conv2d(8→16, k=5, p=2) | `(4, 16, 14, 14)` |
| ReLU | `(4, 16, 14, 14)` |
| MaxPool2d(2, 2) | `(4, 16, 7, 7)` |
| Flatten | `(4, 784)` — because $16 \times 7 \times 7 = 784$ |
| Linear(784→120) | `(4, 120)` |
| ReLU | `(4, 120)` |
| Linear(120→10) | `(4, 10)` |

The `nn.Linear` needs **784** input features (the flattened feature volume $16 \times 7 \times 7$).

### ✏️ Exercise 3.2 — Parameter Count

For the CNN in Exercise 3.1, compute the number of parameters in each layer:

**CNN:**

| Layer | Calculation | Parameters |
|---|---|---|
| Conv2d(1→8, k=5) | $8 \times (1 \times 5 \times 5) + 8$ | 208 |
| Conv2d(8→16, k=5) | $16 \times (8 \times 5 \times 5) + 16$ | 3,216 |
| Linear(784→120) | $784 \times 120 + 120$ | 94,200 |
| Linear(120→10) | $120 \times 10 + 10$ | 1,210 |
| **Total** | | **98,834** |

**Session 9 MLP (784→256→128→10):**

| Layer | Parameters |
|---|---|
| Linear(784→256) | 200,960 |
| Linear(256→128) | 32,896 |
| Linear(128→10) | 1,290 |
| **Total** | **235,146** |

The CNN has **2.4× fewer parameters** but (as we'll see) achieves better accuracy. Most of the CNN's parameters are in the first FC layer — the conv layers are very efficient.

---

## 4. Classic Architecture: LeNet-5 {#lenet}

### The Architecture That Started It All

LeNet-5 (Yann LeCun, 1998) was designed for handwritten digit recognition — the exact task we've been working on! It was used by the US Postal Service to read ZIP codes.

```
Input: 1 × 32 × 32 (we'll adapt to 28×28)

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Conv(1→6, 5×5) → ReLU → Pool(2×2)                            │
│       ↓                                                         │
│  Conv(6→16, 5×5) → ReLU → Pool(2×2)                           │
│       ↓                                                         │
│  Flatten → FC(400→120) → ReLU → FC(120→84) → ReLU → FC(84→10)│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**For our 28×28 MNIST input** (no padding on convolutions):

| Layer | Output shape | Parameters |
|---|---|---|
| Input | `(1, 28, 28)` | — |
| Conv(1→6, k=5) | `(6, 24, 24)` | 156 |
| ReLU | `(6, 24, 24)` | — |
| MaxPool(2, 2) | `(6, 12, 12)` | — |
| Conv(6→16, k=5) | `(16, 8, 8)` | 2,416 |
| ReLU | `(16, 8, 8)` | — |
| MaxPool(2, 2) | `(16, 4, 4)` | — |
| Flatten | `(256)` | — |
| FC(256→120) | `(120)` | 30,840 |
| ReLU | `(120)` | — |
| FC(120→84) | `(84)` | 10,164 |
| ReLU | `(84)` | — |
| FC(84→10) | `(10)` | 850 |
| **Total** | | **44,426** |

Only **44K parameters** — half the size of our Session 9 MLP!

### Why Depth Matters

A single convolution layer detects **simple** features (edges, gradients). Stacking layers lets the network build a **hierarchy**:

```
Layer 1 (Conv1):    Edges, gradients
                         ↓ combine
Layer 2 (Conv2):    Corners, curves, textures
                         ↓ combine
Layer 3 (FC):       Digit parts (loops, strokes)
                         ↓ combine
Layer 4 (FC):       Whole digits (0-9)
```

Each layer builds on the previous one. This is the power of **deep** learning — not just more parameters, but more levels of abstraction.

### ✏️ Exercise 4.1 — Receptive Field

The **receptive field** is the region of the input that influences a single output neuron.

**Part A:** After one $5 \times 5$ conv (no padding), each output pixel "sees" a $5 \times 5$ patch of the input. What is the receptive field after a $2 \times 2$ max pool?

**Part B:** After Conv1 ($5 \times 5$) + Pool ($2 \times 2$) + Conv2 ($5 \times 5$), what is the receptive field on the original input?

**Solution:**

**Part A:** The $2 \times 2$ pool combines $2 \times 2$ adjacent conv outputs. Each of those already sees $5 \times 5$ input pixels. The pools overlap by 4 pixels in each dimension (because stride 1 in conv), so the receptive field after pool is:

$5 + (2 - 1) = 6$ in each dimension → $\mathbf{6 \times 6}$ receptive field.

More precisely: the pool output at position $(i,j)$ looks at conv outputs $(2i, 2j)$ through $(2i+1, 2j+1)$, which look at input pixels $(2i, 2j)$ through $(2i+5, 2j+5)$.

**Part B:** After Conv2 ($5 \times 5$), each Conv2 output sees a $5 \times 5$ region of Conv1's pooled output. Each of those sees $6 \times 6$ input pixels. With the additional $5 \times 5$ spread and stride-2 pooling:

Receptive field = $5 + (5 - 1) \times 2 = 5 + 8 = 13$ → approximately $\mathbf{14 \times 14}$ input pixels.

A single neuron deep in the network "sees" **half the image**! This is how local operations build global understanding.

### 💻 Exercise 4.2 — Verify Receptive Field in Code (optional)

**Task:** Confirm the receptive field calculation empirically. Pass an impulse signal (all zeros except one pixel = 1) through the conv+pool layers, and observe which output neurons are non-zero — that region is the receptive field.

```python
import torch
import torch.nn as nn

# Build Conv1 + Pool1 from LeNet-5 (no padding, 5×5 kernel)
conv1 = nn.Conv2d(1, 1, kernel_size=5, bias=False)
pool1 = nn.MaxPool2d(2, 2)
conv2 = nn.Conv2d(1, 1, kernel_size=5, bias=False)

# Fill all kernels with 1s so any non-zero input propagates
nn.init.constant_(conv1.weight, 1.0)
nn.init.constant_(conv2.weight, 1.0)

# Create an impulse: 28×28 image, single pixel at center set to 1
impulse = torch.zeros(1, 1, 28, 28)
impulse[0, 0, 13, 13] = 1.0    # center pixel

with torch.no_grad():
    after_conv1 = conv1(impulse)           # (1, 1, 24, 24)
    after_pool1 = pool1(after_conv1)       # (1, 1, 12, 12)
    after_conv2 = conv2(after_pool1)       # (1, 1,  8,  8)

mask = (after_conv2[0, 0] > 0)
print(f"After conv1:  {after_conv1.shape}, non-zero: {(after_conv1 > 0).sum().item()} pixels")
print(f"After pool1:  {after_pool1.shape}, non-zero: {(after_pool1 > 0).sum().item()} pixels")
print(f"After conv2:  {after_conv2.shape}, non-zero: {mask.sum().item()} pixels")

rows = torch.where(mask.any(dim=1))[0]
cols = torch.where(mask.any(dim=0))[0]
print(f"\nOutput support rows: {rows.min().item()} → {rows.max().item()} "
      f"(span: {rows.max().item() - rows.min().item() + 1})")
print(f"Output support cols: {cols.min().item()} → {cols.max().item()} "
      f"(span: {cols.max().item() - cols.min().item() + 1})")
```

Expected output (approximately):
```
After conv1:  torch.Size([1, 1, 24, 24]), non-zero: 25 pixels  (5×5 patch)
After pool1:  torch.Size([1, 1, 12, 12]), non-zero: 9 pixels   (3×3 after halving)
After conv2:  torch.Size([1, 1,  8,  8]), non-zero: 1 pixel    (single neuron affected)
```

The **one output neuron** that fires was triggered by a $\approx 14 \times 14$ region of the original image — consistent with our analytical calculation.

---

## 5. Training CNNs: Practical Considerations {#training}

### Data Augmentation

CNNs learn from spatial patterns, so we can create more training data by applying **random spatial transformations**:

```python
import torchvision.transforms as T

train_transform = T.Compose([
    T.RandomRotation(10),           # Rotate ±10°
    T.RandomAffine(0, translate=(0.1, 0.1)),  # Shift up to 10%
    T.ToTensor(),
])

test_transform = T.Compose([
    T.ToTensor(),                    # No augmentation for test!
])
```

**Key rule:** Augment **training** data only. Test data must reflect real-world conditions.

**Common augmentations for MNIST:**

| Augmentation | Effect | Why it helps |
|---|---|---|
| Rotation (±10°) | Slightly tilted digits | People write at different angles |
| Translation (10%) | Shifted position | Digits aren't always centered |
| Scaling (±10%) | Slightly larger/smaller | Handwriting size varies |

### Batch Training with DataLoader

For large datasets, we can't fit all images in memory at once. PyTorch's `DataLoader` handles **mini-batch** training:

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# Training loop now iterates over batches:
for epoch in range(n_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

This is the **batch SGD** we discussed in Session 5, now automated!

### ✏️ Exercise 5.1 — Augmentation Reasoning

For each dataset, which augmentations make sense?

**Solution:**

| Dataset | Rotation? | H-flip? | V-flip? | Color jitter? |
|---|---|---|---|---|
| MNIST digits | ✅ Small (±15°) | ❌ (6 ≠ mirrored 6) | ❌ (6 ≠ 9 upside down) | ❌ (grayscale) |
| Cats vs dogs | ✅ Small | ✅ (a cat facing left is still a cat) | ❌ (upside-down cats are rare) | ✅ (lighting varies) |
| Satellite images | ✅ Full 360° | ✅ | ✅ (no "up" in satellite view) | ✅ (seasons, time of day) |

The key principle: only augment in ways that **preserve the label**. A horizontally flipped "b" becomes "d" — that would confuse the model!

### 💻 Exercise 5.2 — Visualize Augmentation Effects

**Task:** Load a single MNIST image and display it alongside several augmented versions.

```python
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Load MNIST without any transform, to get a raw PIL image
raw_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
raw_img, label = raw_dataset[0]   # PIL Image

aug_none = transforms.Compose([transforms.ToTensor()])

aug_rotate = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

aug_translate = transforms.Compose([
    transforms.RandomAffine(0, translate=(0.15, 0.15)),
    transforms.ToTensor(),
])

aug_combined = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomAffine(0, translate=(0.15, 0.15)),
    transforms.ToTensor(),
])

augmentations = {
    "Original": aug_none,
    "Rotation ±15°": aug_rotate,
    "Translation 15%": aug_translate,
    "Combined": aug_combined,
}

# Display: 4 columns (one per augmentation), 4 rows (4 random samples each)
fig, axes = plt.subplots(4, 4, figsize=(12, 12))

for col, (name, aug) in enumerate(augmentations.items()):
    axes[0, col].set_title(name, fontsize=12, fontweight='bold')
    for row in range(4):
        img_t = aug(raw_img)
        axes[row, col].imshow(img_t.squeeze(), cmap='gray')
        axes[row, col].axis('off')

plt.suptitle(f'Augmentation Comparison — Digit {label}', fontsize=16)
plt.tight_layout()
plt.show()
```

**Observation:** Each row shows a **different random draw** of the same augmentation. Notice how the combined transform generates genuinely varied training samples while always preserving the digit identity.

---

# Part II — Build, Train, Explore

---

## 6. Implement LeNet-5 in PyTorch {#implement-lenet}

### 💻 Exercise 6.1 — Define LeNet-5

**Task:** Implement LeNet-5 as an `nn.Module`. Use the architecture from Section 4, adapted for $28 \times 28$ input.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

class LeNet5(nn.Module):
    """
    LeNet-5 adapted for 28×28 MNIST.
    
    Architecture:
        Conv(1→6, 5×5) → ReLU → MaxPool(2×2)
        Conv(6→16, 5×5) → ReLU → MaxPool(2×2)
        Flatten → FC(256→120) → ReLU → FC(120→84) → ReLU → FC(84→10)
    """
    
    def __init__(self):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1: Conv(1→6, k=5) → ReLU → MaxPool(2,2)
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 2: Conv(6→16, k=5) → ReLU → MaxPool(2,2)
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        # After features: shape is (batch, 16, 4, 4) → flatten to 256
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)   # Flatten: (batch, 16, 4, 4) → (batch, 256)
        x = self.classifier(x)
        return x

# Create and verify
model = LeNet5()
print(model)

# Verify with a dummy input
dummy = torch.randn(2, 1, 28, 28)   # batch=2, channels=1, 28×28
out = model(dummy)
print(f"\nInput:  {dummy.shape}")
print(f"Output: {out.shape}")  # Should be (2, 10)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### 💻 Exercise 6.2 — Verify Shapes Layer by Layer

**Task:** Write a diagnostic function that prints the tensor shape after every layer.

```python
def trace_shapes(model, input_shape=(1, 1, 28, 28)):
    """Print the tensor shape after each layer in the model."""
    x = torch.randn(input_shape)
    print(f"{'Input':>30s}: {list(x.shape)}")
    
    for name, layer in model.features.named_children():
        x = layer(x)
        print(f"{str(layer):>30s}: {list(x.shape)}")
    
    x = x.view(x.size(0), -1)
    print(f"{'Flatten':>30s}: {list(x.shape)}")
    
    for name, layer in model.classifier.named_children():
        x = layer(x)
        print(f"{str(layer):>30s}: {list(x.shape)}")

trace_shapes(LeNet5())
```

Expected output:
```
                         Input: [1, 1, 28, 28]
        Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1)): [1, 6, 24, 24]
                          ReLU(): [1, 6, 24, 24]
   MaxPool2d(kernel_size=2, stride=2, ...): [1, 6, 12, 12]
       Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1)): [1, 16, 8, 8]
                          ReLU(): [1, 16, 8, 8]
   MaxPool2d(kernel_size=2, stride=2, ...): [1, 16, 4, 4]
                       Flatten: [1, 256]
         Linear(in_features=256, out_features=120, ...): [1, 120]
                          ReLU(): [1, 120]
          Linear(in_features=120, out_features=84, ...): [1, 84]
                          ReLU(): [1, 84]
           Linear(in_features=84, out_features=10, ...): [1, 10]
```

---

## 7. Train on MNIST (Full Dataset) {#train-mnist}

### 💻 Exercise 7.1 — Load MNIST with DataLoader

**Task:** Set up the MNIST dataset with data augmentation for training and DataLoaders for batching.

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load datasets
train_dataset = datasets.MNIST(root='./data', train=True, download=True, 
                                transform=train_transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, 
                               transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

print(f"Training batches: {len(train_loader)} (of size 64)")
print(f"Test batches:     {len(test_loader)} (of size 256)")

# Verify a batch
images, labels = next(iter(train_loader))
print(f"Batch shape: {images.shape}, Labels shape: {labels.shape}")
# Expected: (64, 1, 28, 28) and (64,)
```

### 💻 Exercise 7.2 — Write the Batch Training Loop

**Task:** Write a complete training loop using DataLoader.

```python
def train_cnn(model, train_loader, test_loader, n_epochs=10, lr=0.001):
    """
    Train a CNN with batch SGD.
    
    Returns: train_losses, test_accs (per epoch)
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    test_accs = []
    
    for epoch in range(n_epochs):
        # ── Train ──
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        train_losses.append(avg_loss)
        
        # ── Evaluate on test set ──
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        test_acc = correct / total * 100
        test_accs.append(test_acc)
        
        print(f"Epoch {epoch+1:2d}/{n_epochs}: "
              f"Train Loss = {avg_loss:.4f} | Test Acc = {test_acc:.2f}%")
    
    return train_losses, test_accs
```

### 💻 Exercise 7.3 — Train and Plot

**Task:** Train LeNet-5 for 10 epochs and plot the training loss and test accuracy.

```python
model = LeNet5()
train_losses, test_accs = train_cnn(model, train_loader, test_loader, n_epochs=10, lr=0.001)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(range(1, len(train_losses)+1), train_losses, 'b-o', linewidth=2, markersize=6)
ax.set_xlabel('Epoch', fontsize=14)
ax.set_ylabel('Training Loss', fontsize=14)
ax.set_title('LeNet-5 Training Loss', fontsize=16)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(range(1, len(test_accs)+1), test_accs, 'g-o', linewidth=2, markersize=6)
ax.axhline(y=97, color='red', linestyle='--', linewidth=2, label='MLP baseline (~97%)')
ax.set_xlabel('Epoch', fontsize=14)
ax.set_ylabel('Test Accuracy (%)', fontsize=14)
ax.set_title('LeNet-5 Test Accuracy', fontsize=16)
ax.set_ylim(95, 100)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nFinal test accuracy: {test_accs[-1]:.2f}%")
print(f"Session 9 MLP baseline: ~97%")
```

### 💻 Exercise 7.4 — Confusion Matrix

**Task:** Build and display the $10 \times 10$ confusion matrix on the test set.

```python
# Collect all predictions
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        all_preds.append(preds)
        all_labels.append(labels)

all_preds = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()

# Build confusion matrix
cm = np.zeros((10, 10), dtype=int)
for true, pred in zip(all_labels, all_preds):
    cm[true, pred] += 1

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cm, cmap='Blues')
for i in range(10):
    for j in range(10):
        color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                color=color, fontsize=9)

ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xlabel('Predicted', fontsize=14)
ax.set_ylabel('True', fontsize=14)
ax.set_title(f'LeNet-5 Confusion Matrix (Acc: {test_accs[-1]:.2f}%)', fontsize=16)
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.show()

# Per-digit accuracy
print("\nPer-digit accuracy:")
for d in range(10):
    acc = cm[d, d] / cm[d].sum() * 100
    print(f"  Digit {d}: {acc:.1f}% ({cm[d, d]}/{cm[d].sum()})")
```

---

## 8. Visualize What the CNN Learned {#visualize}

### The Key Question

We hand-designed edge and blur kernels in Session 10. Now the CNN **learned** its own kernels through backpropagation. What did it discover?

### 💻 Exercise 8.1 — Visualize Learned Filters

**Task:** Extract and display the 6 filters learned by Conv1 (the first convolutional layer).

```python
conv1_weights = model.features[0].weight.data.clone()

print(f"Conv1 filter shape: {conv1_weights.shape}")

fig, axes = plt.subplots(1, 6, figsize=(15, 3))
for i in range(6):
    filt = conv1_weights[i, 0].numpy()   # shape (1, 5, 5) → (5, 5)
    ax = axes[i]
    ax.imshow(filt, cmap='gray')
    ax.set_title(f'Filter {i}', fontsize=11)
    ax.axis('off')

plt.suptitle('LeNet-5 Conv1: Learned Filters (5×5)', fontsize=16)
plt.tight_layout()
plt.show()
```

**Observation:** You should see filters that resemble **edge detectors** at various angles, **gradient detectors**, and maybe a **blob detector**. The CNN reinvented what took decades of computer vision research — automatically, from data!

### 💻 Exercise 8.2 — Visualize Feature Maps

**Task:** Pass a single MNIST digit through the network and display the feature maps after each conv layer.

```python
# Get a test digit
test_img, test_label = test_dataset[0]
print(f"Digit: {test_label}, shape: {test_img.shape}")

model.eval()
with torch.no_grad():
    x = test_img.unsqueeze(0)      # Add batch dim: (1, 1, 28, 28)
    
    # After Conv1 + ReLU
    after_conv1 = model.features[1](model.features[0](x))              # (1, 6, 24, 24)
    # After Pool1
    after_pool1 = model.features[2](after_conv1)                        # (1, 6, 12, 12)
    # After Conv2 + ReLU
    after_conv2 = model.features[4](model.features[3](after_pool1))     # (1, 16, 8, 8)
    # After Pool2
    after_pool2 = model.features[5](after_conv2)                        # (1, 16, 4, 4)

fig = plt.figure(figsize=(18, 12))

# Original
ax = fig.add_subplot(4, 8, 1)
ax.imshow(test_img.squeeze(), cmap='gray')
ax.set_title(f'Input (digit {test_label})', fontsize=11)
ax.axis('off')

# Conv1 feature maps (6 maps)
for i in range(6):
    ax = fig.add_subplot(4, 8, 9 + i)   # Second row
    fmap = after_conv1[0, i].numpy()
    ax.imshow(fmap, cmap='viridis')
    ax.set_title(f'Conv1-{i}', fontsize=9)
    ax.axis('off')

# Conv2 feature maps (16 maps in 2 rows of 8)
for i in range(16):
    row = 2 + i // 8    # rows 2 and 3
    col = i % 8
    ax = fig.add_subplot(4, 8, row * 8 + col + 1)
    fmap = after_conv2[0, i].numpy()
    ax.imshow(fmap, cmap='viridis')
    ax.set_title(f'C2-{i}', fontsize=8)
    ax.axis('off')

plt.suptitle('Feature Maps Through LeNet-5', fontsize=16)
plt.tight_layout()
plt.show()
```

**Observations to look for:**
- **Conv1 maps**: Should look like edges/gradients applied to the digit — some highlight horizontal strokes, others vertical
- **Conv2 maps**: More abstract — combinations of edges that detect curves, junctions, and parts of digits
- **Some maps may be mostly dark**: the filter didn't fire for this particular digit, but it would for others

### 💻 Exercise 8.3 — Compare Feature Maps Across Digits

**Task:** Pick 3 different digits (e.g., 0, 1, 7) and show the Conv1 feature maps for each.

```python
digits_to_show = [0, 1, 7]
digit_images = []
for d in digits_to_show:
    idx = next(i for i in range(len(test_dataset)) if test_dataset[i][1] == d)
    digit_images.append(test_dataset[idx][0])

fig, axes = plt.subplots(3, 7, figsize=(18, 8))

for row, (d, img) in enumerate(zip(digits_to_show, digit_images)):
    # Original
    axes[row, 0].imshow(img.squeeze(), cmap='gray')
    axes[row, 0].set_title(f'Digit {d}', fontsize=12)
    axes[row, 0].axis('off')
    
    # Feature maps
    with torch.no_grad():
        fmaps = model.features[1](model.features[0](img.unsqueeze(0)))
    
    for i in range(6):
        ax = axes[row, i+1]
        ax.imshow(fmaps[0, i].numpy(), cmap='viridis')
        ax.axis('off')
        if row == 0:
            ax.set_title(f'Filter {i}', fontsize=10)

plt.suptitle('Same Filters, Different Digits', fontsize=16)
plt.tight_layout()
plt.show()
```

**Key observation:** The same filter produces **different patterns** for different digits. Filter 2 might highlight the horizontal bar in "7" and the top curve in "0" — both are horizontal features, but they appear in different places. This is **weight sharing** in action: one filter, many locations.

### 💻 Exercise 8.4 — What Gets Misclassified?

**Task:** Find some misclassified digits, show them with their Conv1 feature maps, and analyze why the CNN failed.

```python
# Find misclassified examples
wrong_indices = np.where(all_preds != all_labels)[0]
print(f"Misclassified: {len(wrong_indices)} out of {len(all_labels)} "
      f"({len(wrong_indices)/len(all_labels)*100:.2f}%)")

# Show first 5 misclassified digits with their feature maps
fig, axes = plt.subplots(5, 7, figsize=(18, 13))

for row in range(min(5, len(wrong_indices))):
    idx = wrong_indices[row]
    img, true_label = test_dataset[idx]
    pred_label = all_preds[idx]
    
    # Original
    axes[row, 0].imshow(img.squeeze(), cmap='gray')
    axes[row, 0].set_title(f'True: {true_label}\nPred: {pred_label}', 
                            fontsize=11, color='red')
    axes[row, 0].axis('off')
    
    # Feature maps
    with torch.no_grad():
        fmaps = model.features[1](model.features[0](img.unsqueeze(0)))
    
    for i in range(6):
        axes[row, i+1].imshow(fmaps[0, i].numpy(), cmap='viridis')
        axes[row, i+1].axis('off')

plt.suptitle('Misclassified Digits: Where the CNN Fails', fontsize=16)
plt.tight_layout()
plt.show()
```

**Write in your notebook:** For 2–3 of the misclassified digits, can you see **why** the CNN was confused? Do the feature maps look ambiguous?

---

## 9. Architecture Experiments {#experiments}

### 💻 Exercise 9.1 — CNN vs MLP: Head-to-Head

**Task:** Train an MLP and a CNN on the same data and training setup, and compare fairly.

```python
# MLP baseline (from Session 9, adapted for DataLoader)
class MLP_MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),           # (batch, 1, 28, 28) → (batch, 784)
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )
    
    def forward(self, x):
        return self.net(x)

mlp = MLP_MNIST()
cnn = LeNet5()

print("Training MLP...")
mlp_losses, mlp_accs = train_cnn(mlp, train_loader, test_loader, n_epochs=10, lr=0.001)

print("\nTraining CNN...")
cnn_losses, cnn_accs = train_cnn(cnn, train_loader, test_loader, n_epochs=10, lr=0.001)

mlp_params = sum(p.numel() for p in mlp.parameters())
cnn_params = sum(p.numel() for p in cnn.parameters())

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(1, 11), mlp_accs, 'b-o', linewidth=2, label=f'MLP ({mlp_params:,} params)')
ax.plot(range(1, 11), cnn_accs, 'g-o', linewidth=2, label=f'CNN ({cnn_params:,} params)')
ax.set_xlabel('Epoch', fontsize=14)
ax.set_ylabel('Test Accuracy (%)', fontsize=14)
ax.set_title('MLP vs CNN on MNIST', fontsize=16)
ax.legend(fontsize=12)
ax.set_ylim(95, 100)
ax.grid(True, alpha=0.3)
plt.show()

print(f"\nMLP parameters:  {mlp_params:,}")
print(f"CNN parameters:  {cnn_params:,}")
print(f"CNN / MLP ratio: {cnn_params/mlp_params:.2f}")
```

### 💻 Exercise 9.2 — Architecture Variations

**Task:** Modify LeNet-5 and measure the impact. Implement **three variants** and compare.

```python
# Variant A: Deeper — add a third conv block
class DeepCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),    # → 6×14×14
            nn.Conv2d(6, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),   # → 16×7×7
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),  # → 32×3×3
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 3 * 3, 120),
            nn.ReLU(),
            nn.Linear(120, 10),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Variant B: Wider — more filters
class WideCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 5), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# Variant C: Tiny — minimal CNN (1 conv layer)
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 5), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Linear(8 * 12 * 12, 10)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
```

**Train all variants and compare:**

```python
variants = {
    "LeNet-5": LeNet5(),
    "Deep (3 conv)": DeepCNN(),
    "Wide (32→64)": WideCNN(),
    "Tiny (1 conv)": TinyCNN(),
}

results = {}

for name, model_v in variants.items():
    n_params = sum(p.numel() for p in model_v.parameters())
    print(f"\n{'='*50}")
    print(f"Training {name} ({n_params:,} parameters)")
    print(f"{'='*50}")
    
    losses, accs = train_cnn(model_v, train_loader, test_loader, n_epochs=10, lr=0.001)
    results[name] = {"losses": losses, "accs": accs, "params": n_params}

fig, ax = plt.subplots(figsize=(12, 7))

for name, res in results.items():
    ax.plot(range(1, 11), res["accs"], '-o', linewidth=2, markersize=5,
            label=f'{name} ({res["params"]:,} params, {res["accs"][-1]:.2f}%)')

ax.set_xlabel('Epoch', fontsize=14)
ax.set_ylabel('Test Accuracy (%)', fontsize=14)
ax.set_title('CNN Architecture Comparison on MNIST', fontsize=16)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### ✏️ Exercise 9.3 — Analysis Questions

1. **Which architecture achieved the best accuracy?** Is the best one also the biggest?
2. **TinyCNN has only 1 conv layer.** How does it compare to the Session 9 MLP? What does this tell you about the value of even a single convolution?
3. **DeepCNN has 3 conv layers.** Did it beat LeNet-5? After pooling 3 times ($28 \to 14 \to 7 \to 3$), the feature maps are only $3 \times 3$. Is this too small?
4. **WideCNN uses dropout in the FC layer.** Based on Session 8, why is this a good idea here?
5. You have a new task: classify $64 \times 64$ RGB images into 100 classes. Based on what you've learned, sketch an architecture (layers, filter sizes, channels). How many conv+pool blocks would you use?

**Discussion:**

1. **Best accuracy vs. biggest:** WideCNN (32→64 channels) often wins despite having more params than LeNet-5 but fewer than DeepCNN. Bigger is not always better.

2. **TinyCNN vs MLP:** TinyCNN typically outperforms the Session 9 MLP (~97%) despite having far fewer parameters. Even one conv layer exploits spatial structure that a flat MLP can't.

3. **DeepCNN with 3×3 maps:** Feature maps become very small after 3 pools. The 3×3 spatial resolution may be too coarse for MNIST — the network loses positional information. On larger images (64×64+), a third pool block is beneficial.

4. **Dropout in WideCNN:** The first FC layer of WideCNN has $64 \times 4 \times 4 = 1024$ input features. A large FC layer is prone to overfitting; Dropout(0.5) acts as an ensemble of sub-networks and reduces co-adaptation.

5. **64×64 RGB architecture (example):**  
   `Conv(3→32, 3×3, p=1) → ReLU → Pool(2,2)` → 32×32  
   `Conv(32→64, 3×3, p=1) → ReLU → Pool(2,2)` → 16×16  
   `Conv(64→128, 3×3, p=1) → ReLU → Pool(2,2)` → 8×8  
   `Flatten(8192) → FC(512) → Dropout(0.5) → FC(100)`  
   3 conv+pool blocks for a 64×64 input is a reasonable starting point.

---

## 10. Summary {#summary}

### What We Learned

✅ **Max pooling**: Keep strongest activations, reduce spatial dimensions by half  
✅ **CNN architecture**: Conv→ReLU→Pool (feature extractor) → Flatten → FC (classifier)  
✅ **The funnel**: Channels grow, spatial dims shrink  
✅ **LeNet-5**: The classic CNN — 44K parameters, 99%+ on MNIST  
✅ **DataLoader**: Batch training on full datasets  
✅ **Data augmentation**: Random transforms to improve generalization  
✅ **Feature map visualization**: See what each filter detects  
✅ **Architecture experiments**: Deeper, wider, and smaller variants

### Key Insights

1. **CNNs beat MLPs on images by exploiting spatial structure:**
   - Fewer parameters through weight sharing
   - Translation invariance through convolutions + pooling
   - Hierarchical features through stacking

2. **What CNNs learn automatically:**
   - Layer 1: Edge detectors (just like our Sobel kernels from Session 10!)
   - Layer 2: Combinations of edges → corners, curves, textures
   - FC layers: Combine spatial features into class predictions

3. **Architecture matters, but not as much as you'd think:**
   - Even a tiny 1-conv-layer CNN beats an MLP
   - Going deeper helps, but with diminishing returns on simple tasks
   - Wider (more filters) can be as effective as deeper

### The Complete Pipeline (Sessions 1–11)

```
Session 2:  Perceptron        → a single neuron classifies
Session 4:  MLP               → hidden layers solve XOR
Session 5:  Gradient descent   → automatic weight learning
Session 6:  Backpropagation    → trains deep networks
Session 7:  Softmax + CCE      → multi-class classification
Session 8:  Regularization     → prevents overfitting
Session 9:  PyTorch            → frameworks automate everything
Session 10: Convolutions       → exploit spatial structure
Session 11: CNNs               → state-of-the-art image classification
```

From a single neuron that computes $y = \text{step}(wx + b)$ to a convolutional network that classifies handwritten digits at 99%+ accuracy — in 11 sessions!

### What's Next?

**Session 12: Final Project & Best Practices**

In the final session:
- **End-to-end ML pipeline**: From data loading to results analysis
- **Professional practices**: Code organization, experiment tracking, reproducibility
- **Final project**: Apply everything to a new challenge
  - Option A: Image classification on a new dataset
  - Option B: MLP vs CNN comparison study
  - Option C: Custom architecture design challenge

**The goal:** Demonstrate your complete understanding by building something on your own!

### Before Next Session

**Prepare your final project:**
1. Choose one of the three project options (or propose your own)
2. Think about your approach: What architecture? What hyperparameters? How will you evaluate?
3. Review Sessions 7–11: loss functions, regularization, training loop, CNN architecture
4. Bring questions!

---

**End of Session 11** 🎓

**You now understand:**
- ✅ How to design and build CNN architectures
- ✅ How to train on real datasets with DataLoader
- ✅ What CNNs learn and why they work

**Next up:** The Final Project — putting it all together! 🚀
