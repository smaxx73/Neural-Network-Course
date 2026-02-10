# Session 12 (2h): Final Project & Best Practices

## Learning Objectives

- Design and implement an end-to-end ML pipeline from raw data to evaluation
- Apply professional ML practices: reproducibility, experiment tracking, code organization
- Critically analyze model performance and justify architectural choices
- Present and defend technical work to peers

---

## Session Structure & Timing

| Block | Duration | Content |
|-------|----------|---------|
| 1. Pipeline Review & Live Demo | 25 min | End-to-end walkthrough |
| 2. Professional Practices | 15 min | Code organization, reproducibility |
| 3. Guided Project Work | 40 min | Students build their pipeline |
| 4. Presentations & Peer Review | 30 min | Show & tell, feedback |
| 5. Course Wrap-up | 10 min | What's next, resources |

---

## Block 1: End-to-End Pipeline Review (25 min)

### 1.1 The ML Pipeline as a Whole

Revisit the full pipeline students have built across the course, now seen as a unified system rather than isolated pieces:

```
Raw Data → Preprocessing → Model Definition → Training Loop → Evaluation → Analysis
```

Walk through each stage, referencing which session introduced it:

| Stage | Key Decisions | Session Reference |
|-------|---------------|-------------------|
| Data loading | Format, normalization, splits | S1 (NumPy), S8 (splits) |
| Preprocessing | Transforms, augmentation | S10–S11 (images) |
| Model definition | Architecture, activations | S2 (perceptron), S4 (MLP), S11 (CNN) |
| Loss & optimizer | Loss function, learning rate, optimizer | S5 (SGD), S7 (cross-entropy), S8 (Adam) |
| Training loop | Epochs, batching, validation | S6 (backprop), S9 (PyTorch) |
| Evaluation | Metrics, confusion matrix, visualization | S7 (metrics), S8 (learning curves) |

### 1.2 Live Coding Demo: Complete Pipeline

The instructor codes a clean, minimal pipeline on the **EuroSAT** dataset (satellite imagery, 10 land-use classes) in ~15 minutes, narrating each decision. Operational framing: *"You are classifying terrain type from satellite observation."*

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ── 1. Data ──────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3444, 0.3803, 0.4078],   # EuroSAT RGB means
                         std=[0.0924, 0.0650, 0.0539])
])

train_set = datasets.EuroSAT('./data', download=True, transform=transform)

# EuroSAT has no predefined split → we create train/val/test (70/15/15)
n = len(train_set)
n_train, n_val = int(0.70 * n), int(0.15 * n)
n_test = n - n_train - n_val
train_subset, val_subset, test_subset = torch.utils.data.random_split(
    train_set, [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(42)  # reproducibility!
)

train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_subset,   batch_size=256)
test_loader  = DataLoader(test_subset,  batch_size=256)

# Classes: AnnualCrop, Forest, HerbaceousVegetation, Highway,
#          Industrial, Pasture, PermanentCrop, Residential, River, SeaLake

# ── 2. Model ─────────────────────────────────────────────
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),   # 64x64 → 64x64 (RGB input)
            nn.ReLU(),
            nn.MaxPool2d(2),                   # 64x64 → 32x32
            nn.Conv2d(16, 32, 3, padding=1),  # 32x32 → 32x32
            nn.ReLU(),
            nn.MaxPool2d(2),                   # 32x32 → 16x16
            nn.Conv2d(32, 64, 3, padding=1),  # 16x16 → 16x16
            nn.ReLU(),
            nn.MaxPool2d(2),                   # 16x16 → 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = SimpleCNN()

# ── 3. Training setup ────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ── 4. Training loop with validation ─────────────────────
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X, y in loader:
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += X.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for X, y in loader:
        logits = model(X)
        loss = criterion(logits, y)
        total_loss += loss.item() * X.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += X.size(0)
    return total_loss / total, correct / total

history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

for epoch in range(10):
    t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer)
    v_loss, v_acc = evaluate(model, val_loader, criterion)
    history["train_loss"].append(t_loss)
    history["val_loss"].append(v_loss)
    history["train_acc"].append(t_acc)
    history["val_acc"].append(v_acc)
    print(f"Epoch {epoch+1:2d} | "
          f"Train loss {t_loss:.4f} acc {t_acc:.3f} | "
          f"Val loss {v_loss:.4f} acc {v_acc:.3f}")

# ── 5. Final evaluation on test set ─────────────────────
test_loss, test_acc = evaluate(model, test_loader, criterion)
print(f"\nTest accuracy: {test_acc:.3f}")

# Class names for analysis
classes = ['AnnualCrop', 'Forest', 'HerbaceousVeg', 'Highway', 'Industrial',
           'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
```

**Key teaching moments during the demo:**

- `Conv2d(3, 16, ...)` — 3 input channels because satellite images are RGB, unlike MNIST's single channel
- The three-way split (70/15/15) — EuroSAT has no predefined split, so we must create one ourselves
- The seed in `random_split` → reproducibility
- `model.train()` vs `model.eval()` → dropout behavior changes
- `@torch.no_grad()` → memory efficiency during evaluation
- Separate test set used **only once**, at the very end

---

## Block 2: Professional Practices (15 min)

### 2.1 Code Organization

Show a clean project structure students should aim for:

```
project/
├── data/                  # Raw and processed data (gitignored)
├── notebooks/
│   └── exploration.ipynb  # EDA and quick experiments
├── src/
│   ├── data.py            # Dataset loading, transforms, splits
│   ├── model.py           # Model architecture(s)
│   ├── train.py           # Training loop and evaluation
│   └── utils.py           # Plotting, metrics, helpers
├── results/
│   ├── figures/           # Saved plots
│   └── logs/              # Training logs
├── README.md
└── requirements.txt
```

**Rule of thumb:** if you copy-paste code between notebooks, extract it into `src/`.

### 2.2 Reproducibility Checklist

Present this as a checklist students can use for any future ML project:

- [ ] **Random seeds** fixed: `torch.manual_seed(42)`, `numpy.random.seed(42)`
- [ ] **Data splits** are deterministic (seeded) and documented
- [ ] **Dependencies** listed in `requirements.txt` (`pip freeze > requirements.txt`)
- [ ] **Hyperparameters** stored in a config dict or file, not scattered in code
- [ ] **Results** (metrics, plots) saved to disk, not just printed
- [ ] **Model checkpoints** saved for the best validation performance

```python
# Example: config dict at the top of your script
config = {
    "batch_size": 64,
    "lr": 1e-3,
    "epochs": 10,
    "model": "SimpleCNN",
    "optimizer": "Adam",
    "dropout": 0.3,
    "seed": 42,
}
```

### 2.3 Experiment Tracking (Lightweight)

For this course level, a simple approach: save a JSON log per experiment.

```python
import json
from datetime import datetime

def save_experiment(config, history, test_acc, path="results/logs"):
    record = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "history": history,
        "test_accuracy": test_acc,
    }
    filename = f"{path}/exp_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(filename, "w") as f:
        json.dump(record, f, indent=2)
```

Mention that in industry, tools like **Weights & Biases** or **MLflow** automate this, but the principle is the same: never lose track of what you ran and what it produced.

---

## Block 3: Guided Project Work (40 min)

### Project Brief

Students choose **one** of three options and build it during the session. Each option reuses code from previous sessions, so the focus is on **integration and comparison**, not writing from scratch.

#### Option A: Satellite Terrain Classification

**Dataset:** EuroSAT (10 classes: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake)

**Operational framing:** *You are building an automated terrain classification system from satellite imagery to support mission planning and situational awareness.*

**Task:**
1. Load and explore the dataset (class distribution, sample visualization per class)
2. Implement a CNN (can start from the demo, modify architecture)
3. Train with proper train/val/test split
4. Report: accuracy, confusion matrix, identify hardest classes
5. Visualize feature maps on a few test images — what spatial features does the network detect?

**Deliverable:** Notebook with results and a 2–3 sentence conclusion about which terrain types the model confuses and why (e.g., PermanentCrop vs AnnualCrop, Pasture vs HerbaceousVegetation).

#### Option B: MLP vs CNN on Satellite Imagery

**Dataset:** EuroSAT

**Operational framing:** *Does spatial structure in satellite images matter? Compare a model that ignores pixel layout (MLP) with one that exploits it (CNN).*

**Task:**
1. Implement an MLP (from Session 9 code, adapted for flattened 64×64×3 input) and a CNN (from Session 11 code)
2. Train both with **the same** hyperparameters where applicable (lr, epochs, optimizer)
3. Match parameter counts as closely as possible (document the counts)
4. Compare: accuracy, training speed, overfitting behavior
5. Plot learning curves side by side

**Deliverable:** Notebook with a comparison table and learning curve plots, plus a short written analysis (5–10 sentences) explaining why the CNN outperforms on spatially-structured data.

#### Option C: Architecture Ablation Study

**Dataset:** EuroSAT

**Operational framing:** *Your unit needs to deploy a terrain classifier with limited compute. Which architectural choices give the best accuracy-efficiency tradeoff?*

**Task:** Starting from a baseline CNN, systematically change **one thing at a time** and measure the effect:

| Experiment | Change |
|------------|--------|
| Baseline | 2 conv layers, ReLU, max pool, no dropout |
| +Dropout | Add dropout (0.3) |
| +BatchNorm | Add batch normalization |
| Deeper | Add a third conv layer |
| Wider | Double the number of filters |
| Sigmoid | Replace ReLU with sigmoid |

**Deliverable:** Table of results with accuracy and training time for each variant, plus a ranked list of which changes helped most.

### Starter Code & Scaffolding

Provide students a notebook template with:
- Data loading already done
- Empty functions to fill: `build_model()`, `train()`, `evaluate()`, `plot_results()`
- A results table template to fill in
- Markdown cells prompting them for written analysis

### Instructor Circulates

During the 40 minutes, the instructor moves between students/groups to:
- Debug common issues (shape mismatches, forgotten `.to(device)`, etc.)
- Ask probing questions: *"Why did you choose this architecture?"*, *"What does the confusion matrix tell you?"*
- Ensure students are documenting their choices

---

## Block 4: Presentations & Peer Review (30 min)

### Format

- Each student or pair presents for **3–4 minutes** (screen share their notebook)
- Structure: dataset → approach → key results → one surprise or lesson learned
- **2 minutes** of questions from peers and instructor

### Evaluation Rubric

Share this rubric before presentations so students know what's expected:

| Criterion | Excellent (5) | Good (3) | Needs Work (1) |
|-----------|---------------|----------|-----------------|
| **Pipeline completeness** | Full pipeline, proper splits, test used once | Missing one element | Major gaps |
| **Code quality** | Clean, documented, config separated | Mostly clean | Hard to follow |
| **Analysis depth** | Confusion matrix discussed, errors analyzed | Basic metrics reported | Only accuracy |
| **Presentation clarity** | Clear narrative, justified choices | Understandable | Disorganized |
| **Reproducibility** | Seeds set, config saved, results logged | Partially reproducible | Not reproducible |

### Peer Feedback

Each student writes **one** piece of constructive feedback for each presentation on a shared document or sticky notes:
- *"One thing I liked: ..."*
- *"One question or suggestion: ..."*

---

## Block 5: Course Wrap-up (10 min)

### What You've Learned: The Big Picture

Revisit the progression map from the syllabus and highlight what students can now do:

```
Session 1–2:  "A neuron computes a weighted sum and applies a function"
Session 3–4:  "Networks can learn their own weights"
Session 5–7:  "Calculus tells us HOW to update weights optimally"
Session 8–9:  "Good engineering prevents overfitting and speeds up work"
Session 10–12: "Specialized architectures exploit data structure"
```

**Key takeaway:** Every concept in this course builds on the one before it. You now understand the foundational ideas behind most modern deep learning.

### What's Next: Paths Forward

Briefly introduce what lies beyond this course, so students know where to go:

| Direction | Topics | Resources |
|-----------|--------|-----------|
| **Deeper architectures** | ResNets, transfer learning, fine-tuning | CS231n (Stanford), fastai course |
| **Sequences & language** | RNNs, LSTMs, Transformers, attention | CS224n (Stanford), "Attention Is All You Need" paper |
| **Generative models** | GANs, VAEs, diffusion models | Goodfellow et al. Chapter 20 |
| **Practical ML** | Data pipelines, deployment, MLOps | Full Stack Deep Learning course |
| **Theory** | Optimization theory, generalization bounds | "Understanding Deep Learning" (Prince, 2023) |

### Recommended Immediate Next Steps

1. **Redo one project from scratch** without looking at course code — this is the best test of understanding
2. **Read one paper** from the classics: LeNet (1998), AlexNet (2012), or ResNet (2015)
3. **Try transfer learning:** load a pretrained model and fine-tune it on a small custom dataset — this is how most real-world CV is done today

---

## Materials to Prepare

- [ ] EuroSAT dataset pre-downloaded on lab machines (or Colab-ready notebook with `datasets.EuroSAT(download=True)`)
- [ ] Starter notebook template with scaffolding for each project option
- [ ] Evaluation rubric (printed or shared digitally)
- [ ] Peer feedback forms
- [ ] "What's Next" resource links document for students to take home
