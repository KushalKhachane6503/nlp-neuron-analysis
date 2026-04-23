#  Neuron-Level Analysis of Transformer Models for Task-Specific Representation Discovery

##  Overview

This project presents **JACO (Joint Adaptive Computation and Optimization)** — a novel framework for understanding and exploiting **neuron-level structure in transformer models** such as BERT and RoBERTa.

Unlike traditional approaches that analyze attention heads or weights, this work focuses on **activation-level neuron behavior**, revealing how a small subset of neurons dynamically drives most of the model’s computation.

---

##  Novelty

This work introduces several **new and non-trivial insights** into transformer behavior:

###  1. Variance-Based Neuron Taxonomy

We propose a **data-driven, unsupervised classification of neurons** based purely on activation variance:

* **Core neurons** → stable, input-invariant
* **Collaborative neurons** → context-sensitive
* **Fragile neurons** → highly input-adaptive

 This is **probe-free**, generalizable, and works across models and tasks.

---

###  2. Emergent Neuron Routing (New Insight)

We show that transformers naturally exhibit **implicit routing behavior**:

* Only **~20% of neurons retain 80–98% of representational energy**
* Neuron selection is **input-dependent (RSS = 0.11–0.28)**

 This suggests transformers behave like **implicit mixture-of-experts systems**

---

###  3. Complexity-Adaptive Routing (CAR)

We introduce a **dynamic neuron selection mechanism**:

* Allocates **10–40% neurons per input**
* Improves retention by **+1% to +17% over fixed routing**

 First step toward **input-aware sparse inference at neuron level**

---

###  4. Role-Aware Training (NRRT / JACO Loss)

We design a **role-regularized objective function**:

* Preserves performance while reducing active neurons by **up to 83%**
* Encourages stable + efficient neuron utilization

---

### 🔬 5. New Fundamental Insight

> Core neurons are **3.7× more impactful** than fragile neurons

 Contrary to intuition:

* High-variance ≠ important
* Stability ≠ redundancy

---

###  6. Early Specialization Discovery

* **62.5% neurons specialize at epoch 0**

 Indicates:

> functional roles are largely determined during **pre-training**

---

##  Methodology

### Models

* BERT-base (110M parameters)
* RoBERTa-base (125M parameters)

### Datasets

* MNLI
* QQP
* MRPC
* AG News
* SST-2

### Pipeline

1. Extract hidden layer activations
2. Compute per-neuron variance
3. Classify neurons (Core / Collaborative / Fragile)
4. Apply routing (fixed + adaptive)
5. Evaluate using:

   * Retention (information preserved)
   * RSS (routing stability)
6. Perform masking + ablation experiments

---

##  Results

### 🔹 Efficiency vs Information

| Model   | Active Neurons | Retention |
| ------- | -------------- | --------- |
| BERT    | 20%            | ~81%      |
| RoBERTa | 20%            | ~98%      |

---

### 🔹 Key Observations

*  Routing is **input-dependent (low RSS)**

*  Activity concentrates in:

  * BERT → Layers 9–11
  * RoBERTa → Layers 8–9

*  Core neurons dominate:

  * 3.7× higher impact
  * largest embedding shifts when removed

*  CAR improves retention across all datasets

---

### 🔹 Ablation Highlights

* Variance-based routing vs random → **+49% retention**
* Stable across thresholds (10–30%)
* Layer importance:

  * Late layers outperform early layers significantly
* JACO loss matches baseline with **83% compute reduction**

---

##  Tech Stack

* Python
* PyTorch
* HuggingFace Transformers
* NumPy, Pandas
* Matplotlib / Seaborn

---

##  How to Run

```bash
pip install -r requirements.txt
```

```bash
jupyter notebook
```

---

##  Project Structure

```
notebooks/   → experiments & analysis  
results/     → plots, tables, outputs  
paper/       → ACL-style research paper  
```

---

##  Key Takeaways

* Transformer computation is **highly sparse and structured**
* Neuron roles are **not random — they are organized and meaningful**
* Significant compute savings are possible **without performance loss**
* Models exhibit **implicit routing behavior without explicit design**

---

##  Future Work

* Extend to large-scale models (LLaMA, GPT)
* Apply to real-time adaptive inference systems
* Explore neuron-level pruning and compression
* Build interpretability tools using neuron roles

---

##  Author

**Kushal Khachane**

IIT Bhilai
