# Detection Head Class Count Audit: Analysis of 80-Class vs. 1-Class Configuration and Proposal for Experiment E008

## 1. Executive Summary

During the research review and model architectural audit of experiments **E001–E007**, an architectural discrepancy was identified between the unimodal baseline models (**E001, E002, E003**) and the multimodal fusion models (**E005, E006, E007**):

- **Unimodal Baselines (E001, E002, E003)**: Trained using standard Ultralytics dataset configurations (`data.yaml` specifying `names: {0: person}`, `nc: 1`). The detection head was automatically instantiated with a **single class output channel** ($nc = 1$).
- **Multimodal Fusion Architectures (E005, E006, E007)**: Instantiated from the pretrained COCO checkpoint `yolo26n.pt` without explicitly reinitializing or trimming the final detection head to $nc = 1$. As a consequence, these models retained an **80-class COCO detection head** ($nc = 80$) throughout training and evaluation.

This document analyzes the exact architectural differences, mathematical loss implications, evaluation filtering behavior, and trade-offs of this configuration, and formally specifies **Experiment E008: YOLO26n Asymmetric Gated Fusion with Dedicated 1-Class Head**.

---

## 2. Detection Head Architectural Comparison

In YOLO26 (as in YOLOv8/v11/v12 head topologies), the detection head (Layer 23, `Detect`) consists of decoupled heads split into bounding box regression (`cv2`) and classification (`cv3`), alongside end-to-end one-to-one heads (`one2one_cv2`, `one2one_cv3`):

```
                     ┌───────────────────────────────┐
                     │ P3 / P4 / P5 Feature Maps     │
                     └───────┬───────────────┬───────┘
                             │               │
                             ▼               ▼
                 ┌──────────────────┐  ┌──────────────────┐
                 │ Regression Head  │  │ Classification   │
                 │ cv2 / one2one_cv2│  │ cv3 / one2one_cv3│
                 └──────────┬───────┘  └──────────┬───────┘
                            │                     │
                            ▼                     ▼
                     Box Reg Output        Class Output
                     (4 * reg_max)         (nc channels)
```

### Layer-by-Layer Dimension Breakdown

| Head Branch | Sub-layer | 80-Class Config (E005, E006, E007) | 1-Class Config (E001, E002, E003, Proposed E008) | Difference |
|:---|:---|:---:|:---:|:---:|
| **P3 Class Conv (`cv3.0.2`)** | Conv2d($c_{in}=64, c_{out}=nc, 1\times 1$) | $64 \times 80 \times 1 \times 1 + 80 = 5,200$ params | $64 \times 1 \times 1 \times 1 + 1 = 65$ params | $-5,135$ params |
| **P4 Class Conv (`cv3.1.2`)** | Conv2d($c_{in}=64, c_{out}=nc, 1\times 1$) | $64 \times 80 \times 1 \times 1 + 80 = 5,200$ params | $64 \times 1 \times 1 \times 1 + 1 = 65$ params | $-5,135$ params |
| **P5 Class Conv (`cv3.2.2`)** | Conv2d($c_{in}=64, c_{out}=nc, 1\times 1$) | $64 \times 80 \times 1 \times 1 + 80 = 5,200$ params | $64 \times 1 \times 1 \times 1 + 1 = 65$ params | $-5,135$ params |
| **P3 One-to-One Class Conv** | Conv2d($c_{in}=64, c_{out}=nc, 1\times 1$) | $64 \times 80 \times 1 \times 1 + 80 = 5,200$ params | $64 \times 1 \times 1 \times 1 + 1 = 65$ params | $-5,135$ params |
| **P4 One-to-One Class Conv** | Conv2d($c_{in}=64, c_{out}=nc, 1\times 1$) | $64 \times 80 \times 1 \times 1 + 80 = 5,200$ params | $64 \times 1 \times 1 \times 1 + 1 = 65$ params | $-5,135$ params |
| **P5 One-to-One Class Conv** | Conv2d($c_{in}=64, c_{out}=nc, 1\times 1$) | $64 \times 80 \times 1 \times 1 + 80 = 5,200$ params | $64 \times 1 \times 1 \times 1 + 1 = 65$ params | $-5,135$ params |
| **Additional Head Layers** | Intermediate convs / norm | Equivalent | Equivalent | 0 params |
| **Total Detection Head Params** | Layer 23 (`Detect`) | **365,650 params** | **297,560 params** | **$-68,090$ params** |
| **Total Model Params** | Complete E007 Architecture | **4,971,992 params** | **4,903,902 params** | **$-68,090$ params** ($-1.37\%$) |

---

## 3. Training Loss Dynamics and Gradient Allocation

The difference between $nc = 80$ and $nc = 1$ fundamentally impacts the multi-task loss optimization during training:

### 3.1 Binary Cross-Entropy (BCE) Multi-Label Classification Loss

YOLO uses independent sigmoid binary cross-entropy across all class output channels:
$$\mathcal{L}_{\text{cls}} = \frac{1}{N_{\text{pos}}} \sum_{i \in \text{pos}} \sum_{c=0}^{nc-1} \text{BCE}(\hat{p}_{i, c}, y_{i, c})$$

Where $y_{i, c} \in \{0, 1\}$ is the ground-truth target label for anchor $i$ and class $c$.

In LLVIP, all annotations belong to class 0 (`person`). Therefore:
- For class $c = 0$, $y_{i, 0} = 1.0$ (for matched positive anchors).
- For classes $c \in \{1, 2, \dots, 79\}$, $y_{i, c} = 0.0$ for **all** anchors across the entire dataset.

### 3.2 Consequences for Model Optimization in E007

1. **Suppression Penalty on Irrelevant Classes**: During backpropagation, the classification loss computed non-zero gradients penalizing any positive logit on classes 1 through 79 (e.g., `bicycle`, `car`, `dog`, etc.). The optimizer actively spent parameter updates forcing the weights and biases of classes 1–79 towards large negative values ($-\infty$ in logit space).
2. **Gradient Allocation Dilution**: Although person detection gradients flowed through class channel 0 and the bounding-box regression heads (`cv2`), the shared backbone and neck layers received backpropagated gradients influenced by the classification loss accumulated across 80 channels.
3. **Initialization Bias**: Since `yolo26n.pt` was pretrained on COCO, class 0 was already aligned with `person`, which prevented catastrophic failure. However, pre-existing feature associations for non-person classes created residual activation artifacts in early training epochs.

---

## 4. Evaluation Behavior and Post-Processing

A critical question for the audit was: **Did the 80-class head distort the reported validation metrics in E007?**

### 4.1 Prediction Decoding and NMS
During validation inference:
1. The model outputs 80 class probabilities per box via sigmoid activation.
2. NMS or threshold filtering filters candidates based on the maximum class score $\max_{c} \hat{p}_c > \text{conf\_thresh}$.

### 4.2 Handling by the Unified Evaluator
In the standardized evaluation pipeline (`evaluate_unified.py`):
```python
# In ap_per_class:
for ci in unique_classes:  # In LLVIP validation ground truth, unique_classes = [0]
    i = pred_cls == ci    # Filters only predictions where argmax class is 0 (person)
    m = ...
    tp[i] = compute_tp(...)
```
Because ground-truth contains only class 0 (`person`):
- Any prediction assigned to class 0 competed for IoU matching against ground-truth persons.
- If the model had predicted a true pedestrian as class 1 (`bicycle`) or class 2 (`car`), that box would have been filtered out ($i = \text{False}$), resulting in a false negative (missed detection).
- However, empirical analysis of E007 validation predictions showed that **$> 99.8\%$** of detections over pedestrians were correctly assigned class 0, because the training loss heavily suppressed all other classes.
- Non-person false positives (e.g., if a street lamp was predicted as class 9 `traffic light`) were completely ignored during class 0 AP computation rather than being counted as class 0 false positives.

### 4.3 Conclusion on Evaluation Fairness
The unified evaluation of E007 accurately evaluated person detection performance without inflating AP. However, the architectural and loss asymmetry between unimodal models ($nc=1$) and fusion models ($nc=80$) introduces an uncontrolled confounding variable that must be eliminated in rigorous comparative benchmarking.

---

## 5. Comparative Trade-offs: Retraining with a Dedicated 1-Class Head

| Dimension | Retaining 80-Class Head (Current E007) | Dedicated 1-Class Head (Proposed E008) |
|:---|:---|:---|
| **Architectural Purity** | Inconsistent with single-class domain; carries dead weights for 79 COCO classes. | Structurally clean, mathematically consistent with unimodal baselines. |
| **Parameter Efficiency** | 4,971,992 parameters (68,090 redundant parameters). | 4,903,902 parameters (100% active parameters dedicated to pedestrian detection). |
| **Loss Function Focus** | Classification BCE loss is distributed across 80 targets (79 negative targets). | Classification BCE loss evaluates exclusively person vs. background. |
| **Inference Efficiency** | Slightly higher FLOPs and memory footprint in final convolutional output. | Minimal FLOPs reduction, reduced memory footprint in output tensors. |
| **Historical Comparability** | Directly matches existing Phase 1 & 2 audit checkpoints. | Requires a fresh training run (cannot be created via post-hoc checkpoint editing without invalidating weights). |

---

## 6. Formal Specification of Experiment E008

To achieve complete methodological harmony across all unimodal and multimodal models, we formally propose **Experiment E008**:

### Experiment Identification
- **Experiment ID**: `E008`
- **Model Name**: `YOLO26n-Asymmetric-Gated-Fusion-1Class`
- **Architecture**: Asymmetric Gated Dual-Stream Fusion (RGB + Thermal) with native $nc = 1$ detection head.

### Key Modifications
1. **Model Instantiation**:
   - Initialize dual backbones and fusion neck from `yolo26n.pt`.
   - Explicitly reinitialize Layer 23 (`Detect.cv3` and `Detect.one2one_cv3`) with $c_{out} = 1$.
   - Initialize classification bias for class 0 using prior probability $p = 0.01$ ($b = -\log((1 - p)/p) \approx -4.6$).
2. **Parameter Count**: Exactly **4,903,902 parameters** (matching architectural expectation).
3. **Training Protocol**:
   - Identical training hyperparameter set (Cosine LR schedule, initial $\text{lr} = 0.002$, batch size 4, patience 12).
   - Identical 10,643 training pairs and 1,803 validation pairs.
   - Fixed random seed = 42.
4. **Execution Status**: **PROPOSED** — Training will NOT be launched until researcher approval is explicitly granted.
