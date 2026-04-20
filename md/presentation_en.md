---
marp: true
theme: default
paginate: true
header: "Progress Update: Span-Level Error Neurons"
footer: "Internal Discussion Only | 2026-04-15"
math: katex
---

# Neurons vs. Translation Errors 🤖
## Recent Progress & findings

- **Goal**: Find which "brain cells" (neurons) in xCOMET care about span errors.
- **Current status**: We've aligned the data and run the first regression.

---

### What's under the hood? 🛠️

1. **Extraction (`extract_hidden.py`)**:
   - Ripped out hidden states from `xCOMET-XL`.
   - Dimension: **2560** neurons per token.
   - Fixed a tricky alignment bug to make sure MT tokens match their neuron spikes.

2. **The Probe (`regression.py`)**:
   - Used a Logistic Regression to "sniff" for error patterns.
   - Formula: $\hat{P} = \sigma(\mathbf{w}^T \mathbf{x} + b)$
   - Weights $\mathbf{w}$ tell us who the "troublemakers" or "policemen" are.

---

### Quick Look: The Numbers 📊

Based on ~73k tokens (WMT ESA data):

- **Overall Accuracy**: 90%
- **Error Detection**:
  - We caught **92%** of errors (Recall).
  - But **57%** of our "error" guesses were false alarms (Precision 0.43).
- **Takeaway**: Neurons are very sensitive to error signals, but maybe *too* sensitive.

---

### Who are the Key Players? 🔍

Top performing neurons found so far:

| Rank | Neuron # | Weight | Role |
| :--- | :---: | :---: | :--- |
| 1 | **309** | 0.97 | 🚨 Error Detector (+) |
| 2 | **2208** | 0.65 | 🚨 Error Detector (+) |
| 3 | **1937** | -0.60 | ✅ Correctness Sentinel (-) |

- **Positive weights**: The neuron fires when it sees an error.
- **Negative weights**: The neuron fires when things look "good".

---

### Next Steps 🚀

- [ ] Check if these same 10 neurons work for other languages (Generalization).
- [ ] Try a "kill switch" (Ablation) — if we mute these neurons, does the model lose its mind?
- [ ] Clean up the test scripts.

---

# Chat / Feedback?
🙌
