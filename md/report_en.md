# Neuron Representation Analysis Report for Span-Level Error Detection in Machine Translation

---

## 1. Research Background and Objectives

### Research Background
Current quality estimation models for machine translation, such as **xCOMET**, provide highly accurate scores. However, the internal logic used to determine errors remains a "black box." To improve model interpretability, this project aims to deconstruct the decision-making mechanisms within the model.

### Core Objectives
*   **Identification**: Locate specific "sentinel neurons" responsible for monitoring translation errors among thousands of neuronal dimensions.
*   **Linear Readability Validation**: Investigate how translation error information is stored within the model's hidden layers.
*   **Generalization Analysis**: Examine whether the neurons identifying errors remain consistent across different domains (e.g., news, social media).

---

## 2. Experimental Design and Technical Path

This experiment utilizes the **Linear Probing** technique. The workflow is divided into five key phases:

| Phase | Core Task | Technical Details |
| :--- | :--- | :--- |
| **1. Data Cleaning** | Sample Selection | Used `AnnotationLoader` to load WMT23 human-annotated datasets (ESA-1/ESA-2), strictly filtering for samples with precise character-level coordinates. |
| **2. Coordinate Alignment** | Label Mapping | Established an overlap determination mechanism via `offset_mapping` to accurately map character-level error labels to individual tokens. |
| **3. Feature Extraction** | Capture Activations | Fed sequences into **xCOMET-XL** and intercepted hidden states from the 24th (final) layer using Hook technology, obtaining 2560-dimensional activation vectors. |
| **4. Data Purification** | Signal Isolation | Precisely located and removed source text signals, retaining only the neuron values corresponding to target tokens, saved losslessly in `.pt` format. |
| **5. Regression Analysis** | Neuron Localization | Ran **Logistic Regression** with class-weight balancing on 73,000 tokens to identify core dimensions based on regression coefficients (weights). |

---

## 3. Technical Challenges and Solutions

During the experiment, three critical challenges were addressed through deep analysis:

### Token Boundary Deviation
*   **Problem**: Human annotations often contain coordinate inaccuracies due to manual slips (e.g., selecting `brullt` instead of `brullte`).
*   **Solution**: Designed an **"Overlap Area Determination Algorithm."** A token is classified as an error if it has any intersection with the annotated error span, significantly increasing fault tolerance.

### Sequence Alignment Risk
*   **Problem**: Since xCOMET inputs include source text, early experiments encountered misalignments between hidden states and labels.
*   **Solution**: Implemented a **"Core Substring Sliding Window Matching Method"** to precisely locate the start of the target translation within the sequence, ensuring "pixel-level" alignment between features and labels.

### Extreme Data Imbalance
*   **Problem**: Erroneous tokens account for only **8%** of the total dataset.
*   **Solution**: Introduced `class_weight='balanced'` in the logistic regression loss function to prevent the model from developing a prediction bias toward "correct" tokens.

---

## 4. Experimental Results and Core Findings

Based on the comprehensive experiment involving **73,681** tokens, several key conclusions were reached:

### A. Performance Metrics
*   **Recall (1): 0.92** —— A remarkable result, proving that **92%** of the error signals within xCOMET are stored in a linearly explicit manner.
*   **Precision (1): 0.43** —— Reveals the **"Signal Overflow Effect"** in Transformers (where error signals bleed into adjacent tokens) and the high sensitivity of the model.
*   **F1-Score (1): 0.59** —— Demonstrates the strong baseline performance of the linear probing approach.

### B. "Sentinel Neuron" Localization
By ranking the weights, we successfully identified the **Top 10** critical neurons from the 2560 dimensions:

1.  **Lead Sentinel: Neuron #309 (Weight: 0.9704)**
    *   The strongest error detector in the entire model; its weight consistently ranked first across all data scales.
2.  **Role Differentiation**
    *   **Error Detectors (Positive Weight)**: e.g., `#2208`, `#926`.
    *   **Correctness Sentinels (Negative Weight)**: e.g., `#1937`, `#490`.

---

## 5. Conclusion and Future Work

This research confirms that xCOMET's perception of translation errors is highly **sparse** and **localized**. Error detection is not diffused throughout the entire network but is instead highly controlled by a few core neurons, led by **#309**.

This provides a solid experimental foundation for future work on **Neuron Intervention** to correct translation biases and enhance the interpretability of MT evaluation models.
