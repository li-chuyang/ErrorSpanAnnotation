Title: Identifying Neurons that Control Span-Level Errors

### Abstract
Span-level error estimation models used in machine translation estimate translation quality by identifying errors at the span level. These models are often applied to unseen language pairs and domains at inference time, but there has been limited investigation into whether they generalize appropriately to such settings, or whether such generalization is even possible. In this study, we analyze these models at the neuron level and reveal the challenges involved in identifying span-level errors.

### Data
The dataset is sourced from the [WMT ErrorSpanAnnotation](https://github.com/wmt-conference/ErrorSpanAnnotation) project.

### Method
We employ **Linear Regression** (specifically Logistic Regression as a linear probe) to estimate the relationship between the labels of error spans and the internal representations $\mathbf{h}$ of the model:

$$ P(y=1 | \mathbf{h}) = \text{sigmoid}(\mathbf{w}^T \mathbf{h} + b) $$

Where:
- $\mathbf{h} \in \mathbb{R}^d$ is the hidden state vector extracted from the encoder.
- $y \in \{0, 1\}$ represents whether a token belongs to an error span.
- $\mathbf{w}$ represents the weights assigned to each neuron, indicating its sensitivity to errors.
