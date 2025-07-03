
# 🧠 Mini Report

## 🔍 ML Classification (Iris Dataset)

**Model Architecture**:
- Input: 4 features
- Hidden Layer: 10 neurons, ReLU
- Output: 3 neurons (softmax via CrossEntropy)

**Hyperparameters**:
- Optimizer: SGD (lr=0.01)
- Epochs: 50

**Final Accuracy**:
- Train: ~100%
- Test: ~96%

**Interpretation**: Model learned well and generalized effectively on the Iris dataset.

## 🤖 GenAI: GPT-2 Experiment

**Prompt**: "Once upon a time"  
**Temperatures**: 0.7 and 1.0

| Temp | Output Summary | Notes |
|------|----------------|-------|
| 0.7  | Logical, story-like | Coherent, less creative |
| 1.0  | More varied, surprising | Creative, but risk of incoherence |

## 💡 Key Learnings

- Learned training loop logic in PyTorch (forward, backward, update)
- Learned how temperature impacts generation creativity
- Found manual accuracy tracking insightful
