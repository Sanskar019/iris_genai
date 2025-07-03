# Iris Classification + GenAI Project

## 🧠 ML Classifier (Iris Dataset)
- Model: 2-layer PyTorch NN (4 -> 10 -> 3)
- Optimizer: SGD, Loss: CrossEntropy
- Train/Test Split: 80/20, Normalized
- Epochs: 50, Learning Rate: 0.01

## 🤖 GenAI with GPT-2
- Top-k Sampling (k=50)
- Temperatures: 0.7 and 1.0
- Model: GPT-2 (small) from Hugging Face

## 📦 Requirements
```
pip install torch pandas numpy matplotlib scikit-learn transformers
```

## ▶️ How to Run
```bash
python iris_classifier.py
python generate.py
```
# iris_genai
