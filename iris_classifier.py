
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv")
X = df.drop("species", axis=1).values
y = df["species"].astype("category").cat.codes.values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = IrisNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

train_acc_list = []
test_acc_list = []

for epoch in range(50):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    _, predicted_train = torch.max(outputs, 1)
    train_acc = (predicted_train == y_train).sum().item() / y_train.size(0)

    with torch.no_grad():
        test_outputs = model(X_test)
        _, predicted_test = torch.max(test_outputs, 1)
        test_acc = (predicted_test == y_test).sum().item() / y_test.size(0)

    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)

    print(f"Epoch {epoch+1:2d}: Loss = {loss.item():.4f}, Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}")

plt.plot(train_acc_list, label="Train Accuracy")
plt.plot(test_acc_list, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy vs. Epoch")
plt.savefig("accuracy_plot.png")
plt.show()

print(f"\n✅ Final Test Accuracy: {test_acc_list[-1]*100:.2f}%")
